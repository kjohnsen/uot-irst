function [S,L,tau] = solver_RASL(Y, lambda, gamma, opts)
% Robust alignment by sparse and low-rank decomposition (RASL) solver. Supports
% original and EMD regularized formulations.
%
% Inputs
%   Y         cell array with non-vectorized images
%   lambda    sparsity penalty weight
%   gamma     rank penalty weight
%   opts
%     .max_iters_outer           max iterations on outer loop
%     .tolerance_outer           tolerance on tau for stopping criterion
%     .tform_type                transform type ('affine', 'translation',
%                                'rotation')
%     .roi_padding               width of ROI padding
%     .reference_frame           align to this frame at the end
%     .print_every_outer         show debugging info every x iterations
%     .show_debug_plots_outer    show plots in debugging output
%     .use_ot                    use EMD/UOT regularization
%     .kappa                     EMD regularization weight
%     .mu                        mass growth/decay weight
%     .rho                       ADMM step size
%     See documentation in inner loop solver for following options
%     .max_iters_inner
%     .tolerance_inner
%     .print_every_inner
%     .show_debug_plots_inner
%     .beckopts.tau1
%     .beckopts.tau2
%     .beckopts.maxiter
%
% Outputs
%   S      sparse matrix
%   L      low rank matrix
%   tau    transform parameters

img_size = size(Y{1});

% default parameters
max_iters_outer = 50;
tolerance_outer = 1e-4;
tform_type = 'rotation';
roi_padding = 10;
reference_frame = 1;
print_every_outer = 0;
show_debug_plots_outer = false;
use_ot = false;
kappa = 0.1;
mu = 5;

% set options
if exist('opts', 'var')
if isfield(opts, 'max_iters_outer'), max_iters_outer = opts.max_iters_outer; end
if isfield(opts, 'tform_type'), tform_type = opts.tform_type; end
if isfield(opts, 'roi_padding'), roi_padding = opts.roi_padding; end
if isfield(opts, 'tolerance_outer'), tolerance_outer = opts.tolerance_outer; end
if isfield(opts, 'print_every_outer'), print_every_outer = opts.print_every_outer; end
if isfield(opts, 'use_ot'), use_ot = opts.use_ot; end
if isfield(opts, 'kappa'), kappa = opts.kappa; end
if isfield(opts, 'mu'), mu = opts.mu; end
end

% start and end indices of region of interest
roi.x = [roi_padding+1 img_size(2)-roi_padding];
roi.y = [roi_padding+1 img_size(1)-roi_padding];

% initialization
roi.size = [diff(roi.y) diff(roi.x)] + 1;
opts.roi = roi;
RA = imref2d(roi.size, roi.x+[-0.5, 0.5], roi.y+[-0.5, 0.5]);
[u, v] = meshgrid(roi.x(1):roi.x(2), roi.y(1):roi.y(2));
u = u(:); v = v(:);
K = numel(Y);
N = prod(roi.size);
tform = get_transform(tform_type);

tau = repmat(tform.init, 1, K);
I = zeros(N, K);
Ix = cell(K, 1);
Iy = cell(K, 1);
Iu = cell(K, 1);
Iv = cell(K, 1);
J = cell(K, 1);
S = zeros(N, K);
L = zeros(N, K);
Q = cell(K,1);
R = cell(K,1);
% compute gradients
for k = 1:K
    [Ix{k}, Iy{k}] = imgradientxy(Y{k}, 'intermediate');
end

% RASL main loop
for t = 1:max_iters_outer
    prev_tau = tau;
    
    if ~mod(t, print_every_outer)
        fprintf('Executing RASL outer loop iteration %u...\n', t);
    end
    
    % compute transformed images and Jacobians
    for k = 1:K
        T_mtx = tform.projective_mtx(tau(:, k));
        T_proj2d = projective2d(inv(T_mtx)');
        I(:,k) = vec(imwarp(Y{k}, T_proj2d, 'OutputView', RA));
        Iu{k} = vec(imwarp(Ix{k}, T_proj2d, 'OutputView', RA));
        Iv{k} = vec(imwarp(Iy{k}, T_proj2d, 'OutputView', RA));

        J{k} = tform.img_jacobian(tau(:,k), Iu{k}, Iv{k}, u, v);
        [Q{k}, R{k}] = qr(J{k},0);
    end
    
    % inner loop
    if use_ot
        [S,L,d_tau] = solver_RASL_UOT_inner_ADMM(I, lambda, gamma, ...
                                                 kappa, mu, roi.size, ...
                                                 Q, S, L, opts);
    else
        [S,L,d_tau,diagnostics] = solver_RASL_inner_ADMM(I, lambda, gamma, Q, S, L, opts);
    end
    
    for k = 1:K
        d_tau(:, k) = R{k}\d_tau(:,k);
    end
    tau = tau + d_tau;
    
    % subtract mean to prevent all images from unnecessary shifting
    tau = tform.rm_global_phase(tau);
    
    % debug output
    if ~mod(t, print_every_outer)
        if show_debug_plots_outer  
            clf(1);
            plot_ind = 1;
            subplot(211); imagesc(reshape(S(:,plot_ind),roi.size)); axis image; colormap gray;
            subplot(212); imagesc(reshape(L(:,plot_ind),roi.size)); axis image; colormap gray;
            rmse = norm(I-S-L,'fro')/norm(I,'fro')
        end

        % compute objective function value
        obj_val_ell2 = 0;
        for k = 1:K
            obj_val_ell2 = obj_val_ell2 + 0.5*norm(I(:,k)+J{k}*d_tau(:,k)-S(:,k)-L(:,k), 'fro')^2;
        end
        obj_val_ell1 = lambda*norm(S(:), 1);
        obj_val_nuc = gamma*sum(svd(L,'econ'));
        obj_val = obj_val_ell2 + obj_val_ell1 + obj_val_nuc;
        fprintf('Objective,ell1,ell2,nuclear=%g, %g, %g, %g\n',...
                obj_val, obj_val_ell1, obj_val_ell2, obj_val_nuc);
    end
    
    residual = norm(prev_tau-tau, 'fro');
    if residual < tolerance_outer, break; end
end

% normalize transformations to reference frame and do final solve
tau = normalize_affine_kth(tau, reference_frame);
for k = 1:K
    T_mtx = tform.projective_mtx(tau(:, k));
    T_proj2d = projective2d(inv(T_mtx)');
    I(:,k) = vec(imwarp(Y{k}, T_proj2d, 'OutputView', RA));
end
if use_ot
    [S,L,~] = solver_RPCA_UOT_Beckman_ADMM(roi.size,I,{speye(size(I,1))},lambda,gamma,kappa,mu,opts)
else
    [S,L]=solver_RPCA_denoising_ADMM(I,lambda,gamma,opts);
end
end % end main function

% aux functions
function J = img_jacobian_rotation(tau, Iu, Iv, u, v)
J_theta = Iu.*(-sin(tau(1)).*u - cos(tau(1)).*v) + ...
          Iv.*(cos(tau(1)).*u - sin(tau(1)).*v);
J = J_theta(:);
end

function J = img_jacobian_translation(tau, Iu, Iv, u, v)
J = [Iu(:), Iv(:)];
end

function J = img_jacobian_affine(tau, Iu, Iv, u, v)
J = [vec(Iu.*u), vec(Iu.*v), vec(Iu), ...
     vec(Iv.*u), vec(Iv.*v), vec(Iv)];
end

function T = projective_mtx_rotation(tau)
T = [cos(tau), -sin(tau), 0; sin(tau), cos(tau) 0;  0 0 1];
end

function T = projective_mtx_translation(tau)
T = [1 0 tau(1); 0 1 tau(2); 0 0 1];
end

function tau = subtract_param_mean(tau)
tau = tau - mean(tau, 2);
end

function T = get_transform(tform_type)
switch tform_type
    case 'rotation'
        T.P = 1;
        T.img_jacobian = @img_jacobian_rotation;
        T.projective_mtx = @projective_mtx_rotation;
        T.rm_global_phase = @subtract_param_mean;
        T.init = 0;
    case 'translation'
        T.P = 2;
        T.img_jacobian = @img_jacobian_translation;
        T.projective_mtx = @projective_mtx_translation;
        T.rm_global_phase = @subtract_param_mean;
        T.init = [0 0]';
    case 'affine'
        T.P = 6;
        T.img_jacobian = @img_jacobian_affine;
        T.projective_mtx = @projective_mtx_affine;
        T.rm_global_phase = @normalize_affine_mean;
        T.init = [1 0 0 0 1 0]';
    otherwise
        error('Unsupported transform.');
end
end

function x = vec(x)
    x = reshape(x, [], 1);
end
