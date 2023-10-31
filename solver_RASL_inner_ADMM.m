function [S,L,tau, diagnostics] = ...
    solver_RASL_inner_ADMM(Y, lambda, gamma, J, S, L, opts)
% Solve the RASL inner loop with nonnegativity constraints on S, L
% min  (1/2)*sum_i { sum_square(y_i + Ji*tau_i - s_i - l_i) }
%      + lambda*||S(:)||_1
%      + gamma*||L||_*
% subject to S, L >= 0

%% Initialization
rho = 0.1;   % ADMM step size
max_iters = 1000;
tolerance = 1e-1;
print_every = 0;
show_debug_plots = true;

% beta = 1e-2; % \ell_2 regularizer on tau

if exist('opts', 'var')
if isfield(opts, 'rho'), rho = opts.rho; end
if isfield(opts, 'max_iters_inner'), max_iters = opts.max_iters_inner; end
if isfield(opts, 'tolerance_inner'), tolerance = opts.tolerance_inner; end
if isfield(opts, 'print_every_inner'), print_every = opts.print_every_inner; end
if isfield(opts, 'show_debug_plots_inner'), show_debug_plots = opts.show_debug_plots_inner; end
end

[N, K] = size(Y);
P = size(J{1}, 2);

J_decomp = cell(K, 1);
for k = 1:K
    J_decomp{k} = decomposition(J{k});
%     J_decomp{k} = decomposition(J{k}'*J{k} + beta*eye(P));  % Tik reg
end

% S = zeros(N, K);
% L = zeros(N, K);
tau = zeros(P, K);
% T = zeros(N, K);
T = L;
A = zeros(N, K);
D = zeros(N, K);
residual = nan(max_iters, 2);

%% ADMM loop
for ii = 1:max_iters
    prevT = T(:);

    T = Prox_NuclearNorm(L+A, gamma/rho);
    L = max(0, (1/(1+rho)) * ( rho*(T-A) + (Y+D-S) ) );
    S = Prox_NonNeg_l1(Y+D-L, lambda);
    D = -Y+S+L;
    for k = 1:K
        tau(:, k) = J_decomp{k}\D(:, k);
        D(:, k) = J{k}*tau(:, k);
    end
    
    A = A + L-T;
    
    residual(ii, 1) = norm(L(:)-T(:));
    residual(ii, 2) = rho*norm(T(:)-prevT);
    
    if ~mod(ii, print_every)
        fprintf('RASL inner loop iteration %u: res1=%g, res2=%g\n',...
            ii, residual(ii, 1), residual(ii, 2));
        if show_debug_plots
            subplot(221); imdebug(L, 'L', opts.roi.size);
            subplot(222); imdebug(S, 'S', opts.roi.size);
            subplot(224); semilogy(residual); title('residual');
            pause()
        end
    end

    if residual(ii,1) < tolerance && residual(ii,2) < tolerance, break; end
end

diagnostics.residual = residual(1:k, :);
diagnostics.iters = k;

end


%% Auxiliary functions

function S = Prox_NonNeg_l1(A,rho)
S = max(0,A-rho);
end

function L = Prox_NuclearNorm(A,rho)
prox_l1 = @(x) max(0,x-rho) - max(0,-x-rho);
[U,S,V] = svd(A,'econ');
L = U*diag(prox_l1(diag(S)))*V';
end

function imdebug(img, title_str, imsize)
imagesc(reshape(img(:, 1),imsize));
axis image;
colormap gray;
colorbar
title(title_str);
end
