function [S,L,tau, diagnostics] = ...
    solver_RASL_UOT_inner_ADMM(Y, lambda, gamma, kappa, mu, imsize, J, S, L, opts)
% Solve the RASL inner loop with nonnegativity constraints on S, L
% min  (1/2)*sum_i { sum_square(y_i + Ji*tau_i - s_i - l_i) }
%      + lambda*||S(:)||_1
%      + gamma*||L||_*
%      + kappa*sum_i^(T-1) { uot(s_i, s_(i+1)) }
% subject to S, L >= 0

%% Initialization
rho = 0.1;   % ADMM step size
max_iters = 1000;
tolerance = 1e-1;
print_every = 0;
show_debug_plots = true;
beckopts.tau1 = 0.1;
beckopts.tau2 = 1.0;
beckopts.maxiter = 1;

% beta = 1e-2; % \ell_2 regularizer on tau

if exist('opts', 'var')
if isfield(opts, 'rho'), rho = opts.rho; end
if isfield(opts, 'max_iters_inner'), max_iters = opts.max_iters_inner; end
if isfield(opts, 'tolerance_inner'), tolerance = opts.tolerance; end
if isfield(opts, 'print_every_inner'), print_every = opts.print_every_inner; end
if isfield(opts,'show_debug_plots_inner'), show_debug_plots = opts.show_debug_plots_inner; end
if isfield(opts,'beck_tau1'), beckopts.tau1 = opts.beck_tau1; end
if isfield(opts,'beck_tau2'), beckopts.tau2 = opts.beck_tau2; end
if isfield(opts,'beck_maxiter'), beckopts.maxiter = opts.beck_maxiter; end

% if isfield(opts, 'beta'), beta = opts.beta; end
end

[N, K] = size(Y);
P = size(J{1}, 2);

% Initialization
Div = GenerateDivergenceMatrices(imsize);
f = ones(K,1)*(1+2*rho); f(1) = 1+rho; f(end) = 1+rho;
S_prox_threshold = lambda*repmat(1./f', N, 1);
idx1 = 1:K-1;
idx2 = 2:K;
idxZ = 1:N;
idxW = N+1:2*N;

J_decomp = cell(K, 1);
for k = 1:K
    J_decomp{k} = decomposition(J{k});
%     J_decomp{k} = decomposition(J{k}'*J{k} + beta*eye(P));
end

tau = zeros(P, K);
T = L;
A = zeros(N, K);
B = zeros(N,K-1);
C = zeros(N,K-1);
JTau = zeros(N, K);
ZW = zeros(2*N,K-1); % auxiliary variable

M = complex(zeros(N,K-1));
R = zeros(N,K-1);
D = zeros(N,K-1);

residual = nan(max_iters, 2);

%% ADMM loop
for ii = 1:max_iters
    prevTZW = [T(:); ZW(:)];
    
    % Solve for S
    F = Y + JTau - L;
    F(:,idx1) = F(:,idx1) + rho*(ZW(idxZ,:) + B);
    F(:,idx2) = F(:,idx2) + rho*(ZW(idxW,:) + C);
    F = bsxfun(@rdivide, F, f');
    S = Prox_NonNeg_l1(F, S_prox_threshold);

    % Solve for L
    L = max(0, (1/(1+rho)) * ( rho*(T-A) + (Y+JTau-S) ) );
    
    % Solve for T
    T = Prox_NuclearNorm(L+A, gamma/rho);
    
    % Solve for tau
    JTau = -Y+S+L;
    for k = 1:K
        tau(:, k) = J_decomp{k}\JTau(:, k);
        JTau(:, k) = J{k}*tau(:, k);
    end
    
    % solve for Z and W
    [ZW,M,R,D] = ...
        Prox_Beckman([S(:,idx1)-B;S(:,idx2)-C],mu,rho/kappa,Div,ZW,M,R,D,beckopts);
    
    % gradient ascent on dual
    A = A + L-T;
    B = B + ZW(idxZ,:)-S(:,idx1);
    C = C + ZW(idxW,:)-S(:,idx2);
    
    residual(ii, 1) = norm([vec(L-T);...
                            vec(S(:,idx1)-ZW(idxZ,:));...
                            vec(S(:,idx2)-ZW(idxW,:))]);
    residual(ii, 2) = rho*norm([vec(T);vec(ZW)]-prevTZW);
    
    if ~mod(ii, print_every)
        fprintf('RASL inner loop iteration %u: res1=%g, res2=%g\n',...
            ii, residual(ii, 1), residual(ii, 2));
        if show_debug_plots
            subplot(221); imdebug(L, imsize, 'L');
            subplot(222); imdebug(S, imsize, 'S');
            subplot(223); imdebug(R, imsize, 'R');
            subplot(224); semilogy(residual); title('residual');
            pause()
        end
    end

    if residual(ii,1) < tolerance && residual(ii,2) < tolerance, break; end
end

diagnostics.residual = residual(1:k, :);

end


%% Auxiliary functions

function S = Prox_NonNeg_l1(A,rho)
S = max(0,A-rho);
end

function r = Prox_L1(r,rho)
r = sign(r).*max(0,abs(r)-rho);
end

function L = Prox_NuclearNorm(A,rho)
prox_l1 = @(x) max(0,x-rho) - max(0,-x-rho);
[U,S,V] = svd(A,'econ');
L = U*diag(prox_l1(diag(S)))*V';
end

function m = Prox_L21(m,rho)
abs_m = abs(m);
m = (1 - rho./abs_m).*m;
m(abs_m<rho) = 0;
end

function [x,m,r,d] = Prox_Beckman(y,mu,rho,D_op,x,m,r,d,opts)
tau1    = opts.tau1;
tau2    = opts.tau2;
maxiter = opts.maxiter;
A_op = @(x) x(1:end/2,:) - x(end/2+1:end,:);
At_op = @(z) [z;-z];
K_op = @(m,x,r) real(conj(D_op)*m) + A_op(x) - r;
K_op_mxr = K_op(m,x,r);
for k = 1:maxiter
    prevm = m; prevx = x; prevr = r; prev_K_op_mxr = K_op_mxr;
    % Solve M (L2 norm shinkage)
    m = Prox_L21(prevm-tau1*conj(D_op')*d,tau1);
    % Solve x (Standard least-squares)
    x = (rho*tau1)/(1+rho*tau1)*y + 1/(1+rho*tau1)*(prevx-tau1*At_op(d));
    % Solve r (L1 norm shinkage)
    r = Prox_L1(prevr+tau1*d, mu*tau1);
    % solve d (over-relaxation)
    K_op_mxr = K_op(m,x,r);
    d = d + tau2*( 2*K_op_mxr - prev_K_op_mxr );
end
end

function D = GenerateDivergenceMatrices(imsize)
Dx = speye(imsize(1)*imsize(2)) - circshift(speye(imsize(1)*imsize(2)),1);
Dx(1,imsize(1)*imsize(2)) = 0;
Dx(:,imsize(1):imsize(1):imsize(1)*imsize(2)) = 0;
Dy = speye(imsize(1)*imsize(2)) - circshift(speye(imsize(1)*imsize(2)),imsize(1));
Dy(:,end-imsize(1)+1:end) = 0;
D = Dx + 1i*Dy;
end

function imdebug(img, imsize, title_str)
imagesc(reshape(img(:, 1),imsize));
axis image;
colormap gray;
colorbar
title(title_str);
end
