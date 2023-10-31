% Demo solver for low-rank plus sparse decomposition with optimal transport
% (EMD) regularization and spatial affine transformation

clc; clear;
rng(1);  % for reproducibility

%% generate data matrix
% set problem parameters
ds = 1/8;  % downsample background image by this much per dimension
K = 4;  % number of targets
T = 7;  % number of frames

% load background frame
template_frame_path = '../../data/frame70.mat';
template_frame = getfield(load(template_frame_path, 'frame'), 'frame');
template_frame = template_frame-min(template_frame(:));
template_frame = imresize(template_frame, ds);
template_frame = 1000*template_frame/max(template_frame(:));
imsize = size(template_frame);
N = numel(template_frame);
L = repmat(template_frame(:), 1, T);  % low rank component

% generate some sparse targets
target_opts.y_min = floor(205*ds);
target_opts.y_max = floor(265*ds);
target_opts.x_min = floor(100*ds);
target_opts.x_max = floor(1750*ds);
target_opts.show_targets = false;
target_power = 1000*0.3;
S = target_power*sparse_targets(imsize, K, T, target_opts);  % sparse component

% data matrix
D = L + S;

%% define transformations
% a general affine transformation matrix has the form
% T = [a b c; d e f; 0 0 1]
% which maps points (x,y) via
%
% T*[x; y ; 1]
%
% i.e., x <- a*x + b*y + c, y <- d*x + e*y + f
%
% Note: MATLAB's image processing toolbox uses the transpose of this form with
% coordinates in row vectors instead.
%
% Store the transform parameters in a 6xT matrix with columns [a b c d e f]':
tform_type = 'affine';


% transform parameter set 1 (default): translation-only.
taus = [ ...
         [[1 0 0] [0 1 0]]',...
         [[1 0 -1] [0 1 0]]',...
         [[1 0 0] [0 1 1]]',...
         [[1 0 2] [0 1 0]]',...
         [[1 0 -2] [0 1 0]]',...
         [[1 0 -1] [0 1 -1]]',...
         [[1 0 1] [0 1 1]]',...
         ];

% transform parameter set 2: translation/rotation/scaling/skewing
% this set of transform parameters demos more variety, but the interpolation
% involved in warping/unwarping the images blurs the targets and edges in the
% background.
% theta = 1;
% taus = [ ...
%          ... % identity (reference frame)
%          [[1 0 0] [0 1 0]]',...
%          ... % translate in x direction
%          [[1 0 -2] [0 1 0]]', ...
%          ... % translate in y direction
%          [[1 0 0] [0 1 1]]',...
%          ... % rotate by theta degrees
%          [[cosd(theta) -sind(theta) 0] [sind(theta) cosd(theta) 0]]',...
%          ... % rotate by theta degrees
%          [[cosd(2*theta) -sind(2*theta) 0] [sind(2*theta) cosd(2*theta) 0]]',...
%          ... % scale in both directions (i.e., zoom out)
%          [[0.98 0 0] [0 0.98 0]]',...
%          ... % skew
%          [[1 0.05 0] [0 1 0]]',...
%      ];

% transform parameter set 3: identity (for debugging)
% taus = repmat([1 0 0 0 1 0]', 1, T);



%% apply transformations to data
RA = imref2d(imsize);  % define image domain after warping to be same as input
Y = cell(T, 1);
for ii = 1:T
    T_mtx = projective_mtx_affine(taus(:, ii));
    tform = projective2d(inv(T_mtx)');
    frame = reshape(D(:, ii), imsize);
    Y{ii} = imwarp(frame, tform, 'OutputView', RA);
end

%% show images and transforms
% Define size of border around region of interest (ROI)
%
% The low-rank/sparse decomposition is performed over the region of interest to
% avoid boundary effects due to transformation.
%
% For best results, choose an ROI such that when the inverse affine
% transformation is applied to the frames, the result matches the original
% frames inside of the ROI.
roi_padding = 5;
roi_x_ind = (roi_padding+1):(imsize(2)-roi_padding);
roi_y_ind = (roi_padding+1):(imsize(1)-roi_padding);
roi_size = imsize-2*roi_padding;
roi_pos = [roi_padding+[1 1] ...
           (imsize(2)-2*roi_padding) (imsize(1)-2*roi_padding)];
c_range = [min(D(:,1)) max(D(:,1))];
show_input_data = false;  % set to true to preview the original and transformed images
if show_input_data
for ii = 1:T
    clf
    Y_orig = reshape(D(:,ii), imsize);
    Y_affine = Y{ii};
    tau = taus(:,ii);
    RA = imref2d(size(Y_orig));
    T_mtx = projective_mtx_affine(tau);
    tform = projective2d(T_mtx');
    Y_rec = imwarp(Y_affine, tform, 'OutputView', RA, 'interp', 'linear');
    
    subplot(311);
    imagesc(Y_orig); axis image; colormap gray;
    caxis(c_range);
    rectangle('position', roi_pos, 'edgecolor', 'r')
    title(sprintf('original frame %u', ii));

    subplot(312);
    imagesc(Y_affine); axis image; colormap gray;
    caxis(c_range);
    rectangle('position', roi_pos, 'edgecolor', 'r')
    title(sprintf('transformed frame [%g %g %g; %g %g %g]',...
                  tau(1), tau(2), tau(3), tau(4), tau(5), tau(6)));

    subplot(313);
    imagesc(Y_rec); axis image; colormap gray;
    caxis(c_range);
    Y_orig_roi = Y_orig(roi_y_ind, roi_x_ind);
    Y_rec_roi = Y_rec(roi_y_ind, roi_x_ind);
    title(sprintf('inverse transformed frame, rmse=%g', ...
                   (norm(Y_rec_roi-Y_orig_roi,'fro')/norm(Y_orig_roi,'fro'))^2));
    rectangle('position', roi_pos, 'edgecolor', 'r')
    
    fprintf('Showing frame %u/%u. Press any key to continue...\n', ii, T);
    pause()
end
end

%% set algorithm parameters and solve
lambda = 0.1;
gamma = lambda*sqrt(prod(imsize-2*roi_padding));
solver_opts.kappa = 0.1;
solver_opts.mu = 10;
solver_opts.rho = 0.01;

solver_opts.roi_padding = roi_padding;
solver_opts.tform_type = tform_type;
solver_opts.max_iters_outer = 150;
solver_opts.max_iters_inner = 250;
solver_opts.print_every_inner = 50;
solver_opts.print_every_outer = 1;
solver_opts.tolerance = 1e-3;
solver_opts.tolerance_outer = 1e-3;
solver_opts.use_ot = true;  % use EMD regularization?
solver_opts.show_debug_plots_inner = false;
solver_opts.show_debug_plots_outer = false;
solver_opts.reference_frame = 1;

[S_est,L_est,tau_est] = solver_RASL(Y, lambda, gamma, solver_opts);

%% print/plot results
disp('True transform parameters:')
disp(taus)
disp('Estimated transform parameters:')
disp(inv_affine_params(normalize_affine_kth(tau_est, 1)))

% show correct transform recovery and corresponding targets
clf;
caxis_ = [min(D(:)) max(D(:))];
r = 4; c = 1;
for ii = 1:T
    subplot(r, c, 1); plot_frame(D(:, ii), caxis_, imsize);
    subplot(r, c, 2); plot_frame(Y{ii}, caxis_);
    rectangle('position', roi_pos, 'edgecolor', 'r')
    S_crop = reshape(S(:,ii), imsize);
    S_crop = S_crop(roi_y_ind, roi_x_ind);
    subplot(r, c, 3); plot_frame(S_crop(:), [], roi_size);
    subplot(r, c, 4); plot_frame(S_est(:, ii), [], roi_size);
    fprintf('Showing recovered frame %u/%u. Press any key to continue...\n', ii, T);
    pause()
end

%% helper functions
function plot_frame(frame, caxis_, imsize)
if exist('imsize', 'var'), frame = reshape(frame, imsize); end
imagesc(frame);
axis image;
colormap gray;
if ~isempty(caxis_), caxis(caxis_); end
set(gca, 'xtick', []);
set(gca, 'ytick', []);
end

