function S = sparse_targets(imsize, K, T, opts)
% Generate a sparse matrix S whose T columns are each vectorized 2D images
% containing sparse targets which move to an adjacent location from frame to
% frame. Targets can move horizontally, vertically, diagonally, or stay
% stationary. When they move, they move by `speed` pixels in each direction.

% default options
speed = 1;
x_min = 1;
x_max = imsize(2);
y_min = 1;
y_max = imsize(1);
show_targets = false;

% set options
if exist('opts', 'var')
if isfield(opts, 'speed'), speed = opts.speed; end
if isfield(opts, 'x_min'), x_min = opts.x_min; end
if isfield(opts, 'x_max'), x_max = opts.x_max; end
if isfield(opts, 'y_min'), y_min = opts.y_min; end
if isfield(opts, 'y_max'), y_max = opts.y_max; end
if isfield(opts, 'show_targets'), show_targets = opts.show_targets; end
end

N = prod(imsize);
S = sparse(N, T);

% generate initial positions
[i, j] = ind2sub(imsize, randsample(N, K));

for t = 1:T
    i = jitter_coord(i, [y_min y_max], speed);
    j = jitter_coord(j, [x_min x_max], speed);
    temp = sparse(i, j, 1, imsize(1), imsize(2), K);
    S(:, t) = temp(:);
end

if show_targets
    gcf; clf(gcf);
    for t = 1:T
        frame = reshape(S(:, t), imsize);
        imagesc(frame);
        axis image
        pause();
    end
end

end %  main function

function out = jitter_coord(in, bounds, speed)
    out = in;
    for ii = 1:numel(in)
        dir = randi([-1 1]);
        new = in(ii) + speed*dir;
        new = min(new, bounds(2));
        new = max(new, bounds(1));
        out(ii) = new;
    end
end
