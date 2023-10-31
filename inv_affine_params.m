function tau = inv_affine_params(tau)
% compute parameters for inverse transform
for ii = 1:size(tau, 2)
    G = projective_mtx_affine(tau(:, ii));
    G_normalized = inv(G);
    tau(:, ii) = vec(transpose(G_normalized(1:2, :)));
end
end
