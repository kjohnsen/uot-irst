function tau = normalize_transform_by_matrix(tau, T)
% apply the matrix T^-1 to each transform matrix associated /w parameters in tau
for ii = 1:size(tau, 2)
    G = projective_mtx_affine(tau(:, ii));
    G_normalized = T\G;
    tau(:, ii) = vec(transpose(G_normalized(1:2,:)));
end
end
