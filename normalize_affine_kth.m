function tau = normalize_affine_kth(tau, k)
% normalize affine transform parameters w.r.t. k'th frame
T = projective_mtx_affine(tau(:, k));
tau = normalize_transform_by_matrix(tau, T);
end
