function tau = normalize_affine_mean(tau)
% normalize affine transform parameters w.r.t. mean transform
tau_mean = mean(tau, 2);
T = projective_mtx_affine(tau_mean);
tau = normalize_transform_by_matrix(tau, T);
end
