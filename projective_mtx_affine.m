function T = projective_mtx_affine(tau)
T = [reshape(tau(1:3), 1, 3); reshape(tau(4:6), 1, 3); [0 0 1]];
end
