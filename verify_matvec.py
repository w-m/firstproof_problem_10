import numpy as np


def khatri_rao(A, B):
    # column-wise Kronecker, A (I x r), B (J x r) -> (IJ x r)
    I, r = A.shape
    J, r2 = B.shape
    assert r == r2
    return np.einsum('ir,jr->ijr', A, B).reshape(I * J, r)


def build_selection(mask_flat):
    # mask_flat: boolean of length N
    idx = np.where(mask_flat)[0]
    N = mask_flat.size
    q = idx.size
    S = np.zeros((N, q))
    S[idx, np.arange(q)] = 1.0
    return S, idx


def implicit_matvec(x_vec, K, Z, obs_idx_pairs, lam):
    # x_vec = vec(X) with X (n x r), Z (M x r)
    n = K.shape[0]
    M = Z.shape[0]
    r = Z.shape[1]
    X = x_vec.reshape(n, r, order="F")

    G = K @ X  # n x r

    # gather u_t = (KX)[i,:] dot Z[j,:]
    u = np.empty(len(obs_idx_pairs))
    for t, (i, j) in enumerate(obs_idx_pairs):
        u[t] = G[i, :] @ Z[j, :]

    # accumulate H = U~ Z where U~ has entries u_t at (i,j)
    H = np.zeros((n, r))
    for t, (i, j) in enumerate(obs_idx_pairs):
        H[i, :] += u[t] * Z[j, :]

    Y = K @ H + lam * (K @ X)
    return Y.reshape(n * r, order="F")


def main():
    rng = np.random.default_rng(0)
    # small tensor dims
    n1, n2, n3 = 4, 3, 2
    k = 1  # mode-1 is RKHS
    n = n1
    M = n2 * n3
    N = n * M
    r = 2
    lam = 0.3

    # SPD kernel
    A = rng.standard_normal((n, n))
    K = A @ A.T + 0.5 * np.eye(n)

    # factor matrices for modes 2 and 3
    A2 = rng.standard_normal((n2, r))
    A3 = rng.standard_normal((n3, r))
    Z = khatri_rao(A3, A2)  # (n3*n2) x r; corresponds to unfolding column index j

    # random observation mask
    q = 9
    mask_flat = np.zeros(N, dtype=bool)
    mask_flat[rng.choice(N, size=q, replace=False)] = True
    S, obs_flat = build_selection(mask_flat)

    # convert flat indices into (i,j) pairs for mode-1 unfolding (column-major vec)
    # vec stacks columns: column j has rows i=0..n-1
    obs_pairs = [(idx % n, idx // n) for idx in obs_flat]

    # explicit system matrix
    ZkronK = np.kron(Z, K)  # (N x nr)
    Aexp = ZkronK.T @ (S @ S.T) @ ZkronK + lam * np.kron(np.eye(r), K)

    # test vector
    x = rng.standard_normal(n * r)
    y_exp = Aexp @ x
    y_imp = implicit_matvec(x, K, Z, obs_pairs, lam)

    rel_err = np.linalg.norm(y_exp - y_imp) / np.linalg.norm(y_exp)
    print(f"relative matvec error: {rel_err:.3e}")
    assert rel_err < 1e-12


if __name__ == "__main__":
    main()
