import numpy as np


def khatri_rao(A, B):
    I, r = A.shape
    J, r2 = B.shape
    assert r == r2
    return np.einsum('ir,jr->ijr', A, B).reshape(I * J, r)


def build_selection(mask_flat):
    idx = np.where(mask_flat)[0]
    N = mask_flat.size
    q = idx.size
    S = np.zeros((N, q))
    S[idx, np.arange(q)] = 1.0
    return S, idx


def pcg(A_mv, b, M_inv, tol=1e-10, maxit=500):
    x = np.zeros_like(b)
    r = b - A_mv(x)
    z = M_inv(r)
    p = z.copy()
    rz = float(r @ z)
    bnorm = np.linalg.norm(b)

    for it in range(maxit):
        Ap = A_mv(p)
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rel = np.linalg.norm(r) / (bnorm + 1e-30)
        if rel < tol:
            return x, it + 1, rel
        z = M_inv(r)
        rz_new = float(r @ z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, maxit, rel


def apply_A0inv_Avec(x, K, G, lam):
    # A0A = G ⊗ I + lam (I ⊗ K^{-1})
    n = K.shape[0]
    r = G.shape[0]
    X = x.reshape(n, r, order="F")

    evalK, U = np.linalg.eigh(K)
    evalG, V = np.linalg.eigh(G)

    Xh = U.T @ X @ V
    denom = evalG[None, :] + lam / evalK[:, None]
    Yh = Xh / denom
    Y = U @ Yh @ V.T
    return Y.reshape(n * r, order="F")


def main():
    rng = np.random.default_rng(0)

    n1, n2, n3 = 12, 9, 7
    n = n1
    M = n2 * n3
    N = n * M
    r = 3
    lam = 0.6

    # SPD kernel and factors
    A = rng.standard_normal((n, n))
    K = A @ A.T + 0.1 * np.eye(n)

    A2 = rng.standard_normal((n2, r))
    A3 = rng.standard_normal((n3, r))
    Z = khatri_rao(A3, A2)
    G = Z.T @ Z

    # mask / observations
    q = 250
    mask_flat = np.zeros(N, dtype=bool)
    mask_flat[rng.choice(N, size=q, replace=False)] = True
    S, obs_flat = build_selection(mask_flat)
    P = S @ S.T
    obs_pairs = [(idx % n, idx // n) for idx in obs_flat]

    def Z_row(j):
        i2 = j % n2
        i3 = j // n2
        return A3[i3, :] * A2[i2, :]

    # synthetic unfolding T (zeros at missing), B = T Z, rhs = vec(B)
    T_vec = rng.standard_normal(N) * mask_flat.astype(float)
    T = T_vec.reshape((n, M), order="F")
    B = T @ Z
    b = B.reshape(n * r, order="F")

    # implicit A(A)= (Z⊗I)^T P (Z⊗I) vec(A) + lam (I⊗K^{-1}) vec(A)
    # apply K^{-1} via solves
    L = np.linalg.cholesky(K)

    def K_solve(X):
        # solves K Y = X for Y
        Y = np.linalg.solve(L, X)
        return np.linalg.solve(L.T, Y)

    def A_mv(x):
        X = x.reshape(n, r, order="F")
        # gather/scatter using predicted entries U = X Z^T
        H = np.zeros((n, r))
        for (i, j) in obs_pairs:
            z = Z_row(j)
            u = X[i, :] @ z
            H[i, :] += u * z
        Y = H + lam * K_solve(X)
        return Y.reshape(n * r, order="F")

    def M_inv(x):
        return apply_A0inv_Avec(x, K, G, lam)

    # explicit matrix for verification
    ZkronI = np.kron(Z, np.eye(n))
    Aexp = ZkronI.T @ P @ ZkronI + lam * np.kron(np.eye(r), np.linalg.inv(K))

    x_pcg, iters, rel = pcg(A_mv, b, M_inv, tol=1e-10, maxit=500)
    x_star = np.linalg.solve(Aexp, b)

    err = np.linalg.norm(x_pcg - x_star) / np.linalg.norm(x_star)
    print(f"A-variable PCG iters={iters}, relres={rel:.2e}, sol relerr={err:.2e}")
    assert err < 1e-8


if __name__ == "__main__":
    main()
