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


def apply_A0inv_vec(x, K, G, lam):
    n = K.shape[0]
    r = G.shape[0]
    X = x.reshape(n, r, order="F")

    evalK, U = np.linalg.eigh(K)
    evalG, V = np.linalg.eigh(G)

    Xh = U.T @ X @ V
    denom = (evalK[:, None] ** 2) * evalG[None, :] + lam * evalK[:, None]
    Yh = Xh / denom
    Y = U @ Yh @ V.T
    return Y.reshape(n * r, order="F")


def pcg(A_mv, b, M_inv, x0=None, tol=1e-10, maxit=200):
    n = b.size
    x = np.zeros(n) if x0 is None else x0.copy()

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


def main():
    rng = np.random.default_rng(0)

    n1, n2, n3 = 10, 8, 6
    n = n1
    M = n2 * n3
    N = n * M
    r = 3
    lam = 0.5

    # kernel + factors
    A = rng.standard_normal((n, n))
    K = A @ A.T + 0.1 * np.eye(n)
    A2 = rng.standard_normal((n2, r))
    A3 = rng.standard_normal((n3, r))
    Z = khatri_rao(A3, A2)
    G = Z.T @ Z

    # observation mask
    q = 200
    mask_flat = np.zeros(N, dtype=bool)
    mask_flat[rng.choice(N, size=q, replace=False)] = True
    S, obs_flat = build_selection(mask_flat)
    P = S @ S.T

    # explicit A only used for verification
    ZkronK = np.kron(Z, K)
    Aexp = ZkronK.T @ P @ ZkronK + lam * np.kron(np.eye(r), K)

    # observation list in (i,j) unfolding coordinates
    obs_pairs = [(idx % n, idx // n) for idx in obs_flat]

    def Z_row(j):
        i2 = j % n2
        i3 = j // n2
        return A3[i3, :] * A2[i2, :]

    b = rng.standard_normal(n * r)

    def A_mv(x):
        X = x.reshape(n, r, order="F")
        Gx = K @ X
        H = np.zeros((n, r))
        for (i, j) in obs_pairs:
            z = Z_row(j)
            u = Gx[i, :] @ z
            H[i, :] += u * z
        Y = K @ H + lam * Gx
        return Y.reshape(n * r, order="F")

    def M_inv(x):
        return apply_A0inv_vec(x, K, G, lam)

    x_pcg, iters, rel = pcg(A_mv, b, M_inv, tol=1e-10, maxit=200)
    x_star = np.linalg.solve(Aexp, b)

    err = np.linalg.norm(x_pcg - x_star) / np.linalg.norm(x_star)
    print(f"PCG iters={iters}, final relres={rel:.2e}, sol relerr={err:.2e}")
    assert err < 1e-8


if __name__ == "__main__":
    main()
