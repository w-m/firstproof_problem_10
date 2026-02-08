import numpy as np


def apply_A0inv_vec(x, K, G, lam):
    # A0 = G ⊗ (K@K) + lam (I ⊗ K), K SPD symmetric, G SPD symmetric.
    n = K.shape[0]
    r = G.shape[0]
    X = x.reshape(n, r, order="F")

    # eigendecompositions
    evalK, U = np.linalg.eigh(K)
    evalG, V = np.linalg.eigh(G)

    # transform
    Xh = U.T @ X @ V

    # divide elementwise by (sigma_a * lambda_b^2 + lam * lambda_b)
    denom = (evalK[:, None] ** 2) * evalG[None, :] + lam * evalK[:, None]
    Yh = Xh / denom

    # inverse transform
    Y = U @ Yh @ V.T
    return Y.reshape(n * r, order="F")


def main():
    rng = np.random.default_rng(0)
    n, r = 5, 3
    lam = 0.7

    A = rng.standard_normal((n, n))
    K = A @ A.T + 1.0 * np.eye(n)

    B = rng.standard_normal((r, r))
    G = B @ B.T + 1.0 * np.eye(r)

    A0 = np.kron(G, K @ K) + lam * np.kron(np.eye(r), K)

    x = rng.standard_normal(n * r)
    y1 = apply_A0inv_vec(x, K, G, lam)
    y2 = np.linalg.solve(A0, x)

    rel = np.linalg.norm(y1 - y2) / np.linalg.norm(y2)
    print(f"relative error: {rel:.3e}")
    assert rel < 1e-12


if __name__ == "__main__":
    main()
