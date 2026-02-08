import time
import numpy as np


def matvec_cached(X, K, i_idx, Z_rows, lam):
    # X: n x r, i_idx: q, Z_rows: q x r (each row is z_t)
    n, r = X.shape
    q = i_idx.size

    G = K @ X  # n x r

    # u_t = <G[i_t,:], z_t>
    u = np.einsum("qr,qr->q", G[i_idx, :], Z_rows)

    # H[i_t,:] += u_t z_t
    H = np.zeros((n, r))
    np.add.at(H, i_idx, u[:, None] * Z_rows)

    Y = K @ H + lam * G
    return Y


def main():
    rng = np.random.default_rng(0)

    n = 300
    r = 20
    q = 50_000
    lam = 0.5

    A = rng.standard_normal((n, n))
    K = A @ A.T / n + 0.1 * np.eye(n)
    X = rng.standard_normal((n, r))

    i_idx = rng.integers(0, n, size=q, dtype=np.int64)
    Z_rows = rng.standard_normal((q, r))

    # warmup
    matvec_cached(X, K, i_idx, Z_rows, lam)

    t0 = time.perf_counter()
    Y = matvec_cached(X, K, i_idx, Z_rows, lam)
    t1 = time.perf_counter()

    print(f"n={n} r={r} q={q} time={t1-t0:.3f}s, ||Y||_F={np.linalg.norm(Y):.3e}")


if __name__ == "__main__":
    main()
