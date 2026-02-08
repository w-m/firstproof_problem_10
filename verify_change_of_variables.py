import numpy as np


def khatri_rao(A, B):
    I, r = A.shape
    J, r2 = B.shape
    assert r == r2
    return np.einsum('ir,jr->ijr', A, B).reshape(I * J, r)


def observed_indices(mask_flat):
    return np.where(mask_flat)[0]


def main():
    rng = np.random.default_rng(0)

    n1, n2, n3 = 5, 4, 3
    n = n1
    M = n2 * n3
    N = n * M
    r = 2
    lam = 0.4

    A = rng.standard_normal((n, n))
    K = A @ A.T + 0.5 * np.eye(n)
    Kinv = np.linalg.inv(K)

    A2 = rng.standard_normal((n2, r))
    A3 = rng.standard_normal((n3, r))
    Z = khatri_rao(A3, A2)

    # sparse mask
    q = 25
    mask_flat = np.zeros(N, dtype=bool)
    mask_flat[rng.choice(N, size=q, replace=False)] = True
    obs_flat = observed_indices(mask_flat)

    # data unfolding T with zeros at missing
    T_vec = rng.standard_normal(N) * mask_flat.astype(float)  # only observed values kept
    T = T_vec.reshape((n, M), order="F")

    B = T @ Z

    # W-system (as in prompt)
    ZkronK = np.kron(Z, K)
    ZkronK_obs = ZkronK[obs_flat, :]
    Aw = ZkronK_obs.T @ ZkronK_obs + lam * np.kron(np.eye(r), K)
    bw = np.kron(np.eye(r), K) @ B.reshape(n * r, order="F")
    w = np.linalg.solve(Aw, bw)
    W = w.reshape(n, r, order="F")
    A_from_W = K @ W

    # A-system (change of variables A=KW)
    ZkronI = np.kron(Z, np.eye(n))
    ZkronI_obs = ZkronI[obs_flat, :]
    Aa = ZkronI_obs.T @ ZkronI_obs + lam * np.kron(np.eye(r), Kinv)
    ba = B.reshape(n * r, order="F")
    a = np.linalg.solve(Aa, ba)
    A_direct = a.reshape(n, r, order="F")

    rel = np.linalg.norm(A_from_W - A_direct) / np.linalg.norm(A_direct)
    print(f"relative A mismatch: {rel:.3e}")
    assert rel < 1e-10


if __name__ == "__main__":
    main()
