"""
Stress-test the claims in final_proof.tex:
1. Matvec correctness for various tensor sizes and sparsity levels
2. Preconditioner correctness (eigendecomposition formula)
3. Full PCG convergence and solution accuracy
4. Edge cases: very sparse observations, rank-deficient K, large rank r
"""
import numpy as np
from scipy.linalg import eigh


def khatri_rao(A, B):
    I, r = A.shape
    J = B.shape[0]
    return np.einsum('ir,jr->ijr', A, B).reshape(I * J, r)


def explicit_system(K, Z, obs_flat, lam, n, r):
    ZkronK = np.kron(Z, K)
    ZkronK_obs = ZkronK[obs_flat, :]
    A = ZkronK_obs.T @ ZkronK_obs + lam * np.kron(np.eye(r), K)
    return A


def implicit_matvec(x_vec, K, z_rows, obs_pairs, lam, n, r):
    X = x_vec.reshape(n, r, order="F")
    G = K @ X
    H = np.zeros((n, r))
    for idx, (i, j) in enumerate(obs_pairs):
        z = z_rows[idx]
        u = G[i, :] @ z
        H[i, :] += u * z
    Y = K @ H + lam * G
    return Y.reshape(n * r, order="F")


def apply_precond_inv(x, K, Gram, lam, alpha):
    n = K.shape[0]
    r = Gram.shape[0]
    X = x.reshape(n, r, order="F")
    evalK, U = eigh(K)
    evalG, V = eigh(Gram)
    Xh = U.T @ X @ V
    denom = alpha * (evalK[:, None] ** 2) * evalG[None, :] + lam * evalK[:, None]
    Yh = Xh / denom
    Y = U @ Yh @ V.T
    return Y.reshape(n * r, order="F")


def pcg(A_mv, b, M_inv, tol=1e-10, maxit=500):
    n = b.size
    x = np.zeros(n)
    res = b.copy()
    z = M_inv(res)
    p = z.copy()
    rz = float(res @ z)
    bnorm = np.linalg.norm(b)
    for it in range(maxit):
        Ap = A_mv(p)
        alpha = rz / float(p @ Ap)
        x += alpha * p
        res -= alpha * Ap
        rel = np.linalg.norm(res) / (bnorm + 1e-30)
        if rel < tol:
            return x, it + 1, rel
        z = M_inv(res)
        rz_new = float(res @ z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, maxit, rel


def test_case(name, n1, n2, n3, k, r, q, lam, K_nugget, seed=42):
    """Run a single test case and return results."""
    rng = np.random.default_rng(seed)
    dims = [n1, n2, n3]
    n = dims[k - 1]
    M = np.prod([dims[i] for i in range(3) if i != k - 1])
    N = n * M

    # Kernel matrix
    A_kern = rng.standard_normal((n, n))
    K = A_kern @ A_kern.T + K_nugget * np.eye(n)

    # Factor matrices (for modes != k)
    factors = []
    for i in range(3):
        if i != k - 1:
            factors.append(rng.standard_normal((dims[i], r)))

    if len(factors) == 2:
        Z = khatri_rao(factors[1], factors[0])
    else:
        Z = factors[0]

    Gram = Z.T @ Z

    # Observation mask
    q_actual = min(q, N)
    mask_flat = np.zeros(N, dtype=bool)
    mask_flat[rng.choice(N, size=q_actual, replace=False)] = True
    obs_flat = np.where(mask_flat)[0]
    obs_pairs = [(idx % n, idx // n) for idx in obs_flat]

    # Cache z_t rows
    z_rows = []
    for (i, j) in obs_pairs:
        if len(factors) == 2:
            n_other_1 = factors[0].shape[0]
            i1 = j % n_other_1
            i2 = j // n_other_1
            z_rows.append(factors[1][i2, :] * factors[0][i1, :])
        else:
            z_rows.append(factors[0][j, :].copy())

    alpha = q_actual / N

    # Test 1: Matvec correctness
    A_exp = explicit_system(K, Z, obs_flat, lam, n, r)
    x = rng.standard_normal(n * r)
    y_exp = A_exp @ x
    y_imp = implicit_matvec(x, K, z_rows, obs_pairs, lam, n, r)
    matvec_err = np.linalg.norm(y_exp - y_imp) / np.linalg.norm(y_exp)

    # Test 2: Preconditioner correctness
    A0_exp = alpha * np.kron(Gram, K @ K) + lam * np.kron(np.eye(r), K)
    y_prec = apply_precond_inv(x, K, Gram, lam, alpha)
    y_prec_exp = np.linalg.solve(A0_exp, x)
    prec_err = np.linalg.norm(y_prec - y_prec_exp) / np.linalg.norm(y_prec_exp)

    # Test 3: SPD check
    eigvals_A = np.linalg.eigvalsh(A_exp)
    spd_ok = eigvals_A.min() > 0

    # Test 4: Full PCG solve
    b = rng.standard_normal(n * r)
    mv = lambda x: implicit_matvec(x, K, z_rows, obs_pairs, lam, n, r)
    pc = lambda x: apply_precond_inv(x, K, Gram, lam, alpha)
    x_pcg, iters, relres = pcg(mv, b, pc, tol=1e-10, maxit=500)
    x_star = np.linalg.solve(A_exp, b)
    sol_err = np.linalg.norm(x_pcg - x_star) / np.linalg.norm(x_star)

    # Test 5: RHS computation (B = TZ via sparse accumulation)
    T_flat = np.zeros(N)
    obs_vals = rng.standard_normal(q_actual)
    T_flat[obs_flat] = obs_vals
    T_mat = T_flat.reshape(n, M, order="F")
    B_direct = T_mat @ Z
    B_sparse = np.zeros((n, r))
    for idx, (i, j) in enumerate(obs_pairs):
        B_sparse[i, :] += obs_vals[idx] * z_rows[idx]
    rhs_err = np.linalg.norm(B_direct - B_sparse) / (np.linalg.norm(B_direct) + 1e-30)

    print(f"[{name}] n={n}, M={M}, r={r}, q={q_actual}, lam={lam}")
    print(f"  Matvec err:  {matvec_err:.2e} {'PASS' if matvec_err < 1e-12 else 'FAIL'}")
    print(f"  Precond err: {prec_err:.2e} {'PASS' if prec_err < 1e-12 else 'FAIL'}")
    print(f"  SPD:         {spd_ok} {'PASS' if spd_ok else 'FAIL'}")
    print(f"  RHS err:     {rhs_err:.2e} {'PASS' if rhs_err < 1e-12 else 'FAIL'}")
    print(f"  PCG iters:   {iters}, relres={relres:.2e}, sol_err={sol_err:.2e} "
          f"{'PASS' if sol_err < 1e-8 else 'FAIL'}")
    print()
    return matvec_err < 1e-12 and prec_err < 1e-12 and spd_ok and sol_err < 1e-8 and rhs_err < 1e-12


if __name__ == "__main__":
    results = []

    # Standard case
    results.append(test_case("Standard", 8, 6, 5, 1, 3, 50, 0.5, 0.5))

    # Very sparse
    results.append(test_case("Very sparse", 8, 6, 5, 1, 3, 10, 0.5, 0.5))

    # Large rank
    results.append(test_case("Large rank", 6, 5, 4, 1, 5, 40, 0.3, 0.5))

    # Small regularization
    results.append(test_case("Small lambda", 8, 6, 5, 1, 3, 50, 0.001, 0.5))

    # Large regularization
    results.append(test_case("Large lambda", 8, 6, 5, 1, 3, 50, 100.0, 0.5))

    # Nearly singular K (small nugget)
    results.append(test_case("Near-singular K", 8, 6, 5, 1, 3, 50, 0.5, 0.01))

    # Different mode (k=2)
    results.append(test_case("Mode k=2", 5, 8, 6, 2, 3, 50, 0.5, 0.5))

    # Dense observations (q close to N)
    results.append(test_case("Dense obs", 6, 5, 4, 1, 3, 100, 0.5, 0.5))

    # n >> r
    results.append(test_case("n >> r", 15, 4, 3, 1, 2, 40, 0.5, 0.5))

    # r close to n
    results.append(test_case("r ~ n", 6, 5, 4, 1, 5, 40, 0.5, 0.5))

    print("=" * 60)
    all_pass = all(results)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
