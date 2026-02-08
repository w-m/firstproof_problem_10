# PCG for RKHS CP-ALS mode-k subproblem (missing data)

We solve for `W ∈ R^{n×r}` in

`[(Z ⊗ K)^T S S^T (Z ⊗ K) + λ (I_r ⊗ K)] vec(W) = (I_r ⊗ K) vec(B)`

with `K ∈ R^{n×n}` SPD kernel, `S` selecting the `q` observed tensor entries, `Z` the Khatri–Rao product of the other CP factors, and `B = T Z`.

## Why PCG
The matrix is symmetric positive definite (if `K ≻ 0`, `λ>0`), so conjugate gradients applies; preconditioning reduces iterations. PCG avoids forming the dense `(nr)×(nr)` matrix (and avoids any `O(N)` work with `N = nM`).

## Matvec `y = A x` without forming `A`
Reshape `x = vec(X)` with `X ∈ R^{n×r}`.
Use `vec(K X Z^T) = (Z ⊗ K) vec(X)` and implement masking via the observed index list.

Per matvec:
1. `G ← K X`  (`O(n^2 r)`).
2. For each observed entry `t=1..q` with unfolding index `(i_t, j_t)`:
   - fetch/compute `z_t = Z[j_t,:]` (compute on-the-fly from CP factors if `Z` is not stored),
   - `u_t ← <G[i_t,:], z_t>` (dot in `R^r`),
   - accumulate `H[i_t,:] += u_t z_t`.
   This realizes `H = (S S^T vec(KXZ^T)) reshaped * Z` in `O(q r)` time (or `O(q d r)` if each `z_t` is formed from `d-1` factors).
3. `Y ← K H + λ K X`  (`O(n^2 r)`; reuse `G`), and return `vec(Y)`.

Overall: `O(n^2 r + q r)` time and `O(nr + q)` memory per matvec; independent of `N`.

## RHS without forming `T`
Compute `B = T Z` by iterating over the `q` observations:
`B[i_t,:] += t_t z_t` (sparse MTTKRP), cost `O(qr)` (or `O(qdr)` if forming `z_t` on-the-fly), then compute `K B` in `O(n^2 r)`.

## Preconditioner
A practical Kronecker preconditioner drops the mask (`S S^T ≈ I`), giving

`A0 = (Z^T Z) ⊗ (K^2) + λ (I ⊗ K)`.

- Compute `G = Z^T Z` cheaply without forming `Z` using the standard Khatri–Rao Gram identity:
  `Z^T Z = (A_1^T A_1) * ... * (A_{k-1}^T A_{k-1}) * (A_{k+1}^T A_{k+1}) * ... * (A_d^T A_d)`,
  where `*` is the Hadamard (entrywise) product.
- Apply `A0^{-1}` via eigendecompositions of `K` and `G` (or Schur/Sylvester solves): per apply about `O(n^2 r + n r^2)` after one-time `O(n^3 + r^3)` setup.

Per PCG iteration cost ≈ matvec + preconditioner apply = `O(n^2 r + q r)` plus lower-order terms.
