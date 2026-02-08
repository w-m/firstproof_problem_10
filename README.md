This repo contains a short writeup and small verification scripts for solving the RKHS-mode CP-ALS subproblem with missing data using PCG.

- `notes_pcg_rkhs_cp.tex` / `notes_pcg_rkhs_cp.pdf`: main technical note (matvec derivation, preconditioner, complexity).
- `answer_pcg_rkhs_cp.tex` / `answer_pcg_rkhs_cp.pdf`: shorter answer-style writeup.
- `proof.tex` / `proof.pdf`: self-contained proof-style writeup (\le 5 pages).
- `final_proof.tex` / `final_proof.pdf`: polished version.
- `pcg_rkhs_cp_summary.md`: condensed summary.
- `verify_matvec.py`: verifies the implicit matvec against an explicit Kronecker construction restricted to the observed rows (small random instance; no explicit selection matrix).
- `verify_precond.py`: verifies the closed-form Kronecker preconditioner application.
- `demo_pcg_solver.py`: runs PCG with the Kronecker preconditioner (using scaling alpha=q/N) on a small system.
- `verify_change_of_variables.py`: verifies the equivalent formulation that solves for `A_k = K W` directly (small explicit system; no selection matrix).
- `demo_pcg_A_variable.py`: runs PCG on the `A_k`-variable system with a Kronecker-sum preconditioner (using scaling alpha=q/N).

Run scripts with:

```bash
uv run --with numpy <script>.py
```
