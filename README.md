This repo contains a short writeup and small verification scripts for solving the RKHS-mode CP-ALS subproblem with missing data using PCG.

- `notes_pcg_rkhs_cp.tex` / `notes_pcg_rkhs_cp.pdf`: main technical note (matvec derivation, preconditioner, complexity).
- `pcg_rkhs_cp_summary.md`: condensed summary.
- `verify_matvec.py`: verifies the implicit matvec against an explicit Kronecker+selection construction (small random instance).
- `verify_precond.py`: verifies the closed-form Kronecker preconditioner application.
- `demo_pcg_solver.py`: runs PCG with the Kronecker preconditioner on a small system.
- `verify_change_of_variables.py`: verifies the equivalent formulation that solves for `A_k = K W` directly.
- `demo_pcg_A_variable.py`: runs PCG on the `A_k`-variable system with a Kronecker-sum preconditioner.

Run scripts with:

```bash
uv run --with numpy <script>.py
```
