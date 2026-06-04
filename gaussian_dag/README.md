# gaussian_dag — library source

Core library modules for mutual-information evaluation and gradient-based
optimisation on linear Gaussian DAGs. All public symbols are re-exported
from the top-level package and listed in `../README.md`.

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public-API re-exports. The eight symbols listed in the top-level README are exactly the ones imported here. |
| `krecursion.py` | Forward K-recursion over a DAG: `compute_k_blocks`, plus the helpers `get_K` (Hermitian-flip aware accessor) and `hermitianize` (`(A + A^H) / 2`). |
| `information.py` | Log-det mutual information from K-blocks: `mutual_information_from_k` (Schur-complement + Cholesky) and the bare-bones `logdet_hpd` it builds on. |
| `optimize.py` | The `pga_ascent` outer loop: constant-step projected gradient ascent on a user-supplied MI closure with an optional projector callback. Topology-agnostic. |
| `projections.py` | Closed-form projections onto Frobenius-norm budgets: `project_frobenius_ball` (single matrix) and `project_total_power` (shared budget across many matrices). |

### Design notes

- Every tensor allocated inside this package inherits `device` and
  `dtype` from its inputs. There is no hard-coded `device="cpu"`; the
  library runs unchanged on CUDA when PyTorch's complex-linalg support
  is available there.
- The K-recursion stores only the canonical lower-triangular blocks
  `K[(j, k)]` for `j >= k`; reads of the upper half go through
  `get_K(K, a, b)`, which applies the Hermitian flip
  `K_{ab} = K_{ba}^H` on the fly.
- `pga_ascent` is intentionally minimal: no momentum, no line search, no
  early stopping. The objective is *ascended* (signs are handled
  internally so that the caller's closure returns the MI directly).
