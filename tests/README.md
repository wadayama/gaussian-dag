# gaussian_dag — test suite

29 tests covering the four library modules and a small GPU smoke layer.
Run the full suite from the repository root:

```bash
uv run pytest -q
```

Expected outcome on a CPU-only machine: **27 passed, 2 skipped** (the two
GPU smoke tests skip themselves cleanly when CUDA is unavailable). On a
CUDA-enabled machine all 29 tests should pass.

## Files

### `test_krecursion.py` — forward K-recursion (7 tests)

| Test | What it checks |
| --- | --- |
| `test_key_coverage_chain` | `compute_k_blocks` returns exactly the canonical keys `{(j, k) : 0 <= k <= j}` for a chain DAG. |
| `test_key_coverage_diamond` | Same coverage check on the 4-node diamond. |
| `test_hermitianize_idempotent_and_hermitian` | `hermitianize(A)` is exactly Hermitian and idempotent under repeated application. |
| `test_get_K_hermitian_flip` | `get_K(K, a, b)` returns the canonical block for `a >= b` and `K[(b, a)].mH` for `a < b`. |
| `test_chain_deterministic` | On a deterministic chain DAG, the K-blocks match the analytical closed form (no Monte Carlo). |
| `test_diamond_cross_covariance` | On the diamond, the parent cross-covariance at the merging node is correctly assembled (self-consistency of the recursion formula). |
| `test_diamond_brute_force_K33` | `K_{3,3}` from `compute_k_blocks` matches a closed-form expansion of `V_3` in terms of the independent sources `(X, Z_1, Z_2, Z_3)`; an end-to-end correctness check that never refers to intermediate K-recursion outputs. |

### `test_information.py` — log-det MI layer (6 tests)

| Test | What it checks |
| --- | --- |
| `test_logdet_hpd_identity` | `logdet_hpd(I) == 0` in any dimension. |
| `test_logdet_hpd_vs_slogdet` | `logdet_hpd(A)` matches `torch.linalg.slogdet(A).logabsdet` on random HPD matrices (independent route). |
| `test_single_link_mimo_classical_mi` | MI from the K-recursion on `Y = H X + Z` matches `log det(I + H Σ_X H^H / σ²)` (closed form). |
| `test_mi_nonneg_diamond` | `I(X; Y) >= 0` on the diamond DAG (sanity check on a general topology). |
| `test_logdet_hpd_raises_on_singular_matrix` | A rank-deficient PSD input triggers a clear `ValueError` (via `cholesky_ex`); the message lists the three suggested remedies. |
| `test_logdet_hpd_jitter_recovers_singular_case` | A small `jitter > 0` lets `logdet_hpd` succeed on an otherwise singular input (finite output). |

### `test_gradients.py` — autograd correctness (2 tests)

| Test | What it checks |
| --- | --- |
| `test_autograd_basic_precoder` | `F.grad` is populated and non-trivial after `I.backward()` for `Y = (H F) X + Z`. |
| `test_finite_difference_precoder` | The autograd Wirtinger gradient agrees with a central finite-difference gradient on the single-link MIMO closure. |

### `test_montecarlo.py` — Monte-Carlo cross-check (1 test)

| Test | What it checks |
| --- | --- |
| `test_diamond_monte_carlo` | The K-recursion blocks agree with the empirical covariance estimated from a large Monte Carlo sample on the diamond DAG. |

### `test_optimize.py` — projections and PGA (11 tests)

| Test | What it checks |
| --- | --- |
| `test_project_frobenius_ball_inside_is_identity` | Inside the ball → identity. |
| `test_project_frobenius_ball_outside_lies_on_boundary` | Outside the ball → rescaled to the boundary. |
| `test_project_frobenius_ball_rejects_nonpositive_P` | `P <= 0` raises. |
| `test_project_total_power_outside` | Multiple matrices outside the shared-budget ball: a single common scale brings the stacked vector onto the boundary. |
| `test_project_total_power_inside_is_identity` | Multiple matrices already inside → identity. |
| `test_pga_increases_mi_mimo_with_frobenius_ball` | PGA improves MI on a constrained single-link MIMO over a few iterations. |
| `test_pga_rejects_invalid_args` | `pga_ascent` raises on invalid arguments (non-positive step size, non-positive iteration count, parameters without `requires_grad`). |
| `test_pga_accepts_functional_projector` | A projector that returns new tensors (no `.copy_` inside) is honoured by `pga_ascent` via auto-copy; protects against silently discarded projections. |
| `test_pga_functional_projector_wrong_length_raises` | A functional projector returning the wrong number of tensors raises rather than silently mis-aligning. |
| `test_pga_unused_parameter_raises_clear_error` | A parameter declared with `requires_grad=True` but never used in the closure produces a clear `RuntimeError` identifying the offending param index (not an opaque `add_(step * None)` crash). |
| `test_input_covariance_water_filling_convergence` | The virtual-edge construction `X = Q X̃` optimised by PGA reaches within 0.05 nats of the classical water-filling reference within 200 iterations (loose tolerance for round-off robustness). |

### `test_gpu_smoke.py` — CUDA smoke (2 tests, skipped if no CUDA)

| Test | What it checks |
| --- | --- |
| `test_k_recursion_on_cuda_matches_cpu` | The forward K-recursion + MI evaluation produces the same value on CPU and CUDA. |
| `test_pga_step_on_cuda` | A short PGA loop on CUDA strictly increases the MI, confirming that gradients flow through the K-recursion on GPU. |
