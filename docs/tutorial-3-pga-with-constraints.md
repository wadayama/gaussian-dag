# Tutorial 3 — PGA optimisation with constraints

Tutorials 1 and 2 *evaluated* mutual information on fixed DAGs. This tutorial
*optimises* it: we insert a learnable precoder `F`, then maximise `I(X; Y)`
under a Frobenius power budget using projected gradient ascent (PGA).

By the end you will understand:

- How to mark a tensor as controllable (`requires_grad_(True)`).
- How PyTorch's complex autograd produces the Wirtinger gradient.
- How to call `pga_ascent` with a closure and a projector.
- The two ready-made projectors:
  `project_frobenius_ball` (single matrix) and
  `project_total_power` (shared budget across many matrices).

The reference problem is single-link MIMO with a precoder: `Y = H F X + Z`.
PGA on this problem converges to the classical water-filling optimum — a
useful sanity check.

---

## 1. Define the problem

```python
import torch
from gaussian_dag import (
    compute_k_blocks, mutual_information_from_k,
    pga_ascent, project_frobenius_ball,
)

torch.manual_seed(0)
d, sigma, P = 3, 0.5, 5.0
dtype = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H = torch.randn(d, d, dtype=dtype, device=DEVICE)
Sigma_X = torch.eye(d, dtype=dtype, device=DEVICE)
Sigma_Z = (sigma ** 2) * torch.eye(d, dtype=dtype, device=DEVICE)

# The controllable precoder F: start small, ask autograd to track it.
F = (0.1 * torch.randn(d, d, dtype=dtype, device=DEVICE)).requires_grad_(True)
```

`F.requires_grad_(True)` is the signal to PyTorch that this tensor is a
*leaf* of the autograd graph. Every operation that consumes `F` will be
recorded; when we eventually call `.backward()` on the MI scalar,
`F.grad` will be populated.

For a real-valued scalar `I` and a complex leaf `F`, PyTorch's `.grad`
contains `2 · ∂I/∂F*` (twice the Wirtinger conjugate-side derivative). The
factor of 2 is the real-Euclidean steepest-ascent direction on the real and
imaginary parts of `F`. `pga_ascent` ascends in that direction directly; the
factor of 2 is absorbed into your step size.

---

## 2. The MI closure

PGA needs a callable that, given the current state of the parameters,
builds the autograd graph and returns the scalar MI:

```python
def compute_mi():
    K = compute_k_blocks(
        num_nodes=2,
        parents={1: [0]},
        edge_mats={(1, 0): H @ F},     # A_{1,0} = H F, F is controllable
        input_cov=Sigma_X,
        noise_covs={1: Sigma_Z},
    )
    return mutual_information_from_k(K, output_node=1, input_node=0)
```

Note how `F` enters the *edge matrix*, not the noise or the input. Anything
captured in the closure that depends on `F` will flow gradients back to
`F.grad`.

---

## 3. The projector

A projector takes the parameter list and returns each iterate to the
feasible set. `pga_ascent` accepts two equivalent calling conventions:

**(a) In-place** — the projector mutates `params` directly and returns
`None`:

```python
def projector(params):
    for p in params:
        p.copy_(project_frobenius_ball(p, P))
```

**(b) Functional** — the projector returns a sequence of new tensors,
one per parameter; `pga_ascent` copies them into place for you:

```python
def projector(params):
    return [project_frobenius_ball(p, P) for p in params]
```

Either is fine. The built-in projectors `project_frobenius_ball` and
`project_total_power` return new tensors (they do not mutate the input),
so convention (b) lets you wire them in without an explicit `.copy_`
wrapper.

`project_frobenius_ball(A, P)` returns `A · min(1, sqrt(P) / ‖A‖_F)`, the
Euclidean projection of `A` onto `{X : ‖X‖_F² ≤ P}`.

---

## 4. Run PGA

```python
history = pga_ascent(
    compute_mi,
    [F],
    step_size=0.05,
    num_iters=200,
    projector=projector,
)
print(f"initial MI = {history[0]:.4f} nats")
print(f"final MI   = {history[-1]:.4f} nats")
print(f"||F||_F^2  = {(F.detach().norm() ** 2).item():.4f} "
      f"(budget {P})")
```

`pga_ascent` returns a list of MI values, one per iteration (evaluated at
the parameter values *before* that iteration's update). The final MI should
match the classical water-filling solution for this channel.

You can confirm that against a closed-form computation:

```python
import numpy as np

with torch.no_grad():
    H_np = H.cpu().numpy()
    eigvals = np.linalg.eigvalsh(H_np.conj().T @ H_np) / (sigma ** 2)

def water_filling(eigvals, P):
    eigvals = np.sort(eigvals)[::-1]
    n = len(eigvals)
    for k in range(n, 0, -1):
        mu = (P + (1.0 / eigvals[:k]).sum()) / k
        p = np.maximum(mu - 1.0 / eigvals[:k], 0.0)
        if (p > 0).all():
            return float(np.log(1 + eigvals[:k] * p).sum())
    return 0.0

print(f"water-filling MI = {water_filling(eigvals, P):.4f} nats")
```

The two numbers should agree to ~6 digits in IEEE double precision.

---

## 5. Shared-budget PGA across many matrices

For a DAG with several controllable factors sharing a *single* total power
budget,

```
sum_m ||A_m||_F^2 <= P_total,
```

use `project_total_power` instead. It applies a single common scale factor
to every matrix:

```python
from gaussian_dag import project_total_power

A_a = (0.5 * torch.randn(d, d, dtype=dtype, device=DEVICE)).requires_grad_(True)
A_b = (0.5 * torch.randn(d, d, dtype=dtype, device=DEVICE)).requires_grad_(True)
P_total = 4.0

def shared_projector(params):
    return project_total_power(params, P_total)

# Use shared_projector in your pga_ascent call exactly as before.
# (Both calling conventions work; the functional form above is shorter
# and avoids a subtle footgun where the `.copy_` step is forgotten.)
```

A single common scale factor preserves the *relative* magnitudes of the
matrices, which is the exact Euclidean projection of the stacked vector
onto a single Frobenius ball. This is the same projection used by the
multi-layer experiment in Tutorial 5.

---

## 6. What is next?

- **Tutorial 4** introduces parameter sharing: a single controllable
  matrix appearing in multiple edges (relay broadcast).
- **Tutorial 5** reproduces Figure 5 of the paper end-to-end.
