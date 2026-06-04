# Tutorial 1 — Installation and your first MI evaluation

This first tutorial walks through installing `gaussian-dag` and computing the
mutual information of a single-link MIMO channel as a sanity check.

By the end of this tutorial you will:

- Have a working `gaussian-dag` environment.
- Understand the model `Y = H X + Z` as a 2-node linear Gaussian DAG.
- Have evaluated `I(X; Y)` with one call to `compute_k_blocks` +
  `mutual_information_from_k`.

---

## 1. Install the library

`gaussian-dag` is a small Python package built on PyTorch. It is recommended
that you use [`uv`](https://docs.astral.sh/uv/) to manage the virtual
environment.

```bash
# Clone the repository.
git clone https://github.com/wadayama/gaussian-dag.git
cd gaussian-dag

# Install dependencies into a fresh .venv (Python >= 3.12 required).
uv sync
```

Confirm the install:

```bash
uv run pytest
```

You should see all tests pass (a couple may be skipped if no CUDA device is
available — that is expected).

---

## 2. The model

We start with the simplest channel: single-link MIMO.

```
X  ─►  [ H ]  ─►  Y = H X + Z,
                      Z ~ CN(0, σ² I_d).
```

In DAG language this is a 2-node graph:

- Node `V_0 = X` is the (unique) root, with `X ~ CN(0, Σ_X)`.
- Node `V_1 = Y` is a non-root, with parent `V_0` and edge transform
  `A_{1,0} = H`.

For circular complex Gaussian `(X, Y)` with positive-definite covariances,
the mutual information has the log-determinant form

```
I(X; Y)  =  log det Σ_Y  −  log det Σ_{Y|X},
```

with `Σ_{Y|X} = Σ_Y − Σ_{YX} Σ_X^{−1} Σ_{XY}` the Schur-complement
conditional covariance. `gaussian-dag` evaluates this directly from the
K-recursion.

---

## 3. Compute `I(X; Y)`

```python
import torch
from gaussian_dag import compute_k_blocks, mutual_information_from_k

torch.manual_seed(0)
d, sigma = 3, 0.5

# Device-agnostic: same code runs on CPU or CUDA.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed channel and covariances.
H = torch.randn(d, d, dtype=torch.complex128, device=DEVICE)
Sigma_X = torch.eye(d, dtype=torch.complex128, device=DEVICE)
Sigma_Z = (sigma ** 2) * torch.eye(d, dtype=torch.complex128, device=DEVICE)

# Build the DAG: 2 nodes, one edge (0 -> 1) carrying H.
K = compute_k_blocks(
    num_nodes=2,
    parents={1: [0]},            # node 0 is the root and need not be listed
    edge_mats={(1, 0): H},       # A_{1,0} = H
    input_cov=Sigma_X,
    noise_covs={1: Sigma_Z},
)

mi = mutual_information_from_k(K, output_node=1, input_node=0)
print(f"I(X; Y) = {mi.item():.4f} nats")
```

What just happened:

- `compute_k_blocks` propagated the covariance `Σ_X` through the DAG and
  produced the canonical K-blocks `K[(0,0)], K[(1,0)], K[(1,1)]`.
- `mutual_information_from_k` read `Σ_Y = K[(1,1)]`, `Σ_{YX} = K[(1,0)]`,
  `Σ_X = K[(0,0)]`, formed the Schur complement, and evaluated the log-det
  difference. Both steps are differentiable.

The returned `mi` is a scalar PyTorch tensor in **nats**.

> **Running on GPU.** The `DEVICE = torch.device("cuda" if ...)` line
> above auto-detects CUDA. Every tensor we allocate passes `device=DEVICE`,
> so the K-blocks produced by `compute_k_blocks` live on that device
> automatically, and `mutual_information_from_k` follows. On a CPU-only
> machine this falls back to CPU; no further change required. To force
> CPU on a CUDA machine, edit the single line to
> `DEVICE = torch.device("cpu")`. See `examples/` for the same pattern
> applied to the full paper-reproduction scripts.

---

## 4. What is next?

- **Tutorial 2** builds a richer DAG (the diamond DAG) and looks inside the
  K-blocks to understand cross-covariance structure.
- **Tutorial 3** turns the channel into an optimisation problem by inserting
  a learnable precoder and running projected gradient ascent.
- **Tutorial 4** introduces parameter sharing (relay-broadcast semantics).
- **Tutorial 5** reproduces Figure 5 of the accompanying paper.
