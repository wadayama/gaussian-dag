# Tutorial 2 — Building a DAG and reading K-blocks

In Tutorial 1 we used a 2-node DAG with one edge. This tutorial builds a
larger DAG (the *diamond*) and shows how to read individual K-blocks. The
diamond is the smallest DAG where branching and merging matter — the
parent cross-covariance at the merging node is essential for the correct
output covariance.

By the end of this tutorial you will understand:

- The `parents` / `edge_mats` dictionaries that specify a DAG.
- The canonical storage of K-blocks (only `K[(j, k)]` for `j >= k`).
- The `get_K` accessor and the Hermitian flip `K_{ab} = K_{ba}^H` for
  `a < b`.
- Why merging nodes depend on parent cross-covariances.

---

## 1. The diamond DAG

```
            ┌── V_1 ──┐
   V_0 ─────┤         ├──► V_3
            └── V_2 ──┘
```

The node equations are

```
V_0 = X         ~ CN(0, I_d)
V_1 = A_{1,0} V_0 + Z_1
V_2 = A_{2,0} V_0 + Z_2
V_3 = A_{3,1} V_1 + A_{3,2} V_2 + Z_3.
```

Two paths branch at `V_0` and merge at `V_3`. The merging block `K_{3,3}`
will involve the parent cross-covariance `K_{2,1}` — the central point of
this tutorial.

(The accompanying paper uses 1-based indexing — `V_1` is the root, `V_4` is
the sink. The library uses 0-based indexing; the structure is the same.)

---

## 2. Specifying the DAG

```python
import torch
from gaussian_dag import compute_k_blocks, get_K

torch.manual_seed(0)
d, sigma = 2, 0.3

dtype = torch.complex128
randn_c = lambda *s: torch.randn(*s, dtype=dtype)

# Edge matrices.
A_10 = 0.5 * randn_c(d, d)   # branch precoder, V_0 -> V_1
A_20 = 0.5 * randn_c(d, d)   # branch precoder, V_0 -> V_2
A_31 = randn_c(d, d)         # merge,           V_1 -> V_3
A_32 = randn_c(d, d)         # merge,           V_2 -> V_3

Sigma_X = torch.eye(d, dtype=dtype)
Sigma_Z = (sigma ** 2) * torch.eye(d, dtype=dtype)

K = compute_k_blocks(
    num_nodes=4,
    parents={1: [0], 2: [0], 3: [1, 2]},
    edge_mats={
        (1, 0): A_10,
        (2, 0): A_20,
        (3, 1): A_31,
        (3, 2): A_32,
    },
    input_cov=Sigma_X,
    noise_covs={1: Sigma_Z, 2: Sigma_Z, 3: Sigma_Z},
)
```

After the call:

- `K[(0,0)] = Σ_X`.
- `K[(1,1)], K[(2,2)], K[(3,3)]` are the node self-covariances.
- `K[(1,0)], K[(2,0)], K[(3,0)], K[(2,1)], K[(3,1)], K[(3,2)]` are the
  cross-covariances (all those with `j >= k`).
- `K[(1,2)]` is **not** stored — use `get_K(K, 1, 2)` and the library will
  return `K[(2,1)].conj().T` for you.

---

## 3. Reading K-blocks: cross-covariance at the merge

```python
K_21 = get_K(K, 2, 1)        # parent cross-cov at the merging node V_3
print(f"||K_{{2,1}}||_F = {torch.linalg.norm(K_21).item():.4f}")

# K_{3,3} depends on K_{2,1} and K_{1,2} = K_{2,1}^H via
#   K_{3,3} = A_{3,1} K_{1,1} A_{3,1}^H
#            + A_{3,1} K_{1,2} A_{3,2}^H
#            + A_{3,2} K_{2,1} A_{3,1}^H
#            + A_{3,2} K_{2,2} A_{3,2}^H
#            + Sigma_3.
cross_contribution = (
    A_31 @ get_K(K, 1, 2) @ A_32.mH
    + A_32 @ get_K(K, 2, 1) @ A_31.mH
)
print(f"||cross contribution||_F = "
      f"{torch.linalg.norm(cross_contribution).item():.4f}")
print(f"||K_{{3,3}}||_F           = "
      f"{torch.linalg.norm(K[(3, 3)]).item():.4f}")
```

The cross contribution is a non-trivial fraction of the merging block
`K_{3,3}`. If you naively built `Σ_Y` by tracking only the diagonal blocks
`K_{j,j}` and ignored the parent cross-covariance, the resulting matrix
would be **wrong** — and so would the mutual information.

---

## 4. Hermitian-flip storage convention

To save memory, only the canonical half `{(j, k) : j >= k}` is stored. The
library reads the other half via the Hermitian flip:

```python
flipped = get_K(K, 1, 2)               # not stored directly
expected = K[(2, 1)].conj().T          # K_{1,2} = K_{2,1}^H
assert torch.allclose(flipped, expected)
```

You almost never need to do this by hand — `get_K(K, a, b)` does it for
you. Use it whenever you need a block in either order.

---

## 5. What is next?

- **Tutorial 3** turns the branch precoders `A_{1,0}, A_{2,0}` into trainable
  controllable factors and shows how PGA maximises `I(V_0; V_3)` under a
  shared Frobenius-ball budget.
- **Tutorial 4** introduces parameter sharing.
- **Tutorial 5** reproduces Figure 5 of the paper end-to-end.
