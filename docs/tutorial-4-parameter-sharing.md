# Tutorial 4 — Parameter sharing (relay broadcast)

So far each edge has carried its own independent controllable factor. This
tutorial introduces *parameter sharing*: a single controllable matrix
appearing in multiple edge factorisations simultaneously. This is the
natural mathematical statement of a *relay node*: a relay processes its
input **once** and broadcasts the result to all outgoing edges.

By the end you will understand:

- How to identify multiple edge positions with one autograd leaf.
- How PyTorch's reverse-mode AD accumulates the gradient correctly.
- How the broadcast semantics models a relay.

---

## 1. The relay-broadcast model

Consider a relay node `i` with multiple outgoing edges. Physically, the
relay does *one* internal processing operation, then forwards the result
through each outgoing channel. We capture that by sharing the
controllable factor `F_i`:

```
For every j with i ∈ Pa(j):
    A_{j,i} = H_{j,i} · F_i,
```

where `H_{j,i}` is the fixed channel and `F_i` is the relay's *shared*
controllable processing matrix.

If instead each outgoing edge had its own independent `F_{j,i}`, the relay
would be doing a different processing for every destination — physically
that means a separate transmitter per destination inside the same node,
which is not how a relay node works.

---

## 2. Build a minimal example

We use the simplest non-trivial structure: a 3-node chain `V_0 → V_1 → V_2`
where the relay `V_1` broadcasts to... well, just one downstream node, so
sharing is trivial here. The next example will be a real broadcast.

```python
import torch
from gaussian_dag import (
    compute_k_blocks, mutual_information_from_k,
    pga_ascent, project_frobenius_ball,
)

torch.manual_seed(0)
d, sigma, P = 3, 0.4, 3.0
dtype = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H_1 = torch.randn(d, d, dtype=dtype, device=DEVICE)    # fixed channel V_0 -> V_1
H_2 = torch.randn(d, d, dtype=dtype, device=DEVICE)    # fixed channel V_1 -> V_2
F_1 = (0.1 * torch.randn(d, d, dtype=dtype, device=DEVICE)).requires_grad_(True)

Sigma_X = torch.eye(d, dtype=dtype, device=DEVICE)
Sigma_Z = (sigma ** 2) * torch.eye(d, dtype=dtype, device=DEVICE)

def compute_mi():
    edge_mats = {
        (1, 0): H_1,
        (2, 1): H_2 @ F_1,         # relay V_1 processing
    }
    K = compute_k_blocks(
        num_nodes=3,
        parents={1: [0], 2: [1]},
        edge_mats=edge_mats,
        input_cov=Sigma_X,
        noise_covs={1: Sigma_Z, 2: Sigma_Z},
    )
    return mutual_information_from_k(K, output_node=2, input_node=0)
```

This is a single-relay chain; there is only one outgoing edge from `V_1`,
so sharing has nothing to do yet. Now let us add a real broadcast.

---

## 3. Real broadcast: one relay, two destinations

Imagine `V_1` broadcasts to two distinct sinks `V_2` and `V_3`:

```
                ┌──► V_2
   V_0 ──► V_1 ─┤
                └──► V_3
```

The relay does one processing `F_1` and pushes the result through each
outgoing channel `H_2`, `H_3`:

```python
H_1 = torch.randn(d, d, dtype=dtype, device=DEVICE)   # V_0 -> V_1
H_2 = torch.randn(d, d, dtype=dtype, device=DEVICE)   # V_1 -> V_2
H_3 = torch.randn(d, d, dtype=dtype, device=DEVICE)   # V_1 -> V_3
F_1 = (0.1 * torch.randn(d, d, dtype=dtype, device=DEVICE)).requires_grad_(True)

def compute_mi_broadcast():
    edge_mats = {
        (1, 0): H_1,
        (2, 1): H_2 @ F_1,         # SAME F_1 reused here
        (3, 1): H_3 @ F_1,         # ... and here.
    }
    K = compute_k_blocks(
        num_nodes=4,
        parents={1: [0], 2: [1], 3: [1]},
        edge_mats=edge_mats,
        input_cov=Sigma_X,
        noise_covs={1: Sigma_Z, 2: Sigma_Z, 3: Sigma_Z},
    )
    # I(V_0; V_2) as one example output of interest.
    return mutual_information_from_k(K, output_node=2, input_node=0)
```

Notice the **same** tensor `F_1` is referenced in both
`edge_mats[(2, 1)]` and `edge_mats[(3, 1)]`. There is no need to declare
this sharing to PyTorch — it is implicit in the fact that the same leaf
tensor appears twice in the autograd graph.

---

## 4. Why reverse-mode AD handles this transparently

When we call `compute_mi_broadcast().backward()`, PyTorch traverses the
graph in reverse and **accumulates** gradient contributions through every
path that uses `F_1`. The chain rule does the right thing:

```
∂I / ∂F_1*  =  (gradient through edge (2, 1))
              + (gradient through edge (3, 1)).
```

That is exactly what is needed for a relay-broadcast model: the relay's
processing matrix should respond to its effect on **all** downstream
recipients, not just one.

You do not have to write any of this by hand. One `.backward()` call and
`F_1.grad` is correctly populated.

---

## 5. Implications for PGA

`pga_ascent` does not care that some parameters are shared across multiple
edges — it just sees a list of leaf tensors. The example above can be
optimised exactly like Tutorial 3:

```python
def projector(params):
    for p in params:
        p.copy_(project_frobenius_ball(p, P))

history = pga_ascent(
    compute_mi_broadcast, [F_1],
    step_size=0.05, num_iters=200, projector=projector,
)
```

The gradient flowing back to `F_1` already accounts for every outgoing
edge — so PGA optimises the relay processing under the joint effect on
all downstream nodes.

---

## 6. What is next?

- **Tutorial 5** is the capstone: it reproduces Figure 5 of the paper, a
  multi-layer Gaussian network with 9 broadcast relays optimised under a
  *shared* total-power budget. Everything you have learned in tutorials
  1–4 comes together there.
