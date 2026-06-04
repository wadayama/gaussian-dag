# Tutorial 5 — Reproducing Figure 5 of the paper

This capstone tutorial walks through the multi-layer Gaussian network
experiment of Fig. 5 of the accompanying paper: 11 nodes spanning 5 layers
(1 source + 3 relay layers of 3 nodes each + 1 sink), 17 edges, per-node
dimension `d = 4`. Each of the 9 relays carries a controllable processing
matrix `F_i`, shared across that relay's outgoing edges. Projected gradient
ascent jointly optimises `{F_i}_{i=1}^{9}` under a shared total power
budget `P = 36`. The end-to-end MI rises from ~4.56 nats (uniform
allocation) to ~9.28 nats.

All of this is implemented as `examples/multilayer_network.py`. This
tutorial explains what is in that script piece by piece. To run it
end-to-end at any time:

```bash
uv run python examples/multilayer_network.py
```

The script writes
`examples/results/multilayer_network.npz` and
`examples/figures/multilayer_network.pdf`.

---

## 1. Build the random topology

The DAG is layered: layer 0 is the source, layers 1–3 are relay layers of
3 nodes each, and layer 4 is the sink. Each node in layer `ell` connects
to each node of layer `ell - 1` with probability `0.6`, with two
safety nets:

- Every node must have at least one parent.
- Every non-sink node must have at least one child.

Per-edge channels are independent `CN(0, 1)` matrices (shape `d × d`).

This is encapsulated in `build_random_network(...)`. The choice
`network_seed = 7` matches Fig. 5 of the paper.

```python
M, parents, edges, node_layer, H = build_random_network(
    num_layers=5, layer_width=3, d=4,
    edge_prob=0.6, channel_sigma2=1.0, seed=7,
)
print(f"M = {M}, |E| = {len(edges)}")    # Expected: M = 11, |E| = 17
```

---

## 2. Allocate the controllable matrices

The source emits an isotropic signal `X ~ CN(0, I_d)`; only relays carry
processing matrices. With 9 relays and `d = 4`, the budget `P = 36` is
chosen so that the uniform initialisation

```
F_i = sqrt(P / (9 d)) · I_d        = sqrt(1) · I_d  = I_d
```

corresponds to *identity processing* at every relay (i.e., each relay
forwards its received signal unchanged). This is a natural baseline.

```python
import torch
DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_relays = M - 2          # 9
scale = (36.0 / (n_relays * 4)) ** 0.5   # = 1.0
F_list = [
    (scale * torch.eye(4, dtype=DTYPE, device=DEVICE)).clone().requires_grad_(True)
    for _ in range(n_relays)
]
```

---

## 3. The broadcast composition

Each relay's `F_i` is shared across all its outgoing edges (Tutorial 4):

```python
def assemble_edge_mats(F_list, H, parents):
    edge_mats = {}
    for j in parents:
        for i in parents[j]:
            edge_mats[(j, i)] = (
                H[(j, i)] if i == 0
                else H[(j, i)] @ F_list[i - 1]
            )
    return edge_mats
```

- Source-outgoing edges (`i == 0`) carry only the fixed channel.
- Relay-outgoing edges (`i >= 1`) carry `H_{j,i} · F_i`, with `F_i` shared
  across all destinations `j`.

PyTorch's reverse-mode AD will accumulate gradient contributions from every
edge using `F_i` into a single `F_i.grad` — no special handling required.

---

## 4. End-to-end MI

The MI of interest is between the source `V_0` and the sink `V_{M-1}`:

```python
from gaussian_dag import compute_k_blocks, mutual_information_from_k

def endpoint_mi(F_list, H, parents, M, d, noise_var):
    eye = torch.eye(d, dtype=F_list[0].dtype, device=F_list[0].device)
    K = compute_k_blocks(
        num_nodes=M,
        parents=parents,
        edge_mats=assemble_edge_mats(F_list, H, parents),
        input_cov=eye,
        noise_covs={j: noise_var * eye for j in range(1, M)},
    )
    return mutual_information_from_k(K, output_node=M - 1, input_node=0)
```

One forward sweep of `compute_k_blocks` propagates **all** node-pair
covariances along the DAG; `mutual_information_from_k` then reads the
required blocks and returns the differentiable scalar MI.

---

## 5. PGA with a shared-budget projector

The 9 matrices share a single total power budget. We use
`project_total_power`:

```python
from gaussian_dag import pga_ascent, project_total_power

def projector(params):
    projected = project_total_power(params, total_power=36.0)
    for p, p_proj in zip(params, projected):
        p.copy_(p_proj)

history = pga_ascent(
    lambda: endpoint_mi(F_list, H, parents, M, 4, noise_var=1.0),
    F_list, step_size=0.05, num_iters=120, projector=projector,
)
print(f"MI: {history[0]:.4f} -> {history[-1]:.4f} nats")
```

A single common scale factor is applied to *every* `F_i`, preserving the
relative magnitudes that the gradient has discovered.

---

## 6. Expected output

Run the full script:

```bash
uv run python examples/multilayer_network.py
```

You should see (on CPU, IEEE double precision):

```
=== Summary ===
network: M = 11 nodes, 17 edges
I(source; sink): 4.5569 (uniform) -> 9.2763 (optimised)
total power Sum||F_i||^2 = 36.0000 (budget 36.0)
per-node power: min = 2.285  max = 7.195  (uniform share = 4.000)
```

The numbers `4.5569`, `9.2763`, and the power range `2.285–7.195` are
exactly those reported in Fig. 5 of the paper (rounded to two decimal
places there).

The optimised allocation is **non-uniform**: PGA discovered that the
shared budget should be redistributed across relays (some receive more
than the uniform 4.000 share, some less), which is the qualitative point
of Fig. 5.

---

## 7. Where to go from here

- Edit the constants at the top of `examples/multilayer_network.py` —
  larger `NUM_LAYERS`, denser `EDGE_PROB`, different `NETWORK_SEED` — and
  rerun to see how the optimised allocation changes.
- Replace the deterministic-channel model with a stochastic one by
  resampling `H` inside the closure and averaging the MI (Bussgang- or
  fading-style extensions are listed as future work in the paper).
- Apply the same template to a topology of your own: define `parents`,
  build the per-edge channels `H`, decide on the parameter-sharing
  pattern, choose a projector, and call `pga_ascent`. The library does
  not assume any particular topology.

Congratulations — you have walked through the full `gaussian-dag`
pipeline.
