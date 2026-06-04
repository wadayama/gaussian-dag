"""Multi-layer Gaussian network: end-to-end MI optimisation under a shared budget.

This example reproduces the multi-layer random-network experiment of
Fig. 5 of the accompanying paper: a 5-layer Gaussian DAG (1 source +
3 relay layers of 3 nodes each + 1 sink, M = 11 nodes, 17 edges,
d = 4 dimensions per node) is built with fixed random per-edge channels;
each of the 9 relay nodes carries a controllable processing matrix
F_i in C^{d x d}, shared across that relay's outgoing edges (broadcast
semantics); and projected gradient ascent jointly optimises
{F_i}_{i=1}^{9} under a shared total-power budget

    sum_{i=1}^{9} ||F_i||_F^2 <= P,   P = 36.

The uniform initialisation (F_i = sqrt(P/(9 d)) * I) corresponds to
identity processing at every relay. PGA discovers a non-uniform
allocation and substantially raises the end-to-end MI.

Outputs (written next to this file):
    results/multilayer_network.npz   - MI history + final powers + topology
    figures/multilayer_network.pdf   - 2-panel figure (network + MI curve)

Run with:
    uv run python examples/multilayer_network.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gaussian_dag import (
    compute_k_blocks,
    mutual_information_from_k,
    pga_ascent,
    project_total_power,
)

DTYPE = torch.complex128

# ---------------------------------------------------------------------------
# Configuration (Fig. 5 reference setting)
# ---------------------------------------------------------------------------
NUM_LAYERS = 5             # L: source + (L - 2) relay layers + sink
LAYER_WIDTH = 3            # nodes per relay layer
D = 4                      # per-node vector dimension
EDGE_PROB = 0.6            # layer-to-layer random connection probability
CHANNEL_SIGMA2 = 1.0       # per-edge complex Gaussian channel variance
NOISE_VAR = 1.0            # per-node white-noise variance
TOTAL_POWER = 36.0         # shared budget sum_i ||F_i||_F^2 <= P
NETWORK_SEED = 7           # random topology + channels (deterministic)
INIT_SEED = 42             # PyTorch seed for the optimisation closure
NUM_ITERS = 120
STEP_SIZE = 0.05


# ===========================================================================
# Random network generation
# ===========================================================================
def build_random_network(
    num_layers: int,
    layer_width: int,
    d: int,
    edge_prob: float,
    channel_sigma2: float,
    seed: int,
) -> tuple[int, dict[int, list[int]], list[tuple[int, int]], np.ndarray, dict]:
    """Generate a random layered Gaussian DAG and fixed per-edge channels.

    Layer 0 is the source (1 node). Layers 1..L-2 are relay layers
    (`layer_width` nodes each). Layer L-1 is the sink (1 node). Each node in
    a layer connects to each node of the previous layer with probability
    `edge_prob`; every node is guaranteed at least one parent and (if
    non-sink) at least one child.

    Channel entries are i.i.d. CN(0, channel_sigma2) per edge.

    Returns
    -------
    M : int
        Total number of nodes (indices 0..M-1).
    parents : dict[int, list[int]]
        parents[j] is the sorted list of parent indices of node j (for j >= 1).
    edges : list[tuple[int, int]]
        Sorted list of (j, i) pairs, one per edge i -> j.
    node_layer : np.ndarray
        Length-M array giving the layer index of each node.
    H : dict[tuple[int, int], torch.Tensor]
        Fixed per-edge channel matrices, shape (d, d) each, complex128.
    """
    # Build layers.
    layers = [[0]]
    idx = 1
    for _ in range(num_layers - 2):
        layers.append(list(range(idx, idx + layer_width)))
        idx += layer_width
    layers.append([idx])  # sink
    M = idx + 1
    node_layer = np.zeros(M, dtype=np.int64)
    for ell, layer in enumerate(layers):
        for n in layer:
            node_layer[n] = ell

    # Random parent selection (NumPy RNG for reproducible topology).
    rng = np.random.RandomState(seed)
    parents: dict[int, list[int]] = {}
    for ell in range(1, num_layers):
        prev, cur = layers[ell - 1], layers[ell]
        for n in cur:
            mask = rng.rand(len(prev)) < edge_prob
            if not mask.any():
                mask[rng.randint(len(prev))] = True
            parents[n] = [prev[k] for k in range(len(prev)) if mask[k]]
        # Guarantee every previous-layer node has at least one child here.
        for p in prev:
            if not any(p in parents[n] for n in cur):
                parents[cur[rng.randint(len(cur))]].append(p)
    for n in parents:
        parents[n] = sorted(set(parents[n]))
    edges = [(j, i) for j in sorted(parents) for i in parents[j]]

    # Fixed per-edge complex Gaussian channels.
    torch_rng = torch.Generator().manual_seed(seed + 1)
    scale = (channel_sigma2 / 2.0) ** 0.5
    H: dict[tuple[int, int], torch.Tensor] = {}
    for (j, i) in edges:
        real = torch.randn(d, d, generator=torch_rng, dtype=torch.float64)
        imag = torch.randn(d, d, generator=torch_rng, dtype=torch.float64)
        H[(j, i)] = scale * torch.complex(real, imag).to(DTYPE)

    return M, parents, edges, node_layer, H


# ===========================================================================
# Differentiable MI(source; sink) under shared-budget PGA
# ===========================================================================
def assemble_edge_mats(
    F_list: list[torch.Tensor],
    H: dict[tuple[int, int], torch.Tensor],
    parents: dict[int, list[int]],
) -> dict[tuple[int, int], torch.Tensor]:
    """Compose A_{ji} = H_{ji} F_i for non-source edges; A_{j0} = H_{j0}.

    F_list[i - 1] is the controllable processing matrix of relay node i
    (i = 1, ..., M - 2). The source (i = 0) carries no controllable factor.
    The same F_list[i - 1] is reused for every outgoing edge of relay i,
    realising the broadcast (parameter-sharing) semantics of a relay node.
    """
    edge_mats: dict[tuple[int, int], torch.Tensor] = {}
    for j in parents:
        for i in parents[j]:
            edge_mats[(j, i)] = H[(j, i)] if i == 0 else H[(j, i)] @ F_list[i - 1]
    return edge_mats


def endpoint_mi(
    F_list: list[torch.Tensor],
    H: dict[tuple[int, int], torch.Tensor],
    parents: dict[int, list[int]],
    M: int,
    d: int,
    noise_var: float,
) -> torch.Tensor:
    """Compute I(V_0; V_{M-1}) through the K-recursion."""
    ref = F_list[0]
    eye = torch.eye(d, dtype=ref.dtype, device=ref.device)
    K = compute_k_blocks(
        num_nodes=M,
        parents=parents,
        edge_mats=assemble_edge_mats(F_list, H, parents),
        input_cov=eye,
        noise_covs={j: noise_var * eye for j in range(1, M)},
    )
    return mutual_information_from_k(K, output_node=M - 1, input_node=0)


def run_optimisation() -> dict[str, np.ndarray]:
    torch.manual_seed(INIT_SEED)

    M, parents, edges, node_layer, H = build_random_network(
        NUM_LAYERS, LAYER_WIDTH, D, EDGE_PROB, CHANNEL_SIGMA2, NETWORK_SEED,
    )
    n_relays = M - 2  # source and sink carry no processing matrix.

    # Uniform initialisation: identity processing at every relay, equal share.
    scale = (TOTAL_POWER / (n_relays * D)) ** 0.5
    F_list = [
        (scale * torch.eye(D, dtype=DTYPE)).clone().requires_grad_(True)
        for _ in range(n_relays)
    ]
    power_init = np.array(
        [float(f.detach().norm() ** 2) for f in F_list], dtype=np.float64,
    )

    def projector(params: list[torch.Tensor]) -> None:
        projected = project_total_power(params, TOTAL_POWER)
        for p, p_proj in zip(params, projected):
            p.copy_(p_proj)

    def closure() -> torch.Tensor:
        return endpoint_mi(F_list, H, parents, M, D, NOISE_VAR)

    mi_history = pga_ascent(
        closure, F_list,
        step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
    )

    with torch.no_grad():
        mi_final = float(closure().item())
    power_final = np.array(
        [float(f.detach().norm() ** 2) for f in F_list], dtype=np.float64,
    )
    F_final = np.stack([f.detach().numpy() for f in F_list])

    return {
        "mi_history": np.array(mi_history, dtype=np.float64),
        "mi_init": np.array(float(mi_history[0]), dtype=np.float64),
        "mi_final": np.array(mi_final, dtype=np.float64),
        "power_init": power_init,
        "power_final": power_final,
        "edges": np.array(edges, dtype=np.int64),
        "node_layer": node_layer,
        "F_final": F_final,
        "M": np.array(M, dtype=np.int64),
        "d": np.array(D, dtype=np.int64),
        "total_power": np.array(TOTAL_POWER, dtype=np.float64),
        "num_iters": np.array(NUM_ITERS, dtype=np.int64),
    }


# ===========================================================================
# Plotting (strictly from results.npz)
# ===========================================================================
def plot_results(npz_path: Path, fig_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })

    data = np.load(npz_path)
    edges = data["edges"]
    node_layer = data["node_layer"]
    power_final = data["power_final"]
    mi_history = data["mi_history"]
    M = int(data["M"])

    # Node positions: x = layer, y = centred slot within the layer.
    layer_nodes: dict[int, list[int]] = {}
    for n in range(M):
        layer_nodes.setdefault(int(node_layer[n]), []).append(n)
    pos = {}
    for ell, nodes in layer_nodes.items():
        n_l = len(nodes)
        for k, n in enumerate(sorted(nodes)):
            pos[n] = (float(ell), k - (n_l - 1) / 2.0)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.6, 5.4))

    # Panel (a): the network, relay nodes shaded by allocated power.
    for (j, i) in edges:
        axA.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                 "-", color="k", linewidth=0.5, zorder=1)
    relays = list(range(1, M - 1))
    xs = [pos[n][0] for n in relays]
    ys = [pos[n][1] for n in relays]
    sc = axA.scatter(xs, ys, c=power_final, cmap=mpl.cm.plasma, s=70,
                     edgecolors="k", linewidths=0.5, zorder=2)
    for endpoint in (0, M - 1):
        axA.scatter([pos[endpoint][0]], [pos[endpoint][1]], marker="s", s=70,
                    color="white", edgecolors="k", linewidths=0.8, zorder=2)
    axA.annotate("source", pos[0], textcoords="offset points", xytext=(0, 28),
                 ha="center", fontsize=7)
    axA.annotate("sink", pos[M - 1], textcoords="offset points", xytext=(0, 28),
                 ha="center", fontsize=7)
    axA.set_title("(a) multi-layer Gaussian network")
    axA.set_xlabel("layer")
    axA.set_yticks([])
    axA.set_xlim(-0.5, float(node_layer.max()) + 0.5)
    cbar = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.03)
    cbar.set_label(r"node power $\|F_i\|_F^2$", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    # Panel (b): end-to-end MI vs iteration.
    it = np.arange(len(mi_history))
    axB.axhline(float(mi_history[0]), ls="--", color="0.6", linewidth=0.7,
                label="uniform allocation")
    axB.plot(it, mi_history, "-", color="tab:blue", linewidth=1.0,
             label="optimised")
    axB.set_xlabel("projected-gradient iteration")
    axB.set_ylabel(r"$I(\mathrm{source};\mathrm{sink})$ (nats)")
    axB.set_title("(b) end-to-end mutual information")
    axB.set_xlim(int(it[0]), int(it[-1]))
    axB.legend()

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    here = Path(__file__).resolve().parent

    print("[optimise] maximising I(source; sink) on a random multi-layer "
          "Gaussian network ...")
    res = run_optimisation()

    results_dir = here / "results"
    results_dir.mkdir(exist_ok=True)
    npz_path = results_dir / "multilayer_network.npz"
    np.savez(npz_path, **res)
    print(f"          wrote {npz_path}")

    figures_dir = here / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / "multilayer_network.pdf"
    plot_results(npz_path, fig_path)
    print(f"          wrote {fig_path}")

    pw = res["power_final"]
    print("\n=== Summary ===")
    print(f"network: M = {int(res['M'])} nodes, "
          f"{res['edges'].shape[0]} edges")
    print(f"I(source; sink): {float(res['mi_init']):.4f} (uniform) -> "
          f"{float(res['mi_final']):.4f} (optimised)")
    print(f"total power Sum||F_i||^2 = {float(pw.sum()):.4f} "
          f"(budget {float(res['total_power']):.1f})")
    print(f"per-node power: min = {pw.min():.3f}  max = {pw.max():.3f}  "
          f"(uniform share = {float(res['total_power']) / len(pw):.3f})")


if __name__ == "__main__":
    main()
