"""Example: Diamond DAG mutual-information maximisation via PGA.

DAG topology:
        V_0 = X
       /          \\
    V_1           V_2
       \\          /
        V_3 = Y

Edges:
    A_{10}, A_{20}: controllable branch matrices ("precoders")
    A_{31}, A_{32}: fixed combiner matrices
Constraint:
    ||A_{10}||_F^2 + ||A_{20}||_F^2 <= P  (joint Frobenius budget)

This example highlights the role of the parent-pair cross-covariance
K_{21} = E[V_1 V_2^H] (non-zero because V_1 and V_2 share the ancestor V_0):
the bilinear sum forming K_{33} contains the cross terms
    A_{31} K_{12} A_{32}^H + A_{32} K_{21} A_{31}^H
which are essential to correctness. The example reports the magnitude of
these cross terms relative to ||K_{33}||_F to make the point concrete.

(Note: an "auto-cov-only" baseline that drops the cross-cov inside K_{33}
makes Sigma_Y artificially smaller while the cross-cov in K_{30} = Sigma_{YX}
remains; the resulting Schur complement Sigma_{Y|X} can become non-PSD,
so a "naive MI" is not even well-defined. This itself is evidence that
cross-cov is structurally necessary for the MI computation, not just an
accuracy concern.)

Run: `uv run python examples/diamond_dag.py`
Outputs:
    examples/results/diamond_dag.npz
    examples/figures/diamond_dag_mi.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gaussian_dag import (
    compute_k_blocks,
    mutual_information_from_k,
    pga_ascent,
    project_total_power,
)

DTYPE = torch.complex128

# ----------------------------- configuration -----------------------------

D = 2
SIGMA_NOISE = 0.3
P_BUDGET = 4.0       # joint budget for A_{10} and A_{20}
NUM_ITERS = 100
STEP_SIZE = 0.05
SEED = 42

# Device: auto-detect CUDA, fall back to CPU. The library is device-agnostic;
# changing this to torch.device("cpu") forces CPU execution.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- helpers -----------------------------


def _randn_complex(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    """Random complex tensor on DEVICE.

    The PyTorch generator stays CPU-resident; we sample on CPU and move
    the result to DEVICE.
    """
    r = torch.randn(*shape, dtype=torch.float64, generator=gen)
    i = torch.randn(*shape, dtype=torch.float64, generator=gen)
    return torch.complex(r, i).to(DEVICE)


def _build_diamond_k(A10: torch.Tensor, A20: torch.Tensor,
                     A31: torch.Tensor, A32: torch.Tensor,
                     sigma_x: torch.Tensor, sigma_n: torch.Tensor):
    """Run compute_k_blocks on the diamond DAG."""
    edge_mats = {(1, 0): A10, (2, 0): A20, (3, 1): A31, (3, 2): A32}
    K = compute_k_blocks(
        4,
        {1: [0], 2: [0], 3: [1, 2]},
        edge_mats,
        sigma_x,
        {1: sigma_n, 2: sigma_n, 3: sigma_n},
    )
    return K


def _cross_term_diagnostics(
    A10: torch.Tensor, A20: torch.Tensor,
    A31: torch.Tensor, A32: torch.Tensor,
    sigma_x: torch.Tensor, sigma_n: torch.Tensor,
) -> dict:
    """Return diagnostics on the cross-cov contribution to K_{33}.

    Computes:
      - K_{21} = parent-pair cross-cov,
      - cross terms in K_{33} = A_{31} K_{12} A_{32}^H + A_{32} K_{21} A_{31}^H,
      - the relative Frobenius magnitude of those cross terms inside K_{33}.
    """
    with torch.no_grad():
        K = _build_diamond_k(A10, A20, A31, A32, sigma_x, sigma_n)
        K21 = K[(2, 1)]
        K12 = K21.mH  # Hermitian flip
        K33 = K[(3, 3)]
        cross_part = A31 @ K12 @ A32.mH + A32 @ K21 @ A31.mH
        return {
            "K21_norm": K21.norm().item(),
            "K33_norm": K33.norm().item(),
            "cross_part_norm": cross_part.norm().item(),
            "cross_rel": cross_part.norm().item() / max(K33.norm().item(), 1e-12),
        }


# ----------------------------- main -----------------------------


def main() -> None:
    print("=" * 72)
    print("Example: Diamond DAG MI maximisation via PGA")
    print("=" * 72)
    print(f"Setup: d={D}, sigma_noise={SIGMA_NOISE}, P_budget={P_BUDGET}, "
          f"iters={NUM_ITERS}, step={STEP_SIZE}, seed={SEED}")

    gen = torch.Generator().manual_seed(SEED)

    # Fixed combiner matrices.
    A31_const = _randn_complex((D, D), gen) * 0.5
    A32_const = _randn_complex((D, D), gen) * 0.5

    # Controllable branch matrices (small init for non-trivial PGA progress).
    A10_init = _randn_complex((D, D), gen) * 0.1
    A20_init = _randn_complex((D, D), gen) * 0.1
    A10 = A10_init.clone().requires_grad_(True)
    A20 = A20_init.clone().requires_grad_(True)

    sigma_x = torch.eye(D, dtype=DTYPE, device=DEVICE)
    sigma_n = (SIGMA_NOISE ** 2) * torch.eye(D, dtype=DTYPE, device=DEVICE)

    def compute_mi() -> torch.Tensor:
        K = _build_diamond_k(A10, A20, A31_const, A32_const, sigma_x, sigma_n)
        return mutual_information_from_k(K, output_node=3, input_node=0)

    def projector(params: list[torch.Tensor]) -> None:
        new = project_total_power(params, P_BUDGET)
        for p, np_p in zip(params, new):
            p.copy_(np_p)

    # Initial diagnostics.
    with torch.no_grad():
        initial_mi = compute_mi().item()
        diag_init = _cross_term_diagnostics(
            A10, A20, A31_const, A32_const, sigma_x, sigma_n
        )
    print(f"\nInitial state:")
    print(f"  MI:                                 {initial_mi:.4f} nats")
    print(f"  ||K_21||_F (parent cross-cov):      {diag_init['K21_norm']:.4f}")
    print(f"  ||A_31 K_12 A_32^H + sym||_F:       {diag_init['cross_part_norm']:.4f}")
    print(f"  cross / ||K_33||_F:                 {diag_init['cross_rel']*100:.2f}%")

    # Run PGA with cross-term diagnostics recorded each iteration.
    print(f"\nRunning PGA for {NUM_ITERS} iterations...")
    history_mi: list[float] = []
    history_cross_rel: list[float] = []
    history_K21: list[float] = []
    for _ in range(NUM_ITERS):
        # Zero grads.
        for p in (A10, A20):
            if p.grad is not None:
                p.grad.zero_()
        # Forward + backward on correct MI.
        I = compute_mi()
        I.backward()
        history_mi.append(I.item())
        # Diagnostics (no_grad to skip building autograd graph).
        with torch.no_grad():
            diag = _cross_term_diagnostics(
                A10, A20, A31_const, A32_const, sigma_x, sigma_n
            )
        history_cross_rel.append(diag["cross_rel"])
        history_K21.append(diag["K21_norm"])
        # Update + project.
        with torch.no_grad():
            for p in (A10, A20):
                p.add_(STEP_SIZE * p.grad)
            projector([A10, A20])

    # Re-evaluate MI at the final controllable factors (history_mi[-1] is
    # recorded before the last update in the hand-rolled PGA loop above;
    # compute_mi() returns the MI at the truly final state).
    with torch.no_grad():
        final_mi = float(compute_mi().item())
    norm_total = (A10.detach().norm() ** 2 + A20.detach().norm() ** 2).item()
    with torch.no_grad():
        diag_final = _cross_term_diagnostics(
            A10, A20, A31_const, A32_const, sigma_x, sigma_n
        )

    print(f"\nFinal state:")
    print(f"  MI:                                 {final_mi:.4f} nats")
    print(f"  MI improvement vs initial:          {final_mi - initial_mi:+.4f} nats")
    print(f"  ||A_10||^2 + ||A_20||^2:            {norm_total:.4f}  (budget {P_BUDGET})")
    print(f"  ||K_21||_F (parent cross-cov):      {diag_final['K21_norm']:.4f}")
    print(f"  ||A_31 K_12 A_32^H + sym||_F:       {diag_final['cross_part_norm']:.4f}")
    print(f"  cross / ||K_33||_F:                 {diag_final['cross_rel']*100:.2f}%")

    # Save results.
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    figures_dir = script_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / "diamond_dag.npz",
        history_mi=np.array(history_mi, dtype=np.float64),
        history_cross_rel=np.array(history_cross_rel, dtype=np.float64),
        history_K21=np.array(history_K21, dtype=np.float64),
        initial_mi=initial_mi,
        final_mi=final_mi,
        A10_final=A10.detach().cpu().numpy(),
        A20_final=A20.detach().cpu().numpy(),
        A31_const=A31_const.cpu().numpy(),
        A32_const=A32_const.cpu().numpy(),
        norm_total=norm_total,
        diag_init=diag_init,
        diag_final=diag_final,
        config=dict(
            d=D, sigma_noise=SIGMA_NOISE, P_budget=P_BUDGET,
            num_iters=NUM_ITERS, step_size=STEP_SIZE, seed=SEED,
        ),
    )
    print(f"\nSaved results: {results_dir / 'diamond_dag.npz'}")

    # Two-panel plot: MI history + cross-cov contribution.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    iters = np.arange(NUM_ITERS)

    ax1.plot(iters, history_mi, color="C0", linewidth=1.5,
             label="$I(X;Y)$ via full K-recursion")
    ax1.set_ylabel("Mutual information [nats]")
    ax1.set_title(f"Diamond DAG: PGA on $A_{{10}}, A_{{20}}$ ($d={D}$, $P={P_BUDGET}$)")
    ax1.legend(loc="lower right")
    ax1.grid(True, linewidth=0.4)

    ax2.plot(iters, np.array(history_cross_rel) * 100, color="C1", linewidth=1.5,
             label=r"$\|A_{31} K_{12} A_{32}^H + A_{32} K_{21} A_{31}^H\|_F / \|K_{33}\|_F$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cross-term contribution to $K_{33}$ [%]")
    ax2.legend(loc="upper right")
    ax2.grid(True, linewidth=0.4)

    plt.tight_layout()
    fig_path = figures_dir / "diamond_dag_mi.pdf"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved figure:  {fig_path}")


if __name__ == "__main__":
    main()
