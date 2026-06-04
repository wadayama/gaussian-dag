"""Example: 2-hop Amplify-and-Forward (AF) relay relay-gain optimisation.

DAG:
    V_0 = X  (source)
    V_1 = relay = H_1 X + Z_1     (received at relay)
    V_2 = Y = (H_2 R) V_1 + Z_2   (received at destination, R relay gain)

Edges:
    A_{10} = H_1                  (fixed first-hop channel)
    A_{21} = H_2 @ R              (relay gain R is controllable, sandwiched in
                                   the second-hop channel H_2)
Constraint:
    ||R||_F^2 <= P_R              (Frobenius proxy for relay transmit-power budget)

PGA finds the optimal R; we compare against a uniform-amplification baseline
R = sqrt(P_R / d) * I.

Run: `uv run python examples/af_relay.py`
Outputs:
    examples/results/af_relay.npz
    examples/figures/af_relay_mi.pdf
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
    project_frobenius_ball,
)

DTYPE = torch.complex128

# ----------------------------- configuration -----------------------------

D = 3                  # signal dimension
SIGMA_NOISE = 0.4      # std-dev of relay & destination noise (same for both)
P_R = 3.0              # ||R||_F^2 <= P_R (relay-gain budget)
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


def _mi_for_R(R: torch.Tensor, H1: torch.Tensor, H2: torch.Tensor,
              sigma_x: torch.Tensor, sigma_1: torch.Tensor,
              sigma_2: torch.Tensor) -> torch.Tensor:
    """MI for the 2-hop AF relay parameterised by R."""
    edge_mats = {(1, 0): H1, (2, 1): H2 @ R}
    K = compute_k_blocks(
        3,
        {1: [0], 2: [1]},
        edge_mats,
        sigma_x,
        {1: sigma_1, 2: sigma_2},
    )
    return mutual_information_from_k(K, output_node=2, input_node=0)


# ----------------------------- main -----------------------------


def main() -> None:
    print("=" * 72)
    print("Example: 2-hop AF Relay -- relay-gain optimisation via PGA")
    print("=" * 72)
    print(f"Setup: d={D}, sigma_noise={SIGMA_NOISE}, P_R={P_R}, "
          f"iters={NUM_ITERS}, step={STEP_SIZE}, seed={SEED}")

    gen = torch.Generator().manual_seed(SEED)
    H1 = _randn_complex((D, D), gen)
    H2 = _randn_complex((D, D), gen)
    R_init = _randn_complex((D, D), gen) * 0.1

    sigma_x = torch.eye(D, dtype=DTYPE, device=DEVICE)
    sigma_n = (SIGMA_NOISE ** 2) * torch.eye(D, dtype=DTYPE, device=DEVICE)

    R = R_init.clone().requires_grad_(True)

    def compute_mi() -> torch.Tensor:
        return _mi_for_R(R, H1, H2, sigma_x, sigma_n, sigma_n)

    def projector(params: list[torch.Tensor]) -> None:
        for p in params:
            p.copy_(project_frobenius_ball(p, P_R))

    # Baselines.
    with torch.no_grad():
        # Uniform amplification: R = sqrt(P/d) * I.
        R_uniform = ((P_R / D) ** 0.5) * torch.eye(D, dtype=DTYPE, device=DEVICE)
        mi_uniform = _mi_for_R(R_uniform, H1, H2, sigma_x, sigma_n, sigma_n).item()

        # "No relay" reference: PGA on a tiny relay → effectively no signal forwarded.
        R_tiny = 1e-6 * torch.eye(D, dtype=DTYPE, device=DEVICE)
        mi_no_relay = _mi_for_R(R_tiny, H1, H2, sigma_x, sigma_n, sigma_n).item()

        initial_mi = compute_mi().item()

    print(f"\nBaselines:")
    print(f"  Initial MI (small R):                {initial_mi:.4f} nats")
    print(f"  R = sqrt(P_R/d) * I (uniform amp.):  {mi_uniform:.4f} nats")
    print(f"  No relay (R ~ 0):                    {mi_no_relay:.4f} nats")

    # PGA.
    print(f"\nRunning PGA for {NUM_ITERS} iterations...")
    history = pga_ascent(
        compute_mi, [R],
        step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
    )

    # Re-evaluate MI at the final R (history[-1] is recorded before the last
    # update in pga_ascent; closure() returns the MI at the truly final state).
    with torch.no_grad():
        final_mi = float(compute_mi().item())
    final_norm_sq = (R.detach().norm() ** 2).item()
    print(f"\nResults:")
    print(f"  Final MI:                      {final_mi:.4f} nats")
    print(f"  Improvement over initial:      {final_mi - initial_mi:+.4f} nats")
    print(f"  Improvement over uniform R:    {final_mi - mi_uniform:+.4f} nats")
    print(f"  ||R||_F^2 final:               {final_norm_sq:.4f} (budget {P_R})")

    # Save results.
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    figures_dir = script_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / "af_relay.npz",
        history=np.array(history, dtype=np.float64),
        initial_mi=initial_mi,
        final_mi=final_mi,
        mi_uniform=mi_uniform,
        mi_no_relay=mi_no_relay,
        R_final=R.detach().cpu().numpy(),
        H1=H1.cpu().numpy(),
        H2=H2.cpu().numpy(),
        final_R_norm_sq=final_norm_sq,
        config=dict(
            d=D, sigma_noise=SIGMA_NOISE, P_R=P_R,
            num_iters=NUM_ITERS, step_size=STEP_SIZE, seed=SEED,
        ),
    )
    print(f"\nSaved results: {results_dir / 'af_relay.npz'}")

    # Plot.
    plt.figure(figsize=(7, 4.5))
    iters = np.arange(NUM_ITERS)
    plt.plot(iters, history, label="PGA on $R$", linewidth=1.5)
    plt.axhline(mi_uniform, linestyle="--", color="gray",
                label=f"Uniform $R=\\sqrt{{P_R/d}}\\,I$: {mi_uniform:.3f}")
    plt.axhline(mi_no_relay, linestyle=":", color="red",
                label=f"No relay ($R\\approx 0$): {mi_no_relay:.3f}")
    plt.xlabel("Iteration")
    plt.ylabel("Mutual information $I(X;Y)$ [nats]")
    plt.title(f"2-hop AF relay: relay-gain optimisation ($d={D}$, $P_R={P_R}$)")
    plt.legend(loc="lower right")
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    fig_path = figures_dir / "af_relay_mi.pdf"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved figure:  {fig_path}")


if __name__ == "__main__":
    main()
