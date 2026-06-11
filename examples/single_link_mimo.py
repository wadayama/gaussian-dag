"""Example: Single-link MIMO mutual-information maximization via PGA.

Problem: Y = (H @ F) X + Z, with
    X ~ CN(0, I_d) fixed,
    H constant channel,
    F controllable precoder,  ||F||_F^2 <= P,
    Z ~ CN(0, sigma^2 I_d).

PGA maximises I(X; Y) over F under the Frobenius-ball constraint.
We compare the PGA trajectory against
    (i) the trivial "no precoder" baseline F = sqrt(P/d) * I,
    (ii) the classical waterfilling MI (theoretical optimum under power P).

Run: `uv run python examples/single_link_mimo.py`
Outputs:
    examples/results/single_link_mimo.npz
    examples/figures/single_link_mimo.pdf
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

D = 3                  # MIMO dimension
SIGMA_NOISE = 0.5      # std-dev of receiver noise (Z ~ CN(0, sigma^2 I))
P_BUDGET = 5.0         # Frobenius power budget ||F||_F^2 <= P
NUM_ITERS = 100
STEP_SIZE = 0.05       # PGA step size
SEED = 42

# Device: auto-detect CUDA, fall back to CPU. The library is device-agnostic;
# changing this to torch.device("cpu") forces CPU execution.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- helpers -----------------------------


def _randn_complex(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    """Random complex tensor on DEVICE.

    The PyTorch generator stays CPU-resident (torch.Generator() does not
    follow .cuda()); we sample on CPU and move the result to DEVICE.
    """
    r = torch.randn(*shape, dtype=torch.float64, generator=gen)
    i = torch.randn(*shape, dtype=torch.float64, generator=gen)
    return torch.complex(r, i).to(DEVICE)


def _compute_mi_for_F(F: torch.Tensor, H: torch.Tensor,
                      sigma_x: torch.Tensor, sigma_z: torch.Tensor) -> torch.Tensor:
    """MI of Y = (H @ F) X + Z via the library."""
    edge_mats = {(1, 0): H @ F}
    K = compute_k_blocks(2, {1: [0]}, edge_mats, sigma_x, {1: sigma_z})
    return mutual_information_from_k(K, output_node=1, input_node=0)


def _waterfilling_mi(H: torch.Tensor, sigma_sq: float, P: float,
                     bisection_iters: int = 200) -> tuple[float, np.ndarray]:
    """Optimal MI by waterfilling for Y = HX + Z, X ~ CN(0, Sigma_p), tr(Sigma_p) <= P.

    Returns (mi_opt, p_opt) where p_opt are the optimal eigen-power allocations.
    """
    # Singular values of H (descending order from torch); s_i^2 are the eigenvalues
    # of H^H H = V Lambda V^H.
    # SVD on CPU for portability of the CPU-only reference path below.
    s = torch.linalg.svdvals(H.cpu()).real.double()
    lam = s ** 2  # eigenvalues of H^H H
    noise_eq = sigma_sq / lam.clamp(min=1e-15)  # 1/(s_i^2 / sigma^2)

    # Bisection on water level v such that sum max(v - noise_eq_i, 0) = P.
    v_low = noise_eq.min().item()
    v_high = noise_eq.max().item() + P
    for _ in range(bisection_iters):
        v_mid = 0.5 * (v_low + v_high)
        total = torch.clamp(torch.tensor(v_mid, dtype=torch.float64) - noise_eq, min=0.0).sum().item()
        if total < P:
            v_low = v_mid
        else:
            v_high = v_mid

    p = torch.clamp(torch.tensor(v_high, dtype=torch.float64) - noise_eq, min=0.0)
    mi = torch.sum(torch.log1p(p * lam / sigma_sq)).item()
    return mi, p.cpu().numpy()


# ----------------------------- main -----------------------------


def main() -> None:
    print("=" * 72)
    print("Example: Single-link MIMO MI maximisation via PGA")
    print("=" * 72)
    print(f"Setup: d={D}, sigma_noise={SIGMA_NOISE}, P_budget={P_BUDGET}, "
          f"iters={NUM_ITERS}, step={STEP_SIZE}, seed={SEED}")

    # Reproducible random elements.
    gen = torch.Generator().manual_seed(SEED)
    H = _randn_complex((D, D), gen)
    F_init = _randn_complex((D, D), gen) * 0.1  # small initial precoder

    sigma_x = torch.eye(D, dtype=DTYPE, device=DEVICE)
    sigma_z = (SIGMA_NOISE ** 2) * torch.eye(D, dtype=DTYPE, device=DEVICE)

    # Make F a learnable parameter.
    F = F_init.clone().requires_grad_(True)

    def compute_mi() -> torch.Tensor:
        return _compute_mi_for_F(F, H, sigma_x, sigma_z)

    def projector(params: list[torch.Tensor]) -> None:
        for p in params:
            p.copy_(project_frobenius_ball(p, P_BUDGET))

    # Baseline: uniform allocation F = sqrt(P/d) * I.
    with torch.no_grad():
        F_uniform = ((P_BUDGET / D) ** 0.5) * torch.eye(D, dtype=DTYPE, device=DEVICE)
        mi_uniform = _compute_mi_for_F(F_uniform, H, sigma_x, sigma_z).item()

    # Baseline: theoretical waterfilling optimum.
    mi_optimal, p_optimal = _waterfilling_mi(H, SIGMA_NOISE ** 2, P_BUDGET)

    # Initial MI.
    with torch.no_grad():
        initial_mi = compute_mi().item()
    print(f"\nBaselines:")
    print(f"  Initial MI (small F):      {initial_mi:.4f} nats")
    print(f"  Uniform F = sqrt(P/d)·I:   {mi_uniform:.4f} nats")
    print(f"  Waterfilling optimum:      {mi_optimal:.4f} nats")

    # Run PGA.
    print(f"\nRunning PGA for {NUM_ITERS} iterations...")
    history = pga_ascent(
        compute_mi, [F],
        step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
    )

    # Re-evaluate MI at the final F (history[-1] is recorded before the last
    # update in pga_ascent; closure() returns the MI at the truly final state).
    with torch.no_grad():
        final_mi = float(compute_mi().item())
    final_norm_sq = (F.detach().norm() ** 2).item()
    print(f"\nResults:")
    print(f"  Final MI:        {final_mi:.4f} nats")
    print(f"  ||F||_F^2:       {final_norm_sq:.4f}  (budget {P_BUDGET})")
    print(f"  Gap to optimum:  {mi_optimal - final_mi:.4f} nats")

    # Save results.
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    figures_dir = script_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / "single_link_mimo.npz",
        history=np.array(history, dtype=np.float64),
        initial_mi=initial_mi,
        final_mi=final_mi,
        mi_uniform=mi_uniform,
        mi_optimal=mi_optimal,
        p_optimal=p_optimal,
        final_F=F.detach().cpu().numpy(),
        H=H.cpu().numpy(),
        final_F_norm_sq=final_norm_sq,
        config=dict(
            d=D, sigma_noise=SIGMA_NOISE, P_budget=P_BUDGET,
            num_iters=NUM_ITERS, step_size=STEP_SIZE, seed=SEED,
        ),
    )
    print(f"\nSaved results: {results_dir / 'single_link_mimo.npz'}")

    # Plot MI vs iteration.
    plt.figure(figsize=(7, 4.5))
    iters = np.arange(NUM_ITERS)
    plt.plot(iters, history, label="PGA on $F$", linewidth=1.5)
    plt.axhline(mi_uniform, linestyle="--", color="gray",
                label=f"Uniform $F$: {mi_uniform:.3f}")
    plt.axhline(mi_optimal, linestyle=":", color="red",
                label=f"Waterfilling: {mi_optimal:.3f}")
    plt.xlabel("Iteration")
    plt.ylabel("Mutual information $I(X;Y)$ [nats]")
    plt.title(f"Single-link MIMO MI maximisation ($d={D}$, $P={P_BUDGET}$, "
              f"$\\sigma={SIGMA_NOISE}$)")
    plt.legend(loc="lower right")
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    fig_path = figures_dir / "single_link_mimo.pdf"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved figure:  {fig_path}")


if __name__ == "__main__":
    main()
