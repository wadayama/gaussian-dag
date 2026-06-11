"""Example: Input-covariance shaping via the virtual-edge construction.

We add a virtual standard-Gaussian source upstream of X so that the input
distribution itself becomes optimisable through an ordinary edge matrix Q.

DAG:
    V_0 = X_tilde ~ CN(0, I_d)       (virtual root, fixed standard Gaussian)
    V_1 = X = Q V_0                  (virtual edge -- Q is controllable; tiny noise)
    V_2 = Y = H V_1 + Z_2            (fixed downstream channel H)

The effective input is X = Q X_tilde ~ CN(0, Sigma_X) with Sigma_X = Q Q^H.
The Frobenius constraint ||Q||_F^2 <= P is exactly tr(Sigma_X) <= P, the
standard average-input-power budget.

Optimisation target: I(X_tilde; Y). In the absence of the tiny stabilisation
noise on V_1, the Markov chain X_tilde -> X = Q X_tilde -> Y gives
I(X_tilde; Y) = I(X; Y) without requiring Q to be invertible, and the
optimum coincides with classical MIMO water-filling on H^H H. With
EPS_V1 > 0 in the implementation below, V_1 is X plus a tiny noise, so this
example optimises the corresponding slightly regularised virtual-edge
objective; the difference from the exact I(X; Y) is O(EPS_V1) and is
invisible at the printed precision. We use the water-filling MI as a
reference to confirm that the virtual-edge construction recovers the same
theoretical optimum reached in Example 1 -- without any special-cased
"input precoder" pathway in the K-recursion / PGA code.

Run: `uv run python examples/input_covariance.py`
Outputs:
    examples/results/input_covariance.npz
    examples/figures/input_covariance.pdf
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

D = 3
SIGMA_NOISE = 0.5      # noise std on V_2 = Y
P_INPUT = 5.0          # ||Q||_F^2 <= P_INPUT  (= tr Sigma_X)
EPS_V1 = 1e-8          # tiny noise on V_1 to keep K_{11} numerically PD
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


def _mi_for_Q(Q: torch.Tensor, H: torch.Tensor, sigma_xt: torch.Tensor,
              sigma_1: torch.Tensor, sigma_2: torch.Tensor) -> torch.Tensor:
    """MI for the virtual-edge DAG parameterised by Q (input shaper)."""
    edge_mats = {(1, 0): Q, (2, 1): H}
    K = compute_k_blocks(
        3,
        {1: [0], 2: [1]},
        edge_mats,
        sigma_xt,
        {1: sigma_1, 2: sigma_2},
    )
    # input_node = 0 corresponds to X_tilde (the virtual root).
    return mutual_information_from_k(K, output_node=2, input_node=0)


def _waterfilling_mi(H: torch.Tensor, sigma_z2: float, p_total: float) -> float:
    """Closed-form water-filling MI for Y = H X + Z, ||Sigma_X||_tr = p_total."""
    # Eigendecomposition of H^H H: positive real eigenvalues.
    HhH = (H.conj().T @ H).cpu().numpy()
    eig = np.linalg.eigvalsh(HhH).clip(min=0)
    # Water-fill on inverse channel: power on mode k is max(0, mu - sigma^2/lambda_k).
    inv_chan = np.array([sigma_z2 / lam if lam > 0 else np.inf for lam in eig])
    # Find mu such that sum_k max(0, mu - inv_chan_k) = p_total.
    sorted_inv = np.sort(inv_chan)
    cum = 0.0
    mu = None
    for k in range(1, len(sorted_inv) + 1):
        # Hypothesise the k smallest modes are active.
        candidate = (p_total + sorted_inv[:k].sum()) / k
        if k == len(sorted_inv) or candidate <= sorted_inv[k]:
            mu = candidate
            break
    assert mu is not None
    powers = np.maximum(mu - inv_chan, 0.0)
    return float(np.sum(np.log(1.0 + (eig * powers) / sigma_z2)))


# ----------------------------- main -----------------------------


def main() -> None:
    print("=" * 72)
    print("Example: Input-covariance shaping via virtual edge")
    print("=" * 72)
    print(f"Setup: d={D}, sigma_noise={SIGMA_NOISE}, P_input={P_INPUT}, "
          f"iters={NUM_ITERS}, step={STEP_SIZE}, seed={SEED}")

    gen = torch.Generator().manual_seed(SEED)
    H = _randn_complex((D, D), gen)
    Q_init = _randn_complex((D, D), gen) * 0.1

    sigma_xt = torch.eye(D, dtype=DTYPE, device=DEVICE)  # X_tilde ~ CN(0, I)
    sigma_1 = EPS_V1 * torch.eye(D, dtype=DTYPE, device=DEVICE)
    sigma_2 = (SIGMA_NOISE ** 2) * torch.eye(D, dtype=DTYPE, device=DEVICE)

    Q = Q_init.clone().requires_grad_(True)

    def compute_mi() -> torch.Tensor:
        return _mi_for_Q(Q, H, sigma_xt, sigma_1, sigma_2)

    def projector(params: list[torch.Tensor]) -> None:
        for p in params:
            p.copy_(project_frobenius_ball(p, P_INPUT))

    # Baselines.
    with torch.no_grad():
        # Uniform input shaping: Q = sqrt(P/d) * I  =>  Sigma_X = (P/d) I.
        Q_uniform = ((P_INPUT / D) ** 0.5) * torch.eye(D, dtype=DTYPE, device=DEVICE)
        mi_uniform = _mi_for_Q(Q_uniform, H, sigma_xt, sigma_1, sigma_2).item()

        initial_mi = compute_mi().item()

    mi_wf = _waterfilling_mi(H, SIGMA_NOISE ** 2, P_INPUT)

    print(f"\nBaselines:")
    print(f"  Initial MI (small Q):                  {initial_mi:.4f} nats")
    print(f"  Uniform Q = sqrt(P/d)*I (white input): {mi_uniform:.4f} nats")
    print(f"  Water-filling optimum (theoretical):   {mi_wf:.4f} nats")

    # PGA.
    print(f"\nRunning PGA for {NUM_ITERS} iterations...")
    history = pga_ascent(
        compute_mi, [Q],
        step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
    )

    # Re-evaluate MI at the final Q (history[-1] is recorded before the last
    # update in pga_ascent; closure() returns the MI at the truly final state).
    with torch.no_grad():
        final_mi = float(compute_mi().item())
    final_norm_sq = (Q.detach().norm() ** 2).item()
    gap_to_wf = mi_wf - final_mi

    # Empirical Sigma_X = Q Q^H trace check (should be ~ P_INPUT at the optimum).
    with torch.no_grad():
        Q_final = Q.detach()
        sigma_x_emp = Q_final @ Q_final.conj().T
        tr_sigma_x = sigma_x_emp.diagonal().real.sum().item()

    print(f"\nResults:")
    print(f"  Final MI:                      {final_mi:.4f} nats")
    print(f"  Gap to water-filling optimum:  {gap_to_wf:+.4f} nats")
    print(f"  ||Q||_F^2 final:               {final_norm_sq:.4f} (budget {P_INPUT})")
    print(f"  tr(Sigma_X) = tr(Q Q^H):       {tr_sigma_x:.4f}")

    # Save results.
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    figures_dir = script_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / "input_covariance.npz",
        history=np.array(history, dtype=np.float64),
        initial_mi=initial_mi,
        final_mi=final_mi,
        mi_uniform=mi_uniform,
        mi_waterfilling=mi_wf,
        gap_to_wf=gap_to_wf,
        Q_final=Q.detach().cpu().numpy(),
        Sigma_X_final=sigma_x_emp.cpu().numpy(),
        H=H.cpu().numpy(),
        final_Q_norm_sq=final_norm_sq,
        tr_sigma_x=tr_sigma_x,
        config=dict(
            d=D, sigma_noise=SIGMA_NOISE, P_input=P_INPUT,
            eps_v1=EPS_V1, num_iters=NUM_ITERS, step_size=STEP_SIZE, seed=SEED,
        ),
    )
    print(f"\nSaved results: {results_dir / 'input_covariance.npz'}")

    # Plot.
    plt.figure(figsize=(7, 4.5))
    iters = np.arange(NUM_ITERS)
    plt.plot(iters, history, label="PGA on $Q$ (virtual edge)", linewidth=1.5)
    plt.axhline(mi_uniform, linestyle="--", color="gray",
                label=f"White input $Q=\\sqrt{{P/d}}\\,I$: {mi_uniform:.3f}")
    plt.axhline(mi_wf, linestyle="-.", color="green",
                label=f"Water-filling optimum: {mi_wf:.3f}")
    plt.xlabel("Iteration")
    plt.ylabel(r"Mutual information $I(\widetilde{X};Y)$ [nats]")
    plt.title(f"Input-covariance shaping via virtual edge ($d={D}$, $P={P_INPUT}$)")
    plt.legend(loc="lower right")
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    fig_path = figures_dir / "input_covariance.pdf"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved figure:  {fig_path}")


if __name__ == "__main__":
    main()
