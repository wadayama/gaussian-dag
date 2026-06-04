"""Tests for projection operators and projected gradient ascent (Milestone 8)."""

from __future__ import annotations

import torch

from gaussian_dag.information import mutual_information_from_k
from gaussian_dag.krecursion import compute_k_blocks
from gaussian_dag.optimize import pga_ascent
from gaussian_dag.projections import project_frobenius_ball, project_total_power


DTYPE = torch.complex128


# ----------------------------- helpers -----------------------------


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _hermitian_psd(d: int, *, seed: int) -> torch.Tensor:
    A = _randn_complex(d, d, seed=seed)
    return A @ A.mH + torch.eye(d, dtype=DTYPE)


# ============================================================
# Test 14: project_frobenius_ball
# ============================================================


def test_project_frobenius_ball_inside_is_identity():
    """A matrix inside the ball is returned unchanged."""
    A = _randn_complex(3, 3, seed=1) * 0.1  # small matrix
    P = 100.0  # large budget
    assert (A.norm() ** 2).item() < P
    A_proj = project_frobenius_ball(A, P)
    assert torch.allclose(A_proj, A, atol=1e-14)


def test_project_frobenius_ball_outside_lies_on_boundary():
    """A matrix outside the ball is rescaled to lie exactly on the boundary."""
    A = _randn_complex(3, 3, seed=1) * 10.0  # large matrix
    P = 1.0  # small budget
    assert (A.norm() ** 2).item() > P
    A_proj = project_frobenius_ball(A, P)
    norm_sq = (A_proj.norm() ** 2).item()
    # ||A_proj||_F^2 == P (exact projection).
    assert abs(norm_sq - P) < 1e-10, f"||A_proj||_F^2 = {norm_sq}, P = {P}"
    # Direction preservation: A_proj = (sqrt(P) / ||A||_F) * A.
    expected_scale = (P ** 0.5) / A.norm().item()
    assert torch.allclose(A_proj, expected_scale * A, atol=1e-10)


def test_project_frobenius_ball_rejects_nonpositive_P():
    """Projection raises on P <= 0."""
    A = _randn_complex(2, 2, seed=1)
    import pytest
    with pytest.raises(ValueError):
        project_frobenius_ball(A, 0.0)
    with pytest.raises(ValueError):
        project_frobenius_ball(A, -1.0)


# ============================================================
# Test 15: project_total_power
# ============================================================


def test_project_total_power_outside():
    """Multiple matrices outside total-power ball: uniformly rescaled to lie on it."""
    A1 = _randn_complex(2, 2, seed=1)
    A2 = _randn_complex(3, 3, seed=2)
    A3 = _randn_complex(2, 3, seed=3)
    params = [A1, A2, A3]
    P = 1.0

    total_sq_before = sum((p.norm() ** 2).item() for p in params)
    assert total_sq_before > P, "Test precondition: total > P"

    proj_params = project_total_power(params, P)
    total_sq_after = sum((p.norm() ** 2).item() for p in proj_params)
    assert abs(total_sq_after - P) < 1e-10

    # All matrices scaled by the same factor (preserves relative magnitudes).
    expected_scale = (P / total_sq_before) ** 0.5
    for orig, proj in zip(params, proj_params):
        assert torch.allclose(proj, expected_scale * orig, atol=1e-10)


def test_project_total_power_inside_is_identity():
    """Multiple matrices already inside: unchanged."""
    A1 = _randn_complex(2, 2, seed=1) * 0.1
    A2 = _randn_complex(3, 3, seed=2) * 0.1
    params = [A1, A2]
    P = 100.0
    assert sum((p.norm() ** 2).item() for p in params) < P

    proj_params = project_total_power(params, P)
    for orig, proj in zip(params, proj_params):
        assert torch.allclose(proj, orig, atol=1e-14)


# ============================================================
# Test 16: PGA convergence on single-link MIMO under Frobenius ball
# ============================================================


def test_pga_increases_mi_mimo_with_frobenius_ball():
    """PGA monotonically (with possible small dips) improves MI on a constrained
    single-link MIMO Y = (H F) X + Z under ||F||_F^2 <= P."""
    d = 3
    H = _randn_complex(d, d, seed=1)
    sigma_x = _hermitian_psd(d, seed=10)
    sigma_z = _hermitian_psd(d, seed=11)
    P_budget = 5.0

    # Start at a small F (well inside the ball, modest initial MI).
    F_init = _randn_complex(d, d, seed=2) * 0.1
    F = F_init.clone().requires_grad_(True)

    def compute_mi() -> torch.Tensor:
        edge_mats = {(1, 0): H @ F}
        K = compute_k_blocks(2, {1: [0]}, edge_mats, sigma_x, {1: sigma_z})
        return mutual_information_from_k(K, output_node=1, input_node=0)

    def projector(params: list[torch.Tensor]) -> None:
        for p in params:
            new_p = project_frobenius_ball(p, P_budget)
            p.copy_(new_p)

    initial_mi = compute_mi().item()
    history = pga_ascent(
        compute_mi, [F], step_size=0.02, num_iters=80, projector=projector
    )
    final_mi = history[-1]

    # MI must strictly improve from the small-F initial value.
    assert final_mi > initial_mi + 0.5, (
        f"PGA did not noticeably improve MI: initial = {initial_mi:.4f}, "
        f"final = {final_mi:.4f}"
    )

    # Feasibility: ||F||_F^2 <= P_budget at the final point.
    final_norm_sq = (F.detach().norm() ** 2).item()
    assert final_norm_sq <= P_budget + 1e-8, (
        f"||F||_F^2 = {final_norm_sq:.4f} exceeds budget {P_budget}"
    )

    # The history must be non-empty and contain finite values throughout.
    assert len(history) == 80
    for v in history:
        assert torch.isfinite(torch.tensor(v))


def test_pga_rejects_invalid_args():
    """pga_ascent raises on invalid arguments."""
    import pytest

    F = _randn_complex(2, 2, seed=1).clone().requires_grad_(True)

    def trivial_mi() -> torch.Tensor:
        return (F.conj() * F).real.sum()

    with pytest.raises(ValueError):
        pga_ascent(trivial_mi, [F], step_size=0.0, num_iters=10)
    with pytest.raises(ValueError):
        pga_ascent(trivial_mi, [F], step_size=-0.1, num_iters=10)
    with pytest.raises(ValueError):
        pga_ascent(trivial_mi, [F], step_size=0.1, num_iters=0)
    # Param without requires_grad should be rejected.
    F_no_grad = _randn_complex(2, 2, seed=2)
    with pytest.raises(ValueError):
        pga_ascent(trivial_mi, [F_no_grad], step_size=0.1, num_iters=1)


def test_pga_accepts_functional_projector():
    """Functional projector convention: a projector that returns new tensors
    (without calling .copy_) must still apply the projection correctly. This
    documents the auto-copy contract of pga_ascent and prevents a silent
    footgun where the projection is accidentally discarded."""
    d = 3
    H = _randn_complex(d, d, seed=1)
    sigma_x = _hermitian_psd(d, seed=10)
    sigma_z = _hermitian_psd(d, seed=11)
    P_budget = 5.0

    # Initialise OUTSIDE the budget so projection has visible work to do.
    F_init = _randn_complex(d, d, seed=2) * 5.0
    F = F_init.clone().requires_grad_(True)

    def compute_mi() -> torch.Tensor:
        edge_mats = {(1, 0): H @ F}
        K = compute_k_blocks(2, {1: [0]}, edge_mats, sigma_x, {1: sigma_z})
        return mutual_information_from_k(K, output_node=1, input_node=0)

    # Functional projector: return new tensors, do NOT call .copy_.
    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, P_budget) for p in params]

    pga_ascent(compute_mi, [F], step_size=0.02, num_iters=20, projector=projector)

    # If the functional convention is honoured by pga_ascent, F must lie on
    # or inside the Frobenius ball at the final point.
    final_norm_sq = (F.detach().norm() ** 2).item()
    assert final_norm_sq <= P_budget + 1e-6, (
        f"Functional projector was silently discarded: ||F||_F^2 = "
        f"{final_norm_sq:.4f} exceeds budget {P_budget}."
    )


def test_pga_functional_projector_wrong_length_raises():
    """A functional projector that returns the wrong number of tensors must
    raise rather than silently mis-aligning."""
    import pytest

    F = (_randn_complex(2, 2, seed=1) * 0.1).clone().requires_grad_(True)

    def trivial_mi() -> torch.Tensor:
        return (F.conj() * F).real.sum()

    def bad_projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        # Returns an empty list instead of one tensor per param.
        return []

    with pytest.raises(ValueError):
        pga_ascent(
            trivial_mi, [F], step_size=0.01, num_iters=1, projector=bad_projector,
        )


def test_pga_unused_parameter_raises_clear_error():
    """A parameter that is declared with requires_grad=True but never used in
    compute_mi() must produce a clear RuntimeError, not an opaque TypeError
    from `add_(step_size * None)`. Helps users diagnose closure bugs quickly."""
    import pytest

    used = _randn_complex(2, 2, seed=1).clone().requires_grad_(True)
    unused = _randn_complex(2, 2, seed=2).clone().requires_grad_(True)

    def compute_mi_uses_only_first() -> torch.Tensor:
        # Note: `unused` is in params but is NOT referenced here.
        return (used.conj() * used).real.sum()

    with pytest.raises(RuntimeError) as excinfo:
        pga_ascent(
            compute_mi_uses_only_first,
            [used, unused],
            step_size=0.01,
            num_iters=1,
        )

    msg = str(excinfo.value)
    assert "received no gradient" in msg, (
        f"Error should mention that a param received no gradient; got: {msg!r}"
    )
    assert "params[1]" in msg, (
        f"Error should identify which param (index 1 here); got: {msg!r}"
    )


def test_input_covariance_water_filling_convergence():
    """The virtual-edge construction (X = Q X_tilde) optimised by PGA must
    approach the classical water-filling MI on the single-link channel.

    This is the formal pytest counterpart to examples/input_covariance.py.
    A loose tolerance is used (final MI within 0.05 nats of water-filling)
    so the test is robust against PyTorch-version round-off without being
    so loose as to admit a broken implementation.
    """
    import numpy as np

    d, sigma, P = 3, 0.5, 5.0
    H = _randn_complex(d, d, seed=1)
    sigma_z = (sigma ** 2) * torch.eye(d, dtype=DTYPE)
    eps_v1 = 1e-8

    # Virtual-edge DAG: V_0 = X_tilde (root), V_1 = X = Q V_0 (+ tiny noise),
    # V_2 = Y = H V_1 + Z.
    Q = (0.1 * _randn_complex(d, d, seed=2)).clone().requires_grad_(True)

    def compute_mi() -> torch.Tensor:
        edge_mats = {(1, 0): Q, (2, 1): H}
        K = compute_k_blocks(
            num_nodes=3,
            parents={1: [0], 2: [1]},
            edge_mats=edge_mats,
            input_cov=torch.eye(d, dtype=DTYPE),
            noise_covs={
                1: eps_v1 * torch.eye(d, dtype=DTYPE),
                2: sigma_z,
            },
        )
        return mutual_information_from_k(K, output_node=2, input_node=0)

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, P) for p in params]

    pga_ascent(
        compute_mi, [Q],
        step_size=0.05, num_iters=200, projector=projector,
    )

    with torch.no_grad():
        final_mi = float(compute_mi().item())

    # Reference: classical water-filling on H, with white input and budget P.
    H_np = H.numpy()
    eigvals = np.linalg.eigvalsh(H_np.conj().T @ H_np) / (sigma ** 2)
    eigvals = np.sort(eigvals)[::-1]
    n = len(eigvals)
    for k in range(n, 0, -1):
        mu = (P + (1.0 / eigvals[:k]).sum()) / k
        p_opt = np.maximum(mu - 1.0 / eigvals[:k], 0.0)
        if (p_opt > 0).all():
            break
    mi_wf = float(np.log(1 + eigvals[:k] * p_opt).sum())

    gap = mi_wf - final_mi
    # Loose tolerance: at T = 200, PGA should sit well within 0.05 nats of
    # the water-filling reference for the chosen problem regime.
    assert gap >= -1e-6, f"PGA exceeded water-filling MI; gap = {gap:+.6f}"
    assert gap < 0.05, (
        f"PGA failed to approach water-filling within 0.05 nats: "
        f"final_mi = {final_mi:.4f}, water-filling = {mi_wf:.4f}, "
        f"gap = {gap:+.6f}"
    )
