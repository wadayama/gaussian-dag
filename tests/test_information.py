"""Unit tests for gaussian_dag.information (Milestone 4 + 5).

Milestone 4: logdet_hpd, mutual_information_from_k implementation.
Milestone 5: Single-link MIMO mutual information matches the classical
log-det expression.
"""

from __future__ import annotations

import torch

from gaussian_dag.information import logdet_hpd, mutual_information_from_k
from gaussian_dag.krecursion import compute_k_blocks


DTYPE = torch.complex128


# ----------------------------- helpers -----------------------------


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    """Random complex tensor with i.i.d. standard normal real/imag parts."""
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _hermitian_psd(d: int, *, seed: int) -> torch.Tensor:
    """Random Hermitian positive-definite matrix of size d x d: A A^H + I."""
    A = _randn_complex(d, d, seed=seed)
    return A @ A.mH + torch.eye(d, dtype=DTYPE)


# ============================================================
# Test 8: logdet_hpd(I) = 0 (sanity)
# ============================================================


def test_logdet_hpd_identity():
    """log det of the identity is 0 in any size."""
    for d in (1, 2, 3, 5):
        I_d = torch.eye(d, dtype=DTYPE)
        result = logdet_hpd(I_d)
        assert torch.allclose(
            result, torch.tensor(0.0, dtype=torch.float64), atol=1e-12
        ), f"logdet_hpd(I_{d}) should be 0, got {result.item()}"


# ============================================================
# Test 9: logdet_hpd matches torch.linalg.slogdet for Hermitian PSD
# ============================================================


def test_logdet_hpd_vs_slogdet():
    """logdet_hpd matches torch.linalg.slogdet(...).logabsdet (independent route)."""
    for d, seed in [(2, 41), (3, 42), (5, 43)]:
        sigma = _hermitian_psd(d, seed=seed)
        expected = torch.linalg.slogdet(sigma).logabsdet
        actual = logdet_hpd(sigma)
        assert torch.allclose(actual, expected, atol=1e-10), (
            f"d={d}: logdet_hpd={actual.item():.6e}, slogdet={expected.item():.6e}"
        )


# ============================================================
# Test 9b: logdet_hpd is batch-safe over a leading batch dimension.
# ============================================================


def test_logdet_hpd_batched():
    """A leading batch dim yields one log-det per element, matching the scalar path."""
    d, B = 3, 5
    mats = torch.stack([_hermitian_psd(d, seed=100 + b) for b in range(B)])  # (B, d, d)
    out = logdet_hpd(mats)
    assert out.shape == (B,)
    expected = torch.linalg.slogdet(mats).logabsdet
    assert torch.allclose(out, expected, atol=1e-10)
    for b in range(B):
        assert torch.allclose(out[b], logdet_hpd(mats[b]), atol=1e-12)


def test_mutual_information_from_k_batched():
    """Batched K-blocks give one MI per element, matching per-element evaluation."""
    d, B = 2, 4
    H = torch.stack([_randn_complex(d, d, seed=200 + b) for b in range(B)])  # (B, d, d)
    sigma_x = _hermitian_psd(d, seed=11)
    sigma_1 = _hermitian_psd(d, seed=12)
    K = compute_k_blocks(2, {1: [0]}, {(1, 0): H}, sigma_x, {1: sigma_1})
    mi_b = mutual_information_from_k(K, 1, 0)
    assert mi_b.shape == (B,)
    for b in range(B):
        Kb = compute_k_blocks(2, {1: [0]}, {(1, 0): H[b]}, sigma_x, {1: sigma_1})
        assert torch.allclose(mi_b[b], mutual_information_from_k(Kb, 1, 0), atol=1e-10)


# ============================================================
# Test 10: Single-link MIMO MI matches classical log-det formula (Milestone 5 core).
# Model: V_0 = X ~ CN(0, Sigma_X), V_1 = Y = H X + Z_1, Z_1 ~ CN(0, Sigma_1).
# Classical:
#   I(X; Y) = log det(H Sigma_X H^H + Sigma_1) - log det Sigma_1.
# ============================================================


def test_single_link_mimo_classical_mi():
    """Single-link MIMO MI from K-recursion matches the classical formula."""
    d = 3
    H = _randn_complex(d, d, seed=1)
    sigma_x = _hermitian_psd(d, seed=10)
    sigma_z = _hermitian_psd(d, seed=11)

    # K-recursion route: build the DAG V_0 -> V_1 with edge H and noise Sigma_Z at V_1.
    parents = {1: [0]}
    edge_mats = {(1, 0): H}
    noise_covs = {1: sigma_z}
    K = compute_k_blocks(2, parents, edge_mats, sigma_x, noise_covs)
    I_lib = mutual_information_from_k(K, output_node=1, input_node=0)

    # Classical route: I = log det(Sigma_Z + H Sigma_X H^H) - log det Sigma_Z.
    sigma_y = H @ sigma_x @ H.mH + sigma_z
    I_classical = logdet_hpd(sigma_y) - logdet_hpd(sigma_z)

    assert torch.allclose(I_lib, I_classical, atol=1e-10), (
        f"I_lib = {I_lib.item():.6e}, I_classical = {I_classical.item():.6e}"
    )
    # Sanity: MI is strictly positive for non-degenerate channel + non-zero input cov.
    assert I_lib.item() > 1e-6, (
        f"Expected positive MI for non-degenerate MIMO, got {I_lib.item()}"
    )


# ============================================================
# Test 11: MI non-negativity on diamond DAG.
# A general DAG must yield I(X; Y) >= 0 (information-theoretic property).
# ============================================================


def test_mi_nonneg_diamond():
    """I(X; Y) >= 0 for the diamond DAG (general non-trivial topology)."""
    d = 2
    A10 = _randn_complex(d, d, seed=1)
    A20 = _randn_complex(d, d, seed=2)
    A31 = _randn_complex(d, d, seed=3)
    A32 = _randn_complex(d, d, seed=4)

    sigma_x = _hermitian_psd(d, seed=10)
    sigma_1 = _hermitian_psd(d, seed=11)
    sigma_2 = _hermitian_psd(d, seed=12)
    sigma_3 = _hermitian_psd(d, seed=13)

    parents = {1: [0], 2: [0], 3: [1, 2]}
    edge_mats = {(1, 0): A10, (2, 0): A20, (3, 1): A31, (3, 2): A32}
    noise_covs = {1: sigma_1, 2: sigma_2, 3: sigma_3}
    K = compute_k_blocks(4, parents, edge_mats, sigma_x, noise_covs)

    I = mutual_information_from_k(K, output_node=3, input_node=0)

    # MI must be non-negative (allow tiny float tolerance).
    assert I.item() > -1e-10, f"MI must be non-negative, got {I.item()}"
    # Sanity: should be strictly positive for non-degenerate diamond.
    assert I.item() > 1e-6, f"Expected positive MI for non-degenerate diamond, got {I.item()}"


def test_logdet_hpd_raises_on_singular_matrix():
    """logdet_hpd must raise a clear ValueError on non-HPD input
    (Cholesky failure detection via cholesky_ex)."""
    import pytest

    d = 3
    # Rank-deficient PSD matrix: outer product of a single random vector.
    v = _randn_complex(d, 1, seed=0)
    A = v @ v.mH  # rank 1, so leading minor of order >= 2 fails.

    with pytest.raises(ValueError) as excinfo:
        logdet_hpd(A)

    msg = str(excinfo.value)
    assert "not Hermitian positive definite" in msg
    assert "remedies" in msg.lower(), (
        "Error message should suggest the three remedies (terminal noise PD, "
        f"jitter, smaller step size). Got: {msg!r}"
    )


def test_logdet_hpd_jitter_recovers_singular_case():
    """A small jitter > 0 should restore strict positive-definiteness
    and let logdet_hpd succeed on an otherwise singular input."""
    d = 3
    v = _randn_complex(d, 1, seed=0)
    A = v @ v.mH  # rank 1.

    # With jitter, the matrix becomes PD and logdet is finite.
    val = logdet_hpd(A, jitter=1e-6)
    assert torch.isfinite(val), f"Expected finite logdet with jitter, got {val}"
