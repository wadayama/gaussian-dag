"""Unit tests for gaussian_dag.krecursion (Milestone 2)."""

from __future__ import annotations

import torch

from gaussian_dag.krecursion import compute_k_blocks, get_K, hermitianize


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
# Test 1: Key coverage
# compute_k_blocks returns exactly all canonical keys (j, k) with j >= k.
# ============================================================


def test_key_coverage_chain():
    M = 4
    d = 2
    parents = {1: [0], 2: [1], 3: [2]}
    edge_mats = {(j, j - 1): _randn_complex(d, d, seed=j) for j in range(1, M)}
    input_cov = _hermitian_psd(d, seed=100)
    noise_covs = {j: _hermitian_psd(d, seed=200 + j) for j in range(1, M)}

    K = compute_k_blocks(M, parents, edge_mats, input_cov, noise_covs)

    expected_keys = {(j, k) for j in range(M) for k in range(j + 1)}
    assert set(K.keys()) == expected_keys


def test_key_coverage_diamond():
    M = 4
    d = 2
    parents = {1: [0], 2: [0], 3: [1, 2]}
    edge_mats = {
        (1, 0): _randn_complex(d, d, seed=1),
        (2, 0): _randn_complex(d, d, seed=2),
        (3, 1): _randn_complex(d, d, seed=3),
        (3, 2): _randn_complex(d, d, seed=4),
    }
    input_cov = _hermitian_psd(d, seed=100)
    noise_covs = {j: _hermitian_psd(d, seed=200 + j) for j in range(1, M)}

    K = compute_k_blocks(M, parents, edge_mats, input_cov, noise_covs)

    expected_keys = {(j, k) for j in range(M) for k in range(j + 1)}
    assert set(K.keys()) == expected_keys


# ============================================================
# Test 2: Hermitian flip and hermitianize utility
# ============================================================


def test_hermitianize_idempotent_and_hermitian():
    A = _randn_complex(3, 3, seed=42)
    H = hermitianize(A)
    H2 = hermitianize(H)
    # Hermitian: H == H^H.
    assert torch.allclose(H, H.mH, atol=1e-12)
    # Idempotent.
    assert torch.allclose(H, H2, atol=1e-12)


def test_get_K_hermitian_flip():
    """For a >= b returns canonical; for a < b returns K[(b, a)].mH."""
    M = 3
    d = 2
    parents = {1: [0], 2: [1]}
    edge_mats = {
        (1, 0): _randn_complex(d, d, seed=1),
        (2, 1): _randn_complex(d, d, seed=2),
    }
    input_cov = _hermitian_psd(d, seed=100)
    noise_covs = {1: _hermitian_psd(d, seed=201), 2: _hermitian_psd(d, seed=202)}
    K = compute_k_blocks(M, parents, edge_mats, input_cov, noise_covs)

    for a in range(M):
        for b in range(M):
            if a >= b:
                assert torch.equal(get_K(K, a, b), K[(a, b)])
            else:
                assert torch.equal(get_K(K, a, b), K[(b, a)].mH)


# ============================================================
# Test 3: Deterministic chain DAG
# Two-hop chain: V_0 -> V_1 -> V_2.
# V_1 = A_10 X + Z_1, V_2 = A_21 V_1 + Z_2.
# Expected: K_{22} = A_21 (A_10 Sigma_X A_10^H + Sigma_1) A_21^H + Sigma_2.
# ============================================================


def test_chain_deterministic():
    M = 3
    d = 2

    A10 = _randn_complex(d, d, seed=1)
    A21 = _randn_complex(d, d, seed=2)
    sigma_x = _hermitian_psd(d, seed=10)
    sigma_1 = _hermitian_psd(d, seed=11)
    sigma_2 = _hermitian_psd(d, seed=12)

    parents = {1: [0], 2: [1]}
    edge_mats = {(1, 0): A10, (2, 1): A21}
    noise_covs = {1: sigma_1, 2: sigma_2}

    K = compute_k_blocks(M, parents, edge_mats, sigma_x, noise_covs)

    # Manual closed-form for K_{11} and K_{22}.
    K11_expected = hermitianize(A10 @ sigma_x @ A10.mH + sigma_1)
    K22_expected = hermitianize(A21 @ K11_expected @ A21.mH + sigma_2)

    # Cross block K_{10} = A_10 Sigma_X.
    K10_expected = A10 @ sigma_x

    # Cross block K_{20} = A_21 K_{10} = A_21 A_10 Sigma_X.
    K20_expected = A21 @ K10_expected

    # Cross block K_{21} = A_21 K_{11}.
    K21_expected = A21 @ K11_expected

    assert torch.allclose(K[(0, 0)], hermitianize(sigma_x), atol=1e-10)
    assert torch.allclose(K[(1, 0)], K10_expected, atol=1e-10)
    assert torch.allclose(K[(1, 1)], K11_expected, atol=1e-10)
    assert torch.allclose(K[(2, 0)], K20_expected, atol=1e-10)
    assert torch.allclose(K[(2, 1)], K21_expected, atol=1e-10)
    assert torch.allclose(K[(2, 2)], K22_expected, atol=1e-10)


# ============================================================
# Test 4: Diamond DAG cross-covariance (the cross-cov necessity test).
# DAG: 0->1, 0->2, 1->3, 2->3.
# Verifies that parent-pair cross-covariance K_{21} is non-zero (shared ancestor V_0)
# and that K_{33} includes cross terms A_31 K_{12} A_32^H + A_32 K_{21} A_31^H.
# An auto-covariance-only baseline would miss these and produce a wrong K_{33}.
# ============================================================


def test_diamond_cross_covariance():
    M = 4
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

    K = compute_k_blocks(M, parents, edge_mats, sigma_x, noise_covs)

    # K_{11} = A_10 Sigma_X A_10^H + Sigma_1.
    K11_expected = hermitianize(A10 @ sigma_x @ A10.mH + sigma_1)
    assert torch.allclose(K[(1, 1)], K11_expected, atol=1e-10)

    # K_{22} = A_20 Sigma_X A_20^H + Sigma_2.
    K22_expected = hermitianize(A20 @ sigma_x @ A20.mH + sigma_2)
    assert torch.allclose(K[(2, 2)], K22_expected, atol=1e-10)

    # K_{21} = A_20 K_{01} = A_20 (A_10 Sigma_X)^H = A_20 Sigma_X A_10^H.
    K21_expected = A20 @ sigma_x @ A10.mH
    assert torch.allclose(K[(2, 1)], K21_expected, atol=1e-10)

    # Parent-pair cross-cov K_{21} should be non-zero (shared ancestor V_0).
    assert torch.linalg.norm(K[(2, 1)]).item() > 1e-6, (
        "K_{21} must be non-zero in diamond DAG due to shared ancestor V_0."
    )

    # K_{33} via the bilinear formula over parents {1, 2}.
    K12 = K[(2, 1)].mH  # Hermitian flip.
    K21 = K[(2, 1)]
    K33_expected = (
        A31 @ K11_expected @ A31.mH
        + A31 @ K12 @ A32.mH
        + A32 @ K21 @ A31.mH
        + A32 @ K22_expected @ A32.mH
        + sigma_3
    )
    K33_expected = hermitianize(K33_expected)
    assert torch.allclose(K[(3, 3)], K33_expected, atol=1e-10)

    # The cross terms (the key reason auto-cov alone is insufficient).
    cross_part = A31 @ K12 @ A32.mH + A32 @ K21 @ A31.mH
    assert torch.linalg.norm(cross_part).item() > 1e-6, (
        "Cross terms in K_{33} must be non-zero for diamond DAG."
    )

    # Compare against auto-cov-only baseline (skipping cross terms):
    # this is what a naive implementation would compute. It must differ.
    K33_naive = (
        A31 @ K11_expected @ A31.mH + A32 @ K22_expected @ A32.mH + sigma_3
    )
    assert not torch.allclose(K[(3, 3)], hermitianize(K33_naive), atol=1e-6), (
        "Correct K_{33} must differ from the auto-cov-only naive baseline."
    )


# ============================================================
# Brute-force closed-form K_{33} for the diamond DAG.
#
# This is a stronger check than test_diamond_cross_covariance: it expands
# V_3 explicitly in terms of the *independent* random sources (X, Z_1, Z_2,
# Z_3) and computes K_{33} = E[V_3 V_3^H] from first principles, never
# referring to intermediate K-recursion outputs. If compute_k_blocks
# implements the recursion correctly, K[(3, 3)] must match this closed form.
# ============================================================


def test_diamond_brute_force_K33():
    """K_{33} from compute_k_blocks matches a brute-force expansion of
    V_3 in terms of independent (X, Z_1, Z_2, Z_3)."""
    M = 4
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

    K = compute_k_blocks(M, parents, edge_mats, sigma_x, noise_covs)

    # Closed-form expansion of V_3 = A31(A10 X + Z_1) + A32(A20 X + Z_2) + Z_3
    #                              = (A31 A10 + A32 A20) X
    #                                + A31 Z_1 + A32 Z_2 + Z_3.
    # Since (X, Z_1, Z_2, Z_3) are mutually independent and zero-mean,
    # K_{33} = E[V_3 V_3^H] = signal_term + Z_1 term + Z_2 term + Z_3 term.
    G3 = A31 @ A10 + A32 @ A20
    K33_brute = (
        G3 @ sigma_x @ G3.mH
        + A31 @ sigma_1 @ A31.mH
        + A32 @ sigma_2 @ A32.mH
        + sigma_3
    )

    assert torch.allclose(K[(3, 3)], K33_brute, atol=1e-10), (
        "K_{3,3} from compute_k_blocks disagrees with the brute-force "
        "expansion from independent (X, Z_1, Z_2, Z_3)."
    )
