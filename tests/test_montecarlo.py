"""Monte Carlo validation of K-recursion against empirical covariances (Milestone 3).

Generates samples from a linear Gaussian DAG via forward equations,
then compares empirical covariance blocks
    K_emp[j, k] = (1/N) sum_n V_j^(n) V_k^(n) ^H
to the K-recursion output K[(j, k)] from compute_k_blocks.

This sanity-tests indexing, Hermitian flip, parent ordering, and
noise propagation simultaneously.
"""

from __future__ import annotations

import torch

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


def _sample_complex_gaussian(
    cov: torch.Tensor,
    n: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Sample n vectors from CN(0, cov) via Cholesky.

    Construction: z = (a + i b) / sqrt(2) with a, b ~ N(0, I_d) independent
    gives z ~ CN(0, I_d). Then y = L z with cov = L L^H gives y ~ CN(0, cov).

    Returns a tensor of shape (d, n); each column is one independent sample.
    """
    d = cov.shape[0]
    L = torch.linalg.cholesky(cov)
    a = torch.randn(d, n, dtype=torch.float64, generator=gen)
    b = torch.randn(d, n, dtype=torch.float64, generator=gen)
    z = torch.complex(a, b) / (2.0 ** 0.5)
    return L @ z


# ============================================================
# Test 7: Monte Carlo covariance validation on diamond DAG.
# DAG: V_0 -> V_1, V_0 -> V_2, V_1 -> V_3, V_2 -> V_3.
# ============================================================


def test_diamond_monte_carlo():
    M = 4
    d = 2
    N = 50000  # MC std ~ 1/sqrt(N) ~ 0.45% relative.

    # DAG and parameters (same as test_diamond_cross_covariance).
    parents = {1: [0], 2: [0], 3: [1, 2]}
    A10 = _randn_complex(d, d, seed=1)
    A20 = _randn_complex(d, d, seed=2)
    A31 = _randn_complex(d, d, seed=3)
    A32 = _randn_complex(d, d, seed=4)

    sigma_x = _hermitian_psd(d, seed=10)
    sigma_1 = _hermitian_psd(d, seed=11)
    sigma_2 = _hermitian_psd(d, seed=12)
    sigma_3 = _hermitian_psd(d, seed=13)

    edge_mats = {(1, 0): A10, (2, 0): A20, (3, 1): A31, (3, 2): A32}
    noise_covs = {1: sigma_1, 2: sigma_2, 3: sigma_3}

    # Reference: K-recursion output.
    K = compute_k_blocks(M, parents, edge_mats, sigma_x, noise_covs)

    # Generate independent samples for X, Z_1, Z_2, Z_3.
    gen = torch.Generator().manual_seed(42)
    X = _sample_complex_gaussian(sigma_x, N, gen)
    Z1 = _sample_complex_gaussian(sigma_1, N, gen)
    Z2 = _sample_complex_gaussian(sigma_2, N, gen)
    Z3 = _sample_complex_gaussian(sigma_3, N, gen)

    # Forward generative equations.
    V0 = X
    V1 = A10 @ X + Z1
    V2 = A20 @ X + Z2
    V3 = A31 @ V1 + A32 @ V2 + Z3
    V = {0: V0, 1: V1, 2: V2, 3: V3}

    # Per-block empirical vs reference comparison via Frobenius relative error.
    # Rationale: covariance matrices may have entries of very different magnitudes,
    # so element-wise rtol penalises small entries unfairly. The Frobenius
    # relative norm is the natural matching metric for covariance.
    # Expected MC error: O(1/sqrt(N)) ~ 0.5% at N=50000; threshold 2% gives ~4x margin.
    threshold_rel = 0.02

    for j in range(M):
        for k in range(j + 1):
            K_emp = (V[j] @ V[k].mH) / N
            K_ref = K[(j, k)]
            diff_norm = (K_emp - K_ref).norm().item()
            ref_norm = K_ref.norm().item()
            rel_err = diff_norm / max(ref_norm, 1e-6)
            assert rel_err < threshold_rel, (
                f"Frobenius relative error too large at (j={j}, k={k}): "
                f"rel_err = {rel_err:.4e} "
                f"(||K_emp - K_ref||_F = {diff_norm:.4e}, "
                f"||K_ref||_F = {ref_norm:.4e})"
            )
