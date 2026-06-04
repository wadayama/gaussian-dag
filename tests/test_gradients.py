"""Autograd verification tests for the MI computation (Milestone 6 + 7).

Milestone 6: I.backward() correctly populates .grad on a controllable factor.
Milestone 7: PyTorch autograd matches central finite-difference gradient.

PyTorch complex-gradient convention (important):
    For a real-valued scalar loss L and complex parameter F = a + i b,
    PyTorch stores in F.grad the quantity
        F.grad = dL/da + i * dL/db,
    which equals 2 * (dL/dF^*) in the standard mathematical Wirtinger
    convention (the conjugate Wirtinger derivative dL/dF^*
    has a factor 1/2 from the chain rule for z^*).
    See: pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc.

The finite-difference reference here therefore uses
    grad_fd[i, j] = dL/dRe(F_ij) + i * dL/dIm(F_ij)
(without the 1/2 factor), to be directly comparable to F.grad.
"""

from __future__ import annotations

import torch

from gaussian_dag.information import mutual_information_from_k
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


def _mi_single_link(F: torch.Tensor, H: torch.Tensor,
                    sigma_x: torch.Tensor, sigma_z: torch.Tensor) -> torch.Tensor:
    """Compute I(X; Y) for the single-link MIMO Y = (H @ F) X + Z via the library.

    Used both for autograd and finite-difference paths.
    """
    parents = {1: [0]}
    edge_mats = {(1, 0): H @ F}
    noise_covs = {1: sigma_z}
    K = compute_k_blocks(2, parents, edge_mats, sigma_x, noise_covs)
    return mutual_information_from_k(K, output_node=1, input_node=0)


# ============================================================
# Test 12: autograd populates .grad on a controllable factor (Milestone 6).
# ============================================================


def test_autograd_basic_precoder():
    """I.backward() populates F.grad non-trivially when F is controllable."""
    d = 3
    H = _randn_complex(d, d, seed=1)
    F_init = _randn_complex(d, d, seed=2)
    sigma_x = _hermitian_psd(d, seed=10)
    sigma_z = _hermitian_psd(d, seed=11)

    F = F_init.clone().requires_grad_(True)
    I = _mi_single_link(F, H, sigma_x, sigma_z)

    # I should be a real scalar with requires_grad.
    assert I.dtype == torch.float64, f"Expected real I, got {I.dtype}"
    assert I.requires_grad, "I should have requires_grad=True (autograd graph attached)."

    I.backward()

    # F.grad should now be populated.
    assert F.grad is not None, "F.grad should be populated after I.backward()."
    assert F.grad.dtype == torch.complex128, f"Got dtype {F.grad.dtype}"
    assert F.grad.shape == F.shape, f"Got shape {F.grad.shape}, expected {F.shape}"
    # Non-trivially populated: should not be zero for a generic F.
    grad_norm = F.grad.abs().sum().item()
    assert grad_norm > 1e-6, (
        f"Expected non-trivial gradient, got |F.grad|_1 = {grad_norm:.4e}"
    )


# ============================================================
# Test 13: autograd matches central finite differences (Milestone 7).
# ============================================================


def test_finite_difference_precoder():
    """Autograd Wirtinger gradient matches central FD gradient on H @ F MIMO."""
    d = 2  # small dimension keeps FD inner loop fast
    H = _randn_complex(d, d, seed=1)
    F_init = _randn_complex(d, d, seed=2)
    sigma_x = _hermitian_psd(d, seed=10)
    sigma_z = _hermitian_psd(d, seed=11)

    # Autograd path.
    F = F_init.clone().requires_grad_(True)
    I = _mi_single_link(F, H, sigma_x, sigma_z)
    I.backward()
    grad_autograd = F.grad.detach().clone()

    # Central finite difference (PyTorch convention: no 1/2 factor).
    # For complex parameter F_{ij} = a + i b and real-valued I:
    #   F.grad[i, j] = dI/da + i * dI/db   (= 2 * dI/dF^*_{ij} in standard Wirtinger)
    eps = 1.0e-5
    grad_fd = torch.zeros_like(F_init)
    for i in range(d):
        for j in range(d):
            # Perturb real part.
            F_plus_re = F_init.clone()
            F_plus_re[i, j] = F_plus_re[i, j] + eps
            F_minus_re = F_init.clone()
            F_minus_re[i, j] = F_minus_re[i, j] - eps
            dI_da = (
                _mi_single_link(F_plus_re, H, sigma_x, sigma_z).item()
                - _mi_single_link(F_minus_re, H, sigma_x, sigma_z).item()
            ) / (2.0 * eps)

            # Perturb imaginary part.
            F_plus_im = F_init.clone()
            F_plus_im[i, j] = F_plus_im[i, j] + 1j * eps
            F_minus_im = F_init.clone()
            F_minus_im[i, j] = F_minus_im[i, j] - 1j * eps
            dI_db = (
                _mi_single_link(F_plus_im, H, sigma_x, sigma_z).item()
                - _mi_single_link(F_minus_im, H, sigma_x, sigma_z).item()
            ) / (2.0 * eps)

            # PyTorch's stored gradient convention: dI/da + i * dI/db (no 1/2).
            grad_fd[i, j] = dI_da + 1j * dI_db

    # Compare via Frobenius relative error.
    # Expected error: O(eps^2) ~ 1e-10 truncation + roundoff ~ 1e-11 → ~1e-9.
    diff_norm = (grad_autograd - grad_fd).norm().item()
    ref_norm = grad_autograd.norm().item()
    rel_err = diff_norm / max(ref_norm, 1e-12)
    assert rel_err < 1e-6, (
        f"Autograd vs FD relative error: {rel_err:.4e} "
        f"(||diff||_F = {diff_norm:.4e}, ||grad_autograd||_F = {ref_norm:.4e})"
    )
