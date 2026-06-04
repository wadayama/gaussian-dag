"""Mutual information from K-blocks for linear Gaussian DAGs.

This module provides the log-det based mutual information layer:
    I(X; Y) = log det Sigma_Y  -  log det Sigma_{Y|X}
with the conditional covariance obtained via the Schur complement
    Sigma_{Y|X} = Sigma_Y - Sigma_{YX} Sigma_X^{-1} Sigma_{XY}.

All quantities are computed without forming an explicit matrix inverse:
- log det of Hermitian PSD matrices uses Cholesky factorization,
- the Schur complement uses torch.linalg.solve.
"""

from __future__ import annotations

import torch

from gaussian_dag.krecursion import get_K, hermitianize


def logdet_hpd(A: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    """Cholesky-based log-determinant for Hermitian positive-definite A.

    For Hermitian positive-definite A = L L^H with lower-triangular L and
    real positive diag(L),
        log det A = 2 * sum_i log L_ii.

    The input is symmetrized by (A + A^H)/2 to enforce Hermitian structure
    against floating-point drift before the Cholesky step. The Cholesky
    factorisation itself is performed via ``torch.linalg.cholesky_ex``: on
    failure (matrix not strictly positive-definite), this function raises
    a ``ValueError`` with diagnostic information rather than letting an
    opaque PyTorch error propagate out of the autograd graph.

    Args:
        A: Hermitian positive-definite matrix (shape d x d, complex or
            real). With the default ``jitter=0`` the input must be
            strictly positive-definite; otherwise the Cholesky factorisation
            fails and a ``ValueError`` is raised (see Raises). A
            rank-deficient or merely PSD input can be admitted by passing
            a small ``jitter > 0`` to regularise it back into the PD cone.
        jitter: If > 0, replace A by A + jitter * I before factorization.
            Useful when the underlying matrix is near-singular or rank
            deficient (e.g. because the noise covariance is small or
            structurally degenerate). Use sparingly: a non-zero jitter
            perturbs the log-determinant by order
            ``d * jitter / lambda_min(A)`` and should be reported in any
            experiment that uses it.

    Returns:
        Real scalar tensor: log det A (natural log; nats convention).

    Raises:
        ValueError: if A (after optional jitter) is not strictly positive
            definite. The error message indicates the leading minor where
            positive definiteness failed and suggests three remediations:
            (1) ensure that the terminal noise covariance is strictly
            positive definite so that the model-level regularity assumption
            of the paper holds; (2) pass a small ``jitter > 0`` to absorb
            round-off near-singularity; or (3) when called from inside a
            PGA loop, reduce the step size so that iterates stay inside
            the open positive-definite cone.
    """
    A = hermitianize(A)
    if jitter > 0.0:
        d = A.shape[-1]
        A = A + jitter * torch.eye(d, dtype=A.dtype, device=A.device)
    L, info = torch.linalg.cholesky_ex(A, check_errors=False)
    info_value = int(info.item())
    if info_value != 0:
        raise ValueError(
            "logdet_hpd: input matrix is not Hermitian positive definite "
            f"(Cholesky failed at leading minor of order {info_value}). "
            "Common remedies: (1) ensure the terminal noise covariance is "
            "strictly positive definite (the paper's regularity assumption); "
            "(2) pass jitter>0 to logdet_hpd / mutual_information_from_k "
            "to absorb near-singularity; (3) inside pga_ascent, reduce "
            "step_size so that iterates remain in the positive-definite cone."
        )
    return 2.0 * torch.log(torch.diagonal(L).real).sum()


def mutual_information_from_k(
    K: dict[tuple[int, int], torch.Tensor],
    output_node: int,
    input_node: int = 0,
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Compute I(X; Y) = log det Sigma_Y - log det Sigma_{Y|X} from K-blocks.

    Reads Sigma_Y = K_{yy}, Sigma_{YX} = K_{yx}, Sigma_X = K_{xx}
    from the canonical K dictionary (using Hermitian flip when needed)
    and forms the conditional covariance via the Schur complement
        Sigma_{Y|X} = Sigma_Y - Sigma_{YX} Sigma_X^{-1} Sigma_{XY}
    using a linear solve (no explicit matrix inverse).

    Args:
        K: Dictionary of canonical K-blocks returned by compute_k_blocks.
        output_node: Index of the output node (e.g., M-1).
        input_node: Index of the input node (default 0).
        jitter: Optional jitter passed to logdet_hpd for both
            Sigma_Y and Sigma_{Y|X}.

    Returns:
        Real scalar tensor: I(X; Y) in nats. Differentiable through
        the K-blocks via PyTorch autograd.
    """
    K_yy = get_K(K, output_node, output_node)
    K_yx = get_K(K, output_node, input_node)
    K_xx = get_K(K, input_node, input_node)

    # Schur complement: Sigma_{Y|X} = Kyy - Kyx * Kxx^{-1} * Kxy.
    # Compute (Kxx^{-1} Kxy) via linear solve, where Kxy = Kyx^H.
    Kxx_inv_Kxy = torch.linalg.solve(K_xx, K_yx.mH)
    K_y_given_x = K_yy - K_yx @ Kxx_inv_Kxy

    return logdet_hpd(K_yy, jitter=jitter) - logdet_hpd(K_y_given_x, jitter=jitter)
