"""K-recursion: forward covariance computation for linear Gaussian DAGs.

Model (0-based indexing):
    V_0 = X ~ CN(0, Sigma_X) (unique root)
    V_j = sum_{i in Pa(j)} A_{ji} V_i + Z_j  for j >= 1
    Z_j ~ CN(0, Sigma_j), mutually independent and independent of X.

Canonical storage: K stores only K_{jk} for j >= k (lower-triangular block matrix).
Access to K_{ab} with a < b is performed via the Hermitian flip rule
    K_{ab} = K_{ba}^H.
"""

from __future__ import annotations

import torch


def hermitianize(A: torch.Tensor) -> torch.Tensor:
    """Symmetrize a square matrix by (A + A^H) / 2.

    This enforces exact Hermitian structure on tensors that should be Hermitian
    in theory but may drift due to floating-point round-off.
    """
    return 0.5 * (A + A.mH)


def get_K(
    K: dict[tuple[int, int], torch.Tensor],
    a: int,
    b: int,
) -> torch.Tensor:
    """Return K_{ab}, applying Hermitian flip when a < b.

    K is assumed to store only canonical keys (j, k) with j >= k.
    """
    if a >= b:
        return K[(a, b)]
    return K[(b, a)].mH


def compute_k_blocks(
    num_nodes: int,
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], torch.Tensor],
    input_cov: torch.Tensor,
    noise_covs: dict[int, torch.Tensor],
    *,
    symmetrize_self_blocks: bool = True,
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute all canonical K-blocks K_{jk} for 0 <= k <= j < num_nodes.

    Args:
        num_nodes: Total number of nodes M (indices 0..M-1).
        parents: parents[j] = list of parent indices for node j.
            Must satisfy i < j for every i in parents[j]. Node 0 is the unique
            root and need not appear as a key.
        edge_mats: edge_mats[(j, i)] = A_{ji}, the linear transformation on
            the edge i -> j (shape d_j x d_i).
        input_cov: Sigma_X = Cov(V_0) (shape d_0 x d_0).
        noise_covs: noise_covs[j] = Sigma_j (shape d_j x d_j) for j >= 1.
        symmetrize_self_blocks: If True, apply (A + A^H)/2 to each self-cov
            block K_{jj} to enforce Hermitian structure numerically.

    Returns:
        Dictionary K with keys (j, k) for 0 <= k <= j < num_nodes.
    """
    K: dict[tuple[int, int], torch.Tensor] = {}
    K[(0, 0)] = hermitianize(input_cov) if symmetrize_self_blocks else input_cov

    for j in range(1, num_nodes):
        if j not in parents or len(parents[j]) == 0:
            raise ValueError(f"Non-root node {j} has no parents.")
        for i in parents[j]:
            if i >= j:
                raise ValueError(
                    f"Parent {i} of node {j} violates topological order (i < j)."
                )

        # (1) Cross blocks K_{jk} for k = 0, ..., j-1.
        for k in range(j):
            acc = None
            for i in parents[j]:
                term = edge_mats[(j, i)] @ get_K(K, i, k)
                acc = term if acc is None else acc + term
            K[(j, k)] = acc

        # (2) Self block K_{jj} = sum_{i,i'} A_{ji} K_{ii'} A_{ji'}^H + Sigma_j.
        acc = noise_covs[j]
        for i in parents[j]:
            Aji = edge_mats[(j, i)]
            for ip in parents[j]:
                Ajip = edge_mats[(j, ip)]
                acc = acc + Aji @ get_K(K, i, ip) @ Ajip.mH
        K[(j, j)] = hermitianize(acc) if symmetrize_self_blocks else acc

    return K
