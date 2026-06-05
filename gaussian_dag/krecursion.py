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


def compute_effective_channel(
    num_nodes: int,
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], torch.Tensor],
    noise_covs: dict[int, torch.Tensor],
    *,
    source_dim: int | None = None,
    symmetrize_self_blocks: bool = True,
) -> tuple[dict[int, torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    """Effective-channel representation (G, C) of a linear Gaussian DAG.

    Collapses the DAG to an equivalent single linear Gaussian channel
        Y = G_M X + R_M,   R_M independent of X,
    exposing the effective channel matrices G_j (source-to-node gains) and
    the effective-noise covariance blocks C_{jk} = E[R_j R_k^H].

    The effective channel matrices follow the forward recursion
        G_0 = I_{d_X},
        G_j = sum_{i in Pa(j)} A_{ji} G_i        (j >= 1),
    with G_j of shape (d_j, d_X). The effective-noise blocks obey the *same*
    recursion as compute_k_blocks but with the input covariance set to zero
    (C_{00} = 0); they are obtained here by reusing compute_k_blocks with a
    zero input covariance, which is exact because the K-recursion is affine
    in its input-covariance seed. Together they satisfy the decomposition
        K_{jk} = G_j Sigma_X G_k^H + C_{jk},
    so the mutual information of the channel is
        I(X; Y) = log det(G_M Sigma_X G_M^H + C_{MM}) - log det C_{MM},
    with M = num_nodes - 1.

    Args:
        num_nodes: Total number of nodes M (indices 0..M-1).
        parents: parents[j] = list of parent indices for node j. Must satisfy
            i < j for every i in parents[j]. Node 0 is the unique root and
            need not appear as a key.
        edge_mats: edge_mats[(j, i)] = A_{ji}, the linear transformation on
            the edge i -> j (shape d_j x d_i).
        noise_covs: noise_covs[j] = Sigma_j (shape d_j x d_j) for j >= 1.
            Note: the input covariance Sigma_X is intentionally NOT an
            argument; (G, C) describe the channel and are independent of it.
        source_dim: The source dimension d_X (= dim of node 0). If None, it
            is inferred from any edge into node 0 (edge_mats[(j, 0)] has shape
            (d_j, d_X)). Provide it explicitly when node 0 has no outgoing
            edge, or to override inference.
        symmetrize_self_blocks: If True, apply (A + A^H)/2 to each C self-cov
            block C_{jj} to enforce Hermitian structure numerically (forwarded
            to compute_k_blocks).

    Returns:
        Tuple (G, C):
        - G: dict {j: G_j} for 0 <= j < num_nodes, each of shape (d_j, d_X),
          with G[0] = I_{d_X}.
        - C: dict of canonical blocks C[(j, k)] for 0 <= k <= j < num_nodes
          (same key convention as compute_k_blocks; use get_K for the
          Hermitian flip), with C[(0, 0)] = 0.
        Both G and C are differentiable in edge_mats (and C in noise_covs)
        via PyTorch autograd. Newly allocated tensors inherit dtype/device
        from the input tensors.

    Raises:
        ValueError: if d_X cannot be inferred (node 0 has no outgoing edge and
            source_dim is None), if dtype/device cannot be inferred (no edge
            or noise tensors available), if a provided source_dim disagrees
            with the dimension implied by an edge into node 0, or via the
            topological-order / missing-parent checks inherited from
            compute_k_blocks. C_{MM} must be strictly positive definite for
            the MI identity above to be well defined (the paper's regularity
            assumption); this is the caller's responsibility.
    """
    # Infer d_X and a reference tensor (for dtype/device) from edges into 0.
    edge_into_root = next(
        (edge_mats[(j, 0)] for j in range(1, num_nodes) if (j, 0) in edge_mats),
        None,
    )
    if edge_into_root is not None:
        inferred_dx = edge_into_root.shape[1]
        if source_dim is not None and source_dim != inferred_dx:
            raise ValueError(
                f"source_dim={source_dim} disagrees with the dimension "
                f"{inferred_dx} implied by an edge into node 0."
            )
        d_x = inferred_dx
        ref = edge_into_root
    else:
        if source_dim is None:
            raise ValueError(
                "Cannot infer the source dimension d_X: node 0 has no "
                "outgoing edge. Pass source_dim explicitly."
            )
        d_x = source_dim
        ref = next(iter(edge_mats.values()), None)
        if ref is None:
            ref = next(iter(noise_covs.values()), None)
        if ref is None:
            raise ValueError(
                "Cannot infer dtype/device: edge_mats and noise_covs are both "
                "empty. Provide at least one edge or noise matrix."
            )

    # Effective-noise blocks: K-recursion with a zero input covariance.
    zero_input = torch.zeros(d_x, d_x, dtype=ref.dtype, device=ref.device)
    C = compute_k_blocks(
        num_nodes,
        parents,
        edge_mats,
        zero_input,
        noise_covs,
        symmetrize_self_blocks=symmetrize_self_blocks,
    )

    # Effective channel matrices via the forward gain recursion.
    G: dict[int, torch.Tensor] = {
        0: torch.eye(d_x, dtype=ref.dtype, device=ref.device)
    }
    for j in range(1, num_nodes):
        acc: torch.Tensor | None = None
        for i in parents[j]:
            term = edge_mats[(j, i)] @ G[i]
            acc = term if acc is None else acc + term
        assert acc is not None  # parents[j] non-empty (validated by C above)
        G[j] = acc

    return G, C
