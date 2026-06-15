"""Named-node DAG builder: a declarative front-end over the K-recursion core.

This module is a *pure, backward-compatible addition* to the library. It adds
no new behavior to the numerical core; it only provides a convenience surface
that *lowers* a named-node DAG declaration to the existing functional API
(``compute_k_blocks`` -> ``mutual_information_from_k`` / ``get_K``).

Worked example (a chain ``X -> Y -> Z``)::

    dag = GaussianDAG()
    dag.add_source("X", cov=Sigma_X)
    dag.add_node("Y", parents={"X": H_XY}, noise=N_Y)
    dag.add_node("Z", parents={"Y": H_YZ}, noise=N_Z)

    mi = dag.mi("X", "Z")        # I(X; Z) in nats (differentiable tensor)
    Sigma_Z = dag.cov("Z")       # self-covariance block of node Z

Profile (see builder_implementation.md, spec v0.2): this library implements the
*single-pair* (``mi(source, target)``, no conditioning set) and *single-root*
profiles. Constructs outside those profiles fail loudly:

* a second ``add_source`` (multi-root) raises ``ValueError``;
* there is intentionally no ``cmi`` method (no conditioning), so requesting one
  raises the natural ``AttributeError`` rather than returning a wrong answer.

Matrices may be given either as concrete tensors (used as-is) or by name
(strings resolved at query time via a ``bind={name: tensor}`` mapping). An
unbound name raises ``ValueError`` -- data is never fabricated.

Conventions inherited from the core: node 0 is the unique root, edge keys are
``(j, i)`` for the edge ``i -> j`` with ``i < j`` (topological order); see
``gaussian_dag.krecursion``. Canonical node indices are assigned by a stable
topological sort (Kahn's algorithm, FIFO queue, ties broken by build/call
order), so a single-source DAG always maps its source to index 0.
"""

from __future__ import annotations

from collections import deque
from typing import Union

import torch

from gaussian_dag.information import mutual_information_from_k
from gaussian_dag.krecursion import compute_k_blocks, get_K

# A matrix reference recorded at build time: either a concrete tensor (used
# as-is) or a name (string) resolved at query time via the ``bind`` mapping.
MatrixRef = Union[str, torch.Tensor]


class GaussianDAG:
    """Declarative named-node builder for a single-root linear Gaussian DAG.

    See the module docstring for the worked example, the supported profiles,
    and the matrix-binding rules. The builder is a thin layer: it records
    structure and matrix references at build time, then lowers to the
    library's functional core when a query (``mi`` / ``cov``) runs.
    """

    def __init__(self) -> None:
        # name -> covariance reference (tensor or name string).
        self._sources: dict[str, MatrixRef] = {}
        # name -> (parents: {parent_name: gain_ref}, noise_ref).
        self._nodes: dict[str, tuple[dict[str, MatrixRef], MatrixRef]] = {}
        # Build/call order; used to derive canonical indices (spec section 12).
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_source(self, name: str, *, cov: MatrixRef) -> "GaussianDAG":
        """Declare a source (root) node with covariance ``cov``.

        This library is single-root: declaring a second source raises
        ``ValueError`` (the multi-root profile is not supported).

        Returns ``self`` to allow chaining.
        """
        self._check_new(name)
        if self._sources:
            existing = next(iter(self._sources))
            raise ValueError(
                f"This library is single-root: source {existing!r} is already "
                f"declared, cannot add a second source {name!r}. "
                "Stack roots at the user level if you need multiple sources."
            )
        self._sources[name] = cov
        self._order.append(name)
        return self

    def add_node(
        self,
        name: str,
        *,
        parents: dict[str, MatrixRef],
        noise: MatrixRef,
    ) -> "GaussianDAG":
        """Declare a non-source node from its ``parents`` and own ``noise``.

        ``parents`` maps each parent's name to the gain matrix on that edge.
        Every parent must already be declared (this enforces acyclicity and a
        valid topological order, and catches self-loops). ``parents`` must be
        non-empty -- a parentless node would be a second source, which this
        single-root library does not support.

        Returns ``self`` to allow chaining.
        """
        self._check_new(name)
        if not parents:
            raise ValueError(
                f"Node {name!r} has no parents. A parentless node is a source; "
                "use add_source(name, cov=...). This library allows only one "
                "source."
            )
        for p in parents:
            if p not in self._sources and p not in self._nodes:
                raise ValueError(
                    f"Unknown parent {p!r} of node {name!r}: declare it with "
                    "add_source/add_node before referencing it."
                )
        self._nodes[name] = (dict(parents), noise)
        self._order.append(name)
        return self

    def _check_new(self, name: str) -> None:
        if name in self._sources or name in self._nodes:
            raise ValueError(f"Duplicate node {name!r}.")

    # ------------------------------------------------------------------
    # Lowering: names -> canonical indices / core inputs
    # ------------------------------------------------------------------

    def _canonical_index(self) -> dict[str, int]:
        """Assign canonical 0-based indices via a stable topological sort.

        Kahn's algorithm with a FIFO queue; the queue is seeded, and every tie
        broken, by build/call order (``self._order``). Deterministic for a
        given build script (spec section 12). For a single-source DAG the
        source has in-degree 0 and is enqueued first, so it receives index 0 --
        matching the core's "node 0 is the unique root" convention.
        """
        # Children of each node, and in-degree, in build order.
        children: dict[str, list[str]] = {n: [] for n in self._order}
        indeg: dict[str, int] = {n: 0 for n in self._order}
        for name, (parents, _) in self._nodes.items():
            for p in parents:
                children[p].append(name)
                indeg[name] += 1

        queue: deque[str] = deque(n for n in self._order if indeg[n] == 0)
        index: dict[str, int] = {}
        nxt = 0
        while queue:
            n = queue.popleft()
            index[n] = nxt
            nxt += 1
            for c in children[n]:  # children already in build order
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)

        if len(index) != len(self._order):
            # Unreachable given add_node's pre-declared-parent rule, but guard
            # against a cycle rather than silently dropping nodes.
            raise ValueError("DAG contains a cycle; cannot order nodes.")
        return index

    def _lower_structure(
        self,
    ) -> tuple[list[str], set[int], dict[int, list[int]], set[tuple[int, int]]]:
        """Return the index-based structure (no matrix resolution).

        Yields ``(order, sources, parents, edges)`` where ``order`` lists node
        names by canonical index, ``sources`` is the set of source indices,
        ``parents`` maps each non-source index to its parent indices (in
        insertion order), and ``edges`` is the set of ``(child, parent)`` index
        pairs. Used by the structural-conformance checks (spec section 12).
        """
        idx = self._canonical_index()
        order = [name for name, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        sources = {idx[s] for s in self._sources}
        parents = {
            idx[n]: [idx[p] for p in ps] for n, (ps, _) in self._nodes.items()
        }
        edges = {
            (idx[n], idx[p])
            for n, (ps, _) in self._nodes.items()
            for p in ps
        }
        return order, sources, parents, edges

    @staticmethod
    def _resolve(
        m: MatrixRef, bind: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """Resolve a matrix reference to a concrete tensor.

        A concrete tensor is used as-is; a name (string) is looked up in
        ``bind``. An unbound name raises ``ValueError`` -- never fabricated.
        """
        if isinstance(m, str):
            if bind is None or m not in bind:
                raise ValueError(
                    f"Matrix name {m!r} is not bound. Pass it via "
                    "bind={...} on the query."
                )
            return bind[m]
        return m

    def _lower_core_inputs(
        self, bind: dict[str, torch.Tensor] | None
    ) -> tuple[
        int,
        dict[int, list[int]],
        dict[tuple[int, int], torch.Tensor],
        torch.Tensor,
        dict[int, torch.Tensor],
    ]:
        """Build the (num_nodes, parents, edge_mats, input_cov, noise_covs)
        tuple consumed by ``compute_k_blocks``, resolving every matrix."""
        if not self._sources:
            raise ValueError("No source declared; call add_source(...) first.")
        idx = self._canonical_index()
        source_name = next(iter(self._sources))

        parents = {
            idx[n]: [idx[p] for p in ps] for n, (ps, _) in self._nodes.items()
        }
        edge_mats = {
            (idx[n], idx[p]): self._resolve(g, bind)
            for n, (ps, _) in self._nodes.items()
            for p, g in ps.items()
        }
        noise_covs = {
            idx[n]: self._resolve(nz, bind) for n, (_, nz) in self._nodes.items()
        }
        input_cov = self._resolve(self._sources[source_name], bind)
        return len(self._order), parents, edge_mats, input_cov, noise_covs

    def _require_known(self, name: str) -> None:
        if name not in self._sources and name not in self._nodes:
            raise ValueError(f"Unknown node {name!r}.")

    # ------------------------------------------------------------------
    # Queries (each lowers to the core and returns its result)
    # ------------------------------------------------------------------

    def mi(
        self,
        source: str,
        target: str,
        *,
        bind: dict[str, torch.Tensor] | None = None,
        jitter: float = 0.0,
    ) -> torch.Tensor:
        """Single-pair mutual information ``I(V_source; V_target)`` in nats.

        ``source`` must be the (unique) declared source -- this library models
        ``I(X; Y)`` with ``X`` the root. The result is a differentiable real
        scalar tensor, exactly what ``mutual_information_from_k`` returns.
        """
        self._require_known(source)
        self._require_known(target)
        if source not in self._sources:
            raise ValueError(
                f"source {source!r} is not the root: this library computes "
                "I(X; Y) with X the unique source. Pass the source node as the "
                "first argument."
            )
        idx = self._canonical_index()
        num_nodes, parents, edge_mats, input_cov, noise_covs = (
            self._lower_core_inputs(bind)
        )
        K = compute_k_blocks(num_nodes, parents, edge_mats, input_cov, noise_covs)
        return mutual_information_from_k(
            K, output_node=idx[target], input_node=idx[source], jitter=jitter
        )

    def cov(
        self,
        node: str,
        *,
        bind: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Self-covariance block ``Sigma_node = K_{node,node}``.

        Lowers to ``compute_k_blocks`` and returns the canonical self-block via
        ``get_K``. Differentiable through the edge/noise/source matrices.
        """
        self._require_known(node)
        idx = self._canonical_index()
        num_nodes, parents, edge_mats, input_cov, noise_covs = (
            self._lower_core_inputs(bind)
        )
        K = compute_k_blocks(num_nodes, parents, edge_mats, input_cov, noise_covs)
        return get_K(K, idx[node], idx[node])
