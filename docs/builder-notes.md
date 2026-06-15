# Builder notes — gaussian-dag

Library-specific decisions for the named-node DAG builder, per the template in
`builder_implementation.md` §13. The shared policy lives in that document; only
the choices it delegates to the implementer are recorded here.

- **Conforms to builder_implementation.md spec version:** 0.2
- **Profiles implemented:** `single-pair` (MI between the source and a target,
  no conditioning set) and `single-root` (exactly one source node). The
  `conditional`, `multiroot`, and `stochastic/batch` profiles are **not**
  implemented.
- **Query method(s):**
  - `mi(source, target, *, bind=None, jitter=0.0)` — `I(V_source; V_target)` in
    nats; `source` must be the unique declared source (the model's `X`).
  - `cov(node, *, bind=None)` — the self-covariance block `Σ_node = K_{node,node}`.
  - Returns whatever the core returns: a differentiable real-scalar tensor
    (`mi`) or a covariance-block tensor (`cov`).
- **Class name:** `GaussianDAG` — matches the worked example in spec §3/§14 for
  maximal cross-library recognizability. No collision: the name did not exist
  anywhere in the repo, and it is added to `gaussian_dag/__init__.__all__`
  alongside (never replacing) the existing 9 public symbols.
- **Matrix input:** both **by name (string)** and **as a concrete tensor**. A
  tensor is used as-is; a name is resolved at query time via a
  `bind={name: tensor}` mapping passed to the query. An unbound name raises
  `ValueError` — data is never fabricated (spec §8). No batch axis (not a
  stochastic library).
- **Matrix conventions:** inherited from the core. `complex128` on CPU/CUDA is
  the standard dtype; tensors keep their own dtype/device (device-agnostic).
  Edge keys lower to `(j, i)` for the edge `i → j` with `i < j`; node 0 is the
  unique root. Self-covariance blocks are Hermitian-symmetrized by the core
  (`symmetrize_self_blocks=True`).
- **Module / namespace:** `gaussian_dag/builder.py`, re-exported from
  `gaussian_dag/__init__.py`.
- **Canonical index (spec §12):** stable topological sort — Kahn's algorithm
  with a FIFO queue, seeding and tie-breaking by build/call order. A
  single-source DAG maps its source to index 0, matching the core's root
  convention. Exposed for structural tests via the internal
  `_lower_structure()`.
- **Deliberate divergences from the recommended idioms (§5) and why:**
  - **No `cmi` query.** This library has no conditioning concept (single-pair
    profile, spec §9). There is intentionally no `cmi` method, so a request for
    conditional MI raises the natural `AttributeError` rather than returning a
    wrong answer (satisfies §4.4 "fail loudly").
  - **`mi` requires `source` to be the root.** Faithful to the modeled quantity
    `I(X; Y)` with `X` the unique source; passing a non-source as `source`
    raises `ValueError`.
- **Unsupported constructs and the errors raised:**
  - A second `add_source` (multi-root) → `ValueError` ("single-root").
  - A parentless `add_node` (would be a second source) → `ValueError`.
  - An `add_node` referencing an undeclared parent (also catches self-loops) →
    `ValueError` ("Unknown parent").
  - A duplicate node name → `ValueError` ("Duplicate").
  - An unbound matrix name at query time → `ValueError` ("not bound").
  - `mi` with a non-source `source` argument → `ValueError` ("not the root").
  - Conditional MI (`cmi`) → `AttributeError` (no such method).

## Structural-conformance vectors (spec §12)

`chain` and `diamond` are expressible and verified by
`tests/test_builder.py::test_structure_chain` / `test_structure_diamond`. The
`two-source (MAC-like)` vector is **exempt** (single-root library) but is still
rejected loudly — see `test_second_source_rejected`.
