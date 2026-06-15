"""Unit tests for gaussian_dag.builder (named-node DAG builder).

These tests cover the acceptance criteria of builder_implementation.md
(spec v0.2): source-only, a chain, a node with multiple parents, the
round-trip equivalence to the functional core, the structural-conformance
vectors (section 12), and the loud-failure requirements for constructs outside
this library's single-pair / single-root profiles (section 4.4).
"""

from __future__ import annotations

import pytest
import torch

from gaussian_dag import GaussianDAG
from gaussian_dag.information import mutual_information_from_k
from gaussian_dag.krecursion import compute_k_blocks, get_K


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
# source-only
# ============================================================


def test_source_only_cov_matches():
    d = 3
    Sigma_X = _hermitian_psd(d, seed=100)
    dag = GaussianDAG()
    dag.add_source("X", cov=Sigma_X)
    # cov of the source is its own covariance (hermitianized by the core).
    assert torch.allclose(dag.cov("X"), Sigma_X, atol=1e-12)


# ============================================================
# chain X -> Y -> Z : round-trip vs the functional core (section 11.3)
# ============================================================


def test_chain_mi_roundtrip():
    d = 2
    Sigma_X = _hermitian_psd(d, seed=100)
    H_XY = _randn_complex(d, d, seed=1)
    H_YZ = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=201)
    N_Z = _hermitian_psd(d, seed=202)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sigma_X)
    dag.add_node("Y", parents={"X": H_XY}, noise=N_Y)
    dag.add_node("Z", parents={"Y": H_YZ}, noise=N_Z)
    mi_builder = dag.mi("X", "Z")

    # Same DAG built by hand with the index-based core (X=0, Y=1, Z=2).
    K = compute_k_blocks(
        3,
        {1: [0], 2: [1]},
        {(1, 0): H_XY, (2, 1): H_YZ},
        Sigma_X,
        {1: N_Y, 2: N_Z},
    )
    mi_core = mutual_information_from_k(K, output_node=2, input_node=0)

    assert torch.allclose(mi_builder, mi_core, atol=1e-10)


def test_chain_intermediate_target_roundtrip():
    d = 2
    Sigma_X = _hermitian_psd(d, seed=100)
    H_XY = _randn_complex(d, d, seed=1)
    H_YZ = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=201)
    N_Z = _hermitian_psd(d, seed=202)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sigma_X)
    dag.add_node("Y", parents={"X": H_XY}, noise=N_Y)
    dag.add_node("Z", parents={"Y": H_YZ}, noise=N_Z)

    K = compute_k_blocks(
        3,
        {1: [0], 2: [1]},
        {(1, 0): H_XY, (2, 1): H_YZ},
        Sigma_X,
        {1: N_Y, 2: N_Z},
    )
    # I(X; Y) for the intermediate node.
    assert torch.allclose(
        dag.mi("X", "Y"),
        mutual_information_from_k(K, output_node=1, input_node=0),
        atol=1e-10,
    )
    # cov(Z) self-block.
    assert torch.allclose(dag.cov("Z"), get_K(K, 2, 2), atol=1e-10)


# ============================================================
# multi-parent diamond X -> {Y, W} -> Z : round-trip
# ============================================================


def test_diamond_mi_roundtrip():
    d = 2
    Sigma_X = _hermitian_psd(d, seed=100)
    A_XY = _randn_complex(d, d, seed=1)
    A_XW = _randn_complex(d, d, seed=2)
    A_YZ = _randn_complex(d, d, seed=3)
    A_WZ = _randn_complex(d, d, seed=4)
    N_Y = _hermitian_psd(d, seed=201)
    N_W = _hermitian_psd(d, seed=202)
    N_Z = _hermitian_psd(d, seed=203)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sigma_X)
    dag.add_node("Y", parents={"X": A_XY}, noise=N_Y)
    dag.add_node("W", parents={"X": A_XW}, noise=N_W)
    dag.add_node("Z", parents={"Y": A_YZ, "W": A_WZ}, noise=N_Z)
    mi_builder = dag.mi("X", "Z")

    # Hand-built (X=0, Y=1, W=2, Z=3).
    K = compute_k_blocks(
        4,
        {1: [0], 2: [0], 3: [1, 2]},
        {(1, 0): A_XY, (2, 0): A_XW, (3, 1): A_YZ, (3, 2): A_WZ},
        Sigma_X,
        {1: N_Y, 2: N_W, 3: N_Z},
    )
    mi_core = mutual_information_from_k(K, output_node=3, input_node=0)

    assert torch.allclose(mi_builder, mi_core, atol=1e-10)


# ============================================================
# Structural-conformance vectors (spec section 12, structure only)
# ============================================================


def test_structure_chain():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    dag.add_node("Y", parents={"X": "H_XY"}, noise="N_Y")
    dag.add_node("Z", parents={"Y": "H_YZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "Z"]
    assert sources == {0}
    assert parents == {1: [0], 2: [1]}
    assert edges == {(1, 0), (2, 1)}


def test_structure_diamond():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    dag.add_node("Y", parents={"X": "H_XY"}, noise="N_Y")
    dag.add_node("W", parents={"X": "H_XW"}, noise="N_W")
    dag.add_node("Z", parents={"Y": "H_YZ", "W": "H_WZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "W", "Z"]
    assert sources == {0}
    assert parents == {1: [0], 2: [0], 3: [1, 2]}
    assert edges == {(1, 0), (2, 0), (3, 1), (3, 2)}


# ============================================================
# Binding: name-or-object resolved at query time (spec section 8)
# ============================================================


def test_bind_by_name_matches_concrete():
    d = 2
    Sigma_X = _hermitian_psd(d, seed=100)
    H_XY = _randn_complex(d, d, seed=1)
    N_Y = _hermitian_psd(d, seed=201)

    by_name = GaussianDAG()
    by_name.add_source("X", cov="Sigma_X")
    by_name.add_node("Y", parents={"X": "H_XY"}, noise="N_Y")
    mi_named = by_name.mi(
        "X", "Y", bind={"Sigma_X": Sigma_X, "H_XY": H_XY, "N_Y": N_Y}
    )

    by_obj = GaussianDAG()
    by_obj.add_source("X", cov=Sigma_X)
    by_obj.add_node("Y", parents={"X": H_XY}, noise=N_Y)
    mi_obj = by_obj.mi("X", "Y")

    assert torch.allclose(mi_named, mi_obj, atol=1e-12)


def test_unbound_name_raises():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    dag.add_node("Y", parents={"X": "H_XY"}, noise="N_Y")
    with pytest.raises(ValueError, match="not bound"):
        dag.mi("X", "Y", bind={"Sigma_X": _hermitian_psd(2, seed=1)})


# ============================================================
# Differentiability survives the builder (core autograd preserved)
# ============================================================


def test_mi_is_differentiable_through_builder():
    d = 2
    Sigma_X = torch.eye(d, dtype=DTYPE)
    F = (0.1 * _randn_complex(d, d, seed=7)).requires_grad_(True)
    H = _randn_complex(d, d, seed=8)
    N_Y = _hermitian_psd(d, seed=201)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sigma_X)
    dag.add_node("Y", parents={"X": H @ F}, noise=N_Y)
    mi = dag.mi("X", "Y")
    mi.backward()

    assert F.grad is not None
    assert torch.isfinite(F.grad).all()


# ============================================================
# Loud failures for unsupported constructs (spec section 4.4)
# ============================================================


def test_second_source_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    with pytest.raises(ValueError, match="single-root"):
        dag.add_source("Y", cov="Sigma_Y")


def test_unknown_parent_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    with pytest.raises(ValueError, match="Unknown parent"):
        dag.add_node("Z", parents={"Q": "H"}, noise="N_Z")


def test_duplicate_name_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    with pytest.raises(ValueError, match="Duplicate"):
        dag.add_node("X", parents={"X": "H"}, noise="N")


def test_self_loop_rejected():
    # A self-loop names a not-yet-declared node as its own parent.
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    with pytest.raises(ValueError, match="Unknown parent"):
        dag.add_node("Y", parents={"Y": "H"}, noise="N_Y")


def test_parentless_node_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sigma_X")
    with pytest.raises(ValueError, match="no parents"):
        dag.add_node("Y", parents={}, noise="N_Y")


def test_non_source_as_mi_source_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov=_hermitian_psd(2, seed=1))
    dag.add_node("Y", parents={"X": _randn_complex(2, 2, seed=2)},
                 noise=_hermitian_psd(2, seed=3))
    with pytest.raises(ValueError, match="not the root"):
        dag.mi("Y", "X")


def test_no_conditioning_query():
    # Single-pair profile: there is no cmi method (no conditioning set).
    dag = GaussianDAG()
    assert not hasattr(dag, "cmi")
