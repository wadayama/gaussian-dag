"""Gaussian-DAG: linear Gaussian DAG mutual information via K-recursion."""

from gaussian_dag.builder import GaussianDAG
from gaussian_dag.information import logdet_hpd, mutual_information_from_k
from gaussian_dag.krecursion import (
    compute_effective_channel,
    compute_k_blocks,
    get_K,
    hermitianize,
)
from gaussian_dag.optimize import pga_ascent
from gaussian_dag.projections import project_frobenius_ball, project_total_power

__all__ = [
    "GaussianDAG",
    "compute_effective_channel",
    "compute_k_blocks",
    "get_K",
    "hermitianize",
    "logdet_hpd",
    "mutual_information_from_k",
    "pga_ascent",
    "project_frobenius_ball",
    "project_total_power",
]
