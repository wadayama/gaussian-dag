"""GPU smoke tests.

These run only when a CUDA device is available; otherwise they are skipped
(the public CI does not exercise this path, but the tests are kept in the
suite so that GPU users can verify their installation with ``uv run pytest``).

The strategy is:
  1. Build a tiny linear Gaussian DAG on the GPU.
  2. Run one forward K-recursion + one MI evaluation + one PGA step.
  3. Verify that the result agrees with a CPU reference to within floating
     point tolerance.

This stresses the device-agnostic claim of the library: no tensor is allocated
without inheriting ``device`` and ``dtype`` from its inputs.
"""

from __future__ import annotations

import pytest
import torch

from gaussian_dag import (
    compute_k_blocks,
    mutual_information_from_k,
    pga_ascent,
    project_frobenius_ball,
)

DTYPE = torch.complex128

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device not available",
)


def _build_single_link_problem(device: torch.device, d: int = 3, sigma: float = 0.5):
    """Construct a deterministic single-link MIMO problem on the given device."""
    torch.manual_seed(0)
    H_real = torch.randn(d, d, dtype=torch.float64)
    H_imag = torch.randn(d, d, dtype=torch.float64)
    H = torch.complex(H_real, H_imag).to(DTYPE).to(device)
    Sigma_X = torch.eye(d, dtype=DTYPE, device=device)
    Sigma_Z = (sigma ** 2) * torch.eye(d, dtype=DTYPE, device=device)
    F = (0.1 * H.detach().clone()).requires_grad_(True)
    return H, F, Sigma_X, Sigma_Z


def _evaluate_mi(H, F, Sigma_X, Sigma_Z):
    K = compute_k_blocks(
        num_nodes=2,
        parents={1: [0]},
        edge_mats={(1, 0): H @ F},
        input_cov=Sigma_X,
        noise_covs={1: Sigma_Z},
    )
    return mutual_information_from_k(K, output_node=1, input_node=0)


@cuda_only
def test_k_recursion_on_cuda_matches_cpu():
    """The forward K-recursion + MI evaluation must agree CPU vs GPU."""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    H_cpu, F_cpu, Sx_cpu, Sz_cpu = _build_single_link_problem(cpu)
    H_gpu, F_gpu, Sx_gpu, Sz_gpu = _build_single_link_problem(gpu)

    mi_cpu = _evaluate_mi(H_cpu, F_cpu, Sx_cpu, Sz_cpu)
    mi_gpu = _evaluate_mi(H_gpu, F_gpu, Sx_gpu, Sz_gpu)

    assert mi_gpu.device.type == "cuda"
    assert mi_cpu.device.type == "cpu"
    # Both forward graphs read the same tensors, so the agreement should be
    # tight (numerical jitter only).
    assert abs(mi_cpu.item() - mi_gpu.item()) < 1e-10


@cuda_only
def test_pga_step_on_cuda():
    """One PGA step on CUDA: gradient flows, MI strictly increases."""
    gpu = torch.device("cuda")
    H, F, Sx, Sz = _build_single_link_problem(gpu)

    P = 5.0

    def compute_mi():
        return _evaluate_mi(H, F, Sx, Sz)

    def projector(params):
        for p in params:
            p.copy_(project_frobenius_ball(p, P))

    mi_before = float(compute_mi().item())
    history = pga_ascent(
        compute_mi, [F], step_size=0.05, num_iters=10, projector=projector,
    )
    mi_after = history[-1]

    assert F.device.type == "cuda"
    assert mi_after > mi_before, (
        f"PGA on CUDA failed to increase MI: {mi_before:.6f} -> {mi_after:.6f}"
    )
