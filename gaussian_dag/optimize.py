"""Simple projected gradient ascent (PGA) for linear Gaussian DAG MI maximization.

This module provides a topology-agnostic outer-loop wrapper that ascends a
user-supplied MI closure with constant step size, optionally followed by a
user-supplied projector at each iteration.

Design notes:
- This wrapper depends only on a callable `compute_mi` and on the
  pytorch-tensor-list `params`; it has no knowledge of DAG structure.
- The projector is called inside `torch.no_grad()`. Two calling
  conventions are accepted:

    (a) In-place: the projector mutates `params` directly (e.g., via
        `param.copy_(...)`) and returns `None`.
    (b) Functional: the projector returns a sequence of tensors of the
        same length as `params`; `pga_ascent` then copies each returned
        tensor into the corresponding parameter via `.copy_()`.

  Convention (b) lets users wire `project_frobenius_ball` /
  `project_total_power` (which return new tensors) directly as the
  projector without an explicit `.copy_` wrapper, removing a silent
  footgun where the projection result is accidentally discarded.

- The convention is to *ascend* mutual information directly: `I.backward()`
  is invoked, and parameters are updated via `p.add_(alpha * p.grad)`.
  PyTorch's `.grad` for a complex `p` and real-valued `I` is the natural
  Wirtinger gradient (without the 1/2 factor; equivalently, the real-Euclidean
  steepest-ascent direction on the real and imaginary parts of `p`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def pga_ascent(
    compute_mi: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Callable[[list[torch.Tensor]], None | Sequence[torch.Tensor]] | None = None,
) -> list[float]:
    """Run projected gradient ASCENT on mutual information.

    At each iteration t = 0, 1, ..., num_iters - 1:
        1. Zero out any existing gradients in `params`.
        2. Compute `I = compute_mi()` and call `I.backward()`.
           Record `I.item()` in the history list.
        3. Inside `torch.no_grad()`:
           a. Update each `p in params` via `p.add_(step_size * p.grad)`.
           b. If `projector` is provided, call `projector(params)`. The
              projector may either mutate `params` in place and return
              `None`, or return a sequence of new tensors (one per
              parameter) which `pga_ascent` will copy into place. Mixing
              the two is not supported within one call.

    Args:
        compute_mi: Closure that, given the current state of `params`,
            constructs the autograd graph and returns the scalar MI tensor.
        params: List of leaf tensors with `requires_grad=True`.
        step_size: Constant positive step size.
        num_iters: Number of PGA iterations (must be > 0).
        projector: Optional callable taking `params`. May either mutate in
            place (returning `None`) or return a sequence of new tensors
            (one per parameter), in which case the returned tensors are
            copied into the parameters by `pga_ascent`.

    Returns:
        history: List of length `num_iters`, where history[t] = I.item()
            recorded immediately after the forward pass of iteration t
            (i.e., the MI evaluated at the *pre-update* parameter values for
            iteration t).
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    for p in params:
        if not p.requires_grad:
            raise ValueError("All entries of `params` must have requires_grad=True.")

    history: list[float] = []
    for _ in range(num_iters):
        # 1. Zero out previous gradients (allowed to be None on first iteration).
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        # 2. Forward + backward.
        I = compute_mi()
        I.backward()
        history.append(I.item())
        # 3. Update and project.
        with torch.no_grad():
            for idx, p in enumerate(params):
                if p.grad is None:
                    raise RuntimeError(
                        f"params[{idx}] received no gradient after backward(): "
                        "the parameter has requires_grad=True but does not "
                        "participate in the autograd graph produced by "
                        "compute_mi(). Common causes: (a) the parameter is "
                        "declared but never used in the closure; (b) the "
                        "closure rebinds the parameter to a new tensor "
                        "(e.g. via `F = F.detach()` or in-place arithmetic "
                        "outside torch.no_grad); (c) a typo in a closure-"
                        "captured variable name. Verify that the parameter "
                        "is referenced inside compute_mi() and that the "
                        "returned MI tensor depends on it."
                    )
                p.add_(step_size * p.grad)
            if projector is not None:
                out = projector(params)
                if out is not None:
                    # Functional projector: copy each returned tensor into place.
                    if len(out) != len(params):
                        raise ValueError(
                            f"projector returned {len(out)} tensors, expected "
                            f"{len(params)}."
                        )
                    for p, q in zip(params, out):
                        p.copy_(q)
    return history
