"""Numerical IRAE solvers for the unified fixed-point penalty model.

All active solvers minimize the scaled objective

    R_h(x) + lambda * D(Ax, y),

where ``R_h(x) = 0.5 * ||x - h(x)||_2^2``.  Decreasing ``lambda`` is
equivalent to increasing the quadratic penalty on the constraint ``h(x)=x``.
The positive multiplier may decay, keep, or grow ``lambda`` according to the
degradation-specific continuation schedule.
"""

from __future__ import annotations

import math
from operator import index as integer_index
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from utils.precision import full_precision

Tensor = torch.Tensor

_LAMBDA_MIN = 1e-5
_LAMBDA_WARMUP = 30
_BACKTRACK_FACTOR = 0.7
_MAX_BACKTRACK = 20
_STEP_MIN = 1e-6
# All active solvers start from the same internal step size.
_T_INIT = 0.01
_SPARSE_BACKTRACKING_RULES = {"monotone", "sufficient_decrease"}
_SUFFICIENT_DECREASE_CONSTANT = 1e-4


# Shared validation and continuation utilities.
def _default_device(device: Optional[torch.device]) -> torch.device:
    return device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _prepare_inputs(observation: Tensor, x_init: Optional[Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    observation = observation.to(device)
    initial = observation if x_init is None else x_init.to(device)
    for name, value in (("observation", observation), ("x_init", initial)):
        if value.ndim != 4 or value.numel() == 0 or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite, nonempty floating-point BCHW tensor.")
    if initial.shape != observation.shape:
        raise ValueError("x_init must have the same shape as observation.")
    return observation, initial.detach().clone().requires_grad_(True)


def _check_ae_shape(x: Tensor, ae_model: torch.nn.Module) -> None:
    with torch.no_grad():
        reconstruction = ae_model(x)
    if reconstruction.shape != x.shape:
        raise ValueError(
            "ae_model(x) must have the same shape as x. "
            f"Got ae_model(x).shape={reconstruction.shape}, x.shape={x.shape}."
        )
    if not torch.isfinite(reconstruction).all():
        raise FloatingPointError("ae_model(x) must be finite.")


def _validate_continuation(
    lambda_: float,
    lambda_multiplier: float,
    lambda_check_interval: int,
    max_iteration: int,
) -> None:
    if not math.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError("lambda_ must be finite and positive.")
    if not math.isfinite(lambda_multiplier) or lambda_multiplier <= 0:
        raise ValueError("lambda_multiplier must be finite and positive.")
    if integer_index(lambda_check_interval) <= 0:
        raise ValueError("lambda_check_interval must be positive.")
    if integer_index(max_iteration) < 0:
        raise ValueError("max_iteration must be non-negative.")


def _should_update_lambda(iteration: int, interval: int) -> bool:
    return iteration >= _LAMBDA_WARMUP and iteration % interval == 0


def _multiply_lambda(lambda_: float, multiplier: float) -> float:
    updated = max(lambda_ * multiplier, _LAMBDA_MIN)
    if not math.isfinite(updated):
        raise ValueError("Updated lambda must be finite.")
    return updated


# Sparse L1 PGD is intentionally kept numerically identical to the searched implementation.
@full_precision()
def irae_sparse_l1_pgd(
    observation: Tensor,
    ae_model: torch.nn.Module,
    x_init: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    lambda_: float = 0.01,
    is_lambda_update: bool = True,
    lambda_multiplier: float = 0.8,
    lambda_check_interval: int = 10,
    max_iteration: int = 400,
    backtracking_rule: str = "sufficient_decrease",
    sufficient_decrease_constant: float = _SUFFICIENT_DECREASE_CONSTANT,
    DEBUG: bool = False,
    return_history: bool = False,
):
    """Solve ``R_h(x) + lambda * ||observation-x||_1`` by proximal GD."""
    _validate_continuation(lambda_, lambda_multiplier, lambda_check_interval, max_iteration)
    backtracking_rule = str(backtracking_rule).strip().lower()
    if backtracking_rule not in _SPARSE_BACKTRACKING_RULES:
        raise ValueError(
            f"Unsupported sparse backtracking_rule={backtracking_rule!r}. "
            f"Expected one of {sorted(_SPARSE_BACKTRACKING_RULES)}."
        )
    sufficient_decrease_constant = float(sufficient_decrease_constant)
    if not math.isfinite(sufficient_decrease_constant) or not 0 < sufficient_decrease_constant < 1:
        raise ValueError("sufficient_decrease_constant must be finite and lie in (0, 1).")
    device = _default_device(device)
    observation, x = _prepare_inputs(observation, x_init, device)
    ae_model = _prepare_model(ae_model, device)
    _check_ae_shape(x, ae_model)

    t = _T_INIT
    history_x = [x.detach().cpu()] if return_history else []
    history_metrics = []
    backtrack_failures = 0

    if DEBUG:
        recon_loss = ((x - ae_model(x)) ** 2).sum().item()
        l1_loss = torch.norm(observation - x, p=1).item()
        print(
            f"Sparse-L1 PGD iter 0: obj={0.5 * recon_loss + lambda_ * l1_loss:.4f}, "
            f"recon_loss={recon_loss:.4f}, l1_loss={l1_loss:.4f}, "
            f"lambda_={lambda_:.4e}, step_size={t:.4e}"
        )

    for iteration in range(max_iteration):
        if is_lambda_update and _should_update_lambda(iteration, lambda_check_interval):
            updated_lambda = _multiply_lambda(lambda_, lambda_multiplier)
            if updated_lambda != lambda_:
                lambda_ = updated_lambda
                if DEBUG:
                    print(f"  --> updated lambda_ to {lambda_:.4e} at iter {iteration}")
                # Sparse proximal updates retain the conservative step reset used by its searched schedule.
                # The smooth linear solver below instead carries its accepted step across lambda changes.
                t = min(_T_INIT, t)

        reconstruction = ae_model(x)
        recon_term = 0.5 * ((x - reconstruction) ** 2).sum()
        data_term = lambda_ * torch.norm(observation - x, p=1)
        objective_current = (recon_term + data_term).detach()
        if not torch.isfinite(objective_current):
            raise FloatingPointError("Sparse-L1 objective is not finite.")

        if return_history:
            history_metrics.append(
                {
                    "iter": iteration,
                    "total_loss_scaled": objective_current.item(),
                    "recon_residual_sq": (2 * recon_term).item(),
                    "data_l1": (data_term / lambda_).item(),
                    "lambda": float(lambda_),
                    "step_size": float(t),
                }
            )

        if x.grad is not None:
            x.grad.zero_()
        recon_term.backward()
        grad_x = x.grad.detach().clone()
        if not torch.isfinite(grad_x).all():
            raise FloatingPointError("Sparse-L1 gradient is not finite.")
        grad_norm = torch.norm(grad_x, p=2)

        with torch.no_grad():
            t_current = t
            found = False
            for _ in range(_MAX_BACKTRACK):
                v = x - t_current * grad_x
                x_candidate = observation - torch.sign(observation - v) * torch.max(
                    torch.abs(observation - v) - t_current * lambda_, torch.zeros_like(v)
                )
                reconstruction_candidate = ae_model(x_candidate)
                recon_candidate = 0.5 * ((x_candidate - reconstruction_candidate) ** 2).sum()
                data_candidate = lambda_ * torch.norm(observation - x_candidate, p=1)
                # Preserve the legacy test for reproduction; the default rule certifies composite decrease.
                if backtracking_rule == "monotone":
                    accepted = recon_candidate + data_candidate <= objective_current
                else:
                    # This is exactly the paper's proximal Armijo sufficient-decrease rule.
                    delta_sq = (x_candidate - x).square().sum()
                    required_decrease = sufficient_decrease_constant * delta_sq / (2.0 * t_current)
                    accepted = recon_candidate + data_candidate <= objective_current - required_decrease
                if accepted:
                    found = True
                    break
                t_current *= _BACKTRACK_FACTOR
                if t_current < _STEP_MIN:
                    break

            if found:
                x = x_candidate.detach().requires_grad_(True)
                t = t_current / _BACKTRACK_FACTOR
            else:
                if backtracking_rule == "sufficient_decrease":
                    raise RuntimeError(
                        "Sparse-L1 PGD backtracking failed to satisfy the sufficient-decrease condition."
                    )
                backtrack_failures += 1
                t = max(t_current, _STEP_MIN)

        if x.grad is not None:
            x.grad.zero_()
        if return_history:
            history_x.append(x.detach().cpu().clone())
        if DEBUG and (iteration % 10 == 0 or iteration == max_iteration - 1):
            recon_loss = ((x - ae_model(x)) ** 2).sum().item()
            l1_loss = torch.norm(observation - x, p=1).item()
            print(
                f"Sparse-L1 PGD iter {iteration}: obj={0.5 * recon_loss + lambda_ * l1_loss:.4f}, "
                f"recon_loss={recon_loss:.4f}, l1_loss={l1_loss:.4f}, "
                f"grad_norm={grad_norm:.4f}, lambda_={lambda_:.4e}, step_size={t:.4e}"
            )

    if DEBUG:
        print(
            f"Sparse-L1 PGD finished with {backtrack_failures} backtracking "
            f"failures over {max_iteration} iterations."
        )
    if return_history:
        return x.detach(), history_x, history_metrics
    return x.detach()


# Linear degradations share one smooth finite-penalty solver.
def irae_linear_l2_gd(
    observation: Tensor,
    ae_model: torch.nn.Module,
    forward_operator: Callable[[Tensor], Tensor],
    x_init: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    lambda_: float = 1.0,
    is_lambda_update: bool = True,
    lambda_multiplier: float = 0.8,
    lambda_check_interval: int = 10,
    max_iteration: int = 400,
    DEBUG: bool = False,
    return_history: bool = False,
    operator_name: str = "A",
):
    """Solve ``R_h(x) + 0.5 * lambda * ||A(x)-y||_2^2`` by gradient descent."""
    _validate_continuation(lambda_, lambda_multiplier, lambda_check_interval, max_iteration)
    device = _default_device(device)
    observation, x = _prepare_inputs(observation, x_init, device)
    ae_model = _prepare_model(ae_model, device)
    _check_ae_shape(x, ae_model)

    with torch.no_grad():
        forward_shape = forward_operator(x).shape
    if forward_shape != observation.shape:
        raise ValueError(
            "forward_operator(x) must have the same shape as observation. "
            f"Got forward_operator(x).shape={forward_shape}, observation.shape={observation.shape}."
        )

    def objective(z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        reconstruction = ae_model(z)
        recon_term = 0.5 * ((z - reconstruction) ** 2).sum()
        residual = forward_operator(z) - observation
        discrepancy = 0.5 * residual.square().sum()
        return recon_term + lambda_ * discrepancy, recon_term, discrepancy, residual

    t = _T_INIT
    history_x = [x.detach().cpu()] if return_history else []
    history_metrics = []
    backtrack_failures = 0

    for iteration in range(max_iteration):
        lambda_updated = False
        if is_lambda_update and _should_update_lambda(iteration, lambda_check_interval):
            updated_lambda = _multiply_lambda(lambda_, lambda_multiplier)
            if updated_lambda != lambda_:
                lambda_ = updated_lambda
                lambda_updated = True
                # Unlike sparse proximal PGD, smooth linear GD keeps the accepted step.
                # Armijo backtracking revalidates it against the newly weighted objective.
                if DEBUG:
                    print(f"  --> updated lambda_ to {lambda_:.4e} at iter {iteration}")

        objective_current, recon_term, discrepancy, residual = objective(x)
        objective_detached = objective_current.detach()
        if not torch.isfinite(objective_detached):
            raise FloatingPointError("Linear-L2 objective is not finite.")
        if return_history:
            history_metrics.append(
                {
                    "iter": iteration,
                    "total_loss_scaled": objective_detached.item(),
                    "recon_residual_sq": (2 * recon_term).item(),
                    "data_discrepancy": discrepancy.item(),
                    "data_residual_sq": residual.square().sum().item(),
                    "lambda": float(lambda_),
                    "lambda_updated": lambda_updated,
                    "step_size": float(t),
                }
            )

        if x.grad is not None:
            x.grad.zero_()
        objective_current.backward()
        grad_x = x.grad.detach().clone()
        if not torch.isfinite(grad_x).all():
            raise FloatingPointError("Linear-L2 gradient is not finite.")
        grad_sq = grad_x.square().sum()
        grad_norm = torch.sqrt(grad_sq)

        with torch.no_grad():
            t_current = t
            found = False
            for _ in range(_MAX_BACKTRACK):
                x_candidate = x - t_current * grad_x
                objective_candidate, _, _, _ = objective(x_candidate)
                if objective_candidate <= objective_detached - 0.5 * t_current * grad_sq:
                    found = True
                    break
                t_current *= _BACKTRACK_FACTOR
                if t_current < _STEP_MIN:
                    break
            if found:
                x = x_candidate.detach().requires_grad_(True)
                t = t_current / _BACKTRACK_FACTOR
            else:
                backtrack_failures += 1
                t = max(t_current, _STEP_MIN)

        if x.grad is not None:
            x.grad.zero_()
        if return_history:
            history_x.append(x.detach().cpu().clone())
        if DEBUG and (iteration % 10 == 0 or iteration == max_iteration - 1):
            print(
                f"Linear-L2 GD iter {iteration}: obj={objective_detached.item():.4f}, "
                f"recon_loss={(2 * recon_term).item():.4f}, "
                f"data_discrepancy={discrepancy.item():.4f}, grad_norm={grad_norm:.4f}, "
                f"lambda_={lambda_:.4e}, step_size={t:.4e}, operator={operator_name}"
            )

    if DEBUG:
        print(
            f"Linear-L2 GD finished with {backtrack_failures} backtracking "
            f"failures over {max_iteration} iterations."
        )
    if return_history:
        return x.detach(), history_x, history_metrics
    return x.detach()


# Inpainting supplies the paper's raw masked L2 discrepancy.
def irae_inpainting_l2_gd(
    observation: Tensor,
    mask: Tensor,
    ae_model: torch.nn.Module,
    x_init: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    lambda_: float = 1.0,
    is_lambda_update: bool = True,
    lambda_multiplier: float = 0.8,
    lambda_check_interval: int = 10,
    max_iteration: int = 400,
    DEBUG: bool = False,
    return_history: bool = False,
):
    device = _default_device(device)
    observation = observation.to(device)
    mask = mask.to(device=device, dtype=observation.dtype)
    if not bool(torch.logical_or(mask == 0, mask == 1).all()):
        raise ValueError("mask must be binary: 1 for observed entries and 0 for missing entries.")
    try:
        mask = torch.broadcast_to(mask, observation.shape)
    except RuntimeError as exc:
        raise ValueError("mask must be broadcastable to observation.shape.") from exc
    observed_count = mask.reshape(mask.shape[0], -1).sum(dim=1)
    if not bool((observed_count > 0).all()):
        raise ValueError("Each inpainting sample must contain at least one observed entry.")
    return irae_linear_l2_gd(
        observation=observation,
        ae_model=ae_model,
        forward_operator=lambda z: mask * z,
        x_init=x_init,
        device=device,
        lambda_=lambda_,
        is_lambda_update=is_lambda_update,
        lambda_multiplier=lambda_multiplier,
        lambda_check_interval=lambda_check_interval,
        max_iteration=max_iteration,
        DEBUG=DEBUG,
        return_history=return_history,
        operator_name="M",
    )


# Deblurring uses the same linear solver with a differentiable depthwise convolution.
def _prepare_blur_weight(kernel: Tensor, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    kernel = kernel.detach().clone().to(device=device, dtype=dtype)
    if kernel.dim() == 2:
        kernel = kernel[None, None]
    elif kernel.dim() != 4 or kernel.shape[:2] != (1, 1):
        raise ValueError("kernel must be 2D [kH,kW] or 4D [1,1,kH,kW].")
    if kernel.shape[-2] % 2 == 0 or kernel.shape[-1] % 2 == 0:
        raise ValueError("Only odd-sized blur kernels are supported.")
    if not bool(torch.isfinite(kernel).all()) or float(kernel.abs().sum()) == 0:
        raise ValueError("kernel must be finite and non-zero.")
    return kernel.repeat(channels, 1, 1, 1)


def make_blur_operator(kernel: Tensor, padding_mode: str = "reflect") -> Callable[[Tensor], Tensor]:
    padding_mode = str(padding_mode).strip().lower()
    if padding_mode == "constant":
        padding_mode = "zeros"
    if padding_mode not in {"reflect", "replicate", "circular", "zeros"}:
        raise ValueError("padding_mode must be one of {'reflect','replicate','circular','zeros','constant'}.")
    kernel = kernel.detach().clone()
    cache: Dict[Tuple[torch.device, torch.dtype, int], Tensor] = {}

    def forward_operator(z: Tensor) -> Tensor:
        if z.dim() != 4:
            raise ValueError(f"The blur operator expects NCHW tensors, got z.dim()={z.dim()}.")
        key = (z.device, z.dtype, z.shape[1])
        if key not in cache:
            cache[key] = _prepare_blur_weight(kernel, z.shape[1], z.device, z.dtype)
        weight = cache[key]
        pad_h, pad_w = weight.shape[-2] // 2, weight.shape[-1] // 2
        if padding_mode == "zeros":
            return F.conv2d(z, weight, padding=(pad_h, pad_w), groups=z.shape[1])
        padded = F.pad(z, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
        return F.conv2d(padded, weight, groups=z.shape[1])

    return forward_operator


def irae_deblurring_l2_gd(
    observation: Tensor,
    kernel: Tensor,
    ae_model: torch.nn.Module,
    x_init: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    lambda_: float = 1.0,
    is_lambda_update: bool = True,
    lambda_multiplier: float = 0.8,
    lambda_check_interval: int = 10,
    max_iteration: int = 400,
    DEBUG: bool = False,
    return_history: bool = False,
    padding_mode: str = "reflect",
):
    """Use the paper's raw non-blind deblurring L2 discrepancy."""
    return irae_linear_l2_gd(
        observation=observation,
        ae_model=ae_model,
        forward_operator=make_blur_operator(kernel, padding_mode),
        x_init=x_init,
        device=device,
        lambda_=lambda_,
        is_lambda_update=is_lambda_update,
        lambda_multiplier=lambda_multiplier,
        lambda_check_interval=lambda_check_interval,
        max_iteration=max_iteration,
        DEBUG=DEBUG,
        return_history=return_history,
        operator_name="K",
    )
