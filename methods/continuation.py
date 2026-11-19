"""Thin experiment-facing dispatch for the unified IRAE numerical solvers."""

from __future__ import annotations

import math
from operator import index as integer_index
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from methods.irae_solver import (
    _default_device,
    irae_deblurring_l2_gd,
    irae_inpainting_l2_gd,
    irae_sparse_l1_pgd,
)

Tensor = torch.Tensor


def _normalize_mode(mode: Any) -> str:
    mode = str(mode).strip().lower()
    return {
        "sp": "sparse",
        "salt_pepper": "sparse",
        "salt-and-pepper": "sparse",
        "inpaint": "inpainting",
        "blur": "deblur",
        "deblurring": "deblur",
    }.get(mode, mode)


def _require_tensor(value: Any, name: str, device: torch.device) -> Tensor:
    if value is None:
        raise ValueError(f"degradation['{name}'] is required.")
    if not torch.is_tensor(value):
        raise TypeError(f"degradation['{name}'] must be a torch.Tensor.")
    return value.to(device)


def _validate_solver_input_range(value: Tensor, name: str) -> None:
    if value.ndim != 4 or value.numel() == 0 or not value.is_floating_point():
        raise ValueError(f"{name} must be a finite, nonempty floating-point BCHW tensor.")
    value = value.detach()
    value_min, value_max = float(value.min()), float(value.max())
    if not math.isfinite(value_min) or not math.isfinite(value_max):
        raise ValueError(f"{name} must be a finite, nonempty floating-point BCHW tensor.")
    if value_min < -0.5 or value_max > 1.5:
        raise ValueError(
            f"Required input range check failed for {name}: expected values near [0,1], "
            f"got min={value_min:.4f}, max={value_max:.4f}."
        )


def _format_history_result(result) -> Dict[str, Any]:
    reconstruction, trajectory, metrics = result
    reconstruction = reconstruction.clamp(0.0, 1.0)
    history = list(trajectory)
    history[-1] = reconstruction.detach().cpu().clone()
    return {
        "reconstruction": reconstruction,
        "history": history,
        "history_metrics": metrics,
    }


# Choose the data term for the batch's degradation mode.
def irae_direct(
    *,
    observation: Tensor,
    ae_model: nn.Module,
    degradation: Optional[Dict[str, Any]] = None,
    x_init: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    lambda_: float = 0.1,
    is_lambda_update: bool = True,
    lambda_multiplier: float = 0.8,
    lambda_check_interval: int = 10,
    max_iteration: int = 100,
    backtracking_rule: str = "sufficient_decrease",
    sufficient_decrease_constant: float = 1e-4,
    DEBUG: bool = False,
    return_history: bool = False,
):
    device = _default_device(device)
    if x_init is observation:
        x_init = None
    observation = observation.to(device)
    _validate_solver_input_range(observation, "observation")
    if x_init is not None:
        x_init = x_init.to(device)
        _validate_solver_input_range(x_init, "x_init")

    degradation = dict(degradation or {})
    mode = _normalize_mode(degradation.get("mode", "sparse"))
    common = {
        "observation": observation,
        "ae_model": ae_model,
        "x_init": x_init,
        "device": device,
        "lambda_": lambda_,
        "is_lambda_update": bool(is_lambda_update),
        "lambda_multiplier": float(lambda_multiplier),
        "lambda_check_interval": integer_index(lambda_check_interval),
        "max_iteration": integer_index(max_iteration),
        "DEBUG": bool(DEBUG),
        "return_history": bool(return_history),
    }

    if mode == "sparse":
        result = irae_sparse_l1_pgd(
            backtracking_rule=backtracking_rule,
            sufficient_decrease_constant=sufficient_decrease_constant,
            **common,
        )
    elif mode == "inpainting":
        result = irae_inpainting_l2_gd(
            mask=_require_tensor(degradation.get("mask"), "mask", device),
            **common,
        )
    elif mode == "deblur":
        result = irae_deblurring_l2_gd(
            kernel=_require_tensor(degradation.get("kernel"), "kernel", device),
            padding_mode=str(degradation.get("padding_mode", "reflect")),
            **common,
        )
    else:
        raise ValueError(f"Unsupported degradation mode: {mode}")

    return _format_history_result(result) if return_history else torch.clamp(result, 0.0, 1.0)


# JInit solves three separate finite-penalty problems and warm-starts each next AE.
def irae_jinit(
    *,
    observation: Tensor,
    ae_models: Sequence[nn.Module],
    degradation: Optional[Dict[str, Any]] = None,
    x_init: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    lambda_: float = 0.1,
    lambda_init_gamma: float = 1.0,
    stage_iterations: Optional[Sequence[int]] = None,
    is_lambda_update: bool = True,
    lambda_multiplier: float = 0.8,
    lambda_check_interval: int = 10,
    max_iteration: int = 100,
    backtracking_rule: str = "sufficient_decrease",
    sufficient_decrease_constant: float = 1e-4,
    DEBUG: bool = False,
    return_history: bool = False,
) -> Dict[str, Any]:
    models = list(ae_models)
    if len(models) != 3:
        raise ValueError(f"solver='jinit' requires exactly 3 AE models, got {len(models)}.")
    iterations = list(stage_iterations) if stage_iterations is not None else [max_iteration] * 3
    if len(iterations) != 3:
        raise ValueError(f"solver='jinit' requires exactly 3 stage iterations, got {len(iterations)}.")
    iterations = [integer_index(value) for value in iterations]
    if any(value < 0 for value in iterations):
        raise ValueError("stage iterations must be non-negative.")
    lambda_init_gamma = float(lambda_init_gamma)
    if not math.isfinite(lambda_init_gamma) or lambda_init_gamma <= 0:
        raise ValueError("lambda_init_gamma must be finite and positive.")
    stage_values = [lambda_init_gamma * float(lambda_)] * 2 + [float(lambda_)]

    current = observation if x_init is None else x_init
    stage_outputs = []
    for stage, (model, stage_lambda, stage_iters) in enumerate(zip(models, stage_values, iterations)):
        result = irae_direct(
            observation=observation,
            ae_model=model,
            degradation=degradation,
            x_init=current,
            device=device,
            lambda_=float(stage_lambda),
            is_lambda_update=is_lambda_update,
            lambda_multiplier=lambda_multiplier,
            lambda_check_interval=lambda_check_interval,
            max_iteration=int(stage_iters),
            backtracking_rule=backtracking_rule,
            sufficient_decrease_constant=sufficient_decrease_constant,
            DEBUG=DEBUG,
            return_history=return_history,
        )
        current = result["reconstruction"] if return_history else result
        stage_output = {
            "stage": stage,
            "lambda": float(stage_lambda),
            "max_iteration": int(stage_iters),
            "reconstruction": current,
        }
        if return_history:
            stage_output.update(history=result["history"], history_metrics=result["history_metrics"])
        stage_outputs.append(stage_output)
    return {"reconstruction": current, "stage_outputs": stage_outputs}


def irae_solver(
    *,
    solver: str,
    observation: Tensor,
    degradation: Optional[Dict[str, Any]] = None,
    x_init: Optional[Tensor] = None,
    ae_model: Optional[nn.Module] = None,
    ae_models: Optional[Sequence[nn.Module]] = None,
    **kwargs,
):
    """Run recovery with one autoencoder or a three-stage sequence."""
    solver = str(solver).strip().lower()
    if solver == "direct":
        if ae_model is None:
            raise ValueError("solver='direct' requires ae_model.")
        return irae_direct(
            observation=observation,
            ae_model=ae_model,
            degradation=degradation,
            x_init=x_init,
            **kwargs,
        )
    if solver == "jinit":
        if ae_models is None:
            raise ValueError("solver='jinit' requires ae_models.")
        return irae_jinit(
            observation=observation,
            ae_models=ae_models,
            degradation=degradation,
            x_init=x_init,
            **kwargs,
        )
    raise ValueError(f"Unsupported active IRAE solver: {solver}. Expected 'direct' or 'jinit'.")
