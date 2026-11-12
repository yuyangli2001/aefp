"""Thin experiment-facing dispatch for the unified IRAE numerical solvers."""

from __future__ import annotations

import math
from operator import index as integer_index
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from methods.irae_solver import (
    irae_deblurring_l2_gd,
    irae_inpainting_l2_gd,
    irae_sparse_l1_pgd,
)

Tensor = torch.Tensor


# Small input helpers keep experiment metadata out of the numerical core.
def _default_device(device: Optional[torch.device]) -> torch.device:
    return device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    if value.ndim != 4 or value.numel() == 0 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite, nonempty floating-point BCHW tensor.")
    value_min, value_max = float(value.min()), float(value.max())
    if value_min < -0.5 or value_max > 1.5:
        raise ValueError(
            f"Required input range check failed for {name}: expected values near [0,1], "
            f"got min={value_min:.4f}, max={value_max:.4f}."
        )


def _format_history_result(result) -> Dict[str, Any]:
    reconstruction = torch.clamp(result[0], 0.0, 1.0)
    history = list(result[1]) if len(result) > 1 and result[1] else result[1]
    if history:
        history[-1] = reconstruction.detach().cpu().clone()
    return {
        "reconstruction": reconstruction,
        "history": history,
        "history_metrics": result[2] if len(result) > 2 else None,
    }


def _validate_three(values: Optional[Sequence[Any]], name: str) -> Optional[list]:
    if values is None:
        return None
    values = list(values)
    if len(values) != 3:
        raise ValueError(f"solver='jinit' requires exactly 3 {name}, got {len(values)}.")
    return values


# Direct recovery dispatches one batch to the matching paper data discrepancy.
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
    observation = observation.to(device)
    x_init = observation if x_init is None else x_init.to(device)
    _validate_solver_input_range(observation, "observation")
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
    iterations = _validate_three(stage_iterations, "stage iterations") or [integer_index(max_iteration)] * 3
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
    """Run the active paper solver mode: ``direct`` or ``jinit``."""
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
