"""Run AE recovery stages and yield paired batches for metrics."""

from __future__ import annotations

from operator import index as integer_index
from typing import Any, Dict

import torch

from methods.continuation import irae_solver
from models.loading import load_ae_checkpoint
from utils.configuration import normalize_degradation_mode
from utils.utils_init4solver import build_solver_init


def _iter_loader_batches(eval_loader, *, num_samples, device):
    seen = 0
    for raw in eval_loader:
        batch_size = len(raw["degraded"])
        take = batch_size if num_samples is None else min(batch_size, integer_index(num_samples) - seen)
        if take <= 0:
            break
        params = {}
        for key, value in raw["params"].items():
            if torch.is_tensor(value):
                if key != "kernel" and value.ndim and len(value) == batch_size:
                    value = value[:take]
                value = value.to(device)
            elif isinstance(value, (tuple, list)) and len(value) == batch_size:
                value = value[:take]
            params[key] = value
        yield {
            "clean": raw["clean"][:take].to(device),
            "degraded": raw["degraded"][:take].to(device),
            "index": raw["index"][:take],
            "mode": normalize_degradation_mode(raw["meta"]["mode"]),
            "params": params,
        }
        seen += take


def build_batch_degradation(
    mode: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert stored operator parameters to the solver convention."""
    mode = normalize_degradation_mode(mode)

    if mode == "sparse":
        return {"mode": "sparse"}

    if mode == "inpainting":
        if "missing_mask" not in params:
            raise ValueError("Inpainting requires params['missing_mask'].")
        return {
            "mode": "inpainting",
            "mask": 1.0 - params["missing_mask"].float(),
        }

    if mode == "deblur":
        if "kernel" not in params:
            raise ValueError("Deblur requires params['kernel'].")
        return {
            "mode": "deblur",
            "kernel": params["kernel"],
            "padding_mode": params.get("padding_mode", "reflect"),
        }

    raise ValueError(f"Unsupported degradation mode: {mode}")


def run_recovery_stages(*, solver, models, observation, degradation, x_init, **settings):
    if solver == "feedforward":
        with torch.no_grad():
            return [models[0].get_recon(observation)]
    solver_args = dict(observation=observation, degradation=degradation, x_init=x_init, **settings)
    if solver == "direct":
        return [irae_solver(solver=solver, ae_model=models[0], **solver_args)]
    result = irae_solver(solver=solver, ae_models=models, **solver_args)
    return [stage["reconstruction"] for stage in result["stage_outputs"]]


def iter_irae_recovery_batches(
    *, eval_loader, checkpoint_paths, model_cfg, solver, num_samples=None,
    init_method="observation", device=None, **settings,
):
    """Load the selected AEs once and yield CPU tensors for stage evaluation."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = [load_ae_checkpoint(model_cfg, path, device) for path in checkpoint_paths]
    for batch in _iter_loader_batches(eval_loader, num_samples=num_samples, device=device):
        observation = batch["degraded"]
        initial = observation if solver == "feedforward" else build_solver_init(
            batch=batch, init_method=init_method,
        )
        stages = run_recovery_stages(
            solver=solver, models=models, observation=observation, x_init=initial,
            degradation=build_batch_degradation(batch["mode"], batch["params"]),
            device=device, **settings,
        )
        stages = [stage.detach().cpu() for stage in stages]
        yield {
            "clean": batch["clean"].detach().cpu(), "degraded": observation.detach().cpu(),
            "init": initial.detach().cpu(), "recovered": stages[-1], "stage_recovered": stages,
            "index": batch["index"],
        }
