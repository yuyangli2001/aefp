from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import MethodCapabilities, RecoveryAdapter
from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession
from models.loading import load_ae_checkpoint
from recovery.checkpoints import fixed_ae_paths


class AEAdapter(RecoveryAdapter):
    capabilities = MethodCapabilities(frozenset({"mnist", "cifar10", "celeba-hq-64"}), frozenset({"sparse", "inpainting", "deblur"}))

    def recover(self, batch: RecoveryBatch, config: Mapping[str, Any], session: RecoverySession, device: str) -> torch.Tensor:
        model = session.get_or_load(("ae", config["model_id"], device), lambda: _load_models(config, device, "feedforward")[0])
        with torch.inference_mode():
            return model.get_recon(batch.observation.to(device).float())


def _model_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    architecture = config["checkpoint"]["architecture"]
    dataset = config["dataset"]
    return {"variant": "resconv", "params": {
        "in_channels": 1 if dataset == "mnist" else 3,
        "input_size": 28 if dataset == "mnist" else (32 if dataset == "cifar10" else 64),
        "stage_channels": list(architecture["stage_channels"]),
        "blocks_per_stage": [1] + [2] * (len(architecture["stage_channels"]) - 1),
        "bottleneck_channels": int(architecture["bottleneck_channels"]),
        "norm_type": architecture["norm_type"], "gn_groups": 8,
        "upsample_mode": architecture["upsample_mode"],
    }}


def _load_models(config: Mapping[str, Any], device: str, solver: str):
    model_cfg = _model_cfg(config)
    return [load_ae_checkpoint(model_cfg, path, device)
            for path in fixed_ae_paths(config["checkpoint"], solver=solver)]
