from __future__ import annotations

from typing import Any, Mapping

import torch

from .ae import _load_models
from .base import MethodCapabilities, RecoveryAdapter
from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession


class IRAEAdapter(RecoveryAdapter):
    capabilities = MethodCapabilities(frozenset({"mnist", "cifar10", "celeba-hq-64"}), frozenset({"sparse", "inpainting", "deblur"}), requires_autograd=True)

    def recover(self, batch: RecoveryBatch, config: Mapping[str, Any], session: RecoverySession, device: str) -> torch.Tensor:
        from recovery.batch import build_batch_degradation, run_recovery_stages

        models = session.get_or_load(("irae", config["model_id"], device), lambda: _load_models(config, device, "jinit"))
        observation = batch.observation.to(device).float()
        params = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.degradation_params.items()}
        degradation = build_batch_degradation(config["degradation"]["mode"], params)
        stages = run_recovery_stages(
            solver="jinit", models=models, observation=observation, degradation=degradation, x_init=observation,
            device=torch.device(device), lambda_=float(config["lambda"]),
            lambda_init_gamma=float(config["lambda_init_gamma"]), max_iteration=int(config["max_iteration"]),
            stage_iterations=list(config["jinit_stage_iters"]), is_lambda_update=bool(config["is_lambda_update"]),
            lambda_multiplier=float(config["lambda_multiplier"]), lambda_check_interval=int(config["lambda_check_interval"]),
            backtracking_rule=str(config["backtracking_rule"]),
            sufficient_decrease_constant=float(config["sufficient_decrease_constant"]))
        return stages[-1]
