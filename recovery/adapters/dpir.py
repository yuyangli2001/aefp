from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import MethodCapabilities, RecoveryAdapter
from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession
from recovery.checkpoints import require_checkpoint


class DPIRAdapter(RecoveryAdapter):
    capabilities = MethodCapabilities(frozenset({"mnist", "cifar10", "celeba-hq-64"}), frozenset({"sparse"}))

    def recover(self, batch: RecoveryBatch, config: Mapping[str, Any], session: RecoverySession, device: str) -> torch.Tensor:
        observation = batch.observation.to(device).float()
        model = session.get_or_load(("dpir", config["model_id"], device), lambda: self._load(config, device))
        from baselines.dpir_03.dpir_desparse import _run_dpir_iterations

        with torch.inference_mode():
            return _run_dpir_iterations(model, observation, float(config["lambda"]), float(config["mu0"]),
                                        float(config["mu_rho"]), int(config["iter_num"]), bool(config["x8"]))

    @staticmethod
    def _load(config: Mapping[str, Any], device: str):
        from baselines.dpir_03.dpir_desparse import _load_frozen_prior, get_dataset_spec

        return _load_frozen_prior(get_dataset_spec(config["dataset"]), require_checkpoint(config["checkpoint"]), device)
