from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import MethodCapabilities, RecoveryAdapter
from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession
from recovery.checkpoints import require_checkpoint


class DAEPAdapter(RecoveryAdapter):
    capabilities = MethodCapabilities(frozenset({"mnist", "cifar10", "celeba-hq-64"}), frozenset({"sparse"}), requires_autograd=True)

    def recover(self, batch: RecoveryBatch, config: Mapping[str, Any], session: RecoverySession, device: str) -> torch.Tensor:
        from baselines.daep_02.daep import restore_daep_l1_adapted
        from baselines.daep_02.model import load_daep_dae

        checkpoint = require_checkpoint(config["checkpoint"])
        model = session.get_or_load(("daep", config["model_id"], device),
                                    lambda: load_daep_dae(config["dataset"], checkpoint, device))
        output = restore_daep_l1_adapted(
            batch.observation.to(device).float() * 255.0, model,
            prior_strength=float(config["prior_strength"]), alpha=float(config["alpha"]),
            mu=float(config["mu"]), num_iter=int(config["num_iter"]), seed=int(config["seed"]),
            sample_indices=batch.indices)
        return output / 255.0
