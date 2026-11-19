from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import MethodCapabilities, RecoveryAdapter
from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession
from recovery.checkpoints import require_checkpoint


class DPSAdapter(RecoveryAdapter):
    capabilities = MethodCapabilities(frozenset({"mnist", "cifar10", "celeba-hq-64"}),
                                      frozenset({"sparse"}),
                                      batch_semantics="gradient_coupled", requires_autograd=True)

    def recover(self, batch: RecoveryBatch, config: Mapping[str, Any], session: RecoverySession, device: str) -> torch.Tensor:
        runner = _runner()
        components = session.get_or_load(("dps", config["model_id"], device),
                                         lambda: runner.load_dps_components(config["dataset"], require_checkpoint(config["checkpoint"]), torch.device(device)))
        model, sampler, spec = components
        seed = int(config["seed"]) + int(batch.metadata.get("batch_index", 0))
        return runner.recover_dps_batch(model, sampler, spec, batch.observation,
                                        config["degradation"]["ratio"], float(config["scale"]), seed)


def _runner():
    from baselines.dps_04 import run_dps_sparse
    return run_dps_sparse
