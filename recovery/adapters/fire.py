from __future__ import annotations

import json
from typing import Any, Mapping

import torch

from .base import MethodCapabilities, RecoveryAdapter
from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession
from recovery.checkpoints import require_checkpoint


class _FireAdapter(RecoveryAdapter):
    method = ""
    capabilities = MethodCapabilities(frozenset({"mnist", "cifar10", "celeba-hq-64"}),
                                      frozenset({"sparse"}), batch_semantics="rng_grouped")

    def recover(self, batch: RecoveryBatch, config: Mapping[str, Any], session: RecoverySession, device: str) -> torch.Tensor:
        runner = _runner()
        paths = _checkpoint_paths(config)
        seed = int(config.get("seed", 42)) + int(batch.metadata.get("batch_index", 0))
        # Solver settings are part of the cached object, so distinct presets need distinct entries.
        settings = {key: config[key] for key in ("lambda", "gamma", "sigma_noise_max", "max_iter",
                                                "avg_gdsteps", "equivariant", "average_last")}
        settings["prior_weights"] = config.get("prior_weights", [0.8, 0.2] if self.method == "fire-multi" else [1.0])
        key = (self.method, config["dataset"], config["degradation"]["mode"], tuple(paths), device,
               json.dumps(settings, sort_keys=True))
        with runner.seed_scope(seed):
            model = session.get_or_load(key,
                lambda: runner.build_fire_model(config["dataset"], self.method, config, paths,
                                                torch.device(device), batch.observation[:1].to(device)))
            # Model construction also draws random numbers before loading the frozen weights.
            runner.set_seed(seed)
            return runner.recover_fire_batch(model, batch.observation, torch.device(device))


class FireSingleAdapter(_FireAdapter):
    method = "fire-single"


class FireMultiAdapter(_FireAdapter):
    method = "fire-multi"


def _runner():
    from baselines.fire_05 import run_fire_l1_l2
    return run_fire_l1_l2


def _checkpoint_paths(config: Mapping[str, Any]) -> list[str]:
    manifests = list(config.get("checkpoints", ())) or [config["checkpoint"]]
    return [require_checkpoint(item) for item in manifests]
