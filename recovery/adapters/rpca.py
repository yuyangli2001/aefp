from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .base import MethodCapabilities, RecoveryAdapter
from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession


class RPCAAdapter(RecoveryAdapter):
    capabilities = MethodCapabilities(frozenset({"mnist", "cifar10", "celeba-hq-64"}), frozenset({"sparse"}))

    def recover(self, batch: RecoveryBatch, config: Mapping[str, Any], session: RecoverySession, device: str) -> torch.Tensor:
        from baselines.pcp_01.model import _rpca_single_image

        restored = []
        for image in batch.observation.detach().cpu():
            array = image.squeeze(0).numpy() if image.shape[0] == 1 else image.permute(1, 2, 0).numpy()
            low_rank, _ = _rpca_single_image(array, lamb_mult=float(config["lambda_mult"]))
            low_rank = np.clip(low_rank, 0.0, 1.0)
            tensor = torch.from_numpy(low_rank).float()
            restored.append(tensor.unsqueeze(0) if tensor.ndim == 2 else tensor.permute(2, 0, 1))
        return torch.stack(restored)
