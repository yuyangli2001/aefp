from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from utils.utils_dataloader import get_degradation_dataloader

from .registry import get_adapter, resolve_config
from .schemas import RecoveryBatch, RecoveryResult, normalize_output
from .session import RecoverySession


def recover_dataset(
    dataset: str,
    degradation: Mapping[str, Any],
    method: str,
    config: Mapping[str, Any] | None = None,
    *,
    split: str = "test",
    indices: Sequence[int] | None = None,
    num_samples: int | None = None,
    loader_batch_size: int | None = None,
    num_workers: int = 0,
    device: str | None = None,
    root: str | Path | None = None,
) -> RecoveryResult:
    """Recover dataset samples in the supplied order, with fixed degradation and batch seeds."""
    dataset_name, method_name, degradation_cfg, resolved = resolve_config(dataset, degradation, method, config)
    adapter = get_adapter(method_name)
    caps = adapter.capabilities
    if dataset_name not in caps.datasets or degradation_cfg["mode"] not in caps.degradations:
        raise ValueError(f"Adapter does not support {dataset_name}/{degradation_cfg['mode']}.")

    loader_dataset, image_size = ("celeba-hq-256", 64) if dataset_name == "celeba-hq-64" else (dataset_name, None)
    batch_size = resolved["execution_batch_size"] if loader_batch_size is None else loader_batch_size
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_degradation_dataloader(
        dataset=loader_dataset, split=split, degradation_cfg=degradation_cfg,
        batch_size=batch_size, num_workers=num_workers, root=root, image_size=image_size,
        sample_indices=indices, num_samples=num_samples, shuffle=False,
        generator=torch.Generator().manual_seed(degradation_cfg["seed"]),
    )

    reconstructions, observations, index_batches, sample_ids, labels = [], [], [], [], []
    # Use the batch number for stochastic methods and keep references outside the adapter.
    with RecoverySession() as session:
        for batch_number, raw in enumerate(loader):
            batch_indices = raw["index"]
            ids = [f"{dataset_name}:{split}:{index:06d}" for index in batch_indices.tolist()]
            observation = raw["degraded"]
            batch = RecoveryBatch(observation, raw["params"], ids, batch_indices,
                                  metadata={"batch_index": batch_number})
            with torch.set_grad_enabled(caps.requires_autograd):
                output = adapter.recover(batch, resolved, session, device)
            reconstructions.append(normalize_output(output, len(observation), observation.shape[1:]))
            observations.append(observation.detach().cpu().float())
            index_batches.append(batch_indices)
            sample_ids.extend(ids)
            labels.extend(_batch_labels(raw["label"], len(ids)))

    return RecoveryResult(
        reconstruction=torch.cat(reconstructions) if reconstructions else None,
        observation=torch.cat(observations) if observations else None,
        sample_ids=sample_ids, indices=torch.cat(index_batches) if index_batches else torch.empty(0, dtype=torch.long),
        labels=labels, method=method_name, resolved_config=resolved,
        num_samples=len(sample_ids), metadata={"split": split, "batch_size": int(batch_size),
                                             "degradation_seed": degradation_cfg["seed"]},
    )


def _batch_labels(labels: Any, count: int) -> list:
    if labels is None:
        return [None] * count
    if torch.is_tensor(labels):
        return [label.item() if label.numel() == 1 else label for label in labels.cpu()]
    return list(labels)
