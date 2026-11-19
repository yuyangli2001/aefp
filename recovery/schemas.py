from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor


# Recursively freeze the resolved settings so adapters cannot mutate them.
def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({str(k): freeze(v) for k, v in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(v) for v in item)
        return item

    return freeze(value)


@dataclass(frozen=True)
class RecoveryBatch:
    """Observations and operator parameters for one recovery batch."""

    observation: Tensor
    degradation_params: Mapping[str, Any]
    sample_ids: Sequence[str]
    indices: Tensor
    labels: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    reconstruction: Optional[Tensor]
    observation: Optional[Tensor]
    sample_ids: list[str]
    indices: Tensor
    labels: Optional[Any]
    method: str
    resolved_config: Mapping[str, Any]
    num_samples: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def normalize_output(value: Tensor, expected_batch: int, expected_shape: Sequence[int] | None = None) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Recovery output must be a Tensor, got {type(value)}")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] != expected_batch:
        raise ValueError(f"Expected BCHW output with batch {expected_batch}, got {tuple(value.shape)}")
    if expected_shape is not None and tuple(value.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"Recovery output shape {tuple(value.shape[1:])} differs from observation {tuple(expected_shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError("Recovery output contains non-finite values.")
    # Out-of-place clipping also accepts inference tensors and preserves input aliases.
    return value.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
