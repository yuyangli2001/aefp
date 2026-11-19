from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor

from recovery.schemas import RecoveryBatch
from recovery.session import RecoverySession


@dataclass(frozen=True)
class MethodCapabilities:
    datasets: frozenset[str]
    degradations: frozenset[str]
    batch_semantics: str = "independent"
    requires_autograd: bool = False


class RecoveryAdapter(ABC):
    capabilities: MethodCapabilities

    @abstractmethod
    def recover(
        self,
        batch: RecoveryBatch,
        config: Mapping[str, Any],
        session: RecoverySession,
        device: str,
    ) -> Tensor:
        """Recover observations without access to clean images."""
