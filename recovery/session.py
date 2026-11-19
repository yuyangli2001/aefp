from __future__ import annotations

from typing import Any, Callable, Hashable


class RecoverySession:
    """Small per-run cache so frozen models are loaded only once."""

    def __init__(self) -> None:
        self._cache: dict[Hashable, Any] = {}

    def get_or_load(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def close(self) -> None:
        self._cache.clear()

    def __enter__(self) -> "RecoverySession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
