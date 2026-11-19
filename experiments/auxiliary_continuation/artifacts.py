"""Atomic CSV/JSON artifacts and explicit visualization overwrite checks."""

from __future__ import annotations
import csv
import json
import os
import tempfile
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from .variants import FIELDNAMES


# Publish a complete text artifact only after its writer succeeds.
@contextmanager
def _atomic_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    with _atomic_text(path) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _require_writable(paths: Sequence[Path], force: bool) -> None:
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None and not force:
        raise FileExistsError(f"{existing} exists; pass --force to replace it.")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with _atomic_text(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDNAMES} for row in rows)
