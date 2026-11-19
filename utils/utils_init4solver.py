from __future__ import annotations

from typing import Dict

import torch


# Keep clean and observation initialization without preprocessing.
def normalize_solver_init_method(init_method: str) -> str:
    """Normalize init aliases to the supported runtime modes."""
    key = str(init_method).strip().lower()
    alias = {
        "clean": "clean",
        "clean_init": "clean",
        "observation": "observation",
        "obs": "observation",
        "degraded": "observation",
        "noisy_init": "observation",
    }
    if key not in alias:
        raise ValueError(f"Unsupported init_method: {init_method}")
    return alias[key]


def build_solver_init(
    *,
    batch: Dict[str, torch.Tensor],
    init_method: str,
) -> torch.Tensor:
    """Build the initialization tensor required by the solver."""
    init_mode = normalize_solver_init_method(init_method)
    initial = batch["clean"] if init_mode == "clean" else batch["degraded"]
    if not torch.is_tensor(initial) or initial.ndim != 4 or initial.numel() == 0 or not initial.is_floating_point() or not torch.isfinite(initial).all():
        raise ValueError("Solver initialization requires finite, nonempty floating-point BCHW images.")
    return initial
