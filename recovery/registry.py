from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from utils.degradation import DegradationConfig, to_loader_degradation_cfg

from .schemas import freeze_mapping

PRESET_PATH = Path(__file__).resolve().parent / "presets" / "paper.yaml"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "presets" / "checkpoints.yaml"

ALIASES = {"aefp": "irae", "robust-ae": "irae", "rae": "irae",
           "fire-s": "fire-single", "fire_single": "fire-single", "gd_drunet": "fire-single",
           "fire-m": "fire-multi", "fire_multi": "fire-multi", "multi": "fire-multi"}
DATASET_ALIASES = {"celeba": "celeba-hq-64", "celeba64": "celeba-hq-64", "celeba-hq": "celeba-hq-64"}
ADAPTERS = {"ae": "recovery.adapters.ae:AEAdapter",
            "irae": "recovery.adapters.irae:IRAEAdapter",
            "rpca": "recovery.adapters.rpca:RPCAAdapter",
            "daep": "recovery.adapters.daep:DAEPAdapter",
            "dpir": "recovery.adapters.dpir:DPIRAdapter",
            "dps": "recovery.adapters.dps:DPSAdapter",
            "fire-single": "recovery.adapters.fire:FireSingleAdapter",
            "fire-multi": "recovery.adapters.fire:FireMultiAdapter"}
BASELINE_METHODS = ("rpca", "daep", "dpir", "dps", "fire-single", "fire-multi")


def canonical_method(method: str) -> str:
    name = str(method).strip().lower().replace(" ", "-")
    name = ALIASES.get(name, name)
    if name not in ADAPTERS:
        raise ValueError(f"Unknown method {method!r}; available: {sorted(ADAPTERS)}")
    return name


def canonical_dataset(dataset: str) -> str:
    name = str(dataset).strip().lower().replace("_", "-")
    return DATASET_ALIASES.get(name, name)


# Full signatures avoid silently reusing parameters at a nearby corruption level.
def canonical_degradation(degradation: DegradationConfig) -> tuple[dict[str, Any], str]:
    cfg = to_loader_degradation_cfg(degradation)
    mode = cfg["mode"]
    if mode == "sparse":
        signature = f"salt_pepper_r{_tag(cfg['ratio'])}_{'per_channel' if cfg['per_channel'] else 'shared'}_seed{cfg['seed']}"
        if cfg["data_min"] == cfg["data_max"]:
            raise ValueError("Sparse presets require a positive value range.")
        if (cfg["value_mode"], cfg["data_min"], cfg["data_max"]) != ("salt_pepper", 0.0, 1.0):
            signature += f"_values_{cfg['value_mode']}_{_tag(cfg['data_min'])}_{_tag(cfg['data_max'])}"
    elif mode == "inpainting":
        if "missing_mask" in cfg:
            raise ValueError("Custom missing_mask is not part of a recovery preset.")
        signature = f"{cfg['mask_mode']}_mr{_tag(cfg['mask_ratio'])}_{'per_channel' if cfg['per_channel'] else 'shared'}_seed{cfg['seed']}"
    elif mode == "deblur":
        if "kernel" in cfg:
            raise ValueError("Custom kernel is not part of a recovery preset.")
        signature = f"gaussian_k{cfg['kernel_size']}_sigma{_tag(cfg['sigma'])}_{cfg['padding_mode']}_noise{_tag(cfg['obs_noise_std'])}_seed{cfg['seed']}"
    else:
        signature = "clean"
    return cfg, signature


def _tag(value: float) -> str:
    value = float(value)
    text = str(int(value)) if math.isfinite(value) and value.is_integer() else repr(value)
    return text.replace(".", "p").replace("-", "m")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_config(dataset: str, degradation: Mapping[str, Any] | Any, method: str,
                   override: Mapping[str, Any] | None = None) -> tuple[str, str, dict[str, Any], Mapping[str, Any]]:
    dataset_name, method_name = canonical_dataset(dataset), canonical_method(method)
    degradation_cfg, signature = canonical_degradation(degradation)
    data = _load_yaml(PRESET_PATH)
    key = f"{method_name}/{dataset_name}/{signature}"
    preset = data.get("presets", {}).get(key)
    if preset is None:
        raise KeyError(f"No exact frozen preset for {key}. Test-time nearest-level fallback is disabled.")
    resolved = copy.deepcopy(data.get("method_defaults", {}).get(method_name, {}))
    resolved.update(copy.deepcopy(preset))
    overrides = copy.deepcopy(dict(override or {}))
    fixed = set(overrides) & {"batch_semantics", "execution_batch_size", "output_stage"}
    if fixed:
        raise TypeError(f"Preset execution metadata cannot be overridden: {sorted(fixed)}")
    unknown = set(overrides) - set(resolved)
    if unknown:
        raise TypeError(f"Unknown config override fields for {key}: {sorted(unknown)}")
    resolved.update(overrides)
    resolved.update({"preset_version": data["version"], "preset_key": key, "dataset": dataset_name,
                     "method": method_name, "degradation": degradation_cfg, "degradation_signature": signature,
                     "config_overrides": overrides, "paper_aligned": not bool(overrides)})
    model_id = resolved.get("model_id")
    if model_id:
        checkpoints = _load_yaml(CHECKPOINT_PATH).get("models", {})
        if model_id not in checkpoints:
            raise KeyError(f"Unknown model_id {model_id!r} in preset {key}")
        resolved["checkpoint"] = checkpoints[model_id]
    model_ids = resolved.get("model_ids", ())
    if model_ids:
        checkpoints = _load_yaml(CHECKPOINT_PATH).get("models", {})
        resolved["checkpoints"] = [checkpoints[item] for item in model_ids]
    return dataset_name, method_name, degradation_cfg, freeze_mapping(resolved)


def get_adapter(method: str):
    module_name, class_name = ADAPTERS[canonical_method(method)].split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)()
