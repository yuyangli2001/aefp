"""Configuration for this experiment."""

import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.degradation import normalize_degradation_mode, to_loader_degradation_cfg


def to_dict(cfg_section) -> Dict[str, Any]:
    if cfg_section is None:
        return {}
    if isinstance(cfg_section, dict):
        return dict(cfg_section)
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg_section, resolve=True)


def deep_merge_dicts(*configs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    merged: Dict[str, Any] = {}
    for cfg in configs:
        if not cfg:
            continue
        for key, value in dict(cfg).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
    return merged


def default_num_workers(raw_value):
    if raw_value is not None:
        return int(raw_value)
    if os.name == "nt":
        return 0
    return 16


def parse_optional_int(raw_value) -> Optional[int]:
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "none", "null"}:
        return None
    return int(raw_value)


def normalize_split(split: str) -> str:
    split = str(split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}. Expected one of ['train', 'val', 'test'].")
    return split


def _default_model_params_for_variant(variant: str) -> Dict[str, Any]:
    if variant == "conv":
        return {
            "in_channels": 1,
            "input_size": 28,
            "downsample_indices": [1, 3, 5],
            "encoder_channels": [32, 32, 64, 64, 128, 128, 128, 128, 128],
            "bottleneck_channels": 16,
            "bottleneck_mode": "conv",
            "norm_type": "bn",
            "use_sn": False,
            "upsample_mode": "resize_conv",
        }
    if variant == "resconv":
        return {
            "in_channels": 1,
            "input_size": 28,
            "stage_channels": [32, 64, 128],
            "blocks_per_stage": [1, 2, 2],
            "bottleneck_channels": 16,
            "norm_type": "bn",
            "gn_groups": 8,
            "upsample_mode": "deconv",
        }
    raise ValueError(
        f"Unsupported model variant: {variant}. Only `conv` and `resconv` are supported."
    )


def normalize_model_cfg(model_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = {
        "family": "ae",
        "variant": "conv",
        "params": {},
    }
    if model_cfg:
        cfg["family"] = model_cfg.get("family", cfg["family"])
        cfg["variant"] = model_cfg.get("variant", cfg["variant"])
    variant = str(cfg["variant"]).strip().lower()
    params = _default_model_params_for_variant(variant)
    if model_cfg:
        params.update(dict(model_cfg.get("params", {})))
    cfg["variant"] = variant
    cfg["params"] = params
    params = cfg["params"]
    params["in_channels"] = int(params["in_channels"])
    params["bottleneck_channels"] = int(params["bottleneck_channels"])
    if params["bottleneck_channels"] <= 0:
        raise ValueError("model.params.bottleneck_channels must be a positive integer.")

    if variant == "conv":
        encoder_channels = [int(value) for value in params["encoder_channels"]]
        downsample_indices = [int(value) for value in params["downsample_indices"]]
        if not encoder_channels:
            raise ValueError("model.params.encoder_channels must be a non-empty list.")
        if any(value <= 0 for value in encoder_channels):
            raise ValueError("model.params.encoder_channels must contain positive integers.")
        if downsample_indices != sorted(downsample_indices) or len(downsample_indices) != len(set(downsample_indices)):
            raise ValueError("model.params.downsample_indices must be a sorted list of unique integers.")
        if any(idx < 0 or idx >= len(encoder_channels) for idx in downsample_indices):
            raise ValueError(
                "model.params.downsample_indices must stay within [0, len(encoder_channels) - 1]."
            )
        params["encoder_channels"] = encoder_channels
        params["downsample_indices"] = downsample_indices
        params["bottleneck_mode"] = str(params["bottleneck_mode"]).strip().lower()
        params["use_sn"] = bool(params["use_sn"])
    elif variant == "resconv":
        stage_channels = [int(value) for value in params["stage_channels"]]
        blocks_per_stage = [int(value) for value in params["blocks_per_stage"]]
        if not stage_channels:
            raise ValueError("model.params.stage_channels must be a non-empty list.")
        if any(value <= 0 for value in stage_channels):
            raise ValueError("model.params.stage_channels must contain positive integers.")
        if len(blocks_per_stage) != len(stage_channels):
            raise ValueError("model.params.blocks_per_stage must have the same length as stage_channels.")
        if any(value < 0 for value in blocks_per_stage):
            raise ValueError("model.params.blocks_per_stage must contain non-negative integers.")
        params["stage_channels"] = stage_channels
        params["blocks_per_stage"] = blocks_per_stage
        params["gn_groups"] = int(params["gn_groups"])

    params["norm_type"] = str(params["norm_type"]).strip().lower()
    params["upsample_mode"] = str(params["upsample_mode"]).strip().lower()
    return cfg


def normalize_jinit_psnr_drops(raw_value) -> List[float]:
    if raw_value is None:
        raise ValueError("jinit requires `train.auxiliary.psnr_drops`.")
    values = list(raw_value if isinstance(raw_value, (list, tuple)) else [raw_value])
    parsed = [float(value) for value in values]
    if len(parsed) != 2:
        raise ValueError(f"jinit requires exactly 2 `psnr_drops`, got {len(parsed)}.")
    if not all(math.isfinite(value) and value > 0.0 for value in parsed):
        raise ValueError(f"jinit requires positive `psnr_drops`, got {parsed}.")
    if not parsed[0] > parsed[1]:
        raise ValueError(
            f"jinit requires ordered `psnr_drops=[coarse, fine]` with coarse > fine, got {parsed}."
        )
    return parsed


# Validate the single AE training schema before creating checkpoints.
def normalize_training_cfg(training_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    defaults = {
        "learning_rate": 1e-4, "train_batch_size": 64, "train_subset_size": None,
        "num_workers": None, "force_retrain": False,
        "original": {"epochs": 200},
        "auxiliary": {"psnr_drops": [10.0, 5.0], "jacobian_weight_init": 1e-8,
                      "jacobian_weight_multiplier": 3.0, "epochs_per_candidate": 30,
                      "max_candidates": 14},
    }
    raw = dict(training_cfg or {})
    unknown = set(raw) - set(defaults)
    for section in ("original", "auxiliary"):
        unknown.update(f"{section}.{key}" for key in dict(raw.get(section, {}) or {}) if key not in defaults[section])
    if unknown:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    cfg = deep_merge_dicts(defaults, raw)
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["train_batch_size"] = int(cfg["train_batch_size"])
    cfg["train_subset_size"] = parse_optional_int(cfg["train_subset_size"])
    cfg["num_workers"] = parse_optional_int(cfg["num_workers"])
    cfg["original"]["epochs"] = int(cfg["original"]["epochs"])
    aux = cfg["auxiliary"]
    aux["psnr_drops"] = normalize_jinit_psnr_drops(aux["psnr_drops"])
    for key in ("jacobian_weight_init", "jacobian_weight_multiplier"):
        aux[key] = float(aux[key])
    for key in ("epochs_per_candidate", "max_candidates"):
        aux[key] = int(aux[key])
    if not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0 or cfg["train_batch_size"] <= 0 or cfg["original"]["epochs"] <= 0:
        raise ValueError("Training learning rate, batch size and epochs must be positive.")
    if cfg["train_subset_size"] is not None and cfg["train_subset_size"] <= 0 or cfg["num_workers"] is not None and cfg["num_workers"] < 0:
        raise ValueError("Training subset size must be positive and workers nonnegative.")
    if not math.isfinite(aux["jacobian_weight_init"]) or aux["jacobian_weight_init"] <= 0 or not math.isfinite(aux["jacobian_weight_multiplier"]) or aux["jacobian_weight_multiplier"] <= 1 or min(aux["epochs_per_candidate"], aux["max_candidates"]) <= 0:
        raise ValueError("Auxiliary training requires positive settings and a Jacobian weight multiplier > 1.")
    return cfg


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load one YAML file as a plain dictionary."""
    from omegaconf import OmegaConf

    return to_dict(OmegaConf.load(path))


def collect_prefixed_overrides(task_overrides, *prefixes: str) -> Dict[str, Any]:
    """Collect task overrides for selected prefixes."""
    from omegaconf import OmegaConf

    filtered = []
    for raw_override in task_overrides or []:
        override = raw_override.lstrip("+")
        if any(override.startswith(f"{prefix}.") for prefix in prefixes):
            filtered.append(override)

    if not filtered:
        return {}
    return to_dict(OmegaConf.from_dotlist(filtered))


def resolve_batch_size(
    explicit_value,
    dataset_cfg: Optional[Dict[str, Any]],
    dataset_key: str,
    fallback_value: int,
) -> int:
    if explicit_value is not None:
        return int(explicit_value)

    dataset_cfg = dict(dataset_cfg or {})
    dataset_value = dataset_cfg.get(dataset_key)
    if dataset_value is not None:
        return int(dataset_value)

    return int(fallback_value)


def resolve_experiment_config(
    cfg,
    *,
    project_root: Path,
    hydra_choices: Optional[Dict[str, Any]] = None,
    task_overrides=None,
) -> Dict[str, Any]:
    """Resolve dataset, model, train, and inference config for one run."""
    hydra_choices = dict(hydra_choices or {})
    user_overrides = collect_prefixed_overrides(task_overrides, "model", "train", "reconstruction", "inference")

    dataset_cfg_raw = to_dict(cfg.dataset)
    # `defaults` is a Hydra-reserved key and must remain a list. Keep
    # dataset-specific overrides under `dataset_defaults` instead.
    dataset_defaults = dict(
        dataset_cfg_raw.get("dataset_defaults", dataset_cfg_raw.get("defaults", {})) or {}
    )
    dataset_cfg = {
        key: value for key, value in dataset_cfg_raw.items() if key not in {"defaults", "dataset_defaults"}
    }
    dataset_source_name = str(dataset_cfg.get("source_name", dataset_cfg["name"]))

    model_choice = str(hydra_choices.get("model", "ae_3down"))
    inference_choice = str(hydra_choices.get("inference", cfg.inference.solver))
    base_model_cfg = load_yaml_config(project_root / "configs" / "model" / f"{model_choice}.yaml")
    base_inference_cfg = load_yaml_config(project_root / "configs" / "inference" / f"{inference_choice}.yaml")
    inference_cfg = deep_merge_dicts(base_inference_cfg, user_overrides.get("inference", {}))
    degradation_cfg = to_dict(cfg.degradation)
    if str(inference_cfg.get("solver", "")).strip().lower() == "jinit":
        legacy_jinit_keys = {"blend_init", "blend_alpha"} & set(inference_cfg)
        if legacy_jinit_keys:
            raise ValueError(
                f"Legacy jinit inference keys are not supported: {sorted(legacy_jinit_keys)}."
            )
    degradation_mode = normalize_degradation_mode(degradation_cfg.get("mode", "clean"))
    init_method = str(inference_cfg.get("init_method", "observation")).strip().lower()
    if degradation_mode in {"inpainting", "deblur"} and init_method != "observation":
        raise ValueError(
            f"degradation.mode={degradation_mode} currently supports only "
            f"`inference.init_method=observation`, got {init_method}."
        )

    # Recovery methods share one original/auxiliary training configuration.
    solver = str(inference_cfg["solver"]).strip().lower()
    if solver not in {"direct", "feedforward", "jinit"}:
        raise ValueError(f"Unsupported solver: {solver}")
    train_cfg = normalize_training_cfg(deep_merge_dicts(
        load_yaml_config(project_root / "configs/train/ae.yaml"),
        dict(dataset_defaults.get("train", {}) or {}), to_dict(cfg.train),
    ))

    model_cfg = normalize_model_cfg(
        deep_merge_dicts(
            base_model_cfg,
            dict(dataset_defaults.get("model", {}) or {}),
            user_overrides.get("model", {}),
        )
    )
    model_cfg["params"]["in_channels"] = dataset_cfg["channels"]
    model_cfg["params"]["input_size"] = dataset_cfg["size"]

    runtime_defaults = dict(dataset_defaults.get("runtime", {}) or {})
    reconstruction_cfg = dict(to_dict(cfg.reconstruction))
    user_reconstruction_cfg = dict(user_overrides.get("reconstruction", {}) or {})
    user_inference_cfg = dict(user_overrides.get("inference", {}) or {})

    reconstruction_cfg["batch_size"] = resolve_batch_size(
        user_reconstruction_cfg.get("batch_size"),
        runtime_defaults,
        "reconstruction_batch_size",
        fallback_value=reconstruction_cfg.get("batch_size", 64),
    )
    inference_cfg["batch_size"] = resolve_batch_size(
        user_inference_cfg.get("batch_size"),
        runtime_defaults,
        "inference_batch_size",
        fallback_value=inference_cfg.get("batch_size", 32),
    )

    data_num_workers = default_num_workers(train_cfg.get("num_workers"))
    train_cfg["num_workers"] = data_num_workers

    return {
        "dataset": dataset_cfg,
        "model": model_cfg,
        "train": train_cfg,
        "reconstruction": reconstruction_cfg,
        "inference": inference_cfg,
        "runtime": {
            "dataset_source_name": dataset_source_name,
            "num_workers": data_num_workers,
        },
    }
