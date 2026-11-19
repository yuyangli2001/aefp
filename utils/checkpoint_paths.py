"""Checkpoint paths for this experiment."""

from pathlib import Path
from typing import Any, Dict, List, Sequence
from .configuration import normalize_model_cfg, normalize_training_cfg


# Definitions retain the original numerical behavior and setting comments.


def format_tag_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(format_tag_value(item) for item in value)
    return str(value).replace(".", "p").replace("-", "m")


def build_model_tag(model_cfg: Dict[str, Any], training_cfg: Dict[str, Any]) -> str:
    model_cfg = normalize_model_cfg(model_cfg)
    training_cfg = normalize_training_cfg(training_cfg)
    params = model_cfg["params"]
    variant = model_cfg["variant"]

    parts = [variant]
    if variant == "conv":
        parts.extend(
            [
                f"dsi{format_tag_value(params['downsample_indices'])}",
                f"ch{format_tag_value(params['encoder_channels'])}",
                f"bc{format_tag_value(params['bottleneck_channels'])}",
                f"norm{params['norm_type']}",
                f"sn{int(bool(params['use_sn']))}",
                f"up{params['upsample_mode']}",
                f"bm{params['bottleneck_mode']}",
            ]
        )
    elif variant == "resconv":
        parts.extend(
            [
                f"sch{format_tag_value(params['stage_channels'])}",
                f"bps{format_tag_value(params['blocks_per_stage'])}",
                f"bc{format_tag_value(params['bottleneck_channels'])}",
                f"norm{params['norm_type']}",
                f"up{params['upsample_mode']}",
            ]
        )

    # Nondefault architecture choices must not reuse a compatible but different model directory.
    if params["norm_type"] == "gn" and int(params.get("gn_groups", 8)) != 8:
        parts.append(f"gn{params['gn_groups']}")
    if variant == "conv" and params["bottleneck_mode"] == "linear":
        parts.append(f"ld{params.get('latent_dim', 128)}")

    # Historical directory tokens remain stable after renaming the training configuration.
    original, auxiliary = training_cfg["original"], training_cfg["auxiliary"]
    parts.extend([
        "pjinit", f"dep{format_tag_value(original['epochs'])}", "dl10p0",
        f"jsw{format_tag_value(auxiliary['jacobian_weight_init'])}",
        f"jep{format_tag_value(auxiliary['epochs_per_candidate'])}",
        f"jgf{format_tag_value(auxiliary['jacobian_weight_multiplier'])}",
        f"jmc{format_tag_value(auxiliary['max_candidates'])}", "jl10p0",
        f"lr{format_tag_value(training_cfg['learning_rate'])}",
        f"bs{format_tag_value(training_cfg['train_batch_size'])}",
    ])
    if training_cfg.get("train_subset_size") is not None:
        parts.append(f"ts{format_tag_value(training_cfg['train_subset_size'])}")
    return "_".join(parts)


def direct_ckpt_path(ckpt_dir: Path) -> Path:
    return Path(ckpt_dir) / "direct" / "stage_1.pth"


def jinit_cache_dir(ckpt_dir: Path) -> Path:
    return Path(ckpt_dir) / "jinit" / "cache"


def jinit_cache_ckpt_path(ckpt_dir: Path, stage_idx: int) -> Path:
    return jinit_cache_dir(ckpt_dir) / f"stage_{int(stage_idx)}.pth"


def jinit_cache_metrics_path(ckpt_dir: Path) -> Path:
    return jinit_cache_dir(ckpt_dir) / "metrics.csv"


def jinit_cache_meta_path(ckpt_dir: Path) -> Path:
    return jinit_cache_dir(ckpt_dir) / "meta.json"


def jinit_selected_dir(ckpt_dir: Path, psnr_drops: Sequence[float]) -> Path:
    return Path(ckpt_dir) / "jinit" / f"psnr_drops_{format_tag_value(list(psnr_drops))}"


def jinit_selected_ckpt_path(ckpt_dir: Path, psnr_drops: Sequence[float], stage_idx: int = 1) -> Path:
    return jinit_selected_dir(ckpt_dir, psnr_drops) / f"stage_{int(stage_idx)}.pth"


def jinit_summary_path(ckpt_dir: Path, psnr_drops: Sequence[float]) -> Path:
    return jinit_selected_dir(ckpt_dir, psnr_drops) / "summary.json"
