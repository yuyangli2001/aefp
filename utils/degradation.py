"""Normalize degradation settings for loaders and recovery presets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from operator import index as integer_index
from typing import Any, Literal, Optional, Union

from torch import Tensor

SUPPORTED_DEGRADATION_MODES = {"clean", "sparse", "inpainting", "deblur"}
SUPPORTED_INPAINTING_MASK_MODES = {"bernoulli", "random", "center", "center_box"}


def normalize_degradation_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    return "deblur" if mode == "deblurring" else mode


@dataclass
class BaseDegradationConfig:
    """Describe a degradation; validation/test use deterministic per-sample random draws."""
    mode: str
    seed: int = 42
    deterministic: bool = False


@dataclass
class CleanConfig(BaseDegradationConfig):
    """Return clean images unchanged."""
    mode: Literal["clean"] = "clean"


@dataclass
class SparseConfig(BaseDegradationConfig):
    """Scalar-channel or shared-pixel sparse replacement/impulse settings."""
    mode: Literal["sparse"] = "sparse"
    ratio: float = 0.1
    value_mode: str = "salt_pepper"
    data_min: float = 0.0
    data_max: float = 1.0
    per_channel: bool = True


@dataclass
class InpaintingConfig(BaseDegradationConfig):
    """Missing masks use 1 for missing and 0 for observed entries."""
    mode: Literal["inpainting"] = "inpainting"
    mask_ratio: float = 0.5
    mask_mode: str = "bernoulli"   # bernoulli/random pixel mask or center/center_box block mask
    missing_mask: Optional[Tensor] = None
    per_channel: bool = False


@dataclass
class DeblurConfig(BaseDegradationConfig):
    """A fixed normalized kernel with optional additive observation noise."""
    mode: Literal["deblur"] = "deblur"
    kernel: Optional[Tensor] = None
    kernel_size: int = 17
    sigma: float = 2.0
    padding_mode: str = "reflect"
    obs_noise_std: float = 0.03


DegradationConfig = Union[BaseDegradationConfig, Mapping[str, Any]]


def build_degradation_config(cfg: DegradationConfig) -> BaseDegradationConfig:
    """Normalize and validate a configuration without generating observations."""
    if isinstance(cfg, BaseDegradationConfig):
        out_cfg: BaseDegradationConfig = cfg
    elif isinstance(cfg, Mapping):
        outer = dict(cfg)
        raw_cfg = dict(outer.pop("params", None) or {}) | outer
        mode = normalize_degradation_mode(raw_cfg.pop("mode", "clean"))
        aliases = {
            "sparse": {"noise_ratio": "ratio"},
            "inpainting": {"mask_per_channel": "per_channel", "mask_seed": "seed"},
            "deblur": {"blur_kernel_size": "kernel_size", "blur_sigma": "sigma", "blur_padding_mode": "padding_mode"},
        }
        for alias, name in aliases.get(mode, {}).items():
            if alias in raw_cfg:
                raw_cfg.setdefault(name, raw_cfg.pop(alias))
        if mode == "inpainting":
            if "mask" in raw_cfg and "missing_mask" in raw_cfg:
                raise ValueError("Use only one of `mask` (legacy) or `missing_mask`.")
            if "mask" in raw_cfg:
                raw_cfg["missing_mask"] = raw_cfg.pop("mask")
        raw_cfg["mode"] = mode
        mode_to_cls = {
            "clean": CleanConfig,
            "sparse": SparseConfig,
            "inpainting": InpaintingConfig,
            "deblur": DeblurConfig,
        }
        if mode not in mode_to_cls:
            raise ValueError(f"Unsupported degradation mode: {mode}")
        cls = mode_to_cls[mode]
        try:
            out_cfg = cls(**raw_cfg)
        except TypeError as exc:
            raise TypeError(f"Invalid fields for `{cls.__name__}`: {exc}") from exc
    else:
        raise TypeError(f"`cfg` must be dict or BaseDegradationConfig, got {type(cfg)}")

    mode = normalize_degradation_mode(out_cfg.mode)
    if mode not in SUPPORTED_DEGRADATION_MODES:
        raise ValueError(f"Unsupported degradation mode: {mode}")
    if mode != out_cfg.mode:
        out_cfg = replace(out_cfg, mode=mode)

    if not isinstance(out_cfg.seed, int):
        raise TypeError(f"`seed` must be int, got {type(out_cfg.seed)}")
    if not isinstance(out_cfg.deterministic, bool):
        raise TypeError(f"`deterministic` must be bool, got {type(out_cfg.deterministic)}")

    if isinstance(out_cfg, SparseConfig):
        out_cfg = replace(out_cfg, ratio=float(out_cfg.ratio), value_mode=str(out_cfg.value_mode).strip().lower(),
                          data_min=float(out_cfg.data_min), data_max=float(out_cfg.data_max),
                          per_channel=bool(out_cfg.per_channel))
        if not (0.0 <= out_cfg.ratio <= 1.0):
            raise ValueError(f"`sparse.ratio` must be in [0,1], got {out_cfg.ratio}")
        if not all(map(isfinite, (out_cfg.data_min, out_cfg.data_max))) or out_cfg.data_min > out_cfg.data_max:
            raise ValueError(
                f"`sparse.data_min` must be <= `sparse.data_max`, got "
                f"{out_cfg.data_min} > {out_cfg.data_max}"
            )
        if out_cfg.value_mode not in {"salt_pepper", "salt-and-pepper", "saltpepper", "random_extreme", "extreme", "signed_impulse", "impulse"}:
            raise ValueError(f"Unsupported sparse value_mode: {out_cfg.value_mode}")
    elif isinstance(out_cfg, InpaintingConfig):
        mask_mode = str(out_cfg.mask_mode).strip().lower()
        if mask_mode not in SUPPORTED_INPAINTING_MASK_MODES:
            raise ValueError(
                f"`inpainting.mask_mode` must be one of "
                f"{sorted(SUPPORTED_INPAINTING_MASK_MODES)}, got {out_cfg.mask_mode}"
            )
        out_cfg = replace(out_cfg, mask_mode=mask_mode, mask_ratio=float(out_cfg.mask_ratio),
                          per_channel=bool(out_cfg.per_channel))
        if out_cfg.missing_mask is None:
            if not (0.0 <= float(out_cfg.mask_ratio) <= 1.0):
                raise ValueError(
                    f"`inpainting.mask_ratio` must be in [0,1], got {out_cfg.mask_ratio}"
                )
    elif isinstance(out_cfg, DeblurConfig):
        if integer_index(out_cfg.kernel_size) <= 0:
            raise ValueError(f"`deblur.kernel_size` must be > 0, got {out_cfg.kernel_size}")
        if int(out_cfg.kernel_size) % 2 == 0:
            raise ValueError(
                f"`deblur.kernel_size` must be odd, got {out_cfg.kernel_size}"
            )
        if not isfinite(float(out_cfg.sigma)) or float(out_cfg.sigma) <= 0.0:
            raise ValueError(f"`deblur.sigma` must be > 0, got {out_cfg.sigma}")
        if not isfinite(float(out_cfg.obs_noise_std)) or float(out_cfg.obs_noise_std) < 0.0:
            raise ValueError(
                f"`deblur.obs_noise_std` must be >= 0, got {out_cfg.obs_noise_std}"
            )
        padding = str(out_cfg.padding_mode).strip().lower()
        padding = "constant" if padding == "zeros" else padding
        if padding not in {"reflect", "replicate", "circular", "constant"}:
            raise ValueError(f"Unsupported blur padding_mode: {out_cfg.padding_mode}")
        out_cfg = replace(out_cfg, padding_mode=padding, sigma=float(out_cfg.sigma),
                          obs_noise_std=float(out_cfg.obs_noise_std))

    return out_cfg


def to_loader_degradation_cfg(degradation_cfg: DegradationConfig | None) -> dict[str, Any]:
    config = build_degradation_config(degradation_cfg or {})
    return {key: value for key, value in vars(config).items() if value is not None and key != "deterministic"}
