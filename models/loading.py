"""Construct autoencoders and load their saved state without training."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from models.conv_autoencoder import PureConvAutoencoder, ResConvAE
from utils.configuration import normalize_model_cfg


def build_ae_model(model_cfg: Optional[Dict[str, Any]]):
    cfg = normalize_model_cfg(model_cfg)
    variant = cfg["variant"]
    params = cfg["params"]
    common = {name: params[name] for name in ("in_channels", "input_size", "bottleneck_channels", "norm_type", "upsample_mode")}
    common["gn_groups"] = params.get("gn_groups", 8)

    if variant == "conv":
        return PureConvAutoencoder(
            encoder_channels=params["encoder_channels"],
            downsample_indices=params["downsample_indices"],
            bottleneck_mode=params["bottleneck_mode"],
            latent_dim=params.get("latent_dim", 128),
            use_sn=params["use_sn"],
            **common,
        )

    if variant == "resconv":
        return ResConvAE(
            stage_channels=params["stage_channels"],
            blocks_per_stage=params["blocks_per_stage"],
            **common,
        )

    raise ValueError(f"Unsupported model variant: {variant}")


def load_model_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    obj = torch.load(str(ckpt_path), map_location=device)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)


def load_ae_checkpoint(model_cfg, ckpt_path, device):
    model = build_ae_model(model_cfg).to(device)
    load_model_state(model, ckpt_path, device)
    return model.eval().requires_grad_(False)
