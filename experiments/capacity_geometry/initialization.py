"""Shared clean-reference initialization for capacity and geometry diagnostics."""
from __future__ import annotations
import math
import torch

def normalized_l2(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    error = (x - reference).flatten(1)
    return torch.linalg.vector_norm(error, dim=1) / math.sqrt(error.shape[1])


def build_interpolated_init(
    clean: torch.Tensor,
    observation: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    generator: torch.Generator,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if clean.shape != observation.shape or clean.ndim != 4:
        raise ValueError("clean and observation must share BCHW shape.")
    if not math.isfinite(alpha) or not math.isfinite(beta) or not 0.0 < alpha <= 1.0 or beta < 0.0:
        raise ValueError("Require 0 < alpha <= 1 and beta >= 0.")
    # Reject invalid references before generating or projecting random directions.
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if (not clean.numel() or not clean.is_floating_point() or clean.dtype != observation.dtype
            or clean.device != observation.device or not torch.isfinite(clean).all()
            or not torch.isfinite(observation).all()):
        raise ValueError("Require nonempty finite floating inputs with matching dtype and device.")

    direction = (observation - clean).flatten(1)
    direction_sq = direction.square().sum(1, keepdim=True)
    if bool((direction_sq <= eps).any()):
        raise ValueError("Each observation must differ from its clean reference.")

    noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype).flatten(1)
    noise -= (noise * direction).sum(1, keepdim=True) / direction_sq * direction
    noise_norm = torch.linalg.vector_norm(noise, dim=1, keepdim=True)
    if bool((noise_norm <= eps).any()):
        raise RuntimeError("Degenerate orthogonal perturbation.")

    unit_noise = (noise / noise_norm).view_as(clean)
    direction_image = direction.view_as(clean)
    direction_norm = torch.sqrt(direction_sq).view(-1, 1, 1, 1)
    base = clean + alpha * direction_image
    preclip = base + beta * direction_norm * unit_noise
    initialization = preclip.clamp(0.0, 1.0)
    actual_noise = (initialization - base).flatten(1)
    clean_distance = torch.linalg.vector_norm((initialization - clean).flatten(1), dim=1)
    obs_distance = torch.sqrt(direction_sq[:, 0])
    cosine = (actual_noise * direction).sum(1) / (
        torch.linalg.vector_norm(actual_noise, dim=1) * obs_distance
    ).clamp_min(eps)

    return initialization, {
        "init_error": normalized_l2(initialization, clean),
        "init_distance_ratio": clean_distance / obs_distance.clamp_min(eps),
        "clip_fraction": ((preclip < 0.0) | (preclip > 1.0)).flatten(1).float().mean(1),
        "perturbation_cosine": cosine,
    }
