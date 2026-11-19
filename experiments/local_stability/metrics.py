"""Singular-value-only diagnostics for finite-tolerance local dimension."""

import math

import numpy as np
import torch

from experiments.capacity_geometry import geometry as geo


# Dimension diagnostics need singular values, not singular vectors or projectors.
def linearize(model, image, config):
    J, residual, timing = geo.compute_linearization(model, image, config["chunk_size"])
    spectrum = torch.linalg.svdvals(torch.from_numpy(J - np.eye(len(J)))).numpy()
    return dict(spectrum=spectrum, residual=residual), timing


def validate_measurement(data, size):
    if set(data) != {"spectrum", "residual"}:
        raise ValueError("Expected only spectrum and residual in a rank-diagnostic measurement.")
    s, residual = data["spectrum"], data["residual"]
    if s.shape != (size,) or residual.shape != (size,):
        raise ValueError("Invalid measurement shape.")
    if any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (s, residual)):
        raise ValueError("Measurements must be finite float64 arrays.")
    if (s < 0).any() or (np.diff(s) > 0).any():
        raise ValueError("Invalid singular spectrum.")


def anchor_metrics(data, epsilon):
    s = data["spectrum"]
    rank = int((s <= epsilon).sum())
    return dict(E_recon=float(np.linalg.norm(data["residual"]) / math.sqrt(len(s))), d_hat=rank,
                threshold_distance=float(np.abs(s - epsilon).min()))


# Normalize each anchor's dimension change before averaging across images and directions.
def compare(anchor, perturbed, epsilon):
    base = anchor_metrics(anchor, epsilon)
    k = base["d_hat"]
    s = perturbed["spectrum"]
    other = int((s <= epsilon).sum())
    return dict(**base, perturbed_dimension=other, delta_d=abs(other - k) / max(1, k),
                absolute_dimension_change=abs(other - k), dimension_changed=int(other != k),
                perturbed_threshold_distance=float(np.abs(s - epsilon).min()))
