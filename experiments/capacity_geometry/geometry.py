"""Full reconstruction Jacobians and local geometry diagnostics."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np
import torch


# Precision settings affect both Jacobians and the shared recovery solver.
@contextmanager
def precision_context(cpu_threads=4):
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads <= 0:
        raise ValueError("cpu_threads must be a positive integer.")
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark, torch.get_num_threads())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(cpu_threads)
    try:
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[:2]
        torch.backends.cudnn.benchmark = old[2]
        torch.set_num_threads(old[3])


def validate_thresholds(thresholds):
    if not thresholds or any(not math.isfinite(e) or not 0 < e < 1 for e in thresholds):
        raise ValueError("Require finite thresholds strictly between zero and one.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be unique.")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# Only input derivatives are needed;
def compute_linearization(model, image, chunk_size=16, adaptive=False):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    model.eval().requires_grad_(False)
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float64
    image = torch.as_tensor(image, device=device, dtype=dtype)
    shape = image.shape
    x = image.flatten()

    def reconstruct(z):
        return model(z.reshape(1, *shape)).flatten()

    synchronize(device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    retries = []
    while True:
        try:
            with torch.enable_grad():
                J = torch.func.jacrev(reconstruct, chunk_size=chunk_size)(x)
            break
        except RuntimeError as error:
            message = str(error).lower()
            oom = "out of memory" in message
            unsupported = "vmap" in message or "batching rule" in message
            if not adaptive or chunk_size == 1 or not (oom or unsupported):
                raise
            retries.append(dict(chunk_size=chunk_size, reason="oom" if oom else "vmap"))
            chunk_size = max(1, chunk_size // 2) if oom else 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
    with torch.no_grad():
        residual = reconstruct(x) - x
    synchronize(device)
    if J.shape != (x.numel(), x.numel()) or not torch.isfinite(J).all() or not torch.isfinite(residual).all():
        raise ValueError("Nonfinite or incorrectly shaped reconstruction Jacobian.")
    return J.detach().cpu().numpy(), residual.cpu().double().numpy(), dict(
        seconds=time.perf_counter() - started, chunk_size=chunk_size, retries=retries,
        dtype=str(dtype).removeprefix("torch."), device=str(device),
        peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    )


# Coherence depends on the subspace projector
def subspace_metrics(Q):
    m, dimension = Q.shape
    if dimension == 0:
        return dict(d_hat=0, mu_hat=None, K_hat=None)
    if not np.allclose(Q.T @ Q, np.eye(dimension), atol=1e-8, rtol=1e-8):
        raise ValueError("Tangent proxy basis is not orthonormal.")
    concentration = float(m * np.square(Q).sum(axis=1).max())
    coherence = concentration / dimension
    if not dimension - 1e-7 <= concentration <= m + 1e-7:
        raise ValueError("Coherence bounds failed.")
    return dict(d_hat=dimension, mu_hat=coherence, K_hat=concentration)


def factorize(J):
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or not J.size or J.shape[0] != J.shape[1] or not np.isfinite(J).all():
        raise ValueError("Expected a finite square Jacobian.")
    A = J - np.eye(len(J))
    _, spectrum, Vh = torch.linalg.svd(torch.from_numpy(A), full_matrices=False)
    return A, spectrum.numpy(), Vh.numpy()


def metrics_from_svd(spectrum, Vh, thresholds, bottleneck_dimension):
    validate_thresholds(thresholds)
    rows = []
    for epsilon in thresholds:
        selected = spectrum <= epsilon
        values = subspace_metrics(Vh[selected].T)
        if values["d_hat"] > bottleneck_dimension:
            raise ValueError("Near-invariant dimension exceeds the bottleneck bound.")
        gap = float(spectrum[~selected].min() - spectrum[selected].max()) if selected.any() and (~selected).any() else None
        rows.append(dict(epsilon=float(epsilon), **values,
                         threshold_distance=float(np.abs(spectrum - epsilon).min()), spectral_gap=gap))
    return rows


def geometry_from_jacobian(J, thresholds, bottleneck_dimension):
    _, spectrum, Vh = factorize(J)
    return metrics_from_svd(spectrum, Vh, thresholds, bottleneck_dimension), spectrum


def recovery_metrics(clean, initialization, recovered, model):
    with torch.no_grad():
        initial_norm = torch.linalg.vector_norm((initialization - clean).flatten(1), dim=1)
        restored_norm = torch.linalg.vector_norm((recovered - clean).flatten(1), dim=1)
        scale = math.sqrt(clean[0].numel())
        values = dict(E_rest=restored_norm / scale, E_init=initial_norm / scale,
                      G_init=restored_norm / initial_norm.clamp_min(1e-12),
                      fixed_point_residual=torch.linalg.vector_norm((model(recovered) - recovered).flatten(1), dim=1) / scale)
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise ValueError("Nonfinite recovery metric.")
        return [{key: float(value[i]) for key, value in values.items()} for i in range(len(clean))]
