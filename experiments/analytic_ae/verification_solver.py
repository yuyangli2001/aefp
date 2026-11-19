"""High-accuracy image-space IRAE solver for controlled synthetic verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index

import torch

Tensor = torch.Tensor


class AnalyticSigmoidAE(torch.nn.Module):
    """Exact idempotent sigmoid AE with a prescribed decoder subspace."""

    def __init__(self, A: Tensor, b0: float) -> None:
        super().__init__()
        if A.ndim != 2 or A.numel() == 0 or not A.is_floating_point() or not torch.isfinite(A).all() or not isfinite(b0):
            raise ValueError("A must have shape [m, d].")
        tolerance = 1e-10 if A.dtype == torch.float64 else 1e-5
        if not torch.allclose(A.T @ A, torch.eye(A.shape[1], dtype=A.dtype, device=A.device), atol=tolerance, rtol=0):
            raise ValueError("A must have orthonormal columns.")
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("b", torch.full((A.shape[0],), float(b0), dtype=A.dtype, device=A.device))

    def encode(self, x: Tensor) -> Tensor:
        return (torch.logit(x) - self.b) @ self.A

    def decode(self, z: Tensor) -> Tensor:
        return torch.sigmoid(z @ self.A.T + self.b)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


@dataclass(frozen=True)
class FullSolverSettings:
    lambda_init: float
    lambda_multiplier: float
    lambda_min: float
    stage_min_iterations: int
    stage_max_iterations: int
    final_stage_max_iterations: int
    step_init: float
    step_max: float
    backtrack_factor: float
    max_backtracks: int
    normalized_pg_tolerance: float
    box_epsilon: float

    def __post_init__(self) -> None:
        positive = (self.lambda_init, self.lambda_min, self.step_init, self.step_max)
        if any(not isfinite(value) or value <= 0 for value in positive) or self.lambda_min > self.lambda_init:
            raise ValueError("Penalty and step scales must be finite, positive and ordered.")
        if not 0 < self.lambda_multiplier < 1 or not 0 < self.backtrack_factor < 1 or not 0 < self.box_epsilon < 0.5:
            raise ValueError("Continuation, backtracking or box settings are invalid.")
        if integer_index(self.stage_min_iterations) < 0 or any(integer_index(value) <= 0 for value in (self.stage_max_iterations, self.final_stage_max_iterations, self.max_backtracks)):
            raise ValueError("Solver iteration budgets are invalid.")
        if not isfinite(self.normalized_pg_tolerance) or self.normalized_pg_tolerance < 0:
            raise ValueError("Gradient mapping tolerance must be finite and non-negative.")


@dataclass(frozen=True)
class FullSolverResult:
    estimate: Tensor
    box_hit_count: Tensor
    manifold_error: Tensor
    projected_gain: Tensor
    normalized_pg_mapping: Tensor



def _lambda_path(settings: FullSolverSettings) -> tuple[float, ...]:
    values = [float(settings.lambda_init)]
    while values[-1] > settings.lambda_min:
        value = max(values[-1] * settings.lambda_multiplier, settings.lambda_min)
        if value == values[-1]:
            break
        values.append(float(value))
    return tuple(values)


def _prox_box_l1(v: Tensor, observation: Tensor, threshold: float, epsilon: float) -> tuple[Tensor, Tensor]:
    unboxed = observation + torch.sign(v - observation) * torch.clamp(
        torch.abs(v - observation) - threshold, min=0.0
    )
    hits = ((unboxed < epsilon) | (unboxed > 1.0 - epsilon)).sum(dim=1)
    return unboxed.clamp(epsilon, 1.0 - epsilon), hits


def _reconstruction_term(model: AnalyticSigmoidAE, x: Tensor) -> Tensor:
    return 0.5 * (x - model(x)).square().sum(dim=1)


def _gradient_mapping(
    model: AnalyticSigmoidAE,
    x: Tensor,
    observation: Tensor,
    lambda_: float,
    step: float,
    epsilon: float,
) -> Tensor:
    point = x.detach().requires_grad_(True)
    gradient = torch.autograd.grad(_reconstruction_term(model, point).sum(), point)[0]
    candidate, _ = _prox_box_l1(point - step * gradient, observation, step * lambda_, epsilon)
    return (point.detach() - candidate.detach()) / step


def irae_fullsolver(
    observation: Tensor,
    x_init: Tensor,
    x_star: Tensor,
    model: AnalyticSigmoidAE,
    settings: FullSolverSettings,
) -> FullSolverResult:
    if observation.ndim != 2 or x_init.shape != observation.shape:
        raise ValueError("observation and x_init must share shape [batch, m].")
    if x_star.ndim == 1:
        x_star = x_star.expand_as(observation)
    if x_star.shape != observation.shape:
        raise ValueError("x_star must have shape [m] or [batch, m].")
    if observation.numel() == 0 or any(not value.is_floating_point() or not torch.isfinite(value).all() for value in (observation, x_init, x_star)):
        raise ValueError("Solver inputs must be nonempty, finite floating-point tensors.")
    if any(value.dtype != model.A.dtype or value.device != model.A.device for value in (observation, x_init, x_star)) or observation.shape[1] != model.A.shape[0]:
        raise ValueError("Solver input dtype, device and dimension must match the model.")
    if not bool(((x_init >= settings.box_epsilon) & (x_init <= 1 - settings.box_epsilon)).all()):
        raise ValueError("Initialization must lie inside the analytic logit box.")
    if not bool(((observation > settings.box_epsilon) & (observation < 1 - settings.box_epsilon)).all()):
        raise ValueError("Observations must lie strictly inside the analytic logit box.")

    x = x_init.detach().clone()
    observation = observation.detach()
    x_star = x_star.detach()
    batch, m = x.shape
    init_distance = torch.linalg.vector_norm(x - x_star, dim=1).clamp_min(torch.finfo(x.dtype).eps)
    box_hits = torch.zeros(batch, dtype=torch.int64, device=x.device)
    step = float(settings.step_init)
    path = _lambda_path(settings)

    for stage_index, lambda_ in enumerate(path):
        max_iterations = (
            settings.final_stage_max_iterations if stage_index == len(path) - 1 else settings.stage_max_iterations
        )
        for stage_iteration in range(max_iterations):
            point = x.detach().requires_grad_(True)
            recon = _reconstruction_term(model, point)
            gradient = torch.autograd.grad(recon.sum(), point)[0]
            accepted = False
            trial_step = step
            accepted_hits = torch.zeros_like(box_hits)
            for _ in range(settings.max_backtracks):
                candidate, trial_hits = _prox_box_l1(
                    point - trial_step * gradient,
                    observation,
                    trial_step * lambda_,
                    settings.box_epsilon,
                )
                delta = candidate - point
                upper = recon + (gradient * delta).sum(dim=1) + delta.square().sum(dim=1) / (2.0 * trial_step)
                candidate_recon = _reconstruction_term(model, candidate)
                # Accept steps that satisfy the quadratic majorization bound.
                if bool((candidate_recon <= upper + 1e-12).all()):
                    accepted = True
                    accepted_hits = trial_hits
                    break
                trial_step *= settings.backtrack_factor
            if not accepted:
                raise RuntimeError("IRAE full-solver backtracking failed to find a majorizing step.")

            x = candidate.detach()
            box_hits += accepted_hits
            step = min(trial_step / settings.backtrack_factor, settings.step_max)

            if stage_iteration + 1 >= settings.stage_min_iterations:
                mapping = _gradient_mapping(model, x, observation, lambda_, trial_step, settings.box_epsilon)
                normalized = torch.linalg.vector_norm(mapping, dim=1) / (lambda_ * m**0.5)
                if bool((normalized <= settings.normalized_pg_tolerance).all()):
                    break


    mapping = _gradient_mapping(model, x, observation, lambda_, trial_step, settings.box_epsilon)
    normalized = torch.linalg.vector_norm(mapping, dim=1) / (lambda_ * m**0.5)
    with torch.no_grad():
        manifold_point = model(x)
        manifold_error = torch.linalg.vector_norm(x - manifold_point, dim=1) / m**0.5
        projected_gain = torch.linalg.vector_norm(manifold_point - x_star, dim=1) / init_distance
    return FullSolverResult(x.cpu(), box_hits.cpu(), manifold_error.cpu(),
                            projected_gain.cpu(), normalized.cpu())
