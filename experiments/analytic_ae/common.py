"""Configuration, exact sigmoid autoencoders and reproducible tangent spaces."""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from pathlib import Path
from typing import Sequence
import matplotlib
import numpy as np
import torch
from scipy.linalg import hadamard
from scipy.special import expit
from experiments.analytic_ae.verification_solver import FullSolverSettings


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SynConfig:
    m: int = 256
    dimensions: tuple[int, ...] = (2, 8, 16, 32, 64, 96, 128, 160, 192, 224)
    b0: float = 0.8
    target_mus: tuple[float, ...] = (1.0, 4.0)
    support_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 240, 256)
    num_subspace_instances: int = 4
    num_corruptions_per_subspace: int = 12
    num_init_directions_per_corruption: int = 2
    corruption_amplitude: float = 0.25
    local_radius: float = 0.05
    beta_grid_count: int = 2001
    mu_tolerance: float = 0.03
    box_epsilon: float = 1e-4
    lambda_init: float = 0.02
    lambda_multiplier: float = 0.5
    lambda_min: float = 0.00005
    stage_min_iterations: int = 20
    stage_max_iterations: int = 300
    final_stage_max_iterations: int = 1000
    step_init: float = 0.1
    step_max: float = 1.0
    backtrack_factor: float = 0.5
    max_backtracks: int = 40
    normalized_pg_tolerance: float = 0.01
    success_recovery_gain: float = 0.1
    success_projected_gain: float = 0.1
    success_manifold_error: float = 5e-5
    sign_tolerance: float = 1e-8
    eta_rank_tolerance: float = 1e-10
    eta_boundary_tolerance: float = 1e-6
    eta_residual_tolerance: float = 1e-8
    mismatch_m: int = 64
    mismatch_d: int = 2
    mismatch_num_subspace_instances: int = 4
    mismatch_radius: float = 0.05
    tangent_angles: tuple[float, ...] = (0, 5, 10, 15, 20, 30)
    mismatch_support_sizes: tuple[int, ...] = (4, 15)
    mismatch_delta_over_radius: tuple[float, ...] = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
    mismatch_angle_count: int = 1440
    mismatch_radial_count: int = 48
    mismatch_radial_min_fraction: float = 1e-4
    mismatch_support_angle_count: int = 16384
    mismatch_refinement_starts: int = 6
    mismatch_fit_max_ratio: float = 0.1
    mismatch_audit_ratio: float = 0.1
    mismatch_audit_angle_count: int = 2880
    mismatch_audit_radial_count: int = 72
    mismatch_resolution_error_tolerance: float = 1e-3
    mismatch_resolution_objective_tolerance: float = 1e-5
    mismatch_random_trials_per_subspace: int = 32
    mismatch_random_seed_offset: int = 60_000_000
    mismatch_bootstrap_replicates: int = 2000
    mismatch_bootstrap_seed_offset: int = 61_000_000
    constraint_tolerance: float = 1e-8
    base_seed: int = 20260801
    device: str = "auto"
    output_dir: str = "outputs/analytic_ae"

    def __post_init__(self) -> None:
        for name in ("m", "mismatch_m"):
            value = integer_index(getattr(self, name))
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two.")
        counts = ("num_subspace_instances", "num_corruptions_per_subspace", "num_init_directions_per_corruption",
                  "beta_grid_count", "mismatch_num_subspace_instances", "mismatch_angle_count", "mismatch_radial_count",
                  "mismatch_support_angle_count", "mismatch_refinement_starts", "mismatch_audit_angle_count",
                  "mismatch_audit_radial_count", "mismatch_random_trials_per_subspace", "mismatch_bootstrap_replicates")
        if any(integer_index(getattr(self, name)) <= 0 for name in counts):
            raise ValueError("Experiment counts must be positive integers.")
        for name in ("base_seed", "mismatch_random_seed_offset", "mismatch_bootstrap_seed_offset"):
            if integer_index(getattr(self, name)) < 0:
                raise ValueError("Seeds and seed offsets must be non-negative integers.")
        for values in (self.dimensions, self.support_sizes, self.mismatch_support_sizes):
            if not values or len(set(values)) != len(values) or any(integer_index(value) <= 0 for value in values):
                raise ValueError("Dimension and support grids must contain unique positive integers.")
        if not self.target_mus or len(set(self.target_mus)) != len(self.target_mus) or any(not isfinite(value) or value < 1 for value in self.target_mus):
            raise ValueError("Coherence targets must be finite, unique and at least one.")
        positive = ("corruption_amplitude", "local_radius", "mismatch_radius")
        nonnegative = ("mu_tolerance", "success_recovery_gain", "success_projected_gain", "success_manifold_error", "sign_tolerance",
                       "eta_rank_tolerance", "eta_boundary_tolerance", "eta_residual_tolerance", "constraint_tolerance",
                       "mismatch_resolution_error_tolerance", "mismatch_resolution_objective_tolerance")
        if any(not isfinite(getattr(self, name)) or getattr(self, name) <= 0 for name in positive) or any(not isfinite(getattr(self, name)) or getattr(self, name) < 0 for name in nonnegative) or not isfinite(self.b0):
            raise ValueError("Experiment scales and tolerances must be finite and valid.")
        if not 0 < self.mismatch_radial_min_fraction <= 1 or not 0 < self.mismatch_fit_max_ratio < 1:
            raise ValueError("Radius and fit fractions are invalid.")
        ratios = self.mismatch_delta_over_radius
        if not ratios or len(set(ratios)) != len(ratios) or any(not isfinite(value) or not 0 <= value < 1 for value in ratios):
            raise ValueError("Mismatch levels must be unique finite fractions in [0,1).")
        if 0 not in ratios or self.mismatch_audit_ratio not in ratios or not any(0 < value <= self.mismatch_fit_max_ratio for value in ratios):
            raise ValueError("Mismatch levels must include zero, the audit level and a positive fitting level.")
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must not be empty.")
        self.solver_settings()

    @classmethod
    def from_json(cls, path: Path) -> "SynConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        # Accept metadata and diagnostic settings from existing experiment records.
        for obsolete in ("config_hash", "run_mode", "small_radius_fraction", "pilot_seed_offset", "quick_seed_offset"):
            data.pop(obsolete, None)
        tuple_fields = (
            "dimensions",
            "target_mus",
            "support_sizes",
            "mismatch_support_sizes",
            "tangent_angles",
            "mismatch_delta_over_radius",
        )
        for key in tuple_fields:
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def recovery_pairs(self) -> tuple[tuple[int, float], ...]:
        return tuple(
            (dimension, target)
            for dimension in self.dimensions
            for target in self.target_mus
            if _attainable_coherence(self.m, dimension, target, self.mu_tolerance)
        )

    def output_path(self) -> Path:
        path = Path(self.output_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def solver_settings(self) -> FullSolverSettings:
        return FullSolverSettings(
            lambda_init=self.lambda_init,
            lambda_multiplier=self.lambda_multiplier,
            lambda_min=self.lambda_min,
            stage_min_iterations=self.stage_min_iterations,
            stage_max_iterations=self.stage_max_iterations,
            final_stage_max_iterations=self.final_stage_max_iterations,
            step_init=self.step_init,
            step_max=self.step_max,
            backtrack_factor=self.backtrack_factor,
            max_backtracks=self.max_backtracks,
            normalized_pg_tolerance=self.normalized_pg_tolerance,
            box_epsilon=self.box_epsilon,
        )


# The numpy model is used for exact geometry and the mismatch estimator.
@dataclass(frozen=True)
class PopulationSigmoidAE:
    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or min(self.A.shape) == 0 or self.b.shape != (self.A.shape[0],):
            raise ValueError("Incompatible A and b shapes.")
        if not np.isfinite(self.A).all() or not np.isfinite(self.b).all():
            raise ValueError("Analytic model parameters must be finite.")
        if not np.allclose(self.A.T @ self.A, np.eye(self.A.shape[1]), atol=1e-10):
            raise ValueError("A must have orthonormal columns.")

    def decoder(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return expit(z @ self.A.T + self.b)

    def encoder(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if np.any((x <= 0) | (x >= 1)):
            raise ValueError("The analytic logit encoder requires x in (0, 1)^m.")
        return (np.log(x) - np.log1p(-x) - self.b) @ self.A

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.decoder(self.encoder(x))

    def decoder_jacobian(self, z: np.ndarray) -> np.ndarray:
        x = self.decoder(np.asarray(z, dtype=np.float64))
        return (x * (1.0 - x))[:, None] * self.A


@dataclass(frozen=True)
class SubspaceRecord:
    m: int
    dimension: int
    seed: int
    target_mu: float
    actual_mu: float
    beta: float
    A: np.ndarray


@dataclass(frozen=True)
class WorstCaseCorruption:
    support: np.ndarray
    signs: np.ndarray


@dataclass(frozen=True)
class TangentCertificate:
    eta: float
    sigma_min: float
    residual: float
    rank_full: bool
    lp_success: bool
    certified: bool


@dataclass(frozen=True)
class CandidateSet:
    z: np.ndarray
    x: np.ndarray


@dataclass(frozen=True)
class MismatchSolution:
    z: np.ndarray
    objective: float
    grid_objective: float
    recovery_error: float
    constraint_slack: float
    constraint_active: bool
    latent_bound_active: bool
    refinement_successes: int


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(matrix, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    return Q * signs


def project_to_one_perp(matrix: np.ndarray) -> np.ndarray:
    one = np.ones(matrix.shape[0]) / np.sqrt(matrix.shape[0])
    return matrix - np.outer(one, one @ matrix)


def subspace_coherence(A: np.ndarray) -> float:
    m, d = A.shape
    return m / d * float(np.max(np.sum(A * A, axis=1)))


# Allow targets close to the upper bound set by the all-ones constraint.
def _attainable_coherence(m: int, d: int, target: float, tolerance: float) -> bool:
    return 1.0 <= target <= (m - 1) / d + tolerance


# Interpolate between Hadamard and coordinate subspaces to vary coherence.
def _build_subspaces(
    m: int,
    dimensions: Sequence[int],
    target_mus: Sequence[float],
    instances: int,
    beta_grid_count: int,
    base_seed: int,
    mu_tolerance: float,
) -> list[SubspaceRecord]:
    if m <= 0 or m & (m - 1):
        raise ValueError("m must be a positive power of two.")
    H = hadamard(m).astype(np.float64) / np.sqrt(m)
    betas = np.linspace(0.0, 1.0, beta_grid_count)
    records: list[SubspaceRecord] = []
    for d in dimensions:
        if not 1 <= d < m:
            raise ValueError("Each dimension must satisfy 1 <= d < m.")
        targets = tuple(
            float(target)
            for target in target_mus
            if _attainable_coherence(m, d, target, mu_tolerance)
        )
        if not targets:
            continue
        for instance in range(instances):
            seed = int(base_seed + 1000 * d + instance)
            rng = np.random.default_rng(seed)
            columns = rng.choice(np.arange(1, m), d, replace=False)
            A_inc = H[rng.permutation(m)][:, columns] * rng.choice((-1.0, 1.0), d)
            best: dict[float, tuple[float, float, np.ndarray]] = {
                target: (0.0, 0.0, A_inc.copy()) for target in targets if target == 1.0
            }
            nontrivial = tuple(target for target in targets if target != 1.0)
            if nontrivial:
                coordinates = rng.choice(m, d, replace=False)
                A_coord = orthonormalize(project_to_one_perp(np.eye(m)[:, coordinates]))
                U, _, Vt = np.linalg.svd(A_coord.T @ A_inc)
                A_coord = A_coord @ (U @ Vt)
                for beta in betas:
                    A = orthonormalize((1.0 - beta) * A_inc + beta * A_coord)
                    mu = subspace_coherence(A)
                    for target in nontrivial:
                        error = abs(mu - target)
                        if target not in best or error < best[target][0]:
                            best[target] = (error, float(beta), A.copy())
            for target in targets:
                _, beta, A = best[target]
                records.append(SubspaceRecord(m, d, seed, float(target), subspace_coherence(A), beta, A))
    return records


def build_subspaces(config: SynConfig) -> list[SubspaceRecord]:
    return _build_subspaces(
        config.m,
        config.dimensions,
        config.target_mus,
        config.num_subspace_instances,
        config.beta_grid_count,
        config.base_seed,
        config.mu_tolerance,
    )


def build_mismatch_subspaces(config: SynConfig) -> list[SubspaceRecord]:
    return _build_subspaces(
        config.mismatch_m,
        (config.mismatch_d,),
        (1.0,),
        config.mismatch_num_subspace_instances,
        config.beta_grid_count,
        config.base_seed + 50_000_000,
        config.mu_tolerance,
    )


def _ae_for(A: np.ndarray, b0: float) -> PopulationSigmoidAE:
    return PopulationSigmoidAE(A, np.full(A.shape[0], b0, dtype=np.float64))


def _ambient_point(
    ae: PopulationSigmoidAE, x_star: np.ndarray, unit: np.ndarray, radius: float
) -> np.ndarray:
    low, high = 0.0, 1.0
    while np.linalg.norm(ae.decoder(high * unit) - x_star) < radius:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("Failed to bracket the ambient-radius point.")
    for _ in range(55):
        mid = 0.5 * (low + high)
        if np.linalg.norm(ae.decoder(mid * unit) - x_star) < radius:
            low = mid
        else:
            high = mid
    return ae.decoder(0.5 * (low + high) * unit)


def _seed(base: int, *parts: int) -> int:
    payload = ":".join(str(value) for value in (base, *parts)).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device
