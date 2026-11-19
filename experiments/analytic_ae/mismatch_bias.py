"""Set and tangent mismatch experiments with shared corruption sampling and L1 refinement."""

from __future__ import annotations
from typing import Sequence
from itertools import combinations
import logging
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.linalg import hadamard
from .common import CandidateSet, MismatchSolution, PopulationSigmoidAE, SubspaceRecord, SynConfig, WorstCaseCorruption, _ae_for, _ambient_point, build_mismatch_subspaces

def _angles(count: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack((np.cos(theta), np.sin(theta)), axis=1)


def find_worst_case_corruption(A: np.ndarray, support_size: int, angle_count: int) -> WorstCaseCorruption:
    if A.shape[1] != 2:
        raise ValueError("Worst-case angular search is restricted to d=2.")
    V = _angles(angle_count) @ A.T
    absV = np.abs(V)
    total = absV.sum(axis=1)
    top = np.partition(absV, A.shape[0] - support_size, axis=1)[:, -support_size:].sum(axis=1)
    index = int(np.argmin((total - 2.0 * top) / total))
    direction = V[index]
    support = np.argsort(-np.abs(direction), kind="stable")[:support_size]
    signs = np.sign(direction[support])
    signs[signs == 0] = 1.0
    return WorstCaseCorruption(support, signs)


def build_support_aligned_normal(A: np.ndarray, support: np.ndarray, signs: np.ndarray) -> np.ndarray:
    q = np.zeros(A.shape[0])
    q[support] = signs
    normal = q - A @ (A.T @ q)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise RuntimeError("The support-aligned normal is degenerate.")
    return normal / norm


def build_random_normal(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    normal = rng.normal(size=A.shape[0])
    normal -= A @ (A.T @ normal)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise RuntimeError("The random normal projection is degenerate.")
    return normal / norm


def _build_candidates(
    ae: PopulationSigmoidAE,
    x_star: np.ndarray,
    radius: float,
    angle_count: int,
    radial_count: int,
    min_fraction: float,
) -> CandidateSet:
    units = _angles(angle_count)
    fractions = np.geomspace(min_fraction, 1.0, radial_count)
    boundary = np.empty(angle_count)
    for index, unit in enumerate(units):
        point = _ambient_point(ae, x_star, unit, radius)
        boundary[index] = np.linalg.norm(ae.encoder(point))
    z = (units[:, None, :] * (boundary[:, None] * fractions[None, :])[:, :, None]).reshape(-1, 2)
    z = np.vstack((np.zeros((1, 2)), z))
    return CandidateSet(z, ae.decoder(z))


# Penalize constraint violations during the unconstrained Nelder–Mead search.
def refine_local_l1(decoder, target, observation, radius, starts, tolerance, options,
                    latent_bound=np.inf, require_success=False, candidates=None):
    def gap(z):
        return float(np.abs(decoder(z) - observation).sum())
    def penalized(z):
        x = decoder(z)
        violation = max(0.0, np.linalg.norm(x - target) - radius)
        box_violation = max(0.0, float(np.max(np.abs(z))) - latent_bound)
        return float(np.abs(x - observation).sum() + 1e4 * (violation + box_violation))
    if candidates is None:
        candidates = [(gap(starts[0]), starts[0])]
    successes = 0
    for start in starts:
        result = minimize(penalized, start, method="Nelder-Mead", options=options)
        # Count successful optimizer exits separately from feasibility.
        successes += int(result.success)
        if ((result.success or not require_success)
                and np.linalg.norm(decoder(result.x) - target) <= radius + tolerance):
            candidates.append((gap(result.x), result.x))
    loss, latent = min(candidates, key=lambda item: item[0])
    return np.asarray(latent), loss, successes


def _solve_mismatch(
    config: SynConfig,
    ae: PopulationSigmoidAE,
    pool: CandidateSet,
    x_star: np.ndarray,
    observation: np.ndarray,
    normal: np.ndarray,
    delta: float,
    latent_bound: float,
) -> MismatchSolution:
    shifted = pool.x + delta * normal
    distance = np.linalg.norm(shifted - x_star, axis=1)
    feasible = np.flatnonzero(distance <= config.mismatch_radius + config.constraint_tolerance)
    if not len(feasible):
        raise RuntimeError("The mismatch candidate grid contains no feasible point.")
    objectives = np.abs(shifted[feasible] - observation).sum(axis=1)
    order = np.argsort(objectives)[: config.mismatch_refinement_starts]
    grid_objective = float(objectives[order[0]])
    z_hat, objective, successes = refine_local_l1(
        lambda z: ae.decoder(z) + delta * normal, x_star, observation, config.mismatch_radius,
        pool.z[feasible[order]], config.constraint_tolerance,
        {"xatol": 1e-11, "fatol": 1e-11, "maxiter": 1000}, latent_bound=latent_bound,
    )
    x_hat = ae.decoder(z_hat) + delta * normal
    recovery_error = float(np.linalg.norm(x_hat - x_star))
    slack = config.mismatch_radius - recovery_error
    return MismatchSolution(
        z=z_hat,
        objective=objective,
        grid_objective=grid_objective,
        recovery_error=recovery_error,
        constraint_slack=slack,
        constraint_active=slack <= 1e-3 * config.mismatch_radius,
        latent_bound_active=float(np.max(np.abs(z_hat))) >= 0.99 * latent_bound,
        refinement_successes=successes,
    )


def _run_mismatch_trajectory(
    config: SynConfig,
    record: SubspaceRecord,
    ae: PopulationSigmoidAE,
    pool: CandidateSet,
    audit_pool: CandidateSet,
    x_star: np.ndarray,
    observation: np.ndarray,
    normal: np.ndarray,
    latent_bound: float,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    radius = config.mismatch_radius
    for ratio in config.mismatch_delta_over_radius:
        delta = ratio * radius
        solution = _solve_mismatch(config, ae, pool, x_star, observation, normal, delta, latent_bound)
        resolution_error = np.nan
        resolution_objective = np.nan
        resolution_error_difference = np.nan
        resolution_objective_difference = np.nan
        resolution_objective_pass: object = np.nan
        resolution_solution_pass: object = np.nan
        resolution_pass: object = np.nan
        if np.isclose(ratio, config.mismatch_audit_ratio):
            audit = _solve_mismatch(
                config, ae, audit_pool, x_star, observation, normal, delta, latent_bound
            )
            resolution_error = audit.recovery_error
            resolution_objective = audit.objective
            resolution_error_difference = abs(solution.recovery_error - audit.recovery_error) / radius
            resolution_objective_difference = abs(solution.objective - audit.objective)
            resolution_objective_pass = bool(
                resolution_objective_difference <= config.mismatch_resolution_objective_tolerance
            )
            resolution_solution_pass = bool(
                resolution_error_difference <= config.mismatch_resolution_error_tolerance
            )
            resolution_pass = bool(resolution_objective_pass and resolution_solution_pass)
        support_size = int(metadata["support_size"])
        rho = support_size / config.mismatch_m
        kappa = rho * record.actual_mu * config.mismatch_d
        rows.append(
            {
                **metadata,
                "subspace_seed": record.seed,
                "actual_mu": record.actual_mu,
                "rho_s": rho,
                "kappa": kappa,
                "theoretical_margin": 1.0 - 2.0 * kappa,
                "delta": delta,
                "delta_over_radius": ratio,
                "mismatch_upper_bound": delta,
                "matched_point_error": float(np.linalg.norm(delta * normal)),
                "normal_norm": float(np.linalg.norm(normal)),
                "normal_error": float(np.linalg.norm(record.A.T @ normal)),
                "recovery_error": solution.recovery_error,
                "error_over_radius": solution.recovery_error / radius,
                "amplification": np.nan if delta == 0 else solution.recovery_error / delta,
                "grid_objective": solution.grid_objective,
                "objective_value": solution.objective,
                "objective_improvement": solution.grid_objective - solution.objective,
                "constraint_slack": solution.constraint_slack,
                "constraint_active": solution.constraint_active,
                "latent_bound_active": solution.latent_bound_active,
                "refinement_successes": solution.refinement_successes,
                "resolution_recovery_error": resolution_error,
                "resolution_objective_value": resolution_objective,
                "resolution_error_difference_over_radius": resolution_error_difference,
                "resolution_objective_difference": resolution_objective_difference,
                "resolution_objective_check_pass": resolution_objective_pass,
                "resolution_solution_check_pass": resolution_solution_pass,
                "resolution_check_pass": resolution_pass,
                "z_hat_0": solution.z[0],
                "z_hat_1": solution.z[1],
            }
        )
    return rows


def _prepare_set_mismatch(config, record):
    radius = config.mismatch_radius
    max_delta = radius * max(config.mismatch_delta_over_radius)
    if max_delta >= radius:
        raise ValueError("Every mismatch level must remain below the local radius.")
    ae = _ae_for(record.A, config.b0)
    x_star = ae.decoder(np.zeros(config.mismatch_d))
    pool = _build_candidates(ae, x_star, radius + max_delta, config.mismatch_angle_count,
                             config.mismatch_radial_count, config.mismatch_radial_min_fraction)
    audit_pool = _build_candidates(ae, x_star, radius + max_delta, config.mismatch_audit_angle_count,
                                   config.mismatch_audit_radial_count, config.mismatch_radial_min_fraction)
    latent_bound = 1.05 * float(max(np.max(np.abs(pool.z)), np.max(np.abs(audit_pool.z))))
    return ae, x_star, pool, audit_pool, latent_bound


def run_mismatch_experiment(config: SynConfig, records: Sequence[SubspaceRecord]) -> pd.DataFrame:
    if config.mismatch_d != 2 or len(config.mismatch_support_sizes) != 2:
        raise ValueError("Adversarial set mismatch requires d=2 and two support sizes.")
    rows: list[dict[str, object]] = []
    for record in records:
        ae, x_star, pool, audit_pool, latent_bound = _prepare_set_mismatch(config, record)
        for case, support_size in zip(("easy", "near_boundary"), config.mismatch_support_sizes):
            worst = find_worst_case_corruption(record.A, support_size, config.mismatch_support_angle_count)
            sparse = np.zeros(config.mismatch_m)
            sparse[worst.support] = config.corruption_amplitude * worst.signs
            observation = x_star + sparse
            normal = build_support_aligned_normal(record.A, worst.support, worst.signs)
            rows.extend(
                _run_mismatch_trajectory(
                    config,
                    record,
                    ae,
                    pool,
                    audit_pool,
                    x_star,
                    observation,
                    normal,
                    latent_bound,
                    {
                        "scenario": "controlled",
                        "case": case,
                        "support_size": support_size,
                    },
                )
            )
    return pd.DataFrame(rows)


def _random_mismatch_streams(
    config: SynConfig, subspace_seed: int, support_size: int, trial: int
) -> tuple[np.random.Generator, np.random.Generator, int, int]:
    root = np.random.SeedSequence(
        [config.base_seed + config.mismatch_random_seed_offset, subspace_seed, support_size, trial]
    )
    support_sequence, normal_sequence = root.spawn(2)
    support_seed = int(support_sequence.generate_state(1, dtype=np.uint32)[0])
    normal_seed = int(normal_sequence.generate_state(1, dtype=np.uint32)[0])
    return (
        np.random.default_rng(support_sequence),
        np.random.default_rng(normal_sequence),
        support_seed,
        normal_seed,
    )


def sample_random_corruption(config, subspace_seed, size, trial):
    rng, normal_rng, support_seed, normal_seed = _random_mismatch_streams(config, subspace_seed, size, trial)
    support = np.sort(rng.choice(config.mismatch_m, size, replace=False))
    signs = rng.choice((-1.0, 1.0), size)
    sparse = np.zeros(config.mismatch_m)
    sparse[support] = config.corruption_amplitude * signs
    return support, signs, sparse, normal_rng, support_seed, normal_seed


# Each random trajectory keeps one corruption and one independent normal across all delta levels.
def run_random_mismatch_experiment(
    config: SynConfig, records: Sequence[SubspaceRecord]
) -> pd.DataFrame:
    if config.mismatch_d != 2 or len(config.mismatch_support_sizes) != 2:
        raise ValueError("Random set mismatch requires d=2 and two support sizes.")
    rows: list[dict[str, object]] = []
    for record in records:
        ae, x_star, pool, audit_pool, latent_bound = _prepare_set_mismatch(config, record)
        for case, support_size in zip(("easy", "near_boundary"), config.mismatch_support_sizes):
            for trial in range(config.mismatch_random_trials_per_subspace):
                support, signs, sparse, normal_rng, support_seed, normal_seed = sample_random_corruption(
                    config, record.seed, support_size, trial
                )
                normal = build_random_normal(record.A, normal_rng)
                instance_id = f"s{record.seed}_k{support_size}_r{trial:03d}"
                rows.extend(
                    _run_mismatch_trajectory(
                        config,
                        record,
                        ae,
                        pool,
                        audit_pool,
                        x_star,
                        x_star + sparse,
                        normal,
                        latent_bound,
                        {
                            "scenario": "random",
                            "case": case,
                            "instance_id": instance_id,
                            "trial": trial,
                            "support_size": support_size,
                            "support_seed": support_seed,
                            "normal_seed": normal_seed,
                            "support": ";".join(map(str, support)),
                            "signs": ";".join(f"{int(sign):d}" for sign in signs),
                            "corruption_amplitude": config.corruption_amplitude,
                        },
                    )
                )
    return pd.DataFrame(rows)


# Fit the median curve through the origin over the configured small-shift range.
def summarize_mismatch(config: SynConfig, mismatch: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case, frame in mismatch.groupby("case", sort=False):
        curve = frame.groupby("delta_over_radius", sort=True)["error_over_radius"].median()
        fit = curve[curve.index <= config.mismatch_fit_max_ratio]
        x = fit.index.to_numpy(dtype=np.float64)
        y = fit.to_numpy(dtype=np.float64)
        slope = float(x @ y / (x @ x))
        residual = y - slope * x
        centered = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(residual @ residual) / centered if centered > 0 else 1.0
        seed_slopes = []
        for _, seed_frame in frame.groupby("subspace_seed"):
            seed_fit = seed_frame[seed_frame["delta_over_radius"] <= config.mismatch_fit_max_ratio]
            seed_x = seed_fit["delta_over_radius"].to_numpy(dtype=np.float64)
            seed_y = seed_fit["error_over_radius"].to_numpy(dtype=np.float64)
            seed_slopes.append(float(seed_x @ seed_y / (seed_x @ seed_x)))
        audits = frame["resolution_check_pass"].dropna().astype(bool)
        baseline = frame[np.isclose(frame["delta_over_radius"], 0.0)]
        rows.append(
            {
                "case": case,
                "support_size": int(frame["support_size"].iloc[0]),
                "theoretical_margin": float(frame["theoretical_margin"].median()),
                "fit_max_delta_over_radius": config.mismatch_fit_max_ratio,
                "slope_through_origin": slope,
                "r_squared": r_squared,
                "max_abs_fit_residual": float(np.max(np.abs(residual))),
                "subspace_slope_median": float(np.median(seed_slopes)),
                "subspace_slope_min": float(np.min(seed_slopes)),
                "subspace_slope_max": float(np.max(seed_slopes)),
                "baseline_max_error": float(baseline["recovery_error"].max()),
                "constraint_active_count": int(frame["constraint_active"].sum()),
                "resolution_checks": len(audits),
                "resolution_pass_rate": float(audits.mean()) if len(audits) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_random_mismatch(
    config: SynConfig, mismatch: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for case, frame in mismatch.groupby("case", sort=False):
        # Include every sampled trajectory. Grid agreement is reported as a separate diagnostic.
        trajectories = frame.pivot(
            index=["subspace_seed", "instance_id"], columns="delta_over_radius", values="error_over_radius"
        ).sort_index(axis=1)
        ratios = trajectories.columns.to_numpy(dtype=np.float64)
        values = trajectories.to_numpy(dtype=np.float64)
        fit_mask = ratios <= config.mismatch_fit_max_ratio
        fit_x = ratios[fit_mask]
        denominator = float(fit_x @ fit_x)
        mean_curve = values.mean(axis=0)
        instance_slopes = values[:, fit_mask] @ fit_x / denominator
        subspaces = trajectories.index.get_level_values("subspace_seed").unique().to_numpy()
        blocks = {seed: trajectories.xs(seed, level="subspace_seed").to_numpy() for seed in subspaces}
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    config.base_seed + config.mismatch_bootstrap_seed_offset,
                    int(frame["support_size"].iloc[0]),
                ]
            )
        )
        bootstrap_curves = np.empty((config.mismatch_bootstrap_replicates, len(ratios)))
        for replicate in range(config.mismatch_bootstrap_replicates):
            sampled = []
            for seed in rng.choice(subspaces, size=len(subspaces), replace=True):
                block = blocks[seed]
                sampled.append(block[rng.integers(0, len(block), size=len(block))])
            bootstrap_curves[replicate] = np.concatenate(sampled).mean(axis=0)
        lower, upper = np.percentile(bootstrap_curves, (2.5, 97.5), axis=0)
        bootstrap_slopes = bootstrap_curves[:, fit_mask] @ fit_x / denominator
        mean_slope = float(mean_curve[fit_mask] @ fit_x / denominator)
        residual = mean_curve[fit_mask] - mean_slope * fit_x
        centered = float(np.sum((mean_curve[fit_mask] - mean_curve[fit_mask].mean()) ** 2))
        r_squared = 1.0 - float(residual @ residual) / centered if centered > 0 else 1.0
        support_size = int(frame["support_size"].iloc[0])
        for ratio, mean, low, high in zip(ratios, mean_curve, lower, upper):
            curve_rows.append(
                {
                    "case": case,
                    "support_size": support_size,
                    "delta_over_radius": ratio,
                    "mean_error_over_radius": mean,
                    "ci_lower": low,
                    "ci_upper": high,
                }
            )
        audited = frame[frame["resolution_objective_difference"].notna()]
        objective_pass = (
            audited["resolution_objective_difference"] <= config.mismatch_resolution_objective_tolerance
        )
        solution_pass = (
            audited["resolution_error_difference_over_radius"]
            <= config.mismatch_resolution_error_tolerance
        )
        combined_pass = objective_pass & solution_pass
        baseline = frame[np.isclose(frame["delta_over_radius"], 0.0)]
        primary = frame[frame["delta_over_radius"] <= config.mismatch_fit_max_ratio]
        exploratory = frame[frame["delta_over_radius"] > config.mismatch_fit_max_ratio]
        summary_rows.append(
            {
                "case": case,
                "support_size": support_size,
                "num_subspaces": len(subspaces),
                "num_instances": len(trajectories),
                "theoretical_margin": float(frame["theoretical_margin"].median()),
                "fit_max_delta_over_radius": config.mismatch_fit_max_ratio,
                "mean_curve_slope": mean_slope,
                "slope_ci_lower": float(np.percentile(bootstrap_slopes, 2.5)),
                "slope_ci_upper": float(np.percentile(bootstrap_slopes, 97.5)),
                "r_squared": r_squared,
                "instance_slope_mean": float(np.mean(instance_slopes)),
                "instance_slope_median": float(np.median(instance_slopes)),
                "instance_slope_q25": float(np.percentile(instance_slopes, 25)),
                "instance_slope_q75": float(np.percentile(instance_slopes, 75)),
                "baseline_max_error": float(baseline["recovery_error"].max()),
                "primary_constraint_active_count": int(primary["constraint_active"].sum()),
                "primary_latent_bound_active_count": int(primary["latent_bound_active"].sum()),
                "exploratory_constraint_active_rate": float(exploratory["constraint_active"].mean()),
                "exploratory_latent_bound_active_rate": float(exploratory["latent_bound_active"].mean()),
                "resolution_checks": len(audited),
                "resolution_combined_pass_rate": float(combined_pass.mean()),
                "resolution_objective_pass_rate": float(objective_pass.mean()),
                "resolution_solution_pass_rate": float(solution_pass.mean()),
                "resolution_max_error_difference_over_radius": float(
                    audited["resolution_error_difference_over_radius"].max()
                ),
                "resolution_max_objective_difference": float(
                    audited["resolution_objective_difference"].max()
                ),
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


LATENT_BOUND, R_STAR_RATIO = 0.3, 0.2
RESULT_FILES = ("tangent_mismatch_results.csv", "tangent_mismatch_summary.csv")


# The paired Hadamard columns preserve dimension two and coherence one at every angle.
def build_geometry(config):
    values = np.asarray(config.tangent_angles, dtype=float)
    if (config.mismatch_m, config.mismatch_d, config.b0, config.mismatch_radius) != (64, 2, 0.8, 0.05):
        raise ValueError("Use m=64, d=2, b0=0.8 and R=0.05 for this experiment.")
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all() or len(np.unique(values)) != len(values)
            or np.any((values < 0) | (values > 30)) or 0 not in values or not np.any(values > 0)):
        raise ValueError("Use unique angles in [0, 30], including zero and a positive angle.")
    if (max(config.mismatch_support_sizes) >= 16
            or config.mismatch_audit_angle_count <= config.mismatch_angle_count
            or config.mismatch_audit_radial_count <= config.mismatch_radial_count):
        raise ValueError("Use s<16 and a finer audit grid in both dimensions.")
    H = hadamard(64).astype(float)
    A = H[:, [1, 2]] / 8
    B = np.column_stack((-H[:, 4] * A[:, 1], H[:, 4] * A[:, 0]))
    return A, B


# Target-coordinate hyperplane intersections supply distinct starts at small mismatch.
def anchor_points(Q, logits, center):
    equations = np.column_stack((Q, logits))
    _, indices = np.unique(np.round(equations, 13), axis=0, return_index=True)
    points = [center]
    for first, second in combinations(equations[indices], 2):
        matrix = np.stack((first[:2], second[:2]))
        if abs(np.linalg.det(matrix)) > 1e-12:
            points.append(np.linalg.solve(matrix, [first[2], second[2]]))
    return np.unique(np.round(points, 13), axis=0)


def candidate_grid(ae, target, center, radius, angular, radial):
    phi = np.linspace(0, 2 * np.pi, angular, endpoint=False)
    unit = np.column_stack((np.cos(phi), np.sin(phi)))
    grid = np.vstack((center, (center + np.linspace(0, LATENT_BOUND, radial)[1:, None, None] * unit).reshape(-1, 2)))
    images = ae.decoder(grid)
    feasible = np.linalg.norm(images - target, axis=1) <= radius
    return grid[feasible], images[feasible]


def solve_local_l1(ae, target, observation, pool, anchors, radius, n_refine):
    grid, images = pool
    if not len(grid):
        raise RuntimeError("No feasible grid points.")
    scores = np.abs(images - observation).sum(axis=1)
    starts = np.unique(np.round(np.vstack((grid[np.argsort(scores)[:n_refine]], anchors)), 13), axis=0)
    starts = starts[np.linalg.norm(ae.decoder(starts) - target, axis=1) <= radius]
    candidates = [(float(scores.min()), grid[np.argmin(scores)])]
    candidates += [(float(np.abs(ae.decoder(z) - observation).sum()), z) for z in starts]
    latent, loss, successes = refine_local_l1(
        ae.decoder, target, observation, radius, starts, 1e-10,
        {"xatol": 1e-12, "fatol": 1e-12, "maxiter": 2000},
        require_success=True, candidates=candidates,
    )
    return dict(z=latent, objective=loss, error=float(np.linalg.norm(ae.decoder(latent) - target)),
                grid_objective=float(scores.min()), starts=len(starts), successes=successes)


# Freeze every observation and compare primary and finer solves at each rotation.
def run_tangent_mismatch(config):
    A, B = build_geometry(config)
    base, radius = _ae_for(A, config.b0), config.mismatch_radius
    origin = base.decoder([0, 0])
    alpha = brentq(lambda a: np.linalg.norm(base.decoder([a, 0]) - origin) - R_STAR_RATIO * radius, 1e-10, 1)
    target, r_star = base.decoder([alpha, 0]), R_STAR_RATIO * radius
    patterns = []
    for size in config.mismatch_support_sizes:
        for trial in range(config.mismatch_random_trials_per_subspace):
            _, _, sparse, _, _, _ = sample_random_corruption(config, 0, size, trial)
            patterns.append((size, trial, sparse))
    rows = []
    grids = ((config.mismatch_angle_count, config.mismatch_radial_count),
             (config.mismatch_audit_angle_count, config.mismatch_audit_radial_count))
    for degrees in sorted(config.tangent_angles):
        theta = np.deg2rad(degrees)
        Q = np.cos(theta) * A + np.sin(theta) * B
        ae, epsilon = _ae_for(Q, config.b0), float(np.sin(theta))
        center = np.array([alpha * np.cos(theta), 0])
        anchors = anchor_points(Q, alpha * A[:, 0], center)
        pools = [candidate_grid(ae, target, center, radius, *grid) for grid in grids]
        for size, trial, sparse in patterns:
            observation = target + sparse
            first, final = [solve_local_l1(ae, target, observation, pool, anchors, radius,
                                           config.mismatch_refinement_starts) for pool in pools]
            signal = epsilon * r_star
            error_gap, loss_gap = abs(first['error'] - final['error']), abs(first['objective'] - final['objective'])
            tolerance = max(1e-9 * radius, 0.01 * signal)
            check = (final['error'] <= radius + 1e-10
                     and final['objective'] <= np.abs(ae.decoder(center) - observation).sum() + 1e-10
                     and final['successes'] > 0 and first['successes'] > 0
                     and error_gap <= tolerance and loss_gap <= 1e-9
                     and (degrees != 0 or final['error'] <= 1e-8 * radius))
            rows.append(dict(s=size, trial=trial,
                theta_deg=degrees, epsilon_M=0, epsilon_T=epsilon, r_star=r_star,
                epsilon_T_r_star_over_R=signal / radius, recovery_error_over_R=final['error'] / radius,
                objective=final['objective'],
                grid_objective=final['grid_objective'], primary_error_over_R=first['error'] / radius,
                resolution_error_difference=error_gap, resolution_error_tolerance=tolerance,
                resolution_objective_difference=loss_gap, distinct_starts=final['starts'],
                successful_refinements=final['successes'],
                z_hat_1=float(final['z'][0]), z_hat_2=float(final['z'][1]), check_pass=bool(check)))
        logging.info("Finished tangent angle %.6g (%s cases)", degrees, len(rows))
    return pd.DataFrame(rows)


def _summarize_tangent(frame):
    return frame.groupby(['s', 'theta_deg'], as_index=False).agg(
        epsilon_T=('epsilon_T', 'mean'),
        epsilon_T_r_star_over_R=('epsilon_T_r_star_over_R', 'mean'),
        recovery_error_over_R=('recovery_error_over_R', 'mean'),
        error_min=('recovery_error_over_R', 'min'), error_max=('recovery_error_over_R', 'max'),
        num_trials=('trial', 'size'))


SET_RESULT_FILES = ("results_mismatch.csv", "results_mismatch_summary.csv", "results_mismatch_random.csv",
                    "results_mismatch_random_curve.csv", "results_mismatch_random_summary.csv")
SUITE_FILES = SET_RESULT_FILES + RESULT_FILES + tuple(
    f"{stem}.{extension}" for stem in ("fig_mismatch_bias", "fig_mismatch_bias_combined")
    for extension in ("png", "pdf"))


def run_mismatch_suite(config):
    from .report import plot_adversarial_mismatch, plot_mismatch_comparison

    output = config.output_path()
    output.mkdir(parents=True, exist_ok=True)
    records = build_mismatch_subspaces(config)
    logging.info("Running adversarial and random set mismatch")
    adversarial = run_mismatch_experiment(config, records)
    random = run_random_mismatch_experiment(config, records)
    curve, summary = summarize_random_mismatch(config, random)

    logging.info("Running tangent mismatch")
    tangent = run_tangent_mismatch(config)
    if not tangent.check_pass.all():
        failed = int((~tangent.check_pass).sum())
        raise RuntimeError(f"Tangent mismatch numerical checks failed for {failed} cases.")

    tables = (adversarial, summarize_mismatch(config, adversarial), random, curve, summary,
              tangent, _summarize_tangent(tangent))
    for name, frame in zip(SET_RESULT_FILES + RESULT_FILES, tables):
        frame.to_csv(output / name, index=False)
    plot_adversarial_mismatch(config, adversarial)
    plot_mismatch_comparison(config, curve, tangent)
    return adversarial, random, tangent
