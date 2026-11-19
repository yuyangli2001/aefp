"""Pair tangent dual certificates with image-space numerical AEFP recovery."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog
from experiments.analytic_ae.verification_solver import AnalyticSigmoidAE, irae_fullsolver
from .common import SubspaceRecord, SynConfig, TangentCertificate, _ae_for, _ambient_point, _seed, _torch_device

# This LP computes the sign-aware tangent dual certificate for one corruption.
def tangent_dual_certificate(
    A: np.ndarray,
    support: np.ndarray,
    signs: np.ndarray,
    rank_tolerance: float,
    boundary_tolerance: float,
    residual_tolerance: float,
) -> TangentCertificate:
    clean = np.ones(A.shape[0], dtype=bool)
    clean[support] = False
    B = A[clean]
    c = A[support].T @ signs
    singular_values = np.linalg.svd(B, compute_uv=False)
    sigma_min = float(singular_values[-1]) if B.shape[0] >= A.shape[1] else 0.0
    rank_full = bool(B.shape[0] >= A.shape[1] and sigma_min > rank_tolerance)
    if not rank_full:
        return TangentCertificate(np.nan, sigma_min, np.nan, False, False, False)

    clean_size = B.shape[0]
    identity = np.eye(clean_size)
    A_ub = np.block([[identity, -np.ones((clean_size, 1))], [-identity, -np.ones((clean_size, 1))]])
    result = linprog(
        np.r_[np.zeros(clean_size), 1.0],
        A_ub=A_ub,
        b_ub=np.zeros(2 * clean_size),
        A_eq=np.c_[B.T, np.zeros(A.shape[1])],
        b_eq=c,
        bounds=[(None, None)] * clean_size + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        return TangentCertificate(np.nan, sigma_min, np.nan, True, False, False)
    eta = float(result.fun)
    residual = float(np.max(np.abs(B.T @ result.x[:-1] - c)))
    certified = bool(residual <= residual_tolerance and eta < 1.0 - boundary_tolerance)
    return TangentCertificate(eta, sigma_min, residual, True, True, certified)


def _aggregate_recovery(config: SynConfig, runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["target_mu", "dimension", "support_size"]
    for key, frame in runs.groupby(keys, sort=True):
        clusters = frame.groupby(["subspace_seed", "corruption_seed"], sort=False)
        # A corruption instance succeeds only if every initialization succeeds.
        instance_success = clusters["success"].all()
        certificates = clusters["eta_certificate_pass"].first()
        rows.append(
            {
                "target_mu": key[0],
                "dimension": key[1],
                "support_size": key[2],
                "rho_s": key[2] / config.m,
                "pgd_instance_success_rate": float(instance_success.mean()),
                "eta_certificate_rate": float(certificates.mean()),
                "mu_actual_mean": float(frame["mu_actual"].mean()),
                "num_runs": len(frame),
                "num_clusters": len(instance_success),
            }
        )
    return pd.DataFrame(rows)


def run_recovery_experiment(
    config: SynConfig, records: Sequence[SubspaceRecord]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if any(not 1 <= support <= config.m for support in config.support_sizes):
        raise ValueError("Every recovery support size must satisfy 1 <= s <= m.")
    run_rows: list[dict[str, object]] = []
    device = _torch_device(config.device)
    for record in records:
        ae = _ae_for(record.A, config.b0)
        x_star = ae.decoder(np.zeros(record.dimension))
        model = AnalyticSigmoidAE(
            torch.as_tensor(record.A, dtype=torch.float64, device=device), config.b0
        ).to(device)
        for support_size in config.support_sizes:
            observations: list[np.ndarray] = []
            initializations: list[np.ndarray] = []
            metadata: list[dict[str, object]] = []
            for corruption_index in range(config.num_corruptions_per_subspace):
                corruption_seed = _seed(config.base_seed, record.seed, support_size, corruption_index)
                rng = np.random.default_rng(corruption_seed)
                support = np.sort(rng.choice(config.m, support_size, replace=False))
                signs = rng.choice((-1.0, 1.0), support_size)
                sparse = np.zeros(config.m)
                sparse[support] = config.corruption_amplitude * signs
                observation = x_star + sparse
                if np.any((observation <= config.box_epsilon) | (observation >= 1 - config.box_epsilon)):
                    raise ValueError("A generated observation leaves the analytic logit box.")
                certificate = tangent_dual_certificate(
                    record.A,
                    support,
                    signs,
                    config.eta_rank_tolerance,
                    config.eta_boundary_tolerance,
                    config.eta_residual_tolerance,
                )
                for init_index in range(config.num_init_directions_per_corruption):
                    init_seed = _seed(config.base_seed, record.seed, support_size, corruption_index, init_index)
                    init_rng = np.random.default_rng(init_seed)
                    unit = init_rng.normal(size=record.dimension)
                    unit /= np.linalg.norm(unit)
                    x_init = _ambient_point(ae, x_star, unit, config.local_radius)
                    run_id = (
                        f"d{record.dimension}_mu{record.target_mu:g}_a{record.seed}"
                        f"_s{support_size}_c{corruption_index}_i{init_index}"
                    )
                    observations.append(observation)
                    initializations.append(x_init)
                    metadata.append(
                        {
                            "run_id": run_id,
                            "subspace_seed": record.seed,
                            "corruption_seed": corruption_seed,
                            "init_seed": init_seed,
                            "dimension": record.dimension,
                            "target_mu": record.target_mu,
                            "mu_actual": record.actual_mu,
                            "beta": record.beta,
                            "support_size": support_size,
                            "rho_s": support_size / config.m,
                            "kappa": support_size / config.m * record.actual_mu * record.dimension,
                            "theoretical_margin": 1.0
                            - 2.0 * support_size / config.m * record.actual_mu * record.dimension,
                            "init_distance": config.local_radius,
                            "eta": certificate.eta,
                            "eta_sigma_min": certificate.sigma_min,
                            "eta_lp_residual": certificate.residual,
                            "eta_rank_full": certificate.rank_full,
                            "eta_lp_success": certificate.lp_success,
                            "eta_certificate_pass": certificate.certified,
                        }
                    )

            observation_tensor = torch.as_tensor(np.stack(observations), dtype=torch.float64, device=device)
            init_tensor = torch.as_tensor(np.stack(initializations), dtype=torch.float64, device=device)
            star_tensor = torch.as_tensor(x_star, dtype=torch.float64, device=device)
            result = irae_fullsolver(
                observation_tensor, init_tensor, star_tensor, model, config.solver_settings()
            )
            estimate = result.estimate.numpy()
            for index, meta in enumerate(metadata):
                recovery_distance = np.linalg.norm(estimate[index] - x_star)
                recovery_gain = recovery_distance / float(meta["init_distance"])
                projected_gain = float(result.projected_gain[index])
                manifold_error = float(result.manifold_error[index])
                normalized_mapping = float(result.normalized_pg_mapping[index])
                converged = normalized_mapping <= config.normalized_pg_tolerance
                box_hits = int(result.box_hit_count[index])
                local_reference = projected_gain * float(meta["init_distance"]) <= config.local_radius
                success = bool(
                    recovery_gain <= config.success_recovery_gain
                    and projected_gain <= config.success_projected_gain
                    and manifold_error <= config.success_manifold_error
                    and converged
                    and box_hits == 0
                    and local_reference
                )
                run_rows.append(
                    {
                        **meta,
                        "recovery_gain": recovery_gain,
                        "final_manifold_error": manifold_error,
                        "final_projected_recovery_gain": projected_gain,
                        "final_normalized_pg_mapping": normalized_mapping,
                        "box_hit_count": box_hits,
                        "converged": converged,
                        "final_reference_local": local_reference,
                        "success": success,
                    }
                )
    runs = pd.DataFrame(run_rows)
    return runs, _aggregate_recovery(config, runs)
