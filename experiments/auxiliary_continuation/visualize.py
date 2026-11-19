"""Visualize paired recovery trajectories, objective slices, and endpoint images."""

from __future__ import annotations
import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import numpy as np
import torch
from recovery.batch import _iter_loader_batches, build_batch_degradation
from methods.continuation import irae_direct
from recovery.adapters.ae import _load_models
from recovery.registry import resolve_config
from utils.utils_dataloader import get_degradation_dataloader
from utils.utils_metrics import avg_psnr_pytorch, avg_ssim_pytorch
from utils.precision import full_precision
from utils.figure_layout import panel_heading
from .variants import _degradation, solver_schedule
from .artifacts import _require_writable


# Use the clean-initialized and direct endpoints to span a common plane.
def construct_outcome_plane(observation: torch.Tensor, favorable: torch.Tensor, direct: torch.Tensor,
                            trajectories: Sequence[Sequence[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    p0 = observation.detach().cpu().reshape(-1)
    u1 = favorable.detach().cpu().reshape(-1) - p0
    if not torch.isfinite(u1).all() or u1.norm() <= 1e-12:
        raise ValueError("The favorable endpoint must define a finite nonzero plane direction.")
    u1 = u1 / u1.norm().clamp_min(1e-12)
    direct_offset = direct.detach().cpu().reshape(-1) - p0
    residual = direct_offset - torch.dot(direct_offset, u1) * u1
    ratio = float(residual.norm() / direct_offset.norm().clamp_min(1e-12))
    mode = "outcome"
    if residual.norm() < 1e-6 or ratio < 1e-3:
        points = torch.stack([point.detach().cpu().reshape(-1) for trajectory in trajectories for point in trajectory])
        centered = points - p0
        orthogonal = centered - torch.outer(centered @ u1, u1)
        _, _, basis = torch.pca_lowrank(orthogonal, q=1, center=False)
        residual, mode = basis[:, 0], "trajectory_pca"
    u2 = residual / residual.norm().clamp_min(1e-12)
    if (not torch.isfinite(u2).all() or not torch.isclose(u2.norm(), torch.ones((), dtype=u2.dtype), atol=1e-4)
            or abs(float(u1 @ u2)) > 1e-4):
        raise ValueError("Outcome plane directions must be finite and orthonormal.")
    return p0, u1, u2, torch.tensor(ratio), mode


def project_to_plane(points: Sequence[torch.Tensor], p0: torch.Tensor,
                     u1: torch.Tensor, u2: torch.Tensor) -> np.ndarray:
    coords = []
    for point in points:
        offset = point.detach().cpu().reshape(-1) - p0
        coords.append((float(offset @ u1), float(offset @ u2)))
    return np.asarray(coords, dtype=np.float64)


def compute_objective_grid(model: Any, observation: torch.Tensor, p0: torch.Tensor,
                           u1: torch.Tensor, u2: torch.Tensor, alpha_range: tuple[float, float],
                           beta_range: tuple[float, float], grid_size: int, grid_batch_size: int,
                           lambda_: float, device: torch.device) -> tuple[np.ndarray, ...]:
    if grid_size < 3 or grid_batch_size <= 0 or not math.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("Require grid_size >= 3, positive batches, and finite nonnegative lambda.")
    alphas = np.linspace(*alpha_range, grid_size)
    betas = np.linspace(*beta_range, grid_size)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    coords = torch.tensor(np.column_stack((alpha_grid.ravel(), beta_grid.ravel())),
                          dtype=observation.dtype, device=device)
    p0_device, u1_device, u2_device = p0.to(device), u1.to(device), u2.to(device)
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coords), grid_batch_size):
            pair = coords[start:start + grid_batch_size]
            flat = p0_device + pair[:, :1] * u1_device + pair[:, 1:] * u2_device
            images = flat.reshape(-1, *observation.shape[1:])
            reconstruction = model(images)
            prior = 0.5 * (images - reconstruction).square().flatten(1).sum(1)
            fidelity = (images - observation).abs().flatten(1).sum(1)
            values.append((prior + float(lambda_) * fidelity).cpu())
    objective = torch.cat(values).numpy().reshape(grid_size, grid_size)
    if not np.isfinite(objective).all():
        raise ValueError("Objective grid contains nonfinite values.")
    return alpha_grid, beta_grid, objective


def compute_direction_field(model: Any, observation: torch.Tensor, p0: torch.Tensor,
                            u1: torch.Tensor, u2: torch.Tensor,
                            alpha_range: tuple[float, float], beta_range: tuple[float, float],
                            count: int, lambda_: float, device: torch.device,
                            step_size: float = 0.01) -> tuple[np.ndarray, ...]:
    if count <= 0 or not math.isfinite(lambda_) or lambda_ < 0 or not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("Require positive direction count/step size and finite nonnegative lambda.")
    alphas = np.linspace(*alpha_range, count + 2)[1:-1]
    betas = np.linspace(*beta_range, count + 2)[1:-1]
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    directions = []
    p0_device, u1_device, u2_device = p0.to(device), u1.to(device), u2.to(device)
    model.eval()
    for alpha, beta in zip(alpha_grid.ravel(), beta_grid.ravel()):
        flat = p0_device + float(alpha) * u1_device + float(beta) * u2_device
        image = flat.reshape_as(observation).detach().requires_grad_(True)
        prior = 0.5 * (image - model(image)).square().sum()
        gradient = torch.autograd.grad(prior, image)[0]
        with torch.no_grad():
            proposal = image - step_size * gradient
            difference = observation - proposal
            candidate = observation - difference.sign() * (difference.abs() - step_size * lambda_).clamp_min(0)
            direction = (candidate - image).reshape(-1)
            directions.append((float(direction @ u1_device), float(direction @ u2_device)))
    directions = np.asarray(directions).reshape(count, count, 2)
    if not np.isfinite(directions).all():
        raise ValueError("Direction field contains nonfinite values.")
    return alpha_grid, beta_grid, directions[..., 0], directions[..., 1]


def _metric_pair(clean: torch.Tensor, reconstruction: torch.Tensor) -> tuple[float, float]:
    return (float(avg_psnr_pytorch(clean, reconstruction)),
            float(avg_ssim_pytorch(clean, reconstruction)))


def trajectory_psnr(clean: torch.Tensor, trajectory: Sequence[torch.Tensor]) -> np.ndarray:
    clean_cpu = clean.detach().cpu()
    stacked = torch.cat([point.detach().cpu() for point in trajectory], dim=0)
    mse = (stacked - clean_cpu).square().flatten(1).mean(1).clamp_min(1e-12)
    return (-10.0 * torch.log10(mse)).numpy()


def _save_reconstruction_strip(output_dir: Path, clean: torch.Tensor,
                               endpoints: Mapping[str, torch.Tensor],
                               metrics: Mapping[str, tuple[float, float]]) -> None:
    import matplotlib.pyplot as plt

    items = [("Clean", clean), ("Observation", endpoints["observation"]),
              ("Original end", endpoints["direct"]), ("Coarse", endpoints["coarse"]),
              ("Fine", endpoints["fine"]), ("AEFP end", endpoints["jinit"])]
    # Export readable vector text directly, with one heading line per panel.
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, len(items), figsize=(7.16, 1.35))
        for axis, (name, tensor) in zip(axes, items):
            image = tensor[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            axis.imshow(image.squeeze(), cmap="gray" if image.shape[-1] == 1 else None,
                        vmin=0, vmax=1, interpolation="none")
            metric_key = {"Original end": "direct", "AEFP end": "jinit"}.get(name, name.lower())
            metric = "" if name == "Clean" else f"{metrics[metric_key][0]:.2f} dB"
            panel_heading(axis, name, metric)
            axis.axis("off")
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.86, wspace=0.055)
        fig.savefig(output_dir / "fig_3stage_sample_reconstructions.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(output_dir / "fig_3stage_sample_reconstructions.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


# Recompute displayed metrics from saved endpoints without running restoration again.
def replot_reconstruction_strip(source: Path, output_dir: Path, *, force: bool = False) -> dict[str, Any]:
    _require_writable([output_dir / name for name in ("fig_3stage_sample_reconstructions.png",
        "fig_3stage_sample_reconstructions.pdf", "fig_3stage_metrics.json")], force)
    tensors = torch.load(source, map_location="cpu", weights_only=True)
    keys = ("observation", "direct", "coarse", "fine", "jinit")
    clean = tensors["clean"]
    if (clean.ndim != 4 or clean.shape[0] != 1 or not clean.numel()
            or any(tensors[k].shape != clean.shape for k in keys)
            or any(not torch.isfinite(tensors[key]).all() for key in ("clean", *keys))):
        raise ValueError("Expected matching single-image endpoint tensors.")
    metrics = {key: _metric_pair(clean, tensors[key]) for key in keys}
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_reconstruction_strip(output_dir, clean, tensors, metrics)
    payload = {"source": str(source.resolve()),
               "metrics": {key: {"psnr": p, "ssim": s} for key, (p, s) in metrics.items()}}
    (output_dir / "fig_3stage_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_mechanism_figure(*, output_dir: Path, grids: Mapping[str, tuple[np.ndarray, ...]],
                           directions: Mapping[str, tuple[np.ndarray, ...]],
                           coords: Mapping[str, np.ndarray], metrics: Mapping[str, tuple[float, float]],
                           quality_curves: Mapping[str, np.ndarray],
                           alpha_range: tuple[float, float], beta_range: tuple[float, float],
                           trajectory_stride: int) -> None:
    import matplotlib.pyplot as plt

    original_grid, coarse_grid = grids["original"], grids["coarse"]
    # A zero-valued objective is valid and still needs a finite display scale.
    tiny = np.finfo(np.float64).tiny
    display_grids = {name: (np.log10(grid[2] / grid[2].min()) if grid[2].min() > 0 else
                           np.log10(np.maximum(grid[2].astype(np.float64), tiny)) - np.log10(tiny))
                     for name, grid in grids.items()}
    vmax = max(float(np.percentile(np.concatenate([value.ravel() for value in display_grids.values()]), 99)), 1e-12)
    fig = plt.figure(figsize=(11.2, 4.6))
    grid_spec = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.045),
                                 height_ratios=(4.5, 1.15), wspace=0.16, hspace=0.32)
    axes = [fig.add_subplot(grid_spec[0, 0])]
    axes.extend(fig.add_subplot(grid_spec[0, index], sharex=axes[0], sharey=axes[0])
                for index in (1, 2))
    quality_axes = [fig.add_subplot(grid_spec[1, index]) for index in range(3)]
    colorbar_axis = fig.add_subplot(grid_spec[:, 3])
    backgrounds = (original_grid, coarse_grid, original_grid)
    titles = ("(a) Direct AEFP on original AE", "(b) Coarse auxiliary AE",
              "(c) Coarse → fine → original")
    contours = []
    for axis, grid, title, grid_name in zip(axes, backgrounds, titles,
                                            ("original", "coarse", "original")):
        contours.append(axis.contourf(grid[0], grid[1], display_grids[grid_name],
                                      levels=np.linspace(0.0, vmax, 36), cmap="viridis",
                                      vmin=0.0, vmax=vmax, extend="max"))
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(r"Favorable-outcome direction $u_1$", fontsize=9.5)
        axis.set_xlim(alpha_range)
        axis.set_ylim(beta_range)
        axis.tick_params(labelsize=9)
        axis.grid(alpha=0.15)
    axes[0].set_ylabel(r"Direct residual direction $u_2$", fontsize=9.5)

    for axis, field_name in ((axes[0], "original"), (axes[1], "coarse"), (axes[2], "original")):
        a, b, da, db = directions[field_name]
        x_span, y_span = alpha_range[1] - alpha_range[0], beta_range[1] - beta_range[0]
        norm = np.sqrt((da / x_span) ** 2 + (db / y_span) ** 2) + 1e-12
        axis.quiver(a, b, (da / x_span) / norm * x_span * 0.035,
                    (db / y_span) / norm * y_span * 0.035,
                    color="white", alpha=0.55, angles="xy", scale_units="xy", scale=1,
                    width=0.003, headwidth=3)

    def draw_trajectory(axis, values, color, label):
        axis.plot(values[:, 0], values[:, 1], color=color, linewidth=2.0, label=label, zorder=5)
        axis.scatter(values[::trajectory_stride, 0], values[::trajectory_stride, 1],
                     s=12, color=color, zorder=6)

    draw_trajectory(axes[0], coords["direct"], "#ef4444", "Direct")
    draw_trajectory(axes[1], coords["coarse"], "#f59e0b", "Coarse")
    draw_trajectory(axes[2], coords["coarse"], "#f59e0b", "Coarse")
    draw_trajectory(axes[2], coords["fine"], "#22c55e", "Fine")
    draw_trajectory(axes[2], coords["original"], "#38bdf8", "Original")
    for axis in axes:
        axis.scatter(*coords["observation"][0], marker="o", s=55, color="white",
                     edgecolor="black", label="Observation", zorder=8)
        axis.scatter(*coords["clean"][0], marker="*", s=150, color="#f8fafc",
                     edgecolor="black", label="GT projection", zorder=8)
    axes[0].scatter(*coords["direct"][-1], marker="x", s=70, color="#ef4444",
                    linewidth=2, label="Original end", zorder=8)
    axes[1].scatter(*coords["coarse"][-1], marker="x", s=70, color="#f59e0b",
                    linewidth=2, label="Coarse end", zorder=8)
    axes[2].scatter(*coords["original"][-1], marker="x", s=70, color="#38bdf8",
                    linewidth=2, label="AEFP end", zorder=8)
    boxes = (("Original end", metrics["direct"]), ("Coarse end", metrics["coarse"]),
             ("AEFP end", metrics["jinit"]))
    for axis, (name, pair) in zip(axes, boxes):
        legend = axis.legend(loc="upper left", fontsize=8, framealpha=0.88,
                             borderpad=0.35, labelspacing=0.25, handlelength=1.5,
                             handletextpad=0.5,
                             title=f"{name}\n{pair[0]:.2f} dB | SSIM {pair[1]:.4f}")
        legend._legend_box.align = "left"
        legend.get_title().set_fontsize(8.5)

    quality_axes[0].plot(np.arange(len(quality_curves["direct"])), quality_curves["direct"],
                         color="#ef4444", linewidth=1.8)
    quality_axes[1].plot(np.arange(len(quality_curves["coarse"])), quality_curves["coarse"],
                         color="#f59e0b", linewidth=1.8)
    offset = 0
    for name, color in (("coarse", "#f59e0b"), ("fine", "#22c55e"),
                        ("original", "#38bdf8")):
        values = quality_curves[name]
        quality_axes[2].plot(offset + np.arange(len(values)), values, color=color, linewidth=1.8)
        offset += len(values) - 1
    for boundary in (100, 200):
        quality_axes[2].axvline(boundary, color="0.55", linestyle="--", linewidth=0.8)
    quality_values = np.concatenate(list(quality_curves.values()))
    quality_margin = max(0.5, 0.05 * float(np.ptp(quality_values)))
    for index, axis in enumerate(quality_axes):
        axis.set_xlabel("Iteration", fontsize=9.5)
        axis.set_ylim(float(quality_values.min() - quality_margin),
                      float(quality_values.max() + quality_margin))
        axis.tick_params(labelsize=9)
        axis.grid(alpha=0.2)
        if index == 0:
            axis.set_ylabel("True PSNR (dB)\n(not projected)", fontsize=9.5)
    colorbar = fig.colorbar(contours[0], cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.1, top=0.94)
    fig.savefig(output_dir / "mechanism_3panel.png", dpi=250, bbox_inches="tight")
    fig.savefig(output_dir / "mechanism_3panel.pdf", bbox_inches="tight")
    plt.close(fig)


# The clean-initialized run supplies a reference direction for the common plane.
def recover_paths(clean, observation, degradation, models, config, device):
    base_lambda = float(config['lambda'])
    aux_lambda = float(config['lambda_init_gamma']) * base_lambda
    common = dict(observation=observation, degradation=degradation, device=device,
                  max_iteration=int(config['max_iteration']), return_history=True, **solver_schedule(config))
    direct = irae_direct(ae_model=models[-1], x_init=observation, lambda_=base_lambda, **common)
    oracle = irae_direct(ae_model=models[-1], x_init=clean, lambda_=base_lambda, **common)
    stages, current = [], observation
    for model, stage_lambda in zip(models, (aux_lambda, aux_lambda, base_lambda)):
        result = irae_direct(ae_model=model, x_init=current, lambda_=stage_lambda, **common)
        stages.append(result)
        current = result['reconstruction']
    return direct, oracle, stages


@full_precision()
def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    """Compute and plot one recovery using frozen checkpoints."""
    visual_dir = Path(args.output_dir) / "visual"
    _require_writable([visual_dir / name for name in ("endpoints.pt", "endpoints.csv", "visualization_config.json",
        "mechanism_3panel.png", "mechanism_3panel.pdf", "fig_3stage_sample_reconstructions.png",
        "fig_3stage_sample_reconstructions.pdf")], args.force)
    dataset, ratio = args.datasets[0], float(args.ratios[0])
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    _, _, degradation_cfg, config = resolve_config(dataset, _degradation(ratio), "irae")
    device = torch.device(args.device)
    models = _load_models(config, str(device), "jinit")
    coarse_model, _, original_model = models
    loader_dataset, image_size = ("celeba-hq-256", 64) if dataset == "celeba-hq-64" else (dataset, None)
    loader = get_degradation_dataloader(
        dataset=loader_dataset, split=args.split, degradation_cfg=degradation_cfg,
        batch_size=1, num_workers=0, root=args.data_root, image_size=image_size,
        sample_indices=[args.sample_index], shuffle=False, drop_last=False,
    )
    batch = next(iter(loader))
    clean, observation = batch["clean"].to(device), batch["degraded"].to(device)
    degradation = build_batch_degradation(batch["meta"]["mode"],
                                          {key: value.to(device) if torch.is_tensor(value) else value
                                           for key, value in batch.get("params", {}).items()})
    base_lambda = float(config["lambda"])
    direct_result, oracle_result, stage_results = recover_paths(clean, observation, degradation, models, config, device)
    direct_final_lambda = float(direct_result["history_metrics"][-1]["lambda"])
    landscape_lambda = direct_final_lambda if args.landscape_lambda is None else float(args.landscape_lambda)

    trajectories = ([direct_result["history"], oracle_result["history"]]
                    + [result["history"] for result in stage_results])
    p0, u1, u2, residual_ratio, plane_mode = construct_outcome_plane(
        observation, oracle_result["reconstruction"], direct_result["reconstruction"], trajectories)
    coords = {
        "observation": project_to_plane([observation], p0, u1, u2),
        "clean": project_to_plane([clean], p0, u1, u2),
        "direct": project_to_plane(direct_result["history"], p0, u1, u2),
        "oracle": project_to_plane(oracle_result["history"], p0, u1, u2),
        "coarse": project_to_plane(stage_results[0]["history"], p0, u1, u2),
        "fine": project_to_plane(stage_results[1]["history"], p0, u1, u2),
        "original": project_to_plane(stage_results[2]["history"], p0, u1, u2),
    }
    quality_curves = {
        "direct": trajectory_psnr(clean, direct_result["history"]),
        "coarse": trajectory_psnr(clean, stage_results[0]["history"]),
        "fine": trajectory_psnr(clean, stage_results[1]["history"]),
        "original": trajectory_psnr(clean, stage_results[2]["history"]),
    }
    all_coords = np.concatenate(list(coords.values()))
    spans = np.maximum(np.ptp(all_coords, axis=0), 1e-3)
    lower, upper = all_coords.min(axis=0) - 0.25 * spans, all_coords.max(axis=0) + 0.25 * spans
    alpha_range, beta_range = (float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))
    grids = {
        "original": compute_objective_grid(original_model, observation, p0, u1, u2,
                                           alpha_range, beta_range, args.grid_size,
                                           args.grid_batch_size, landscape_lambda, device),
        "coarse": compute_objective_grid(coarse_model, observation, p0, u1, u2,
                                         alpha_range, beta_range, args.grid_size,
                                         args.grid_batch_size, landscape_lambda, device),
    }
    directions = {
        "original": compute_direction_field(original_model, observation, p0, u1, u2,
                                             alpha_range, beta_range, args.direction_grid,
                                             landscape_lambda, device),
        "coarse": compute_direction_field(coarse_model, observation, p0, u1, u2,
                                           alpha_range, beta_range, args.direction_grid,
                                           landscape_lambda, device),
    }
    endpoints = {
        "observation": observation, "direct": direct_result["reconstruction"],
        "coarse": stage_results[0]["reconstruction"], "fine": stage_results[1]["reconstruction"],
        "jinit": stage_results[2]["reconstruction"], "oracle": oracle_result["reconstruction"],
    }
    metrics = {name: _metric_pair(clean, value) for name, value in endpoints.items()}
    visual_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'clean': clean.detach().cpu(), **{name: value.detach().cpu() for name, value in endpoints.items()}},
               visual_dir / 'endpoints.pt')
    _save_reconstruction_strip(visual_dir, clean, endpoints, metrics)
    _plot_mechanism_figure(output_dir=visual_dir, grids=grids, directions=directions,
                           coords=coords, metrics=metrics, quality_curves=quality_curves,
                           alpha_range=alpha_range,
                           beta_range=beta_range, trajectory_stride=args.trajectory_stride)
    with (visual_dir / "endpoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("endpoint", "psnr", "ssim", "alpha", "beta"))
        writer.writeheader()
        coord_keys = {"observation": "observation", "direct": "direct", "coarse": "coarse",
                      "fine": "fine", "jinit": "original", "oracle": "oracle"}
        for name, pair in metrics.items():
            coordinate = coords[coord_keys[name]][-1]
            writer.writerow({"endpoint": name, "psnr": pair[0], "ssim": pair[1],
                             "alpha": coordinate[0], "beta": coordinate[1]})
    payload = {
        "dataset": dataset, "ratio": ratio, "split": args.split, "sample_index": args.sample_index,
        "seed": 42,
        "solver_precision_policy": "full_precision_deterministic_cudnn",
        "backtracking_rule": config['backtracking_rule'],
        "preset_key": config["preset_key"], "model_id": config["model_id"],
        "checkpoint_dir": config["checkpoint"]["path"],
        "stage_paths": list(config["checkpoint"]["stage_paths"]), "lambda": base_lambda,
        "landscape_lambda": landscape_lambda,
        "direct_final_lambda": direct_final_lambda,
        "landscape_lambda_source": ("direct_final_iteration" if args.landscape_lambda is None
                                    else "command_line_override"),
        "lambda_init_gamma": float(config["lambda_init_gamma"]),
        "stage_iterations": list(config["jinit_stage_iters"]), "plane_mode": plane_mode,
        "direct_residual_ratio": float(residual_ratio), "alpha_range": alpha_range,
        "beta_range": beta_range, "grid_size": args.grid_size,
        "grid_batch_size": args.grid_batch_size, "direction_grid": args.direction_grid,
        "metrics": {name: {"psnr": pair[0], "ssim": pair[1]} for name, pair in metrics.items()},
    }
    with (visual_dir / "visualization_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload
