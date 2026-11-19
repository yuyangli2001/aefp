"""MNIST bottleneck-capacity versus IRAE-recovery diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.ae_trainer import MultiStageCAETrainer
from utils.checkpoint_paths import build_model_tag, direct_ckpt_path
from utils.configuration import load_yaml_config, normalize_model_cfg, normalize_training_cfg
from models.loading import load_ae_checkpoint
from methods.continuation import irae_direct
from utils.utils_dataloader import get_degradation_dataloader

from experiments.capacity_geometry.initialization import normalized_l2, build_interpolated_init


METRICS = (
    "recon_error",
    "init_error",
    "restoration_error",
    "recovery_gain",
    "fixed_point_residual",
    "realized_support_ratio",
    "init_distance_ratio",
    "clip_fraction",
    "perturbation_cosine",
)

# Defaults for the MNIST test sweep over bottleneck widths and training seeds.
FORMAL_BOTTLENECK_CHANNELS = (1, 2, 4, 6, 8, 16, 32)
FORMAL_MODEL_SEEDS = (2, 1024, 2026)
FORMAL_ALPHA = 0.25
FORMAL_BETA = 0.05
FORMAL_LAMBDA = 0.07
FORMAL_SPLIT = "test"


# Keep model, corruption, and initialization randomness reproducible and separate.
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Build x0 = clean + alpha * (obs-clean) + an orthogonal normalized perturbation.


# Load the exact model/training profiles.
def current_configs(args: argparse.Namespace, bottleneck: int) -> tuple[dict, dict, dict]:
    dataset_raw = load_yaml_config(PROJECT_ROOT / "configs/dataset/mnist.yaml")
    dataset_cfg = {
        key: value
        for key, value in dataset_raw.items()
        if key not in {"defaults", "dataset_defaults"}
    }
    model_cfg = load_yaml_config(PROJECT_ROOT / "configs/model/resae_3down.yaml")
    model_cfg["params"]["bottleneck_channels"] = bottleneck
    model_cfg = normalize_model_cfg(model_cfg)
    model_cfg["params"].update(in_channels=1, input_size=28)

    train_cfg = load_yaml_config(PROJECT_ROOT / "configs/train/ae.yaml")
    train_cfg.update(
        train_batch_size=args.train_batch_size,
        train_subset_size=args.train_subset_size,
        num_workers=args.num_workers,
        force_retrain=args.force_retrain,
    )
    train_cfg["original"]["epochs"] = args.epochs
    return dataset_cfg, model_cfg, normalize_training_cfg(train_cfg)


def prepare_model(
    args: argparse.Namespace,
    bottleneck: int,
    model_seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Path]:
    set_random_seed(model_seed)
    dataset_cfg, model_cfg, train_cfg = current_configs(args, bottleneck)
    ckpt_dir = (
        PROJECT_ROOT
        / args.checkpoint_root
        / "mnist"
        / build_model_tag(model_cfg, train_cfg)
        / f"seed_{model_seed}"
    )
    # In evaluation-only mode, load the existing checkpoint directly.
    if not args.eval_only:
        train_loader = get_degradation_dataloader(
            "mnist",
            "train",
            {"mode": "clean"},
            args.train_batch_size,
            args.num_workers,
            image_size=28,
            subset_size=args.train_subset_size,
        )
        trainer = MultiStageCAETrainer(
            dataset_cfg=dataset_cfg,
            clean_train_loader=train_loader,
            ckpt_dir=str(ckpt_dir),
            model_cfg=model_cfg,
            training_cfg=train_cfg,
            device=device,
        )
        trainer.prepare_direct()
    checkpoint = direct_ckpt_path(ckpt_dir)
    return load_ae_checkpoint(model_cfg, checkpoint, device), checkpoint


def evaluate_model(
    args: argparse.Namespace,
    loader,
    model: torch.nn.Module,
    bottleneck: int,
    model_seed: int,
    checkpoint: Path,
    device: torch.device,
) -> list[dict]:
    accumulators = [{name: 0.0 for name in METRICS} | {"count": 0} for _ in range(args.init_repeats)]
    generators = [
        torch.Generator(device=device).manual_seed(args.init_seed + repeat)
        for repeat in range(args.init_repeats)
    ]
    seen = 0
    for batch in loader:
        if args.num_samples is not None and seen >= args.num_samples:
            break
        take = len(batch["clean"])
        if args.num_samples is not None:
            take = min(take, args.num_samples - seen)
        clean = batch["clean"][:take].to(device)
        observation = batch["degraded"][:take].to(device)
        support = ((observation - clean).abs().flatten(1) > 1e-8).float().mean(1)
        with torch.no_grad():
            recon_error = normalized_l2(model.get_recon(clean), clean)

        for repeat, generator in enumerate(generators):
            initialization, init_metrics = build_interpolated_init(
                clean,
                observation,
                alpha=args.alpha,
                beta=args.beta,
                generator=generator,
            )
            recovered = irae_direct(
                observation=observation,
                ae_model=model,
                degradation={"mode": "sparse"},
                x_init=initialization,
                device=device,
                lambda_=args.lambda_,
                is_lambda_update=not args.no_lambda_update,
                lambda_multiplier=args.lambda_multiplier,
                lambda_check_interval=args.lambda_check_interval,
                max_iteration=args.max_iteration,
            )
            with torch.no_grad():
                init_norm = torch.linalg.vector_norm((initialization - clean).flatten(1), dim=1)
                rest_norm = torch.linalg.vector_norm((recovered - clean).flatten(1), dim=1)
                values = {
                    "recon_error": recon_error,
                    "init_error": init_metrics["init_error"],
                    "restoration_error": normalized_l2(recovered, clean),
                    "recovery_gain": rest_norm / init_norm.clamp_min(1e-12),
                    "fixed_point_residual": normalized_l2(model(recovered), recovered),
                    "realized_support_ratio": support,
                    "init_distance_ratio": init_metrics["init_distance_ratio"],
                    "clip_fraction": init_metrics["clip_fraction"],
                    "perturbation_cosine": init_metrics["perturbation_cosine"],
                }
            state = accumulators[repeat]
            state["count"] += take
            for name, value in values.items():
                state[name] += value.sum().item()
        seen += take

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if seen == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    rows = []
    for repeat, state in enumerate(accumulators):
        row = {
            "bottleneck_channels": bottleneck,
            "bottleneck_dimension": 16 * bottleneck,
            "model_seed": model_seed,
            "init_seed": args.init_seed + repeat,
            "num_samples": state["count"],
            "alpha": args.alpha,
            "beta": args.beta,
            "lambda": args.lambda_,
            "parameter_count": parameter_count,
            "checkpoint_path": str(checkpoint.relative_to(PROJECT_ROOT)) if checkpoint.is_relative_to(PROJECT_ROOT) else str(checkpoint),
        }
        row.update({name: state[name] / state["count"] for name in METRICS})
        row["reconstruction_amplification"] = row["restoration_error"] / max(row["recon_error"], 1e-12)
        rows.append(row)
    return rows


def summarize(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Require nonempty capacity records.")
    keys = ["bottleneck_channels", "bottleneck_dimension", "model_seed"]
    if (frame.bottleneck_channels <= 0).any() or not frame.bottleneck_dimension.eq(16 * frame.bottleneck_channels).all():
        raise ValueError("Bottleneck dimensions must equal 16 times their positive channel counts.")
    if frame.duplicated(keys + ["init_seed"]).any():
        raise ValueError("Duplicate model/initialization records would bias the summaries.")
    if frame.num_samples.nunique() != 1 or (frame.num_samples <= 0).any():
        raise ValueError("All runs must evaluate the same nonempty image set size.")
    if not np.isfinite(frame[list(METRICS)].to_numpy()).all() or (frame.recon_error <= 0).any():
        raise ValueError("Metrics must be finite and reconstruction error positive for the ratio.")
    if frame.groupby(keys).init_seed.apply(lambda seeds: tuple(sorted(seeds))).nunique() != 1:
        raise ValueError("All models must share the same initialization seeds.")
    if frame.groupby("bottleneck_channels").model_seed.apply(lambda seeds: tuple(sorted(set(seeds)))).nunique() != 1:
        raise ValueError("All capacities must share the same model seeds.")
    for name in ("alpha", "beta", "lambda"):
        if name in frame and (not np.isfinite(frame[name]).all() or frame[name].nunique() != 1):
            raise ValueError(f"All records must share a finite {name}.")
    frame.to_csv(output_dir / "run_metrics.csv", index=False)
    numeric = list(METRICS) + ["reconstruction_amplification"]
    seed_frame = frame.groupby(keys, as_index=False)[list(METRICS)].mean()
    # R is a ratio of image-averaged errors per model, not an average of per-image ratios.
    seed_frame["reconstruction_amplification"] = seed_frame.restoration_error / seed_frame.recon_error
    seed_frame.to_csv(output_dir / "seed_summary.csv", index=False)

    summaries = []
    for (channels, dimension), group in seed_frame.groupby(
        ["bottleneck_channels", "bottleneck_dimension"]
    ):
        row = {
            "bottleneck_channels": channels,
            "bottleneck_dimension": dimension,
            "num_model_seeds": len(group),
        }
        for name in numeric:
            values = group[name].tolist()
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{name}_mean"] = statistics.mean(values)
            row[f"{name}_std"] = std
            row[f"{name}_ci95"] = 1.96 * std / math.sqrt(len(values))
        summaries.append(row)
    result = pd.DataFrame(summaries).sort_values("bottleneck_dimension")
    result.to_csv(output_dir / "capacity_summary.csv", index=False)
    return result


# Plot capacity diagnostics and a compact error comparison.
def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    summary = summary.sort_values("bottleneck_dimension")
    positions = summary["bottleneck_dimension"].to_numpy()
    tick_labels = summary["bottleneck_dimension"].astype(int).astype(str)
    style = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        # Set text sizes for the compact layout.
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.labelweight": "normal",
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    series = (
        (0, "recon_error", r"$E_{\mathrm{recon}}$", "#D55E00", "-", "o"),
        (0, "restoration_error", r"$E_{\mathrm{rest}}$", "#0072B2", "--", "s"),
        (1, "reconstruction_amplification", r"$\xi$", "#009E73", "-.", "^"),
        (1, "recovery_gain", r"$G_{\mathrm{init}}$", "#CC79A7", "--", "D"),
    )
    outputs = ((path, 2, 7.16), (path.with_name("fig_capacity_tradeoff.png"), 1, 3.5))
    with plt.rc_context(style):
        for output, panels, width in outputs:
            fig, axes = plt.subplots(1, panels, figsize=(width, 2.45), constrained_layout=True)
            axes = np.atleast_1d(axes)
            ratio_axis = axes[0].twinx() if panels == 1 else axes[1]
            handles = []
            for panel, name, label, color, linestyle, marker in series:
                if panels == 1 and name == "recovery_gain":
                    continue
                axis = axes[0] if panel == 0 else ratio_axis
                handle = axis.errorbar(
                    positions,
                    summary[f"{name}_mean"],
                    yerr=summary[f"{name}_std"],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.3,
                    markersize=3.5,
                    capsize=2,
                    elinewidth=0.8,
                    label=label,
                )
                handles.append(handle)
            axes[0].set_ylabel(r"Normalized $\ell_2$ error")
            ratio_axis.axhline(1.0, color="#009E73", linestyle=":", linewidth=0.7, alpha=0.6, zorder=0)
            # Fit the full error-bar extents with modest padding; a zero baseline is unnecessary for lines.
            axes[0].margins(y=0.08)
            ratio_axis.margins(y=0.08)
            axes[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
            if panels == 2:
                axes[0].set_title("(a) Reconstruction and restoration", loc="left", fontweight="bold")
                axes[1].set_title("(b) Restoration diagnostics", loc="left", fontweight="bold")
                axes[1].set_ylabel("Ratio")
            else:
                ratio_axis.set_ylabel(r"Error ratio $\xi=E_{\mathrm{rest}}/E_{\mathrm{recon}}$", color="#009E73")
                ratio_axis.tick_params(axis="y", colors="#009E73")
                ratio_axis.spines["right"].set_color("#009E73")
                ratio_axis.spines["top"].set_visible(False)
                ratio_axis.grid(False)
                axes[0].legend(handles=handles, labels=[r"$E_{\mathrm{recon}}$", r"$E_{\mathrm{rest}}$", r"$\xi$ (right)"],
                               loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                               frameon=False, handlelength=2, columnspacing=1.1)
            for axis in axes:
                axis.set_xlabel(r"Bottleneck dimension $d_b$ (log scale)")
                axis.set_xscale("log", base=2)
                axis.set_xticks(positions, tick_labels)
                axis.grid(axis="y", color="0.88", linewidth=0.5)
                if panels == 2:
                    axis.legend(frameon=False)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
            fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-from", help="Rebuild summaries and figures from run_metrics.csv without inference.")
    parser.add_argument(
        "--bottleneck-channels", type=int, nargs="+", default=list(FORMAL_BOTTLENECK_CHANNELS)
    )
    parser.add_argument("--model-seeds", type=int, nargs="+", default=list(FORMAL_MODEL_SEEDS))
    parser.add_argument("--alpha", type=float, default=FORMAL_ALPHA, help="Frozen initialization interpolation.")
    parser.add_argument("--beta", type=float, default=FORMAL_BETA, help="Frozen orthogonal perturbation scale.")
    parser.add_argument("--init-seed", type=int, default=4200)
    parser.add_argument("--init-repeats", type=int, default=2)
    parser.add_argument("--corruption-ratio", type=float, default=0.1)
    parser.add_argument("--corruption-seed", type=int, default=42)
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=FORMAL_LAMBDA,
        help="Fixed MNIST sparse-corruption weight inherited from the main experiment.",
    )
    parser.add_argument("--max-iteration", type=int, default=100)
    parser.add_argument("--lambda-multiplier", type=float, default=0.8)
    parser.add_argument("--lambda-check-interval", type=int, default=10)
    parser.add_argument("--no-lambda-update", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default=FORMAL_SPLIT)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--train-subset-size", type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-root", default="checkpoints/capacity")
    parser.add_argument("--output-root", default="outputs/capacity_geometry")
    training = parser.add_mutually_exclusive_group()
    training.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoints; fail if weights are missing.")
    training.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    # Fail before data loading or training when a numerical override is invalid.
    for name in ("alpha", "beta", "corruption_ratio", "lambda_", "lambda_multiplier"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite.")
    if not args.bottleneck_channels or any(value <= 0 for value in args.bottleneck_channels):
        raise ValueError("Bottleneck channels must be positive.")
    if len(set(args.bottleneck_channels)) != len(args.bottleneck_channels):
        raise ValueError("Bottleneck channels must be unique.")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        raise ValueError("Model seeds must be unique.")
    if not args.model_seeds:
        raise ValueError("At least one model seed is required.")
    if any(not 0 <= seed < 2**32 for seed in args.model_seeds):
        raise ValueError("Model seeds must be between zero and 2**32 - 1.")
    if not 0 <= args.init_seed < 2**63 - args.init_repeats or args.corruption_seed < 0:
        raise ValueError("Initialization and corruption seeds must be nonnegative and in range.")
    if not 0.0 < args.alpha <= 1.0 or args.beta < 0.0:
        raise ValueError("Require 0 < alpha <= 1 and beta >= 0.")
    if not 0.0 < args.corruption_ratio <= 1.0 or args.lambda_ <= 0.0:
        raise ValueError("Corruption ratio and lambda must be positive.")
    if args.init_repeats <= 0 or args.max_iteration <= 0:
        raise ValueError("Initialization repeats and max iteration must be positive.")
    if min(args.batch_size, args.train_batch_size, args.epochs, args.lambda_check_interval) <= 0:
        raise ValueError("Batch sizes, epochs, and lambda check interval must be positive.")
    if args.lambda_multiplier <= 0.0 or args.num_workers < 0:
        raise ValueError("lambda multiplier must be positive and num workers non-negative.")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("num_samples must be positive or omitted.")
    if args.train_subset_size is not None and args.train_subset_size <= 0:
        raise ValueError("train_subset_size must be positive or omitted.")


def _require_writable_results(output_dir: Path, overwrite: bool) -> None:
    names = ("run_metrics.csv", "seed_summary.csv", "capacity_summary.csv", "capacity_tradeoff.png",
             "capacity_tradeoff.pdf", "fig_capacity_tradeoff.png", "fig_capacity_tradeoff.pdf")
    existing = next((output_dir / name for name in names if (output_dir / name).exists()), None)
    if existing is not None and not overwrite:
        raise FileExistsError(f"{existing} exists; pass --overwrite-results to replace it.")


# The run directory is keyed by the complete resolved formal/override configuration.
def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved = {key: value for key, value in vars(args).items() if key != "plot_from"} | {
        "quick": False,
        "device": str(device),
        "model_config": "resae_3down",
        "training_profile": "jinit_direct_branch",
        "solver": "direct",
    }
    hash_config = {
        key: value
        for key, value in resolved.items()
        if key not in {"overwrite_results", "force_retrain", "output_root"}
    }
    config_hash = hashlib.sha256(json.dumps(hash_config, sort_keys=True).encode()).hexdigest()[:12]
    output_dir = PROJECT_ROOT / args.output_root / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_writable_results(output_dir, args.overwrite_results)
    OmegaConf.save(OmegaConf.create(resolved), output_dir / "resolved_config.yaml")

    loader = get_degradation_dataloader(
        "mnist",
        args.split,
        {
            "mode": "sparse",
            "ratio": args.corruption_ratio,
            "value_mode": "salt_pepper",
            "per_channel": True,
            "seed": args.corruption_seed,
        },
        args.batch_size,
        args.num_workers,
        image_size=28,
        shuffle=False,
    )
    rows = []
    for bottleneck in args.bottleneck_channels:
        for model_seed in args.model_seeds:
            model, checkpoint = prepare_model(
                args, bottleneck, model_seed, device
            )
            rows.extend(
                evaluate_model(
                    args,
                    loader,
                    model,
                    bottleneck,
                    model_seed,
                    checkpoint,
                    device,
                )
            )
    summary = summarize(rows, output_dir)
    plot_summary(summary, output_dir / "capacity_tradeoff.png")
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    # Replot archived runs into a separate output directory without loading models or datasets.
    if args.plot_from:
        source = (PROJECT_ROOT / args.plot_from).resolve()
        output_dir = (PROJECT_ROOT / args.output_root).resolve()
        if source.parent == output_dir:
            raise ValueError("Use a separate output directory to preserve the source records.")
        _require_writable_results(output_dir, args.overwrite_results)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = summarize(pd.read_csv(source).to_dict("records"), output_dir)
        plot_summary(summary, output_dir / "capacity_tradeoff.png")
    else:
        output_dir = run(args)
    print(f"Saved exp_recon_recovery outputs to: {output_dir}")


if __name__ == "__main__":
    main()
