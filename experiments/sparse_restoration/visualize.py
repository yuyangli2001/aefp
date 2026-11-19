"""Select and plot a representative CelebA-HQ-64 sparse-restoration example."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recovery.cache import (
    DEFAULT_METHODS,
    RecoveryCacheConfig as ExperimentConfig,
    load_or_create_method_cache,
    load_or_create_pairs_cache,
    plain,
    resolve_device,
)
from utils.utils_metrics import compute_psnr, compute_ssim
from utils.figure_layout import panel_heading
from recovery.registry import ADAPTERS, canonical_method


DATASET, RATIO, SEED, NUM_CANDIDATES = "celeba-hq-64", 0.2, 42, 100
METHODS = DEFAULT_METHODS
PAPER_MEANS = {
    "ae": {"psnr": 17.57, "ssim": 0.4041},
    "rpca": {"psnr": 24.73, "ssim": 0.7978},
    "daep": {"psnr": 23.81, "ssim": 0.7593},
    "dpir": {"psnr": 32.10, "ssim": 0.9329},
    "dps": {"psnr": 33.13, "ssim": 0.9317},
    "fire-single": {"psnr": 32.94, "ssim": 0.9310},
    "fire-multi": {"psnr": 33.25, "ssim": 0.9360},
    "irae": {"psnr": 34.95, "ssim": 0.9688},
}
COMPARISON_PANELS = (
    ("clean", "Clean"), ("degraded", "Observation"), ("rpca", "PCP"),
    ("ae", "FF-AE"), ("daep", "DAEP"), ("dpir", "DPIR"),
    ("dps", "DPS"), ("fire-single", "FiRe-S"),
    ("fire-multi", "FiRe-M"), ("irae", "AEFP"),
)
PANELS = tuple(panel for panel in COMPARISON_PANELS if panel[0] in ("clean", "degraded", *METHODS))
SCORE_FORMULA = "mean over methods of (abs(psnr-target)/target + abs(ssim-target)/target) / 2"


def per_image_metrics(clean: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, np.ndarray]:
    if clean.shape != reconstruction.shape or clean.ndim != 4:
        raise ValueError("clean and reconstruction must share BCHW shape")
    clean_np = clean.detach().cpu().permute(0, 2, 3, 1).numpy()
    recon_np = reconstruction.detach().cpu().permute(0, 2, 3, 1).numpy()
    return {
        "psnr": np.asarray([compute_psnr(x, y) for x, y in zip(clean_np, recon_np)]),
        "ssim": np.asarray([compute_ssim(x, y) for x, y in zip(clean_np, recon_np)]),
    }


# Rank candidates jointly across every recovery method; dataset index breaks exact ties.
def select_representative(
    metrics: Mapping[str, Mapping[str, np.ndarray]], indices: Sequence[int] | torch.Tensor,
) -> tuple[int, np.ndarray]:
    index_values = [int(value) for value in indices]
    if not metrics or not set(metrics).issubset(PAPER_MEANS) or not index_values:
        raise ValueError('metrics must contain registered paper methods and at least one sample')
    if min(index_values) < 0 or len(set(index_values)) != len(index_values):
        raise ValueError('Candidate indices must be unique and nonnegative.')
    scores = np.zeros(len(index_values), dtype=np.float64)
    for method in metrics:
        target = PAPER_MEANS[method]
        psnr = np.asarray(metrics[method]["psnr"], dtype=np.float64)
        ssim = np.asarray(metrics[method]["ssim"], dtype=np.float64)
        if psnr.shape != scores.shape or ssim.shape != scores.shape or not np.isfinite(psnr).all() or not np.isfinite(ssim).all():
            raise ValueError(f"Invalid metric shape for {method}")
        scores += np.abs(psnr - target["psnr"]) / target["psnr"]
        scores += np.abs(ssim - target["ssim"]) / target["ssim"]
    scores /= 2 * len(metrics)
    position = min(range(len(index_values)), key=lambda pos: (scores[pos], index_values[pos]))
    return position, scores


def save_comparison(
    path: Path,
    images: Mapping[str, torch.Tensor],
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    position: int,
    panels: Sequence[tuple[str, str]] = PANELS,
    zoom_box: tuple[int, int, int, int] | None = (38, 9, 52, 23),
) -> None:
    # Reject malformed panels before opening a figure or saving partial output.
    if not panels or len({key for key, _ in panels}) != len(panels):
        raise ValueError('Comparison panels must be nonempty and unique.')
    reference = images['clean']
    if reference.ndim != 4 or not 0 <= position < len(reference):
        raise ValueError('Invalid selected image position.')
    for key, _ in panels:
        if images[key].shape != reference.shape or not torch.isfinite(images[key]).all():
            raise ValueError('Comparison images must have matching finite BCHW shapes.')
    if zoom_box is not None:
        x0, y0, x1, y1 = zoom_box
        if not (0 <= x0 < x1 <= reference.shape[-1] and 0 <= y0 < y1 <= reference.shape[-2]):
            raise ValueError('Zoom box is outside the image.')
    # Arrange up to five comparison panels per row.
    columns = min(5, len(panels))
    rows = (len(panels) + columns - 1) // columns
    # Stack bold method names above regular PSNR/SSIM values, preserving the dB unit.
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(rows, columns, figsize=(7.16, 3.3 if rows == 2 else 1.7), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (key, label) in zip(axes.flat, panels):
            image = images[key][position].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            cmap = "gray" if image.shape[-1] == 1 else None
            axis.imshow(image.squeeze(), cmap=cmap, vmin=0.0, vmax=1.0, interpolation="none")
            metric = ""
            if key != "clean":
                metric = f"{metrics[key]['psnr'][position]:.2f} dB / {metrics[key]['ssim'][position]:.4f}"
            panel_heading(axis, label, metric, multiline=True)
            # Every panel shows the same pixel crop without enhancement or smoothing.
            if zoom_box is not None:
                x0, y0, x1, y1 = zoom_box
                if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                    raise ValueError("Zoom box is outside the image.")
                color = "#D62728"
                axis.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                         fill=False, edgecolor=color, linewidth=0.8))
                inset = axis.inset_axes((0.025, 0.025, 0.40, 0.40))
                inset.imshow(image[y0:y1, x0:x1].squeeze(), cmap=cmap,
                             vmin=0.0, vmax=1.0, interpolation="none")
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(0.8)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.91,
                            wspace=0.055, hspace=0.28)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def _checkpoint_records(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifests = ([resolved["checkpoint"]] if "checkpoint" in resolved else [])
    manifests += list(resolved.get("checkpoints", ()))
    return [
        {key: plain(manifest.get(key)) for key in ("path", "stage_paths")
         if manifest.get(key) is not None}
        for manifest in manifests
    ]


def build_audit(
    indices: Sequence[int] | torch.Tensor,
    scores: np.ndarray,
    position: int,
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected_index = int(indices[position])
    return {
        "dataset": DATASET, "split": "test", "ratio": RATIO, "corruption_seed": SEED,
        "candidate_range": [0, len(indices) - 1],
        "method_scope": list(records), "reproduces_full_comparison_figure": False,
        "full_method_coverage": set(records) == set(PAPER_MEANS),
        "selection_targets": "historical paper means retained from the original visualization",
        "candidate_indices": [int(value) for value in indices],
        "selection_formula": SCORE_FORMULA, "selected_index": selected_index,
        "selected_score": float(scores[position]),
        "candidate_scores": [
            {"index": int(index), "score": float(score)} for index, score in zip(indices, scores)
        ],
        "methods": {
            method: {
                "target_mean": PAPER_MEANS[method],
                "selected_metrics": {name: float(values[position]) for name, values in metrics[method].items()},
                "preset_key": records[method]["resolved_config"]["preset_key"],
                "checkpoints": _checkpoint_records(records[method]["resolved_config"]),
            }
            for method in records
        },
    }


# Render archived recoveries as recorded, independently of current inference presets.
def replot_from_cache(source: Path, output_dir: Path, sample_index: int,
                      methods: Sequence[str]) -> Path:
    paths = {name: source / f"{name}.pt" for name in ("test_pairs", *methods)}
    records = {name: torch.load(path, map_location="cpu", weights_only=False)
               for name, path in paths.items()}
    pairs = records.pop("test_pairs")
    matches = (pairs["indices"] == sample_index).nonzero().flatten()
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cached sample with index {sample_index}.")
    images = {"clean": pairs["clean"], "degraded": pairs["degraded"]}
    for name, record in records.items():
        if not torch.equal(record["indices"], pairs["indices"]) or any(
            record[key] != pairs[key] for key in ("dataset", "split", "degradation")
        ):
            raise ValueError(f"Cached inputs do not match for {name}.")
        images[name] = record["reconstruction"]
    if any(value.shape != pairs["clean"].shape for value in images.values()):
        raise ValueError("Cached image shapes do not match.")
    metrics = {name: per_image_metrics(images["clean"], value)
               for name, value in images.items() if name != "clean"}
    position = int(matches[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(output_dir / "fig_sparse_restoration_celeba64_r0p2.pdf",
                    images, metrics, position,
                    panels=[panel for panel in COMPARISON_PANELS if panel[0] in images])
    audit = {"source": str(source.resolve()), "sample_index": sample_index,
             "selection": "fixed index", "inference_rerun": False,
             "source_files": {name: str(path.resolve()) for name, path in paths.items()},
             "metrics": {name: {key: float(value[position]) for key, value in pair.items()}
                         for name, pair in metrics.items()}}
    (output_dir / "fig_sparse_metrics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output_dir


def run(args: argparse.Namespace) -> Path:
    if args.plot_from is not None:
        return replot_from_cache(args.plot_from, Path(args.output_dir),
                                 18 if args.sample_index is None else args.sample_index, args.methods)
    device = resolve_device(args.device)
    methods = getattr(args, 'methods', METHODS)
    cfg = ExperimentConfig(
        dataset=DATASET, seed=SEED,
        test_num_samples=None if args.full_test_cache else args.num_candidates, num_workers=0,
        data_root=args.data_root, recovery_cache_root=args.recovery_cache_root,
        recovery_policy=args.recovery_policy,
    )
    pairs, _ = load_or_create_pairs_cache(cfg, RATIO)
    if len(pairs['indices']) < args.num_candidates:
        raise ValueError('Requested display candidates exceed the cached test pool.')
    expected_indices = torch.arange(len(pairs['indices']) if args.full_test_cache else args.num_candidates, dtype=torch.long)
    if not torch.equal(pairs["indices"], expected_indices):
        raise RuntimeError("Expected consecutive candidate indices in dataset order")
    records = {
        method: load_or_create_method_cache(cfg, RATIO, method, pairs, device)[0]
        for method in methods
    }
    # Select display candidates from the exact full-test recoveries used for quantitative evaluation.
    if args.full_test_cache:
        pairs = {key: value[:args.num_candidates] if torch.is_tensor(value) else value for key, value in pairs.items()}
        records = {name: dict(record, reconstruction=record['reconstruction'][:args.num_candidates])
                   for name, record in records.items()}
    clean = pairs["clean"]
    method_metrics = {
        method: per_image_metrics(clean, record["reconstruction"])
        for method, record in records.items()
    }
    observation_metrics = per_image_metrics(clean, pairs["degraded"])
    position, scores = select_representative(method_metrics, pairs["indices"])
    # Use a fixed sample for full-test displays unless another index is requested.
    selected = getattr(args, 'sample_index', None)
    if selected is None and args.full_test_cache:
        selected = 18
    if selected is not None:
        matches = (pairs['indices'] == selected).nonzero().flatten()
        if len(matches) != 1:
            raise ValueError(f'Sample {selected} is outside the requested display candidates.')
        position = int(matches[0])
    images = {"clean": clean, "degraded": pairs["degraded"]} | {
        method: record["reconstruction"] for method, record in records.items()
    }
    display_metrics = {"degraded": observation_metrics, **method_metrics}
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(output_dir / "fig_sparse_restoration_celeba64_r0p2.pdf",
                    images, display_metrics, position,
                    panels=[panel for panel in COMPARISON_PANELS if panel[0] in images])
    audit = build_audit(pairs["indices"], scores, position, method_metrics, records)
    audit.update(selection_mode='fixed_index' if selected is not None else 'representative_score',
                 selection_record=None, requested_index=selected,
                 recovery_scope='full_test' if args.full_test_cache else 'candidate_prefix')
    (output_dir / "selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device")
    parser.add_argument("--plot-from", type=Path, help="Plot a recorded recovery-cache directory without inference or preset migration.")
    parser.add_argument('--methods', nargs='+', type=canonical_method, choices=list(ADAPTERS), default=list(METHODS), help='Recovery methods; aefp is the public alias for irae.')
    parser.add_argument('--sample-index', type=int, help='Fixed test index to display; full-test mode defaults to index 18.')
    parser.add_argument("--num-candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument('--full-test-cache', action='store_true', help='Read full-test caches and then select the display candidates.')
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", default="outputs/sparse_restoration/visual")
    parser.add_argument("--recovery-cache-root", default="outputs/cache/recovery")
    parser.add_argument("--recovery-policy", choices=("auto", "cache-only", "refresh"), default="auto")
    args = parser.parse_args(argv)
    if args.num_candidates <= 0 or (args.sample_index is not None and args.sample_index < 0):
        parser.error('Require positive --num-candidates and nonnegative --sample-index.')
    if len(set(args.methods)) != len(args.methods):
        parser.error('--methods must contain unique canonical methods.')
    return args


if __name__ == "__main__":
    print(f"Saved sparse-restoration visualization to {run(parse_args())}")
