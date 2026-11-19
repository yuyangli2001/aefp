"""Plan and resume the ablation cells owned by the auxiliary experiment."""

from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import torch
from recovery.batch import _iter_loader_batches, build_batch_degradation
from recovery.adapters.ae import _load_models
from recovery.registry import resolve_config
from utils.utils_dataloader import get_degradation_dataloader
from .artifacts import _read_csv, _write_csv, _write_json
from .variants import (
    EDGE_VARIANTS, FULL_TEST_SIZE, FULL_VARIANTS, RATIOS, TABLE_VARIANTS, MetricAccumulator,
    VARIANT_LABELS, _degradation, _ratio_index, run_batch_variants,
)


def variants_for_ratio(ratio: float) -> tuple[str, ...]:
    """Use nine variants at 0.2 and the three 300-step variants at 0.1 and 0.3."""
    return TABLE_VARIANTS if _ratio_index(ratio) == 1 else EDGE_VARIANTS


def reported_result_source(ratio: float, variant: str) -> str | None:
    if variant in {"A0", "A1", "A6"} or (_ratio_index(ratio) == 1 and variant in {"A4", "A5"}):
        return "experiments.sparse_restoration.infer --include-stages"
    return None


def build_plan(datasets: Sequence[str], ratios: Sequence[float], variants: Sequence[str] | None,
               split: str, num_samples: int | None) -> list[dict[str, Any]]:
    """Resolve every cell before touching data or checkpoints."""
    ratios = [RATIOS[_ratio_index(ratio)] for ratio in ratios]
    for name, values in (("datasets", datasets), ("ratios", ratios), ("variants", variants)):
        if values is not None and (not values or len(set(values)) != len(values)):
            raise ValueError(f"Require unique, nonempty {name}.")
    if split not in ("val", "test") or (num_samples is not None and (isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0)):
        raise ValueError("Require val/test split and a positive sample count or None.")
    if variants is not None and not set(variants) <= set(FULL_VARIANTS):
        raise ValueError("Unknown ablation variants.")
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            _, _, _, config = resolve_config(dataset, _degradation(ratio), "irae")
            selected = tuple(variants) if variants else variants_for_ratio(ratio)
            for variant in selected:
                source_path = reported_result_source(ratio, variant)
                rows.append({
                    "dataset": dataset, "ratio": ratio, "variant": variant,
                    "label": VARIANT_LABELS[variant], "split": split,
                    "num_samples": num_samples if num_samples is not None else (FULL_TEST_SIZE[dataset] if split == "test" else None),
                    "action": "external" if source_path else "compute",
                    "source_experiment": "E06" if source_path else "E05",
                    "source_command": source_path or "",
                    "preset_key": config["preset_key"], "model_id": config["model_id"],
                    "checkpoint_dir": config["checkpoint"]["path"],
                    "stage_paths": list(config["checkpoint"]["stage_paths"]),
                    "lambda": config["lambda"],
                })
    return rows


def run_scenario(dataset: str, ratio: float, variants: Sequence[str], split: str,
                 num_samples: int | None, num_workers: int, device: torch.device,
                 data_root: Path | None) -> list[dict[str, Any]]:
    """Load one frozen model triplet and stream one dataset/scenario."""
    _, _, degradation_cfg, config = resolve_config(dataset, _degradation(ratio), "irae")
    if tuple(config["jinit_stage_iters"]) != (100, 100, 100):
        raise ValueError("The frozen auxiliary ablation requires observation init and 100/100/100 stages.")
    models = _load_models(config, str(device), "jinit")
    loader_dataset, image_size = ("celeba-hq-256", 64) if dataset == "celeba-hq-64" else (dataset, None)
    loader = get_degradation_dataloader(
        dataset=loader_dataset, split=split, degradation_cfg=degradation_cfg,
        batch_size=int(config["execution_batch_size"]), num_workers=num_workers,
        root=data_root, image_size=image_size, shuffle=False, drop_last=False,
    )
    accumulator = MetricAccumulator(variants)
    for batch in _iter_loader_batches(loader, num_samples=num_samples, device=device):
        degradation = build_batch_degradation(batch["mode"], batch["params"])
        outputs = run_batch_variants(batch["degraded"], degradation, models, set(variants), config, device)
        accumulator.update(batch["clean"], outputs)
    expected = num_samples if num_samples is not None else len(loader.dataset)
    if accumulator.num_samples != expected:
        raise ValueError(f"Expected {expected} images, evaluated {accumulator.num_samples}.")
    return [
        _result_row(dataset, ratio, name, psnr, ssim, split, accumulator.num_samples,
                    "computed", "", config)
        for name, (psnr, ssim) in accumulator.finalize().items()
    ]


def execute(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resume rows computed by this runner and skip externally reported cells."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    plan = build_plan(args.datasets, args.ratios, args.variants, args.split, args.num_samples)
    owned = {_row_key(row): row for row in plan if row['action'] == 'compute'}
    settings_path = output_dir / "run_config.json"
    settings = dict(split=args.split, num_samples=args.num_samples,
                    data_root=str(Path(args.data_root).resolve()) if args.data_root is not None else None)
    saved_settings = None
    if settings_path.exists():
        try:
            saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except ValueError:
            if not args.force:
                raise
        if not args.force and saved_settings != settings:
            raise ValueError("Evaluation context changed; use --force or a new output directory.")
    cached = _read_csv(metrics_path)
    existing = [] if args.force and saved_settings != settings else [
        row for row in cached if row.get("source") == "computed"
        and row["variant"] in FULL_VARIANTS and not reported_result_source(float(row["ratio"]), row["variant"])
    ]
    if existing and not settings_path.exists():
        raise ValueError("Existing metrics have no run_config.json; use --force or a new output directory.")
    rows = {_row_key(row): row for row in existing}
    if len(rows) != len(existing) and not args.force:
        raise ValueError("Duplicate cached ablation rows; use --force or a new output directory.")
    # Preserve completed results outside the current selection.
    missing: dict[tuple[str, float], list[str]] = defaultdict(list)
    for key, item in owned.items():
        if key in rows and not args.force:
            _validate_row(rows[key], item)
        else:
            missing[key[:2]].append(key[2])
    if list(rows.values()) != cached or not metrics_path.exists():
        _write_csv(metrics_path, list(rows.values()))
    if saved_settings != settings:
        _write_json(settings_path, settings)
    _write_json(output_dir / "plan.json", plan)
    device = torch.device(args.device)
    for (dataset, ratio), variants in missing.items():
        computed = run_scenario(dataset, ratio, variants, args.split, args.num_samples,
                                args.num_workers, device, args.data_root)
        expected = {(dataset, ratio, name, args.split) for name in variants}
        if {_row_key(row) for row in computed} != expected or len(computed) != len(expected):
            raise ValueError("Scenario returned missing or duplicate ablation variants.")
        for row in computed:
            _validate_row(row, owned[_row_key(row)])
        rows.update((_row_key(row), row) for row in computed)
        _write_csv(metrics_path, list(rows.values()))
    return [row for key, row in rows.items() if key in owned]


def _result_row(dataset: str, ratio: float, variant: str, psnr: float, ssim: float,
                split: str, num_samples: int, source: str, source_path: str,
                config: Mapping[str, Any]) -> dict[str, Any]:
    return {"dataset": dataset, "ratio": ratio, "variant": variant,
            "label": VARIANT_LABELS[variant], "psnr": psnr, "ssim": ssim,
            "split": split, "num_samples": num_samples, "source": source,
            "source_path": source_path, "preset_key": config["preset_key"],
            "model_id": config["model_id"],
            "checkpoint_dir": config["checkpoint"]["path"], "lambda": config["lambda"]}


def _row_key(row: Mapping[str, Any]) -> tuple[str, float, str, str]:
    return (str(row["dataset"]), float(row["ratio"]), str(row["variant"]), str(row["split"]))


# Validate identity and actual sample coverage instead of trusting a matching CSV key.
def _validate_row(row: Mapping[str, Any], planned: Mapping[str, Any]) -> None:
    if _row_key(row) != _row_key(planned) or row.get("source") != "computed":
        raise ValueError("Ablation result identity does not match the plan.")
    if any(row.get(name) in (None, "") for name in ("num_samples", "psnr", "ssim", "lambda")):
        raise ValueError("Incomplete ablation metrics; use --force.")
    count = int(row["num_samples"])
    if count <= 0 or (planned["num_samples"] is not None and count != planned["num_samples"]):
        raise ValueError("Ablation sample count does not match the plan; use --force.")
    for name in ("preset_key", "model_id", "checkpoint_dir"):
        if row.get(name) != planned[name]:
            raise ValueError(f"Cached {name} does not match the plan; use --force.")
    if (not all(math.isfinite(float(row[name])) for name in ("psnr", "ssim", "lambda"))
            or not math.isclose(float(row["lambda"]), planned["lambda"], rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError("Nonfinite metrics or a changed lambda; use --force.")
