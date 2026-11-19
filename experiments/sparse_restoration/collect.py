"""
Collect Exp1 Hydra results after runs finish.

The experiment script writes metrics inside each run directory. This script scans
those immutable per-run files and creates aggregate CSV files for comparison.

Examples:
  python experiments/sparse_restoration/collect.py outputs/sparse_restoration/multirun/<date>/<time>
  python experiments/sparse_restoration/collect.py outputs/sparse_restoration/multirun

By default, aggregate CSVs are written under the input directory:
  outputs/sparse_restoration/multirun/<date>/<time>/aggregate
  outputs/sparse_restoration/multirun/aggregate
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


FINAL_FIELDS = [
    "sweep_dir",
    "job_num",
    "run_dir",
    "overrides",
    "dataset",
    "model_tag",
    "model_variant",
    "degradation_mode",
    "degradation_params",
    "inference_split",
    "solver",
    "lambda",
    "init_method",
    "lambda_multiplier",
    "lambda_check_interval",
    "backtracking_rule",
    "sufficient_decrease_constant",
    "is_lambda_update",
    "jinit_stage_iters",
    "observation_psnr",
    "observation_ssim",
    "init_psnr",
    "init_ssim",
    "processed_init_stage",
    "processed_init_label",
    "processed_init_psnr",
    "processed_init_ssim",
    "lambda_init_gamma",
    "lambda_init",
    "aux_lambda_init",
    "num_samples",
    "final_stage",
    "final_label",
    "final_psnr",
    "final_ssim",
    "checkpoint_dir",
    "ckpt_paths",
    "train_psnr_drops",
    "selected_jinit_ckpts",
    "selected_jinit_stages",
    "actual_solver_stage_count",
]

BEST_FIELDS = [*FINAL_FIELDS, "group_rank"]

BACKTRACKING_COMPARISON_FIELDS = [
    "comparison_status",
    "sweep_dir",
    "dataset",
    "model_tag",
    "degradation_mode",
    "degradation_params",
    "inference_split",
    "solver",
    "lambda",
    "init_method",
    "lambda_multiplier",
    "lambda_check_interval",
    "sufficient_decrease_constant",
    "is_lambda_update",
    "jinit_stage_iters",
    "num_samples",
    "monotone_psnr",
    "monotone_ssim",
    "sufficient_decrease_psnr",
    "sufficient_decrease_ssim",
    "delta_psnr",
    "delta_ssim",
    "monotone_run_dir",
    "sufficient_decrease_run_dir",
]

STAGE_FIELDS = [
    "sweep_dir",
    "job_num",
    "run_dir",
    "overrides",
    "dataset",
    "model_tag",
    "model_variant",
    "degradation_mode",
    "degradation_params",
    "inference_split",
    "solver",
    "lambda",
    "init_method",
    "lambda_multiplier",
    "lambda_check_interval",
    "backtracking_rule",
    "sufficient_decrease_constant",
    "is_lambda_update",
    "lambda_init_gamma",
    "lambda_init",
    "aux_lambda_init",
    "jinit_stage_iters",
    "stage",
    "label",
    "psnr",
    "ssim",
    "checkpoint_dir",
    "ckpt_paths",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: List[Dict[str, Any]], preferred_fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in preferred_fields} for row in rows)


def as_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def read_overrides(run_dir: Path) -> str:
    path = run_dir / ".hydra" / "overrides.yaml"
    if not path.exists():
        return ""

    overrides = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            overrides.append(line[2:])
    return " ".join(overrides)


def run_metadata(run_dir: Path) -> Dict[str, Any]:
    sweep_dir = run_dir.parent if (run_dir.parent / "multirun.yaml").exists() else None
    return {
        "sweep_dir": str(sweep_dir) if sweep_dir else "",
        "job_num": run_dir.name if sweep_dir else "",
        "run_dir": str(run_dir),
        "overrides": read_overrides(run_dir),
    }


def summary_fields(run_dir: Path) -> Dict[str, Any]:
    metrics_dir = run_dir / "metrics"
    summary = read_json(metrics_dir / "recovery_summary.json")
    keys = [
        "dataset",
        "model_tag",
        "model_variant",
        "degradation_mode",
        "degradation_params",
        "inference_split",
        "solver",
        "lambda",
        "init_method",
        "inpainting_lambda",
        "deblur_lambda",
        "deblur_noise_std_mode",
        "lambda_multiplier",
        "lambda_check_interval",
        "backtracking_rule",
        "sufficient_decrease_constant",
        "is_lambda_update",
        "lambda_decay",
        "inpainting_constraint",
        "inpainting_data_penalty",
        "inpainting_t",
        "jinit_stage_iters",
        "observation_psnr",
        "observation_ssim",
        "init_psnr",
        "init_ssim",
        "processed_init_stage",
        "processed_init_label",
        "processed_init_psnr",
        "processed_init_ssim",
        "lambda_init_gamma",
        "lambda_init",
        "aux_lambda_init",
        "num_samples",
        "final_stage",
        "final_label",
        "final_psnr",
        "final_ssim",
        "checkpoint_dir",
        "ckpt_paths",
        "train_psnr_drops",
        "selected_jinit_ckpts",
        "selected_jinit_stages",
        "actual_solver_stage_count",
    ]
    return {key: as_scalar(summary[key]) for key in keys if key in summary}


def final_from_stage_rows(stage_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    final = stage_rows[-1]
    return {
        "dataset": final.get("dataset", ""),
        "model_tag": final.get("model_tag", ""),
        "model_variant": final.get("model_variant", ""),
        "degradation_mode": final.get("degradation_mode", ""),
        "degradation_params": final.get("degradation_params", ""),
        "inference_split": final.get("inference_split", ""),
        "solver": final.get("solver", ""),
        "lambda": final.get("lambda", ""),
        "init_method": final.get("init_method", ""),
        "lambda_multiplier": final.get("lambda_multiplier", final.get("lambda_decay", "")),
        "lambda_check_interval": final.get("lambda_check_interval", ""),
        "backtracking_rule": final.get("backtracking_rule", ""),
        "sufficient_decrease_constant": final.get("sufficient_decrease_constant", ""),
        "is_lambda_update": final.get("is_lambda_update", ""),
        "lambda_init_gamma": final.get("lambda_init_gamma", ""),
        "lambda_init": final.get("lambda", ""),
        "aux_lambda_init": final.get("aux_lambda_init", final.get("lambda_init", "")),
        "inpainting_t": final.get("inpainting_t", ""),
        "jinit_stage_iters": final.get("jinit_stage_iters", ""),
        "processed_init_stage": "",
        "processed_init_label": "",
        "processed_init_psnr": "",
        "processed_init_ssim": "",
        "final_stage": final.get("stage", ""),
        "final_label": final.get("label", ""),
        "final_psnr": final.get("psnr", ""),
        "final_ssim": final.get("ssim", ""),
        "checkpoint_dir": final.get("checkpoint_dir", ""),
        "ckpt_paths": final.get("ckpt_paths", ""),
    }


# Normalize initial-weight meanings in memory; historical result files stay untouched.
def normalize_lambda_fields(row):
    if "aux_lambda_init" not in row:
        row["aux_lambda_init"] = row.get("lambda_init", "")
    row["lambda_init"] = row.get("lambda", "")
    return row


def load_run(run_dir: Path):
    metrics_dir = run_dir / "metrics"
    # Current evaluations publish completion only after all metrics and previews are saved.
    status = read_json(metrics_dir / "recovery_status.json")
    if status and status.get("complete") is not True:
        return None, []
    stage_rows = read_csv(metrics_dir / "recovery_stage_metrics.csv")
    if not stage_rows:
        return None, []

    meta = run_metadata(run_dir)
    summary = summary_fields(run_dir)
    if summary:
        summary = normalize_lambda_fields(summary)
    final_rows = read_csv(metrics_dir / "recovery_final_summary.csv")
    final = final_rows[0] if final_rows else final_from_stage_rows(stage_rows)

    final_row = {}
    final_row.update(meta)
    final_row.update(final)
    final_row.update(summary)

    # Normalize historical mode-specific fields into the unified lambda columns.
    if not final_row.get("lambda"):
        mode = str(final_row.get("degradation_mode", ""))
        if mode == "inpainting":
            final_row["lambda"] = final_row.get(
                "inpainting_lambda", final_row.get("inpainting_data_penalty", "")
            )
        elif mode in {"deblur", "deblurring"}:
            final_row["lambda"] = final_row.get("deblur_lambda", "")
        else:
            final_row["lambda"] = final_row.get("sparse_lambda", "")
    if not final_row.get("lambda_multiplier"):
        final_row["lambda_multiplier"] = final_row.get("lambda_decay", "")

    merged_stage_rows = []
    for row in stage_rows:
        merged = {}
        merged.update(meta)
        merged.update(summary)
        merged.update(row)
        if not merged.get("lambda"):
            mode = str(merged.get("degradation_mode", ""))
            if mode == "inpainting":
                merged["lambda"] = merged.get(
                    "inpainting_lambda", merged.get("inpainting_data_penalty", "")
                )
            elif mode in {"deblur", "deblurring"}:
                merged["lambda"] = merged.get("deblur_lambda", "")
            else:
                merged["lambda"] = merged.get("sparse_lambda", "")
        if not merged.get("lambda_multiplier"):
            merged["lambda_multiplier"] = merged.get("lambda_decay", "")
        merged_stage_rows.append(normalize_lambda_fields(merged))

    return normalize_lambda_fields(final_row), merged_stage_rows


def numeric(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def job_sort_value(job_num: Any) -> Tuple[int, Any]:
    job_num = str(job_num)
    return (0, int(job_num)) if job_num.isdigit() else (1, job_num)


def sort_key(row: Dict[str, Any]):
    return (
        str(row.get("sweep_dir", "")),
        job_sort_value(row.get("job_num", "")),
        str(row.get("solver", "")),
        numeric(row, "lambda"),
    )


def group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("degradation_mode", "")),
        str(row.get("degradation_params", "")),
    )


def numeric_sort_value(row: Dict[str, Any], key: str, *, descending: bool = False) -> Tuple[int, float]:
    value = numeric(row, key)
    if math.isnan(value):
        return (1, 0.0)
    return (0, -value if descending else value)


def best_sort_key(row: Dict[str, Any], metric_prefix: str = "final"):
    return (
        numeric_sort_value(row, f"{metric_prefix}_psnr", descending=True),
        numeric_sort_value(row, f"{metric_prefix}_ssim", descending=True),
        str(row.get("solver", "")),
        numeric_sort_value(row, "lambda"),
        numeric_sort_value(row, "lambda_multiplier"),
        str(row.get("init_method", "")),
        job_sort_value(row.get("job_num", "")),
        str(row.get("run_dir", "")),
    )


def collect_best_by_psnr(rows: List[Dict[str, Any]], top_k: int, metric_prefix: str = "final") -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    ranked_rows = []
    for key in sorted(grouped):
        ranked_group = sorted(grouped[key], key=lambda row: best_sort_key(row, metric_prefix))
        for rank, row in enumerate(ranked_group[:top_k], start=1):
            ranked = dict(row)
            ranked["group_rank"] = rank
            ranked_rows.append(ranked)
    return ranked_rows


def collect_best_by_processed_init_psnr(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    return collect_best_by_psnr(rows, top_k, "processed_init")


def collect(root: Path):
    final_rows = []
    stage_rows = []

    for metrics_dir in sorted(root.rglob("metrics")):
        run_dir = metrics_dir.parent
        final_row, run_stage_rows = load_run(run_dir)
        if final_row is None:
            continue
        final_rows.append(final_row)
        stage_rows.extend(run_stage_rows)

    final_rows.sort(key=sort_key)
    stage_rows.sort(key=sort_key)
    return final_rows, stage_rows


# Pair only jobs from the same Hydra sweep so repeated server runs remain independent.
def collect_backtracking_comparisons(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key_fields = [
        "sweep_dir",
        "dataset",
        "model_tag",
        "degradation_mode",
        "degradation_params",
        "inference_split",
        "solver",
        "lambda",
        "init_method",
        "lambda_multiplier",
        "lambda_check_interval",
        "sufficient_decrease_constant",
        "is_lambda_update",
        "jinit_stage_iters",
        "num_samples",
    ]
    grouped: Dict[Tuple[str, ...], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        rule = str(row.get("backtracking_rule", "")).strip().lower()
        if rule not in {"monotone", "sufficient_decrease"}:
            continue
        key = tuple(str(row.get(field, "")) for field in key_fields)
        rules = grouped.setdefault(key, {})
        if rule in rules:
            raise ValueError(f"Ambiguous duplicate backtracking rule {rule} for {key}.")
        rules[rule] = row

    comparisons = []
    for key in sorted(grouped):
        rules = grouped[key]
        monotone = rules.get("monotone")
        sufficient = rules.get("sufficient_decrease")
        anchor = monotone or sufficient or {}
        status = "complete" if monotone and sufficient else (
            "missing_monotone" if sufficient else "missing_sufficient_decrease"
        )
        comparison = {field: anchor.get(field, "") for field in key_fields}
        comparison.update(
            {
                "comparison_status": status,
                "monotone_psnr": "" if not monotone else monotone.get("final_psnr", ""),
                "monotone_ssim": "" if not monotone else monotone.get("final_ssim", ""),
                "sufficient_decrease_psnr": "" if not sufficient else sufficient.get("final_psnr", ""),
                "sufficient_decrease_ssim": "" if not sufficient else sufficient.get("final_ssim", ""),
                "monotone_run_dir": "" if not monotone else monotone.get("run_dir", ""),
                "sufficient_decrease_run_dir": "" if not sufficient else sufficient.get("run_dir", ""),
            }
        )
        if monotone and sufficient:
            comparison["delta_psnr"] = numeric(sufficient, "final_psnr") - numeric(monotone, "final_psnr")
            comparison["delta_ssim"] = numeric(sufficient, "final_ssim") - numeric(monotone, "final_ssim")
        else:
            comparison["delta_psnr"] = ""
            comparison["delta_ssim"] = ""
        comparisons.append(comparison)
    return comparisons


def default_out_dir(root: Path) -> Path:
    return root / "aggregate"


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Exp1 run metrics into aggregate CSV files.")
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs/sparse_restoration/multirun",
        help="One sweep directory, or a parent directory containing many sweeps.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <root>/aggregate.")
    parser.add_argument("--top-k", type=int, default=10, help="Rows to keep per dataset+degradation group in best_by_psnr.csv.")
    parser.add_argument(
        "--require-backtracking-pairs",
        type=int,
        default=None,
        help="Fail after writing outputs unless exactly this many complete backtracking pairs were found.",
    )
    args = parser.parse_args()
    if args.top_k <= 0 or (args.require_backtracking_pairs is not None and args.require_backtracking_pairs < 0):
        parser.error('Require positive --top-k and nonnegative --require-backtracking-pairs.')
    return args


def print_best(rows: List[Dict[str, Any]]):
    print("Top final_psnr by dataset+degradation:")
    current_group = None
    for row in rows:
        row_group = group_key(row)
        if row_group != current_group:
            current_group = row_group
            print(
                f"  group dataset={row_group[0]} "
                f"mode={row_group[1]} "
                f"params={row_group[2]}"
            )
        print(
            f"    rank={row.get('group_rank', '')} "
            f"job={row.get('job_num', '')} "
            f"solver={row.get('solver', '')} "
            f"lambda={row.get('lambda', '')} "
            f"init={row.get('init_method', '')} "
            f"lambda_multiplier={row.get('lambda_multiplier', '')} "
            f"stage_iters={row.get('jinit_stage_iters', '')} "
            f"psnr={row.get('final_psnr', '')} "
            f"ssim={row.get('final_ssim', '')}"
        )


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir(root)

    final_rows, stage_rows = collect(root)
    if not final_rows:
        raise SystemExit(f"No Exp1 recovery metrics found under {root}")

    best_by_psnr = collect_best_by_psnr(final_rows, args.top_k)
    best_by_processed_init_psnr = collect_best_by_processed_init_psnr(final_rows, args.top_k)
    backtracking_comparisons = collect_backtracking_comparisons(final_rows)

    write_csv(out_dir / "final_summary.csv", final_rows, FINAL_FIELDS)
    write_csv(out_dir / "stage_metrics.csv", stage_rows, STAGE_FIELDS)
    write_csv(out_dir / "best_by_psnr.csv", best_by_psnr, BEST_FIELDS)
    write_csv(out_dir / "best_by_processed_init_psnr.csv", best_by_processed_init_psnr, BEST_FIELDS)
    write_csv(
        out_dir / "backtracking_comparison.csv",
        backtracking_comparisons,
        BACKTRACKING_COMPARISON_FIELDS,
    )

    best_by_ssim_path = out_dir / "best_by_ssim.csv"
    if best_by_ssim_path.exists():
        best_by_ssim_path.unlink()

    print(f"Collected {len(final_rows)} runs and {len(stage_rows)} stage rows.")
    complete_pairs = sum(row["comparison_status"] == "complete" for row in backtracking_comparisons)
    print(f"Collected {complete_pairs} complete backtracking comparison pairs.")
    print(f"Wrote aggregate CSVs to {out_dir}")
    print_best(best_by_psnr)
    if args.require_backtracking_pairs is not None and complete_pairs != args.require_backtracking_pairs:
        raise SystemExit(
            f"Expected {args.require_backtracking_pairs} complete backtracking pairs, found {complete_pairs}."
        )


if __name__ == "__main__":
    main()
