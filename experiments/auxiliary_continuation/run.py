"""Evaluate auxiliary continuation and visualize a single recovery."""

from __future__ import annotations
import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
import torch
from .ablation import execute
from .variants import DATASETS, FULL_VARIANTS, RATIOS, _ratio_index
from .visualize import replot_reconstruction_strip, run_visualization


def parse_args(argv: Sequence[str] | None = None, *, allow_sample_limit: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    common.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    common.add_argument("--split", choices=("val", "test"), default="test")
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--data-root", type=Path)
    common.add_argument("--output-dir", type=Path, default=Path("outputs/auxiliary_continuation"))
    common.add_argument("--force", action="store_true")

    commands = parser.add_subparsers(dest="command", required=True)
    ablation = commands.add_parser("run", parents=[common], help="Run the ablation matrix.")
    ablation.add_argument("--variants", nargs="+", choices=FULL_VARIANTS)
    ablation.add_argument("--num-workers", type=int, default=4)
    ablation.set_defaults(num_samples=None)
    if allow_sample_limit:
        ablation.add_argument("--num-samples", type=int)

    visual = commands.add_parser("visualize", parents=[common], help="Draw trajectories or replot saved endpoints.")
    visual.add_argument("--endpoints-from", type=Path, help="Replot a saved endpoints.pt without inference.")
    visual.add_argument("--sample-index", type=int, default=0)
    visual.add_argument("--grid-size", type=int, default=61)
    visual.add_argument("--grid-batch-size", type=int, default=64)
    visual.add_argument("--direction-grid", type=int, default=6)
    visual.add_argument("--trajectory-stride", type=int, default=5)
    visual.add_argument("--landscape-lambda", type=float)
    args = parser.parse_args(argv)
    if args.data_root is not None and len(args.datasets) != 1:
        parser.error("--data-root is dataset-specific; select exactly one dataset")
    if args.command == "run":
        if args.num_workers < 0:
            parser.error("--num-workers must be non-negative")
        return args
    if args.endpoints_from is not None:
        return args
    if len(args.datasets) != 1 or len(args.ratios) != 1:
        parser.error("visualize requires exactly one --datasets value and one --ratios value")
    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.grid_size < 3 or args.grid_batch_size <= 0:
        parser.error("Require --grid-size >= 3 and a positive --grid-batch-size")
    if args.direction_grid <= 0 or args.trajectory_stride <= 0:
        parser.error("--direction-grid and --trajectory-stride must be positive")
    if args.landscape_lambda is not None and (not math.isfinite(args.landscape_lambda) or args.landscape_lambda < 0):
        parser.error("--landscape-lambda must be finite and non-negative")
    args.ratios = [RATIOS[_ratio_index(args.ratios[0])]]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "run":
        rows = execute(args)
        print(f"{len(rows)} results for this selection in {args.output_dir / 'metrics.csv'}")
    else:
        output = args.output_dir / "visual"
        if args.endpoints_from is not None:
            payload = replot_reconstruction_strip(args.endpoints_from, output, force=args.force)
        else:
            payload = run_visualization(args)
        print(f"Saved figures to {output}")
        print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
