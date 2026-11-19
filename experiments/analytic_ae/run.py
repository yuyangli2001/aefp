"""Run the analytic AE recovery and mismatch experiments."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

from .coherence_recovery import run_recovery_experiment
from .common import SynConfig, build_subspaces
from .mismatch_bias import SUITE_FILES, run_mismatch_suite
from .report import plot_recovery_heatmaps

RECOVERY_TABLES = ("results_recovery_runs.csv", "results_recovery_cells.csv")
RECOVERY_FILES = (*RECOVERY_TABLES, "fig_syn_recovery.pdf", "fig_syn_recovery.png")


# Publish one complete configuration when recovery and mismatch share an output directory.
def _write_config(config: SynConfig) -> None:
    path = config.output_path() / "resolved_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(asdict(config), handle, indent=2)
        try:
            os.link(temporary, path)
            return
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    existing = replace(SynConfig.from_json(path), output_dir=config.output_dir)
    if existing != config:
        raise FileExistsError(f"{path} has different experiment settings; use a new output directory.")


def _run_recovery(config: SynConfig) -> None:
    records = build_subspaces(config)
    runs, cells = run_recovery_experiment(config, records)
    for name, frame in zip(RECOVERY_TABLES, (runs, cells)):
        frame.to_csv(config.output_path() / name, index=False)
    plot_recovery_heatmaps(config, cells)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all",
                        choices=("all", "run-recovery", "run-mismatch-suite"))
    parser.add_argument("--config", type=Path, help="JSON overrides for the default experiment settings.")
    parser.add_argument("--output", type=Path, help="Directory for results and figures.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing experiment outputs.")
    args = parser.parse_args(argv)

    config = SynConfig.from_json(args.config) if args.config else SynConfig()
    if args.output is not None:
        config = replace(config, output_dir=str(args.output.resolve()))

    recovery = args.command != "run-mismatch-suite"
    mismatch = args.command != "run-recovery"
    targets = (RECOVERY_FILES if recovery else ()) + (SUITE_FILES if mismatch else ())
    output = config.output_path()
    if not args.overwrite:
        for name in targets:
            path = output / name
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --overwrite.")

    _write_config(config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    if recovery:
        _run_recovery(config)
    if mismatch:
        run_mismatch_suite(config)
    logging.info("Saved experiment results to %s", output)


if __name__ == "__main__":
    main()
