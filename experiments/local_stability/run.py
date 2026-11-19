"""Measure local dimension around clean MNIST images."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.capacity_geometry import geometry as geo
from experiments.capacity_geometry import inputs as io
from experiments.local_stability import inputs, metrics


# Each Jacobian is an atomic resume unit, bound to one exact sample and perturbation.
def location(run, spec, index, radius_index=None, direction_index=None):
    name = f"anchor_{index}" if radius_index is None else f"sample_{index}_r{radius_index}_u{direction_index}"
    identity = dict(model_id=spec["model_id"], sample_index=int(index),
                    radius_index=radius_index, direction_index=direction_index)
    return run / "measurements" / spec["model_id"] / f"{name}.npz", identity


def read_measurement(path, key, identity, config, size, dimension):
    rows, data = io.read_shard(path, key)
    if len(rows) != 1 or rows[0]["identity"] != identity:
        raise ValueError(f"Measurement identity mismatch: {path}")
    timing = rows[0]["timing"]
    if timing["dtype"] != "float64" or timing["chunk_size"] != config["chunk_size"] or timing["retries"]:
        raise ValueError(f"Measurement precision/chunk mismatch: {path}")
    metrics.validate_measurement(data, size)
    retained = int((data["spectrum"] <= max(config["thresholds"])).sum())
    if retained > dimension:
        raise ValueError(f"Near-kernel dimension exceeds bottleneck: {path}")
    return data


def perturb(image, direction, radius):
    return image.astype(np.float64) + radius * np.sqrt(image.size) * direction


def evaluate(run, manifest, device, model_ids=None):
    config = manifest["config"]
    arrays = io.read_npz(run / "inputs.npz")
    inputs.validate_inputs(arrays, manifest)
    selected = set(model_ids or [s["model_id"] for s in manifest["models"]])
    if selected - {s["model_id"] for s in manifest["models"]}:
        raise ValueError("Unknown model_ids; use identifiers in manifest.json.")
    execution = inputs.execution(run, manifest, device)
    with geo.precision_context(config["cpu_threads"]), torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled, benchmark=False, deterministic=True, allow_tf32=False):
        for spec in manifest["models"]:
            if spec["model_id"] not in selected:
                continue
            model = None

            def measure(index, image, radius_index=None, direction_index=None):
                nonlocal model
                path, identity = location(run, spec, index, radius_index, direction_index)
                if not path.exists():
                    if model is None:
                        model = io.load_verified_model(spec, device, torch.float64)
                    data, timing = metrics.linearize(model, image, config)
                    io.write_shard(path, execution["id"], [dict(identity=identity, timing=timing)], **data)
                return read_measurement(path, execution["id"], identity, config, image.size, spec["dimension"])

            for position, (index, image, directions) in enumerate(zip(arrays["indices"], arrays["clean"], arrays["directions"])):
                measure(index, image)
                for radius_index, radius in enumerate(config["radii"]):
                    for direction_index, direction in enumerate(directions):
                        measure(index, perturb(image, direction, radius), radius_index, direction_index)
                print(f"{spec['model_id']}: anchor {position + 1}/{len(arrays['indices'])} complete", flush=True)
            del model


# Later stages use the immutable manifest; CLI overrides apply only during preparation.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "prepare", "evaluate", "summarize"), nargs="?", default="all")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/local_stability"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-run")
    parser.add_argument("--radii", type=float, nargs=2)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--num-anchors", type=int)
    parser.add_argument("--num-directions", type=int)
    parser.add_argument("--dimensions", type=int, nargs="+")
    parser.add_argument("--model-seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-ids", nargs="+", help="Evaluate only selected models; a full report still requires all models.")
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ("source_run", "radii", "split", "num_anchors", "num_directions", "dimensions", "model_seeds")}
    if args.stage not in ("all", "prepare") and (args.config or any(v is not None for v in overrides.values())):
        parser.error("Later stages read frozen settings; omit preparation overrides.")
    if args.model_ids and args.stage != "evaluate":
        parser.error("--model-ids applies only to evaluate.")
    run = args.run_dir.resolve()
    if args.stage in ("all", "prepare"):
        manifest = inputs.prepare(run, inputs.load_config(args.config, **overrides))
        print(f"Prepared {len(manifest['models'])} models, {manifest['anchor_count']} anchors; radii={manifest['config']['radii']}")
        if args.stage == "prepare":
            return
    manifest = inputs.load_run(run)
    if args.stage in ("all", "evaluate"):
        evaluate(run, manifest, torch.device(args.device), args.model_ids)
    if args.stage != "evaluate":
        from experiments.local_stability import report
        with geo.precision_context(manifest["config"]["cpu_threads"]):
            if not report.summarize(run, manifest):
                raise SystemExit("Incomplete measurements; see report/progress.json. No complete table was produced.")


if __name__ == "__main__":
    main()
