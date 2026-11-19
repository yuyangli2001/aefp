"""Stage-based CLI for the capacity-to-geometry diagnostic (no training)."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from experiments.capacity_geometry import geometry as geo
from experiments.capacity_geometry import inputs as io
from methods.continuation import irae_direct


def log(message):
    logger = logging.getLogger("capacity_to_geometry")
    if logger.handlers:
        logger.info(message)
    else:
        print(message, flush=True)


def load_frozen(run, manifest):
    frozen = io.read_json(run / "frozen.json")
    if frozen["manifest_id"] != manifest["id"]:
        raise ValueError("Frozen configuration does not match the current run.")
    return frozen


def freeze_run(run, manifest, epsilon, dtype):
    geo.validate_thresholds([epsilon])
    if dtype not in ("float32", "float64"):
        raise ValueError("Jacobian dtype must be float32 or float64.")
    frozen = dict(manifest_id=manifest["id"], epsilon=epsilon,
                  thresholds=[0.8 * epsilon, epsilon, min(1.2 * epsilon, (1 + epsilon) / 2)],
                  jacobian_dtype=dtype, chunk_size=manifest["config"]["chunk_size"])
    geo.validate_thresholds(frozen["thresholds"])
    frozen["id"] = manifest["id"] + "-geometry"
    if (run / "frozen.json").exists():
        if load_frozen(run, manifest) != frozen:
            raise ValueError("Threshold or precision changed; use a new run directory.")
        return frozen
    io.write_json(run / "frozen.json", frozen)
    return frozen


# Recovery has no threshold dimension and always uses the original fixed batches.
def evaluate_recovery(run, manifest, frozen, spec, model, device):
    cfg, key = manifest["config"], frozen["id"]
    model.float()
    for batch_id, batch in io.input_batches(run, manifest, "test"):
        x = torch.from_numpy(batch["clean"]).to(device)
        y = torch.from_numpy(batch["observation"]).to(device)
        for repeat, seed in enumerate(cfg["init_seeds"]):
            path = run / "capacity" / spec["model_id"] / f"recovery_batch{batch_id}_init{seed}.npz"
            if io.completed(path, key):
                continue
            initial = torch.from_numpy(batch["initializations"][repeat]).to(device)
            started = time.perf_counter()
            recovered = irae_direct(observation=y, ae_model=model, x_init=initial, device=device,
                                    degradation={"mode": "sparse"}, **cfg["solver"])
            metrics = geo.recovery_metrics(x, initial, recovered, model)
            rows = [dict(model_id=spec["model_id"], sample_index=int(index), label=int(batch["labels"][i]),
                         batch_id=batch_id, init_seed=seed, **metrics[i],
                         realized_support_ratio=float(batch["support"][i].mean()),
                         init_distance_ratio=float(batch["init_distance_ratio"][repeat, i]),
                         clip_fraction=float(batch["clip_fraction"][repeat, i])) for i, index in enumerate(batch["indices"])]
            io.write_shard(path, key, rows, recovered=recovered.detach().cpu().numpy(), seconds=np.array(time.perf_counter() - started))
            log(f"recovery {spec['model_id']} batch={batch_id} init={seed} complete")


def evaluate_models(run, manifest, family, device):
    frozen = load_frozen(run, manifest)
    dtype = getattr(torch, frozen["jacobian_dtype"])
    samples = io.input_samples(run, manifest, "test")
    for spec in (s for s in manifest["models"] if s["family"] == family):
        model = io.load_verified_model(spec, device, dtype)
        for position, (index, label, image) in enumerate(samples):
            path = run / family / spec["model_id"] / f"geometry_{index}.npz"
            if io.completed(path, frozen["id"]):
                continue
            J, residual, timing = geo.compute_linearization(model, image, frozen["chunk_size"])
            metrics, spectrum = geo.geometry_from_jacobian(J, frozen["thresholds"], spec["dimension"])
            mse = float(np.mean(residual ** 2))
            rows = [dict(model_id=spec["model_id"], sample_index=index, label=label, **r,
                         E_recon=math.sqrt(mse), clean_psnr=-10 * math.log10(max(mse, 1e-30)),
                         jacobian_frobenius_sq=float(np.square(J.astype(np.float64)).sum()), **timing) for r in metrics]
            io.write_shard(path, frozen["id"], rows, spectrum=spectrum)
            log(f"geometry {spec['model_id']} {position + 1}/{len(samples)} ({timing['seconds']:.2f}s)")
        if family == "capacity":
            evaluate_recovery(run, manifest, frozen, spec, model, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "prepare", "evaluate", "auxiliary", "summarize"), nargs="?", default="all")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/capacity_geometry"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jacobian-dtype", choices=("float32", "float64"), help="Override geometry precision when preparing a run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    preparing = args.stage in ("all", "prepare")
    if not preparing and (args.config is not None or args.jacobian_dtype is not None):
        parser.error("Use configuration overrides only when preparing a run.")
    manifest = None if preparing else io.load_run(run)
    config = io.load_config(args.config) if manifest is None else manifest["config"]
    if args.jacobian_dtype is not None:
        config["jacobian_dtype"] = args.jacobian_dtype
    run.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("capacity_to_geometry")
    for handler in logger.handlers:
        handler.close()
    logger.handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run / f"{args.stage}.log", encoding="utf-8")]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    with geo.precision_context(config["cpu_threads"]):
        if preparing:
            manifest = io.prepare_run(run, config)
            log(f"Prepared {len(manifest['models'])} checkpoints and shared inputs: {run}")
            freeze_run(run, manifest, config["epsilon"], config["jacobian_dtype"])
        for stage, family in (("evaluate", "capacity"), ("auxiliary", "auxiliary")):
            if args.stage in ("all", stage):
                evaluate_models(run, manifest, family, torch.device(args.device))
        if args.stage in ("all", "summarize"):
            from experiments.capacity_geometry.report import summarize_run

            if not summarize_run(run, manifest, load_frozen(run, manifest)):
                raise RuntimeError("Capacity results are incomplete; inspect report/progress.json.")


if __name__ == "__main__":
    main()
