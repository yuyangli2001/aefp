"""Frozen clean anchors, paired directions, checkpoints, and execution identities."""

from pathlib import Path
import platform

import numpy as np
import torch
import yaml

from experiments.capacity_geometry import geometry as geo
from experiments.capacity_geometry import inputs as io

HERE = Path(__file__).resolve().parent


# Fail before creating a run when the scientific protocol is incomplete.
def validate_config(config):
    if not isinstance(config.get("source_run"), str) or not config["source_run"].strip():
        raise ValueError("source_run must name the prepared capacity-study directory.")
    for name in ("num_directions", "chunk_size", "cpu_threads"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config["num_directions"] % 2:
        raise ValueError("num_directions must be even (paired positive/negative axes).")
    for name in ("anchor_seed", "direction_seed"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    count = config["num_anchors"]
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("num_anchors must be positive or null.")
    for name in ("dimensions", "model_seeds"):
        values = config[name]
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values):
            raise ValueError(f"Invalid {name}.")
    if min(config["dimensions"]) <= 0 or config["split"] not in ("test", "val"):
        raise ValueError("Require positive dimensions and a test/val split.")
    radii = config["radii"]
    if not isinstance(radii, list) or len(radii) != 2 or not all(type(r) in (int, float) and np.isfinite(r) and r > 0 for r in radii) or radii[0] >= radii[1]:
        raise ValueError("Supply two finite normalized radii with 0 < r1 < r2.")
    geo.validate_thresholds(config["thresholds"])
    if config["epsilon"] not in config["thresholds"]:
        raise ValueError("epsilon must be included in thresholds.")


def load_config(path=None, **overrides):
    config = yaml.safe_load((HERE / "configs/default.yaml").read_text(encoding="utf-8"))
    if path:
        supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or set(supplied) - set(config):
            raise ValueError("Configuration must be a mapping of known protocol fields.")
        config.update(supplied)
    config.update({k: v for k, v in overrides.items() if v is not None})
    validate_config(config)
    return config


# Per-anchor seeds make directions independent of selection order and resume boundaries.
def make_directions(indices, shape, count, seed):
    directions = []
    for index in indices:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(index)]))
        axes = rng.standard_normal((count // 2, *shape))
        axes /= np.linalg.norm(axes.reshape(len(axes), -1), axis=1).reshape((-1,) + (1,) * len(shape))
        directions.append(np.stack((axes, -axes), axis=1).reshape(count, *shape))
    return np.stack(directions)


# Check paired image coverage before zip-based evaluation can silently truncate it.
def validate_anchors(arrays, count):
    clean, indices, labels = (arrays[name] for name in ("clean", "indices", "labels"))
    if (type(count) is not int or count <= 0 or clean.ndim != 4 or len(clean) != count
            or not all(clean.shape[1:]) or not np.isfinite(clean).all()
            or indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or len(np.unique(indices)) != count
            or labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer)
            or not np.isin(labels, np.arange(10)).all()):
        raise ValueError("Invalid clean anchors, digit labels, indices, or sample coverage.")


def validate_inputs(arrays, manifest):
    count, config = manifest["anchor_count"], manifest["config"]
    validate_anchors(arrays, count)
    pool = manifest["pool_count"]
    if type(pool) is not int or pool < count or count != (config["num_anchors"] or pool):
        raise ValueError("Anchor count does not match the frozen protocol.")
    directions = arrays["directions"]
    expected = (count, config["num_directions"], *arrays["clean"].shape[1:])
    if (directions.shape != expected or directions.dtype != np.float64 or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions.reshape(count, config["num_directions"], -1), axis=2), 1, rtol=1e-12, atol=1e-12)
            or not np.array_equal(directions[:, ::2], -directions[:, 1::2])):
        raise ValueError("Directions must have complete, paired unit-length float64 coverage.")


# Model identifiers also name measurement directories, so each grid cell must be unique.
def validate_models(specs, config):
    expected = {(d, s) for d in config["dimensions"] for s in config["model_seeds"]}
    if (len(specs) != len(expected) or {(s["dimension"], s["seed"]) for s in specs} != expected
            or len({s["model_id"] for s in specs}) != len(specs)):
        raise ValueError("Source does not contain the complete requested model/seed grid with unique model IDs.")
    for spec in specs:
        name = spec["model_id"]
        if (spec["family"] != "capacity" or not isinstance(name, str) or not name or name in (".", "..")
                or "/" in name or "\\" in name):
            raise ValueError("Require capacity models with single-component identifiers.")


def prepare(run, config):
    validate_config(config)
    if (run / "manifest.json").exists():
        manifest = load_run(run)
        if manifest["config"] != config:
            raise ValueError("Configuration changed; use a new run directory.")
        return manifest
    source = (io.ROOT / config["source_run"]).resolve()
    if run.resolve() == source or source in run.resolve().parents:
        raise ValueError("Use a separate output directory, outside the source run.")
    if run.exists() and any(run.iterdir()):
        raise ValueError("Existing artifacts without a manifest; use a new run directory.")
    original = io.read_json(source / "manifest.json")
    expected = {(d, s) for d in config["dimensions"] for s in config["model_seeds"]}
    specs = [dict(s) for s in original["models"] if s["family"] == "capacity" and (s["dimension"], s["seed"]) in expected]
    validate_models(specs, config)
    for spec in specs:
        path = io.local_checkpoint(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        spec["path"] = path.relative_to(io.ROOT).as_posix()
    batches, provenance = [], []
    for item in original["files"]:
        if item["split"] != config["split"]:
            continue
        batches.append(io.read_npz(source / item["path"]))
        provenance.append(item)
    if not batches:
        raise ValueError("No clean anchors in the requested source split.")
    arrays = {k: np.concatenate([b[k] for b in batches]) for k in ("clean", "indices", "labels")}
    size = len(arrays["indices"])
    validate_anchors(arrays, size)
    expected_count = original.get("config", {}).get("test_count" if config["split"] == "test" else "validation_count", size)
    if size != expected_count:
        raise ValueError("Source split is incomplete; prepare the complete capacity-study inputs first.")
    count = config["num_anchors"] or size
    if count > size:
        raise ValueError("num_anchors exceeds the cached pool.")
    positions = np.arange(size) if count == size else io.select_indices(arrays["labels"], count, config["anchor_seed"])
    arrays = {k: v[positions] for k, v in arrays.items()}
    arrays["directions"] = make_directions(arrays["indices"], arrays["clean"].shape[1:], config["num_directions"], config["direction_seed"])
    io.write_npz(run / "inputs.npz", **arrays)
    manifest = dict(schema="local-rank-v2", config=config, models=specs,
                    source_manifest_id=original["id"], source_files=provenance, pool_count=size,
                    anchor_count=count,
                    perturbation="ambient_unclipped_paired_gaussian_unit_directions",
                    dtype="float64")
    manifest["id"] = io.uuid4().hex
    io.write_json(run / "manifest.json", manifest)
    return manifest


# A new experiment owns its inputs and no longer depends on the source run after preparation.
def load_run(run):
    manifest = io.read_json(run / 'manifest.json')
    if manifest.get('schema') != 'local-rank-v2':
        raise ValueError('Expected local-rank-v2; prepare a new run directory.')
    validate_config(manifest['config'])
    validate_models(manifest['models'], manifest['config'])
    if (not isinstance(manifest.get('id'), str) or not manifest['id'] or manifest.get('dtype') != 'float64'
            or manifest.get('perturbation') != 'ambient_unclipped_paired_gaussian_unit_directions'):
        raise ValueError('Invalid run identity or precision/perturbation protocol.')
    if not (run / 'inputs.npz').is_file():
        raise FileNotFoundError(run / 'inputs.npz')
    for spec in manifest['models']:
        if not io.local_checkpoint(spec['path']).is_file():
            raise FileNotFoundError(spec['path'])
    validate_inputs(io.read_npz(run / 'inputs.npz'), manifest)
    return manifest


def execution(run, manifest, device=None):
    path = run / "execution.json"
    previous = io.read_json(path) if path.exists() else None
    if previous is not None:
        if previous.get("manifest_id") != manifest["id"] or not isinstance(previous.get("id"), str) or not previous["id"]:
            raise ValueError("Modified execution identity.")
    if device is None:
        result = io.read_json(path)
    else:
        device = torch.device(device)
        result = dict(manifest_id=manifest["id"], python=platform.python_version(),
                      torch=str(torch.__version__), numpy=np.__version__, device=str(device),
                      gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      tf32=False, cudnn_deterministic=True, cudnn_benchmark=False)
        result["id"] = previous["id"] if previous is not None else manifest["id"]
        io.write_json(path, result)
    if result["manifest_id"] != manifest["id"]:
        raise ValueError("Modified execution identity.")
    return result
