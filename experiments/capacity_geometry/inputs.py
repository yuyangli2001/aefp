"""Prepare shared samples, load configured models, and save experiment results."""

from __future__ import annotations

from uuid import uuid4
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from models.loading import build_ae_model, load_model_state
from utils.configuration import normalize_model_cfg
from experiments.capacity_geometry.initialization import build_interpolated_init
from utils.utils_dataloader import build_degradation_config, get_degradation_dataloader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


# Store run settings and atomic artifacts without content-hash verification.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def write_shard(path, key, rows, **arrays):
    write_npz(path, key=np.array(key), rows=np.array(json.dumps(rows, allow_nan=False)), **arrays)


def read_shard(path, key):
    data = read_npz(path)
    if str(data.pop("key")) != key:
        raise ValueError(f"Artifact belongs to another configuration: {path}")
    return json.loads(str(data.pop("rows"))), data


def completed(path, key):
    if not Path(path).exists():
        return False
    read_shard(path, key)
    return True


def load_config(path=None):
    config = yaml.safe_load(Path(path or HERE / "configs/default.yaml").read_text(encoding="utf-8"))
    for name in ("test_count", "validation_count", "batch_size", "chunk_size", "cpu_threads"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    # Validate scientific controls before creating inputs or loading checkpoints.
    for name in ("test_seed", "validation_seed"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    seeds = config["init_seeds"]
    if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("init_seeds must contain unique nonnegative integer seeds below 2**63.")
    if (not math.isfinite(config["alpha"]) or not 0 < config["alpha"] <= 1
            or not math.isfinite(config["beta"]) or config["beta"] < 0):
        raise ValueError("Require finite 0 < alpha <= 1 and beta >= 0.")
    from experiments.capacity_geometry.geometry import validate_thresholds
    validate_thresholds([config["epsilon"]])
    if config["jacobian_dtype"] not in ("float32", "float64"):
        raise ValueError("jacobian_dtype must be float32 or float64.")
    solver = config["solver"]
    for name in ("lambda_", "lambda_multiplier"):
        if not math.isfinite(solver[name]) or solver[name] <= 0:
            raise ValueError(f"solver.{name} must be finite and positive.")
    for name in ("max_iteration", "lambda_check_interval"):
        if isinstance(solver[name], bool) or not isinstance(solver[name], int) or solver[name] <= 0:
            raise ValueError(f"solver.{name} must be a positive integer.")
    if (not isinstance(solver["is_lambda_update"], bool)
            or solver["backtracking_rule"] not in ("monotone", "sufficient_decrease")
            or not math.isfinite(solver["sufficient_decrease_constant"])
            or not 0 < solver["sufficient_decrease_constant"] < 1):
        raise ValueError("Invalid solver continuation or backtracking settings.")
    build_degradation_config(config["sparse"])
    return config


# Resolve explicitly named assets inside the current project.
def local_checkpoint(path):
    from utils.path_utils import local_path
    return local_path(path)


def model_config(channels):
    config = yaml.safe_load((ROOT / "configs/model/resae_3down.yaml").read_text(encoding="utf-8"))
    config["params"].update(bottleneck_channels=int(channels), in_channels=1, input_size=28)
    return normalize_model_cfg(config)


def checkpoint_specs(config):
    specs = yaml.safe_load((ROOT / config["models"]).read_text(encoding="utf-8"))
    if not specs or len({spec["model_id"] for spec in specs}) != len(specs):
        raise ValueError("Require a nonempty model list with unique model_id values.")
    for spec in specs:
        if (spec["family"] not in ("capacity", "auxiliary") or isinstance(spec["channels"], bool)
                or not isinstance(spec["channels"], int) or spec["channels"] <= 0
                or spec["dimension"] != 16 * spec["channels"]
                or not spec["model_id"] or Path(spec["model_id"]).name != spec["model_id"]
                or spec["model_id"] in (".", "..")):
            raise ValueError(f"Invalid model specification: {spec['model_id']}")
        spec["model_config"] = model_config(spec["channels"])
        if not local_checkpoint(spec["path"]).is_file():
            raise FileNotFoundError(local_checkpoint(spec["path"]))
    return specs




def load_verified_model(spec, device, dtype=torch.float32):
    path = local_checkpoint(spec["path"])
    model = build_ae_model(spec["model_config"])
    load_model_state(model, path, torch.device("cpu"))
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


# Sample before corruption so split-local indices and corruption seeds remain paired.
def select_indices(labels, count, seed):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not np.isin(labels, np.arange(10)).all()
            or isinstance(count, bool) or not isinstance(count, int) or count <= 0):
        raise ValueError("Require one-dimensional digit labels and a positive integer count.")
    rng = np.random.default_rng(seed)
    counts = np.full(10, count // 10)
    counts[rng.permutation(10)[:count % 10]] += 1
    chosen = [rng.choice(np.flatnonzero(labels == digit), int(n), replace=False) for digit, n in enumerate(counts)]
    return rng.permutation(np.concatenate(chosen)).astype(np.int64)


def dataset_labels(dataset):
    if isinstance(dataset, Subset):
        return dataset_labels(dataset.dataset)[dataset.indices]
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    return np.asarray([dataset[i][1] for i in range(len(dataset))])


def make_loader(config, split, indices=None, clean=False):
    return get_degradation_dataloader(
        dataset="mnist", split=split, sample_indices=indices,
        degradation_cfg={"mode": "clean"} if clean else config["sparse"],
        batch_size=config["batch_size"], shuffle=False, num_workers=0,
        root=config["data_root"], pin_memory=False,
    )


def sample_split(config, split, count, seed, clean=False):
    base = make_loader(config, split, clean=True).dataset.base_dataset
    indices = select_indices(dataset_labels(base), count, seed)
    return indices, make_loader(config, split, indices, clean=clean)


def prepare_inputs(run, config):
    files = []
    generators = [torch.Generator().manual_seed(seed) for seed in config["init_seeds"]]
    for split, count, seed in (("test", config["test_count"], config["test_seed"]),
                               ("val", config["validation_count"], config["validation_seed"])):
        _, loader = sample_split(config, split, count, seed, clean=split == "val")
        for batch_id, batch in enumerate(loader):
            x, y = batch["clean"], batch["degraded"]
            arrays = dict(indices=batch["index"].numpy(), labels=np.asarray(batch["label"]), clean=x.numpy())
            if split == "test":
                arrays.update(observation=y.numpy(), support=(abs(y - x) > 1e-8).numpy())
                initializations, diagnostics = [], []
                for generator in generators:
                    x0, values = build_interpolated_init(x, y, alpha=config["alpha"], beta=config["beta"], generator=generator)
                    initializations.append(x0.numpy())
                    diagnostics.append({name: value.numpy() for name, value in values.items()})
                arrays["initializations"] = np.stack(initializations)
                for name in diagnostics[0]:
                    arrays[name] = np.stack([item[name] for item in diagnostics])
            path = run / "inputs" / f"{split}_batch{batch_id}.npz"
            write_npz(path, **arrays)
            files.append(dict(path=path.relative_to(run).as_posix(), split=split, batch_id=batch_id))
    return files


def prepare_run(run, config):
    if (run / "manifest.json").exists():
        manifest = load_run(run)
        if manifest["config"] != config:
            raise ValueError("Configuration changed; use a new run directory.")
        return manifest
    specs = checkpoint_specs(config)
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)
    manifest = dict(config=config, models=specs, files=prepare_inputs(run, config), environment=environment)
    manifest["id"] = uuid4().hex
    write_json(run / "manifest.json", manifest)
    return manifest


def load_run(run):
    manifest = read_json(run / 'manifest.json')
    for item in manifest['files']:
        if not (run / item['path']).is_file():
            raise FileNotFoundError(run / item['path'])
    for spec in manifest['models']:
        if not local_checkpoint(spec['path']).is_file():
            raise FileNotFoundError(spec['path'])
    return manifest


def input_batches(run, manifest, split):
    return [(item["batch_id"], read_npz(run / item["path"])) for item in manifest["files"] if item["split"] == split]


def input_samples(run, manifest, split):
    return [(int(index), int(label), image) for _, batch in input_batches(run, manifest, split)
            for index, label, image in zip(batch["indices"], batch["labels"], batch["clean"])]
