"""Paired evaluation inputs and recovery caches shared across experiments."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from recovery.api import recover_dataset
from recovery.registry import canonical_degradation, resolve_config
from utils.utils_dataloader import get_degradation_dataloader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_CACHE_VERSION = 1
DEFAULT_METHODS = ("ae", "irae")

# Cache settings contain no feature-classification protocol.
@dataclass
class RecoveryCacheConfig:
    dataset: str = "mnist"
    seed: int = 42
    num_workers: int = 0
    test_num_samples: int | None = None
    data_root: str | None = None
    recovery_cache_root: str = "outputs/cache/recovery"
    recovery_policy: str = "auto"
    feature_batch_size: int = 256


# Reused implementations retain their original experiment semantics.
def resolve_device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def degradation_cfg(ratio: float, seed: int) -> dict[str, Any]:
    return {
        "mode": "sparse", "ratio": float(ratio), "value_mode": "salt_pepper",
        "data_min": 0.0, "data_max": 1.0, "per_channel": True, "seed": int(seed),
    }


def build_loader(cfg: RecoveryCacheConfig, split: str, degradation: Mapping[str, Any]):
    loader_dataset, image_size = (
        ("celeba-hq-256", 64) if cfg.dataset == "celeba-hq-64" else (cfg.dataset, None)
    )
    return get_degradation_dataloader(
        dataset=loader_dataset, split=split, degradation_cfg=dict(degradation),
        batch_size=cfg.feature_batch_size, num_workers=cfg.num_workers,
        root=cfg.data_root, image_size=image_size, shuffle=False, drop_last=False,
    )


def _take(batch_size: int, seen: int, limit: int | None) -> int:
    return batch_size if limit is None else max(0, min(batch_size, int(limit) - seen))


def _labels_tensor(labels: Any, count: int) -> torch.Tensor:
    if labels is None or (isinstance(labels, Sequence) and all(label is None for label in labels)):
        return torch.full((count,), -1, dtype=torch.long)
    values = labels[:count] if torch.is_tensor(labels) or isinstance(labels, Sequence) else labels
    return torch.as_tensor(values, dtype=torch.long)


def collect_test_pairs(cfg: RecoveryCacheConfig, ratio: float) -> dict[str, torch.Tensor]:
    if cfg.test_num_samples is not None and cfg.test_num_samples <= 0:
        raise ValueError("Test-pair caches require a positive sample count or None.")
    clean, degraded, labels, indices = [], [], [], []
    seen = 0
    for batch in build_loader(cfg, "test", degradation_cfg(ratio, cfg.seed)):
        take = _take(int(batch["clean"].shape[0]), seen, cfg.test_num_samples)
        if take <= 0:
            break
        clean.append(batch["clean"][:take].cpu())
        degraded.append(batch["degraded"][:take].cpu())
        labels.append(_labels_tensor(batch.get("label"), take).cpu())
        indices.append(torch.as_tensor(batch["index"][:take]).long().cpu())
        seen += take
    if seen == 0:
        raise ValueError("The test-pair loader is empty.")
    return {"clean": torch.cat(clean), "degraded": torch.cat(degraded),
            "labels": torch.cat(labels), "indices": torch.cat(indices)}


def assert_aligned(result, pairs: Mapping[str, torch.Tensor]) -> None:
    labels = _labels_tensor(result.labels, result.num_samples)
    if not torch.equal(result.indices.cpu(), pairs["indices"]) or not torch.equal(labels, pairs["labels"]):
        raise RuntimeError(f"Sample alignment failed for method={result.method}.")
    if result.observation is None or not torch.equal(result.observation, pairs["degraded"]):
        raise RuntimeError(f"Degraded observations differ for method={result.method}.")


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def recovery_cache_dir(cfg: RecoveryCacheConfig, ratio: float) -> Path:
    if cfg.recovery_policy not in {"auto", "refresh", "cache-only"}:
        raise ValueError(f"Unknown recovery cache policy: {cfg.recovery_policy}")
    if cfg.test_num_samples is not None and cfg.test_num_samples <= 0:
        raise ValueError("Recovery caches require a positive sample count or None.")
    _, signature = canonical_degradation(degradation_cfg(ratio, cfg.seed))
    sample_tag = "nall" if cfg.test_num_samples is None else f"n{cfg.test_num_samples}"
    return (PROJECT_ROOT / cfg.recovery_cache_root / cfg.dataset / "test" / signature / sample_tag).resolve()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(plain(value), indent=2), encoding="utf-8")


def validate_pairs_cache(record: Mapping[str, Any], cfg: RecoveryCacheConfig, ratio: float) -> None:
    expected = degradation_cfg(ratio, cfg.seed)
    if record.get("version") != RECOVERY_CACHE_VERSION or record.get("dataset") != cfg.dataset:
        raise ValueError("Recovery test-pair cache version or dataset mismatch.")
    if record.get("split") != "test" or record.get("degradation") != expected:
        raise ValueError("Recovery test-pair cache split or degradation mismatch.")
    if record.get("requested_num_samples") != cfg.test_num_samples:
        raise ValueError("Recovery test-pair cache sample request mismatch.")
    count = int(record["num_samples"])
    if count <= 0:
        raise ValueError("A recovery test-pair cache cannot be empty.")
    for key in ("clean", "degraded", "labels", "indices"):
        if not torch.is_tensor(record.get(key)) or int(record[key].shape[0]) != count:
            raise ValueError(f"Invalid recovery test-pair cache field: {key}.")
    if record["clean"].ndim != 4 or record["clean"].shape != record["degraded"].shape:
        raise ValueError("Recovery test-pair image shapes do not match.")
    if not torch.isfinite(record["clean"]).all() or not torch.isfinite(record["degraded"]).all():
        raise ValueError("Recovery test-pair images must be finite.")


def load_or_create_pairs_cache(cfg: RecoveryCacheConfig, ratio: float) -> tuple[dict[str, Any], Path]:
    directory = recovery_cache_dir(cfg, ratio)
    path = directory / "test_pairs.pt"
    if path.is_file() and cfg.recovery_policy != "refresh":
        record = torch.load(path, map_location="cpu")
        validate_pairs_cache(record, cfg, ratio)
        print(f"[downstream] ratio={ratio}: using cached canonical test pairs")
        return record, path
    if cfg.recovery_policy == "cache-only":
        raise FileNotFoundError(f"Missing required test-pair cache: {path}")
    pairs = collect_test_pairs(cfg, ratio)
    record = {
        "version": RECOVERY_CACHE_VERSION, "dataset": cfg.dataset, "split": "test",
        "degradation": degradation_cfg(ratio, cfg.seed),
        "requested_num_samples": cfg.test_num_samples,
        "num_samples": int(pairs["labels"].numel()), **pairs,
    }
    atomic_torch_save(record, path)
    write_json(directory / "test_pairs_manifest.json", {
        "version": RECOVERY_CACHE_VERSION, "dataset": cfg.dataset, "split": "test",
        "degradation": record["degradation"], "requested_num_samples": cfg.test_num_samples,
        "num_samples": record["num_samples"], "file": path.name, "bytes": path.stat().st_size,
    })
    return record, path


def validate_method_cache(record: Mapping[str, Any], pairs: Mapping[str, Any],
                          method: str, resolved: Mapping[str, Any]) -> None:
    if record.get("version") != RECOVERY_CACHE_VERSION or record.get("method") != resolved["method"]:
        raise ValueError(f"Recovery cache version or method mismatch for {method}.")
    if record.get("dataset") != pairs["dataset"] or record.get("split") != "test":
        raise ValueError(f"Recovery cache dataset or split mismatch for {method}.")
    if (record.get("degradation") != pairs["degradation"] or
            record.get("requested_num_samples") != pairs["requested_num_samples"]):
        raise ValueError(f"Recovery cache input specification mismatch for {method}.")
    if record.get("preset_key") != resolved["preset_key"] or record.get("resolved_config") != plain(resolved):
        raise ValueError(f"Recovery cache preset mismatch for {method}.")
    reconstruction = record.get("reconstruction")
    if not torch.is_tensor(reconstruction) or reconstruction.ndim != 4:
        raise ValueError(f"Invalid cached reconstruction for {method}.")
    if record.get("num_samples") != pairs["num_samples"] or reconstruction.shape != pairs["clean"].shape:
        raise ValueError(f"Recovery cache sample count mismatch for {method}.")
    if not torch.isfinite(reconstruction).all() or reconstruction.min() < 0 or reconstruction.max() > 1:
        raise ValueError(f"Invalid recovery cache values for {method}.")
    if not torch.equal(record["indices"], pairs["indices"]) or not torch.equal(record["labels"], pairs["labels"]):
        raise ValueError(f"Recovery cache sample alignment mismatch for {method}.")


def load_or_create_method_cache(cfg: RecoveryCacheConfig, ratio: float, method: str,
                                pairs: Mapping[str, Any], device: torch.device) -> tuple[dict[str, Any], Path]:
    directory = recovery_cache_dir(cfg, ratio)
    _, method_name, _, resolved = resolve_config(cfg.dataset, degradation_cfg(ratio, cfg.seed), method)
    path = directory / f"{method_name}.pt"
    if path.is_file() and cfg.recovery_policy != "refresh":
        record = torch.load(path, map_location="cpu")
        validate_method_cache(record, pairs, method, resolved)
        print(f"[downstream] ratio={ratio}, method={method_name}: using recovery cache")
        return record, path
    if cfg.recovery_policy == "cache-only":
        raise FileNotFoundError(f"Missing required recovery cache: {path}")

    scope = 'full test set' if cfg.test_num_samples is None else f'{cfg.test_num_samples} requested test samples'
    print(f"[recovery] ratio={ratio}, method={method_name}: recovering {scope}", flush=True)
    started = time.perf_counter()
    result = recover_dataset(
        cfg.dataset, degradation_cfg(ratio, cfg.seed), method_name, config=None,
        split="test", num_samples=cfg.test_num_samples, num_workers=cfg.num_workers,
        device=str(device), root=cfg.data_root,
    )
    recovery_seconds = time.perf_counter() - started
    assert_aligned(result, pairs)
    return save_method_cache(cfg, ratio, method_name, pairs, result.reconstruction,
                             result.resolved_config, recovery_seconds, result.metadata)


# Save aligned recovery tensors for restoration metrics, feature evaluation and figures.
def save_method_cache(cfg, ratio, method, pairs, reconstruction, resolved, seconds, metadata=None):
    method = resolved["method"]
    directory = recovery_cache_dir(cfg, ratio)
    path = directory / f'{method}.pt'
    if reconstruction.shape != pairs['clean'].shape or not torch.isfinite(reconstruction).all():
        raise ValueError('Incomplete or non-finite reconstruction cache.')
    record = {
        "version": RECOVERY_CACHE_VERSION, "dataset": cfg.dataset, "split": "test",
        "method": method, "preset_key": resolved["preset_key"],
        "resolved_config": plain(resolved),
        "degradation": degradation_cfg(ratio, cfg.seed),
        "requested_num_samples": cfg.test_num_samples, "num_samples": len(pairs['indices']),
        "sample_ids": [f'{cfg.dataset}:test:{int(i):06d}' for i in pairs['indices']], "indices": pairs['indices'].cpu(),
        "labels": pairs['labels'].cpu(),
        "reconstruction": reconstruction.cpu(), "metadata": plain(metadata or {}),
        "recovery_seconds": seconds,
    }
    validate_method_cache(record, pairs, method, resolved)
    atomic_torch_save(record, path)
    write_json(directory / f"{method}_manifest.json", {
        key: record[key] for key in ("version", "dataset", "split", "method", "preset_key",
                                     "degradation", "requested_num_samples", "num_samples",
                                     "metadata", "recovery_seconds")
    } | {"sample_id_first": record["sample_ids"][0], "sample_id_last": record["sample_ids"][-1],
         "resolved_config": record["resolved_config"], "file": path.name, "bytes": path.stat().st_size})
    return record, path
