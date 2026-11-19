"""Downstream classification experiment on uniformly degraded test images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from recovery.adapters.ae import _load_models
from recovery.registry import canonical_degradation, canonical_method, resolve_config
from utils.utils_metrics import avg_psnr_pytorch, avg_ssim_pytorch

from recovery.cache import (RECOVERY_CACHE_VERSION, resolve_device, degradation_cfg, build_loader, _take,
    _labels_tensor, plain, atomic_torch_save, write_json, validate_pairs_cache, load_or_create_pairs_cache,
    validate_method_cache, load_or_create_method_cache)

from experiments.feature_recovery.knn import ExactKNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = ("ae", "irae")


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    ratios: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 11])
    seed: int = 42
    feature_batch_size: int = 256
    num_workers: int = 0
    train_num_samples: int | None = None
    test_num_samples: int | None = None
    standardize: bool = False
    knn_query_batch_size: int = 256
    device: str | None = None
    knn_device: str | None = None
    data_root: str | None = None
    output_root: str = "outputs/feature_recovery"
    recovery_cache_root: str = "outputs/cache/recovery"
    recovery_policy: str = "auto"
    run_id: str | None = None

    # Reject invalid requests before creating recovery caches or extracting features.
    def __post_init__(self):
        if self.dataset not in {"mnist", "cifar10"} or not self.ratios or not self.methods:
            raise ValueError("Use a labeled MNIST/CIFAR-10 dataset and nonempty ratios/methods.")
        self.methods = [canonical_method(method) for method in self.methods]
        if len(set(self.methods)) != len(self.methods) or len(set(self.ratios)) != len(self.ratios):
            raise ValueError("Methods and ratios must be unique.")
        for name in ("feature_batch_size", "knn_query_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("train_num_samples", "test_num_samples"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None.")
        if not self.k_values or len(set(self.k_values)) != len(self.k_values) or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in self.k_values):
            raise ValueError("k_values must contain unique positive integers.")
        if self.train_num_samples is not None and max(self.k_values) > self.train_num_samples:
            raise ValueError("Every k must fit the requested training sample count.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between zero and 2**32 - 1.")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        if not isinstance(self.standardize, bool):
            raise ValueError("standardize must be boolean.")
        if self.recovery_policy not in {"auto", "cache-only", "refresh"}:
            raise ValueError("Unsupported recovery policy.")
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id
                or self.run_id in {".", ".."} or "/" in self.run_id or "\\" in self.run_id):
            raise ValueError("run_id must be a single nonempty directory name.")
        for ratio in self.ratios:
            canonical_degradation(degradation_cfg(ratio, self.seed))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_loader_features(model, loader, *, source_key: str, device: torch.device,
                            num_samples: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, indices = [], [], []
    seen = 0
    model.eval()
    for batch in loader:
        take = _take(int(batch[source_key].shape[0]), seen, num_samples)
        if take <= 0:
            break
        with torch.inference_mode():
            encoded = model.transform(batch[source_key][:take].to(device))
        features.append(encoded.flatten(1).cpu())
        labels.append(_labels_tensor(batch["label"], take).cpu())
        indices.append(torch.as_tensor(batch["index"][:take]).long().cpu())
        seen += take
    if not features:
        raise ValueError(f"No features extracted from source_key={source_key}.")
    features, labels, indices = torch.cat(features), torch.cat(labels), torch.cat(indices)
    expected = num_samples if num_samples is not None else len(loader.dataset)
    if (features.ndim != 2 or not features.shape[1] or len(features) != expected
            or not torch.isfinite(features).all() or labels.shape != (expected,) or (labels < 0).any()
            or indices.shape != (expected,) or (indices < 0).any() or len(indices.unique()) != expected):
        raise ValueError("Extracted features, labels, or sample coverage are invalid.")
    return features, labels, indices


def encode_tensor(model, images: torch.Tensor, *, device: torch.device, batch_size: int) -> torch.Tensor:
    if images.ndim != 4 or not len(images) or batch_size <= 0:
        raise ValueError("Require nonempty BCHW images and a positive encoding batch size.")
    parts = []
    model.eval()
    for start in range(0, images.shape[0], batch_size):
        with torch.inference_mode():
            parts.append(model.transform(images[start : start + batch_size].to(device)).flatten(1).cpu())
    encoded = torch.cat(parts)
    if encoded.ndim != 2 or len(encoded) != len(images) or not encoded.shape[1] or not torch.isfinite(encoded).all():
        raise ValueError("Encoded feature shapes or values are invalid.")
    return encoded


def fit_feature_transform(features: torch.Tensor, standardize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or not features.numel() or not torch.isfinite(features).all():
        raise ValueError("Feature-transform fitting requires a finite nonempty matrix.")
    if not standardize:
        return torch.zeros(features.shape[1]), torch.ones(features.shape[1])
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min_(1e-6)
    return mean, scale


def transform_features(features: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if (features.ndim != 2 or not torch.isfinite(features).all() or mean.shape != (features.shape[1],) or scale.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Feature-transform dimensions or scales are invalid.")
    return (features.float() - mean) / scale


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    if (predictions.ndim != 1 or not len(predictions) or labels.shape != predictions.shape
            or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or (labels < 0).any()):
        raise ValueError("Accuracy requires matching nonempty labeled prediction vectors.")
    return float((predictions == labels).float().mean().item())


def quality_metrics(clean: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": float(avg_psnr_pytorch(clean, reconstruction)),
        "ssim": float(avg_ssim_pytorch(clean, reconstruction)),
    }


def feature_metrics(clean_features: torch.Tensor, features: torch.Tensor) -> dict[str, float]:
    return {
        "feature_l2": float((features - clean_features).norm(dim=1).mean().item()),
        "feature_cosine": float(F.cosine_similarity(features, clean_features, dim=1).mean().item()),
    }


def load_feature_ae(cfg: ExperimentConfig, device: torch.device):
    first = degradation_cfg(cfg.ratios[0], cfg.seed)
    _, _, _, resolved = resolve_config(cfg.dataset, first, "ae")
    model = _load_models(resolved, str(device), "feedforward")[0].to(device).eval()
    return model, resolved


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: ExperimentConfig) -> Path:
    cfg.__post_init__()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    knn_device = resolve_device(cfg.knn_device) if cfg.knn_device else device
    run_id = cfg.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (PROJECT_ROOT / cfg.output_root / cfg.dataset / run_id).resolve()
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists; choose a new run_id or output root.")
    for ratio in cfg.ratios:
        resolve_config(cfg.dataset, degradation_cfg(ratio, cfg.seed), "ae")
        for method in cfg.methods:
            resolve_config(cfg.dataset, degradation_cfg(ratio, cfg.seed), method)

    # Prepare all paired recoveries before fitting or scoring the classifier.
    cache_index: dict[str, dict[str, Any]] = {}
    print(f"[downstream] preparing complete recovery cache ({cfg.recovery_policy})")
    for ratio in cfg.ratios:
        pairs, pairs_path = load_or_create_pairs_cache(cfg, ratio)
        if (pairs["labels"].ndim != 1 or pairs["labels"].dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or (pairs["labels"] < 0).any()
                or (cfg.test_num_samples is not None and len(pairs["labels"]) != cfg.test_num_samples)):
            raise ValueError("Classification caches require labels and the complete requested test sample count.")
        ratio_cache = {"pairs": str(pairs_path), "methods": {}}
        cache_index[str(ratio)] = ratio_cache
        for method in cfg.methods:
            record, method_path = load_or_create_method_cache(cfg, ratio, method, pairs, device)
            ratio_cache["methods"][method] = str(method_path)
            del record
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del pairs

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    write_json(output_dir / "recovery_cache_index.json", {
        "version": RECOVERY_CACHE_VERSION, "policy": cfg.recovery_policy, "ratios": cache_index,
    })
    print(f"[downstream] loading final clean AE on {device}")
    feature_ae, feature_ae_config = load_feature_ae(cfg, device)
    train_features, train_labels, train_indices = extract_loader_features(
        feature_ae, build_loader(cfg, "train", {"mode": "clean"}), source_key="clean",
        device=device, num_samples=cfg.train_num_samples,
    )
    val_features, val_labels, _ = extract_loader_features(
        feature_ae, build_loader(cfg, "val", {"mode": "clean"}), source_key="clean", device=device,
    )
    mean, scale = fit_feature_transform(train_features, cfg.standardize)
    train_features = transform_features(train_features, mean, scale)
    val_features = transform_features(val_features, mean, scale)
    knn = ExactKNN(train_features, train_labels, device=knn_device, query_batch_size=cfg.knn_query_batch_size)

    print(f"[downstream] selecting k from {cfg.k_values} on clean validation features")
    val_predictions = knn.predict_many(val_features, cfg.k_values)
    k_scores = {k: accuracy(pred, val_labels) for k, pred in val_predictions.items()}
    best_k = max(k_scores, key=lambda k: (k_scores[k], -k))
    atomic_torch_save({"features": train_features, "labels": train_labels, "indices": train_indices,
                "mean": mean, "scale": scale, "best_k": best_k}, output_dir / "knn_state.pt")
    (output_dir / "knn_selection.json").write_text(
        json.dumps({"scores": k_scores, "best_k": best_k, "feature_shape": list(train_features.shape),
                    "standardize": cfg.standardize}, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    resolved_methods: dict[str, Any] = {}
    clean_test_features = clean_predictions = reference_pairs = None

    def score(source, images, recovery_seconds=0.0):
        features = encode_tensor(feature_ae, images, device=device, batch_size=cfg.feature_batch_size)
        classified = knn.predict_many(transform_features(features, mean, scale), [best_k])[best_k]
        rows.append({"source": source, "ratio": ratio, "accuracy": accuracy(classified, pairs["labels"]),
                     **quality_metrics(pairs["clean"], images), **feature_metrics(clean_test_features, features),
                     "recovery_seconds": recovery_seconds, "num_samples": int(pairs["labels"].numel())})
        return classified

    for ratio in cfg.ratios:
        print(f"[downstream] ratio={ratio}: evaluating cached recoveries")
        ratio_cache = cache_index[str(ratio)]
        pairs = torch.load(ratio_cache["pairs"], map_location="cpu")
        validate_pairs_cache(pairs, cfg, ratio)
        if reference_pairs is None:
            reference_pairs = pairs
            clean_test_features = encode_tensor(feature_ae, pairs["clean"], device=device,
                                                  batch_size=cfg.feature_batch_size)
            clean_predictions = knn.predict_many(
                transform_features(clean_test_features, mean, scale), [best_k])[best_k]
            rows.append({"source": "clean", "ratio": "", "accuracy": accuracy(clean_predictions, pairs["labels"]),
                         "psnr": "", "ssim": "", "feature_l2": 0.0, "feature_cosine": 1.0,
                         "recovery_seconds": 0.0, "num_samples": int(pairs["labels"].numel())})
        elif not all(torch.equal(pairs[key], reference_pairs[key]) for key in ("clean", "labels", "indices")):
            raise RuntimeError("Clean test samples changed across degradation levels.")

        predictions = {"labels": pairs["labels"], "indices": pairs["indices"],
                       "clean": clean_predictions, "degraded": score("degraded", pairs["degraded"])}

        for method in cfg.methods:
            _, method_name, _, resolved = resolve_config(cfg.dataset, degradation_cfg(ratio, cfg.seed), method)
            record = torch.load(ratio_cache["methods"][method], map_location="cpu")
            validate_method_cache(record, pairs, method, resolved)
            predictions[method_name] = score(method_name, record["reconstruction"], float(record["recovery_seconds"]))
            resolved_methods[f"{ratio}/{method_name}"] = record["resolved_config"]
            del record
            if device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_torch_save(predictions, output_dir / f"predictions_ratio_{str(ratio).replace('.', 'p')}.pt")
        write_rows(output_dir / "summary.csv", rows)
        write_json(output_dir / "summary.json", {"feature_ae": plain(feature_ae_config), "best_k": best_k,
            "rows": rows, "resolved_methods": resolved_methods, "recovery_cache_index": cache_index})

    print(f"[downstream] completed: {output_dir}")
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=("mnist", "cifar10"))
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--methods", nargs="+", type=canonical_method, default=list(DEFAULT_METHODS))
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 11],
                        help="Select by clean validation accuracy; ties prefer smaller k. Use 3 for the paper protocol.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-num-samples", type=int)
    parser.add_argument("--test-num-samples", type=int)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--knn-query-batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--knn-device")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default="outputs/feature_recovery")
    parser.add_argument("--recovery-cache-root", default="outputs/cache/recovery")
    parser.add_argument("--recovery-policy", choices=("auto", "cache-only", "refresh"), default="auto")
    parser.add_argument("--run-id")
    return ExperimentConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    run(parse_args())
