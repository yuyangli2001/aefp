"""Train or reuse AE checkpoints, then evaluate configured image degradations.

Use experiments.sparse_restoration.infer to evaluate fixed checkpoints.
"""

import json
import logging
import math
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.ae_trainer import MultiStageCAETrainer
from utils.experiment_io import BatchFieldView, combine_stage_metrics, read_csv, write_csv
from utils.checkpoint_paths import build_model_tag, direct_ckpt_path, format_tag_value, jinit_selected_ckpt_path, jinit_summary_path
from utils.configuration import normalize_split, resolve_experiment_config, to_dict, to_loader_degradation_cfg
from experiments.sparse_restoration.evaluate import evaluate_checkpoint_specs, run_irae_recovery_eval
from utils.utils_dataloader import get_degradation_dataloader
from recovery.cache import write_json


log = logging.getLogger(__name__)


# Seed a fresh run explicitly before constructing datasets or models.
def _seed_run(seed):
    import random
    from operator import index as integer_index
    import numpy as np
    import torch

    seed = integer_index(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("Run seed must be an integer in [0, 2**32).")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def build_degradation_tag(degradation_cfg) -> str:
    mode = str(degradation_cfg["mode"])
    params = degradation_cfg.get("params", {}) or {}
    if not params:
        return mode

    param_tag = "_".join(
        f"{key}{format_tag_value(value)}"
        for key, value in sorted(params.items())
    )
    return f"{mode}_{param_tag}"


def resolve_output_path(root_value: str, project_root: Path) -> Path:
    root_path = Path(root_value)
    if not root_path.is_absolute():
        root_path = project_root / root_path
    return root_path


def build_ae_eval_cache_dir(
    ckpt_dir: Path,
    degradation_cfg,
    reconstruction_split: str,
    eval_num_samples: int,
    solver: str,
) -> Path:
    degradation_tag = build_degradation_tag(degradation_cfg)
    if str(solver).strip().lower() == "feedforward":
        degradation_tag = f"feedforward_{degradation_tag}"
    return (
        ckpt_dir
        / "ae_reconstruction_eval"
        / degradation_tag
        / f"{reconstruction_split}_n{format_tag_value(eval_num_samples)}"
    )


def evaluate_or_load_ae_rows(
    *,
    trainer: MultiStageCAETrainer,
    val_loader,
    inference_cfg,
    dataset_cfg,
    model_cfg,
    degradation_cfg,
    reconstruction_cfg,
    cache_dir: Path,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_metrics_csv = cache_dir / "ae_stage_metrics.csv"

    solver = str(inference_cfg["solver"]).strip().lower()
    # Bind cache reuse to the evaluation protocol.
    context = dict(solver=solver, profile="jinit", dataset=to_dict(dataset_cfg),
                   model=to_dict(model_cfg), degradation=to_dict(degradation_cfg),
                          psnr_drops=trainer.training_cfg.get("auxiliary", {}).get("psnr_drops"))
    summary_path = cache_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        cached_rows = read_csv(cache_metrics_csv)
        valid = (summary.get("context") == context and len(cached_rows) == (3 if solver == "jinit" else 1)
                 and [int(row["stage"]) for row in cached_rows] == list(range(1, len(cached_rows) + 1))
                 and all(Path(row["ckpt_path"]).is_file() and all(math.isfinite(float(row[key]))
                     for key in ("clean_psnr", "clean_ssim", "degraded_psnr", "degraded_ssim")) for row in cached_rows))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        valid = False
    if valid and not trainer.training_cfg.get("force_retrain", False):
        log.info("Using cached AE reconstruction metrics from %s", cache_metrics_csv)
        return cached_rows, True
    summary_path.unlink(missing_ok=True)
    direct_ckpt = direct_ckpt_path(trainer.ckpt_dir)
    checkpoint_specs = []
    if solver == "jinit":
        psnr_drops = list(trainer.training_cfg["auxiliary"]["psnr_drops"])
        checkpoint_specs.append(
            {
                "stage": 1,
                "label": "selected_jinit_coarse",
                "shrinkage_weight": None,
                "l1_weight": 0.0,
                "ckpt_path": str(jinit_selected_ckpt_path(trainer.ckpt_dir, psnr_drops, 1)),
            }
        )
        summary_json = jinit_summary_path(trainer.ckpt_dir, psnr_drops)
        if not summary_json.exists():
            raise FileNotFoundError(f"Missing jinit summary metadata: {summary_json}")
        summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
        selections = list(summary_payload["selections"])
        checkpoint_specs.append(
            {
                "stage": 2,
                "label": "selected_jinit_fine",
                "shrinkage_weight": float(selections[1]["selected_shrinkage_weight"]),
                "l1_weight": 0.0,
                "ckpt_path": str(jinit_selected_ckpt_path(trainer.ckpt_dir, psnr_drops, 2)),
            }
        )
        checkpoint_specs.append(
            {
                "stage": 3,
                "label": "direct",
                "shrinkage_weight": 0.0,
                "l1_weight": 0.0,
                "ckpt_path": str(direct_ckpt),
            }
        )
        checkpoint_specs[0]["shrinkage_weight"] = float(selections[0]["selected_shrinkage_weight"])
    else:
        checkpoint_specs.append(
            {
                "stage": 1,
                "label": "feedforward" if solver == "feedforward" else "direct",
                "shrinkage_weight": 0.0,
                "l1_weight": 0.0,
                "ckpt_path": str(direct_ckpt),
            }
        )
    clean_stage_metrics, degraded_stage_metrics = [
        evaluate_checkpoint_specs(
            checkpoint_specs, BatchFieldView(val_loader, field), model_cfg=trainer.model_cfg,
            device=trainer.device, test_num_samples=int(reconstruction_cfg["eval_num_samples"]),
            evaluation=trainer.selection_evaluation(), save_dir=cache_dir / "images" / field,
        )
        for field in ("clean", "degraded")
    ]
    ae_stage_metrics = combine_stage_metrics(clean_stage_metrics, degraded_stage_metrics)

    ae_rows = []
    for metric in ae_stage_metrics:
        ae_rows.append(
            {
                "dataset": str(dataset_cfg["name"]),
                "model_tag": str(cache_dir.parents[2].name),
                "model_variant": str(model_cfg["variant"]),
                "degradation_mode": str(degradation_cfg["mode"]),
                "degradation_params": str(degradation_cfg.get("params", {})),
                "stage": int(metric["stage"]),
                "shrinkage_weight": float(metric["shrinkage_weight"]),
                "l1_weight": float(metric["l1_weight"]),
                "clean_psnr": float(metric["clean_psnr"]),
                "clean_ssim": float(metric["clean_ssim"]),
                "degraded_psnr": float(metric["degraded_psnr"]),
                "degraded_ssim": float(metric["degraded_ssim"]),
                "ckpt_path": str(metric["ckpt_path"]),
            }
        )

    write_csv(cache_metrics_csv, ae_rows)
    write_json(summary_path,
            {
                "dataset": str(dataset_cfg["name"]),
                "model_tag": str(cache_dir.parents[2].name),
                "model_variant": str(model_cfg["variant"]),
                "degradation_mode": str(degradation_cfg["mode"]),
                "degradation_params": degradation_cfg.get("params", {}),
                "eval_num_samples": int(reconstruction_cfg["eval_num_samples"]),
                "cache_dir": str(cache_dir),
                "context": context,
            },
    )
    log.info("Saved AE reconstruction metrics cache to %s", cache_metrics_csv)
    return ae_rows, False


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    ## 0. Setup logging, output directories, and save resolved config
    logging.basicConfig(level=logging.INFO)
    seed = _seed_run(cfg.get("seed", 42))

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    images_dir = run_dir / "images"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    hydra_cfg = HydraConfig.get()
    resolved_cfg = resolve_experiment_config(
        cfg,
        project_root=PROJECT_ROOT,
        hydra_choices=hydra_cfg.runtime.choices,
        task_overrides=hydra_cfg.overrides.task,
    )
    resolved_cfg["runtime"]["seed"] = seed
    dataset_cfg = resolved_cfg["dataset"]
    dataset_source_name = resolved_cfg["runtime"]["dataset_source_name"]
    model_cfg = resolved_cfg["model"]
    degradation_cfg = to_dict(cfg.degradation)
    training_cfg = resolved_cfg["train"]
    reconstruction_cfg = resolved_cfg["reconstruction"]
    inference_cfg = resolved_cfg["inference"]

    data_num_workers = int(resolved_cfg["runtime"]["num_workers"])
    resolved_omega = OmegaConf.create(resolved_cfg)
    log.info("Resolved experiment config:\n%s", OmegaConf.to_yaml(resolved_omega))
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(resolved_omega), encoding="utf-8")
    log.info(
        "Resolved batch sizes: train=%d, reconstruction=%d, inference=%d; train_subset_size=%s",
        int(training_cfg["train_batch_size"]),
        int(reconstruction_cfg["batch_size"]),
        int(inference_cfg["batch_size"]),
        training_cfg.get("train_subset_size"),
    )
    log.info(
        "Resolved dataset experiment name=%s, source name=%s, image_size=%s",
        dataset_cfg["name"],
        dataset_source_name,
        dataset_cfg["size"],
    )

    model_tag = build_model_tag(model_cfg, training_cfg)
    checkpoint_root = resolve_output_path(str(cfg.output.checkpoint_root), PROJECT_ROOT)
    ckpt_dir = checkpoint_root / dataset_cfg["name"] / model_tag
    log.info("Resolved AE checkpoint directory: %s", ckpt_dir)

    inference_split = normalize_split(inference_cfg.get("split", "test"))

    ## 1. Prepare data loaders
    loader_deg_cfg = to_loader_degradation_cfg(degradation_cfg)

    train_clean_loader = get_degradation_dataloader(
        dataset=dataset_source_name,
        split="train",
        degradation_cfg={"mode": "clean"},
        batch_size=int(training_cfg["train_batch_size"]),
        num_workers=data_num_workers,
        image_size=int(dataset_cfg["size"]),
        subset_size=training_cfg.get("train_subset_size"),
    )
    val_loader = get_degradation_dataloader(
        dataset=dataset_source_name,
        split="val",
        degradation_cfg=loader_deg_cfg,
        batch_size=int(reconstruction_cfg["batch_size"]),
        num_workers=data_num_workers,
        image_size=int(dataset_cfg["size"]),
        shuffle=False,
    )
    clean_val_loader = get_degradation_dataloader(
        dataset=dataset_source_name,
        split="val",
        degradation_cfg={"mode": "clean"},
        batch_size=int(reconstruction_cfg["batch_size"]),
        num_workers=data_num_workers,
        image_size=int(dataset_cfg["size"]),
        shuffle=False,
    )
    inference_loader = get_degradation_dataloader(
        dataset=dataset_source_name,
        split=inference_split,
        degradation_cfg=loader_deg_cfg,
        batch_size=int(inference_cfg["batch_size"]),
        num_workers=data_num_workers,
        image_size=int(dataset_cfg["size"]),
        shuffle=False,
    )
    ## 2. Prepare, train, and evaluate multi-stage CAE models
    trainer = MultiStageCAETrainer(
        dataset_cfg=dataset_cfg,
        clean_train_loader=train_clean_loader,
        ckpt_dir=str(ckpt_dir),
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        device=None,
    )
    # Direct and feedforward share the original AE without preparing auxiliary checkpoints.
    if str(inference_cfg["solver"]).strip().lower() in {"direct", "feedforward"}:
        checkpoint_paths = [trainer.prepare_direct()]
    else:
        checkpoint_paths = trainer.prepare(
            clean_eval_loader=BatchFieldView(clean_val_loader, "clean"),
        )

    reconstruction_split = "val"
    ae_eval_cache_dir = build_ae_eval_cache_dir(
        ckpt_dir,
        degradation_cfg,
        reconstruction_split,
        int(reconstruction_cfg["eval_num_samples"]),
        str(inference_cfg["solver"]),
    )
    ae_rows, ae_cache_hit = evaluate_or_load_ae_rows(
        trainer=trainer,
        val_loader=val_loader,
        inference_cfg=inference_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        degradation_cfg=degradation_cfg,
        reconstruction_cfg=reconstruction_cfg,
        cache_dir=ae_eval_cache_dir,
    )

    ## 3. Run degradation recovery evaluation
    recovery_result = run_irae_recovery_eval(
        eval_loader=inference_loader,
        ckpt_dir=str(ckpt_dir),
        checkpoint_paths=checkpoint_paths,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        degradation_cfg=degradation_cfg,
        inference_cfg=inference_cfg,
        metrics_dir=metrics_dir,
        images_dir=images_dir,
        model_tag=model_tag,
        psnr_drops=training_cfg["auxiliary"]["psnr_drops"],
    )

    write_csv(metrics_dir / "ae_stage_metrics.csv", ae_rows)
    log.info("Saved AE stage metrics to %s", metrics_dir / "ae_stage_metrics.csv")
    log.info("Saved recovery artifacts: %s", recovery_result["artifact_paths"])
    log.info("AE cache hit: %s; AE cache dir: %s", ae_cache_hit, ae_eval_cache_dir)


if __name__ == "__main__":
    main()
