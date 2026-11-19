"""Measure autoencoder and recovery outputs and save stage previews."""

import json
from operator import index as integer_index
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.checkpoint_paths import format_tag_value
from utils.configuration import normalize_degradation_mode, normalize_split, parse_optional_int
from models.loading import load_ae_checkpoint
from utils.experiment_io import collect_preview, eval_metric, save_grid, to_float, write_csv
from recovery.batch import iter_irae_recovery_batches
from utils.utils_metrics import avg_psnr_pytorch, avg_ssim_pytorch
from recovery.cache import write_json


Tensor = torch.Tensor
_PREVIEW_LIMIT = 16


def evaluate_checkpoint_specs(
    checkpoint_specs, eval_img_loader, *, model_cfg, device, test_num_samples, evaluation, save_dir=None,
):
    preview = None
    if save_dir is not None:
        count = _PREVIEW_LIMIT if test_num_samples is None else min(_PREVIEW_LIMIT, test_num_samples)
        preview = collect_preview(eval_img_loader, count).to(device)
        save_grid(preview, save_dir / "input.png", "Input", count)

    results = []
    for spec in checkpoint_specs:
        with torch.random.fork_rng():
            model = load_ae_checkpoint(model_cfg, Path(spec["ckpt_path"]), device)
        ssim, psnr = eval_metric(
            model, eval_img_loader, device=device, test_num_samples=test_num_samples, evaluation=evaluation,
        )
        if preview is not None:
            save_grid(model.get_recon(preview), save_dir / f"stage_{spec['stage']}" / "reconstruction.png",
                      f"{spec['label']} Reconstruction", count)
        results.append(dict(spec, psnr=float(psnr), ssim=float(ssim)))
    return results


def _compute_batch_metrics(clean: Tensor, reconstruction: Tensor) -> Dict[str, float]:
    return {
        "psnr": to_float(avg_psnr_pytorch(clean, reconstruction)),
        "ssim": to_float(avg_ssim_pytorch(clean, reconstruction)),
    }


def _stage_label(solver: str, stage_idx: int, num_stages: int) -> str:
    solver = str(solver).strip().lower()
    if solver == "direct":
        return "direct_final"
    if solver == "feedforward":
        return "feedforward_final"
    if solver == "jinit":
        if num_stages != 3:
            raise ValueError(f"solver='jinit' requires exactly 3 recovery stages, got {num_stages}.")
        labels = ["jinit_coarse", "jinit_fine", "jinit_final"]
        return labels[stage_idx]
    raise ValueError(f"Unsupported solver: {solver}")


def _validate_stage_iterations(stage_iters) -> List[int]:
    values = [integer_index(value) for value in stage_iters]
    if len(values) != 3 or any(isinstance(value, bool) or value <= 0 for value in stage_iters):
        raise ValueError(f"solver='jinit' requires exactly 3 stage_iterations, got {len(values)}.")
    return values


def _resolve_recovery_weight_cfg(
    inference_cfg: Dict[str, Any],
    degradation_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    solver = str(inference_cfg.get("solver", "")).strip().lower()
    mode = normalize_degradation_mode(degradation_cfg["mode"])
    if solver == "feedforward":
        return {
            "mode": mode,
            "lambda": "",
            "lambda_init_gamma": "",
            "lambda_init": "",
            "aux_lambda_init": "",
        }
    if mode not in {"sparse", "inpainting", "deblur"}:
        raise ValueError(f"Unsupported degradation mode: {mode}")
    lambda_ = float(inference_cfg["lambda_init"])
    lambda_init_gamma = float(inference_cfg.get("aux_lambda_scale", 1.0))
    return {
        "mode": mode,
        "lambda": lambda_,
        "lambda_init_gamma": lambda_init_gamma,
        "lambda_init": lambda_,
        "aux_lambda_init": lambda_init_gamma * lambda_ if solver == "jinit" else "",
    }


def _build_recovery_iter_kwargs(
    inference_cfg: Dict[str, Any],
    degradation_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    solver = str(inference_cfg["solver"]).strip().lower()
    weight_cfg = _resolve_recovery_weight_cfg(inference_cfg, degradation_cfg)
    kwargs: Dict[str, Any] = {
        "solver": solver,
        "num_samples": parse_optional_int(inference_cfg.get("num_samples")),
        "device": inference_cfg.get('device'),
    }
    if solver == "feedforward":
        return kwargs
    kwargs.update(
        {
            "init_method": str(inference_cfg.get("init_method", "observation")),
            "lambda_multiplier": float(inference_cfg.get("lambda_multiplier", 0.8)),
            "lambda_check_interval": int(inference_cfg.get("lambda_update_interval", 10)),
            "backtracking_rule": str(inference_cfg.get("backtracking_rule", "sufficient_decrease")),
            "sufficient_decrease_constant": float(
                inference_cfg.get("sufficient_decrease_constant", 1e-4)
            ),
        }
    )
    kwargs["lambda_"] = weight_cfg["lambda"]
    kwargs["is_lambda_update"] = bool(inference_cfg.get("update_lambda", True))

    if solver == "direct":
        kwargs["max_iteration"] = int(inference_cfg["max_iterations"])
        return kwargs

    if solver == "jinit":
        stage_iters = _validate_stage_iterations(inference_cfg["stage_iterations"])
        kwargs["lambda_init_gamma"] = weight_cfg["lambda_init_gamma"]
        kwargs["stage_iterations"] = stage_iters
        kwargs["max_iteration"] = int(stage_iters[-1])
        return kwargs

    raise ValueError(f"Unsupported solver: {solver}")


class _PreviewCollector:
    def __init__(self, preview_limit: int):
        self.preview_limit = int(preview_limit)
        self.clean_batches: List[Tensor] = []
        self.observation_batches: List[Tensor] = []
        self.init_batches: List[Tensor] = []
        self.stage_reconstruction_batches: Dict[int, List[Tensor]] = {}

    def _append(self, store: List[Tensor], images: Tensor):
        current = sum(batch.size(0) for batch in store)
        if current >= self.preview_limit:
            return
        remain = self.preview_limit - current
        store.append(images[:remain].detach().cpu())

    def add_clean(self, images: Tensor):
        self._append(self.clean_batches, images)

    def add_observation(self, images: Tensor):
        self._append(self.observation_batches, images)

    def add_init(self, images: Tensor):
        self._append(self.init_batches, images)

    def add_stage_reconstruction(self, stage: int, images: Tensor):
        if stage not in self.stage_reconstruction_batches:
            self.stage_reconstruction_batches[stage] = []
        self._append(self.stage_reconstruction_batches[stage], images)

    @staticmethod
    def _stack(batches: List[Tensor]) -> Optional[Tensor]:
        if not batches:
            return None
        return torch.cat(batches, dim=0)

    def export(self) -> Dict[str, Any]:
        return {
            "clean": self._stack(self.clean_batches),
            "observation": self._stack(self.observation_batches),
            "init": self._stack(self.init_batches),
            "stage_recovered": {
                stage: self._stack(batches)
                for stage, batches in self.stage_reconstruction_batches.items()
            },
        }


class _RecoveryMetricAccumulator:
    def __init__(self, solver: str):
        self.solver = str(solver)
        self.total_samples = 0
        self.observation = {"psnr": 0.0, "ssim": 0.0}
        self.init = {"psnr": 0.0, "ssim": 0.0}
        self.stage_sums: List[Dict[str, float]] = []

    def update(self, pack: Dict[str, Any]):
        clean = pack["clean"]
        degraded = pack["degraded"]
        init = pack["init"]
        stage_recovered = list(pack["stage_recovered"])
        batch_size = int(clean.size(0))

        if not self.stage_sums:
            self.stage_sums = [{"psnr": 0.0, "ssim": 0.0} for _ in stage_recovered]
        elif len(stage_recovered) != len(self.stage_sums):
            raise ValueError(
                f"Inconsistent number of stage reconstructions: expected {len(self.stage_sums)}, "
                f"got {len(stage_recovered)}."
            )

        obs_metrics = _compute_batch_metrics(clean, degraded)
        init_metrics = _compute_batch_metrics(clean, init)
        self.observation["psnr"] += obs_metrics["psnr"] * batch_size
        self.observation["ssim"] += obs_metrics["ssim"] * batch_size
        self.init["psnr"] += init_metrics["psnr"] * batch_size
        self.init["ssim"] += init_metrics["ssim"] * batch_size

        for idx, reconstruction in enumerate(stage_recovered):
            metrics = _compute_batch_metrics(clean, reconstruction)
            self.stage_sums[idx]["psnr"] += metrics["psnr"] * batch_size
            self.stage_sums[idx]["ssim"] += metrics["ssim"] * batch_size

        self.total_samples += batch_size

    def finalize(
        self,
        *,
        dataset_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        degradation_cfg: Dict[str, Any],
        inference_cfg: Dict[str, Any],
        model_tag: str,
        ckpt_dir: Path,
        psnr_drops=None,
    ) -> Dict[str, Any]:
        if self.total_samples <= 0 or not self.stage_sums:
            raise ValueError("eval_loader produced no samples.")

        inference_split = normalize_split(inference_cfg.get("split", "test"))
        weight_cfg = _resolve_recovery_weight_cfg(inference_cfg, degradation_cfg)
        lambda_ = weight_cfg["lambda"]
        lambda_multiplier = (
            "" if self.solver == "feedforward" else float(inference_cfg.get("lambda_multiplier", 0.8))
        )
        lambda_check_interval = (
            "" if self.solver == "feedforward" else int(inference_cfg.get("lambda_update_interval", 10))
        )
        backtracking_rule = (
            "" if self.solver == "feedforward" else str(
                inference_cfg.get("backtracking_rule", "sufficient_decrease")
            )
        )
        sufficient_decrease_constant = (
            "" if self.solver == "feedforward" else float(
                inference_cfg.get("sufficient_decrease_constant", 1e-4)
            )
        )
        is_lambda_update = (
            ""
            if self.solver == "feedforward"
            else bool(inference_cfg.get("update_lambda", True))
        )
        init_method = "unused" if self.solver == "feedforward" else str(
            inference_cfg.get("init_method", "observation")
        )
        ckpt_dir_str = str(ckpt_dir)
        jinit_stage_iters = ""
        lambda_init_gamma = ""
        lambda_init = lambda_
        aux_lambda_init = ""
        if self.solver == "jinit":
            lambda_init_gamma = weight_cfg["lambda_init_gamma"]
            aux_lambda_init = weight_cfg["aux_lambda_init"]
            jinit_stage_iters = json.dumps(_validate_stage_iterations(inference_cfg["stage_iterations"]))

        stage_rows = []
        for idx, stage_sum in enumerate(self.stage_sums):
            avg_psnr = stage_sum["psnr"] / self.total_samples
            avg_ssim = stage_sum["ssim"] / self.total_samples
            stage_rows.append(
                {
                    "dataset": str(dataset_cfg["name"]),
                    "model_tag": model_tag,
                    "model_variant": str(model_cfg["variant"]),
                    "degradation_mode": str(degradation_cfg["mode"]),
                    "degradation_params": str(degradation_cfg.get("params", {})),
                    "inference_split": inference_split,
                    "solver": self.solver,
                    "lambda": lambda_,
                    "init_method": init_method,
                    "lambda_multiplier": lambda_multiplier,
                    "lambda_check_interval": lambda_check_interval,
                    "backtracking_rule": backtracking_rule,
                    "sufficient_decrease_constant": sufficient_decrease_constant,
                    "is_lambda_update": is_lambda_update,
                    "lambda_init_gamma": lambda_init_gamma,
                    "lambda_init": lambda_init,
                    "aux_lambda_init": aux_lambda_init,
                    "jinit_stage_iters": jinit_stage_iters,
                    "stage": idx + 1,
                    "label": _stage_label(self.solver, idx, len(self.stage_sums)),
                    "psnr": avg_psnr,
                    "ssim": avg_ssim,
                    "checkpoint_dir": ckpt_dir_str,
                }
            )

        final_stage_row = stage_rows[-1]
        final_row = {
            "dataset": str(dataset_cfg["name"]),
            "model_tag": model_tag,
            "model_variant": str(model_cfg["variant"]),
            "degradation_mode": str(degradation_cfg["mode"]),
            "degradation_params": str(degradation_cfg.get("params", {})),
            "inference_split": inference_split,
            "solver": self.solver,
            "lambda": lambda_,
            "init_method": init_method,
            "lambda_multiplier": lambda_multiplier,
            "lambda_check_interval": lambda_check_interval,
            "backtracking_rule": backtracking_rule,
            "sufficient_decrease_constant": sufficient_decrease_constant,
            "is_lambda_update": is_lambda_update,
            "jinit_stage_iters": jinit_stage_iters,
            "observation_psnr": self.observation["psnr"] / self.total_samples,
            "observation_ssim": self.observation["ssim"] / self.total_samples,
            "init_psnr": self.init["psnr"] / self.total_samples,
            "init_ssim": self.init["ssim"] / self.total_samples,
            "processed_init_stage": "",
            "processed_init_label": "",
            "processed_init_psnr": "",
            "processed_init_ssim": "",
            "lambda_init_gamma": lambda_init_gamma,
            "lambda_init": lambda_init,
            "aux_lambda_init": aux_lambda_init,
            "final_stage": final_stage_row["stage"],
            "final_label": final_stage_row["label"],
            "final_psnr": final_stage_row["psnr"],
            "final_ssim": final_stage_row["ssim"],
            "checkpoint_dir": ckpt_dir_str,
        }
        if self.solver == "jinit":
            processed_init_row = stage_rows[1]
            final_row["actual_solver_stage_count"] = len(stage_rows)
            final_row["processed_init_stage"] = processed_init_row["stage"]
            final_row["processed_init_label"] = processed_init_row["label"]
            final_row["processed_init_psnr"] = processed_init_row["psnr"]
            final_row["processed_init_ssim"] = processed_init_row["ssim"]
            if psnr_drops is not None:
                final_row["train_psnr_drops"] = json.dumps(list(psnr_drops))

        summary_payload = dict(final_row)
        summary_payload["num_samples"] = self.total_samples
        summary_payload["num_recovery_stages"] = len(stage_rows)
        summary_payload["stage_rows"] = stage_rows

        return {
            "stage_rows": stage_rows,
            "final_row": final_row,
            "summary_payload": summary_payload,
        }


def _write_recovery_artifacts(
    *,
    metrics_dir: Path,
    images_dir: Path,
    solver: str,
    lambda_: Any,
    init_method: str,
    preview_data: Dict[str, Any],
    observation_metrics: Dict[str, float],
    init_metrics: Dict[str, float],
    stage_rows: List[Dict[str, Any]],
    final_row: Dict[str, Any],
    summary_payload: Dict[str, Any],
) -> Dict[str, str]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    stage_metrics_csv = metrics_dir / "recovery_stage_metrics.csv"
    final_summary_csv = metrics_dir / "recovery_final_summary.csv"
    summary_json = metrics_dir / "recovery_summary.json"
    solver_image_dir = images_dir / "recovery" / solver / f"lambda_{format_tag_value(lambda_)}"
    solver_image_dir.mkdir(parents=True, exist_ok=True)

    write_csv(stage_metrics_csv, stage_rows)
    write_csv(final_summary_csv, [final_row])

    clean_preview = preview_data["clean"]
    if clean_preview is not None:
        save_grid(clean_preview, solver_image_dir / "clean.png", "Clean", _PREVIEW_LIMIT)

    observation_preview = preview_data["observation"]
    if observation_preview is not None:
        save_grid(
            observation_preview,
            solver_image_dir / "observation.png",
            f"Observation PSNR:{observation_metrics['psnr']:.2f} SSIM:{observation_metrics['ssim']:.4f}",
            _PREVIEW_LIMIT,
        )

    init_preview = preview_data["init"]
    if init_preview is not None:
        save_grid(
            init_preview,
            solver_image_dir / f"init_{init_method}.png",
            f"Init ({init_method}) PSNR:{init_metrics['psnr']:.2f} SSIM:{init_metrics['ssim']:.4f}",
            _PREVIEW_LIMIT,
        )

    stage_previews = dict(preview_data["stage_recovered"])
    for row in stage_rows:
        stage_idx = int(row["stage"]) - 1
        stage_preview = stage_previews.get(stage_idx)
        if stage_preview is None:
            continue
        save_grid(
            stage_preview,
            solver_image_dir / f"stage_{row['stage']}" / "reconstruction.png",
            f"{row['label']} PSNR:{row['psnr']:.2f} SSIM:{row['ssim']:.4f}",
            _PREVIEW_LIMIT,
        )

    write_json(summary_json, summary_payload)

    return {
        "stage_metrics_csv": str(stage_metrics_csv),
        "final_summary_csv": str(final_summary_csv),
        "summary_json": str(summary_json),
        "image_dir": str(solver_image_dir),
    }


def run_irae_recovery_eval(
    *,
    eval_loader,
    ckpt_dir,
    checkpoint_paths,
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    degradation_cfg: Dict[str, Any],
    inference_cfg: Dict[str, Any],
    metrics_dir,
    images_dir,
    model_tag: str,
    psnr_drops=None,
    batch_callback=None,
) -> Dict[str, Any]:
    """Aggregate recovery metrics and save stage previews."""
    solver = str(inference_cfg["solver"]).strip().lower()
    weight_cfg = _resolve_recovery_weight_cfg(inference_cfg, degradation_cfg)
    lambda_ = weight_cfg["lambda"]
    init_method = "unused" if solver == "feedforward" else str(
        inference_cfg.get("init_method", "observation")
    )
    ckpt_dir = Path(ckpt_dir)
    metrics_dir = Path(metrics_dir)
    images_dir = Path(images_dir)

    iter_kwargs = _build_recovery_iter_kwargs(inference_cfg, degradation_cfg)
    expected = iter_kwargs["num_samples"]
    if expected is None and hasattr(eval_loader, "dataset"):
        expected = len(eval_loader.dataset)
    if expected is not None and (expected <= 0 or (hasattr(eval_loader, "dataset") and expected > len(eval_loader.dataset))):
        raise ValueError("Requested sample count must be positive and fit the evaluation split.")
    status_path = metrics_dir / "recovery_status.json"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, dict(complete=False, requested_samples=expected))
    preview_collector = _PreviewCollector(_PREVIEW_LIMIT)
    metrics = _RecoveryMetricAccumulator(solver)
    seen_indices = set()

    for pack in iter_irae_recovery_batches(
        eval_loader=eval_loader,
        checkpoint_paths=checkpoint_paths,
        model_cfg=model_cfg,
        **iter_kwargs,
    ):
        # Validate every batch before exporting tensors or accumulating a partial metric.
        clean, stages = pack["clean"], pack["stage_recovered"]
        images = [pack["degraded"], pack["init"], pack["recovered"], *stages]
        if (clean.ndim != 4 or not len(clean) or not torch.isfinite(clean).all()
                or len(stages) != (3 if solver == "jinit" else 1)
                or any(image.shape != clean.shape or not torch.isfinite(image).all() for image in images)
                or not torch.equal(pack["recovered"], stages[-1])):
            raise ValueError("Invalid recovery batch shapes, values, stages, or final reconstruction.")
        indices = pack["index"]
        if (indices.shape != (len(clean),) or indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or (indices < 0).any()
                or len(indices.unique()) != len(indices) or seen_indices.intersection(indices.tolist())):
            raise ValueError("Recovery sample indices must be unique nonnegative integers.")
        seen_indices.update(indices.tolist())
        if batch_callback is not None:
            batch_callback(pack)
        preview_collector.add_clean(pack["clean"])
        preview_collector.add_observation(pack["degraded"])
        preview_collector.add_init(pack["init"])
        for idx, reconstruction in enumerate(pack["stage_recovered"]):
            preview_collector.add_stage_reconstruction(idx, reconstruction)

        metrics.update(pack)

    if expected is not None and metrics.total_samples != expected:
        raise ValueError("Recovery sample coverage does not match the requested count.")
    finalized = metrics.finalize(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        degradation_cfg=degradation_cfg,
        inference_cfg=inference_cfg,
        model_tag=model_tag,
        ckpt_dir=ckpt_dir,
        psnr_drops=psnr_drops,
    )
    ckpt_paths = [str(path) for path in checkpoint_paths]
    if ckpt_paths:
        ckpt_paths_str = json.dumps(ckpt_paths)
        for row in finalized["stage_rows"]:
            row["ckpt_paths"] = ckpt_paths_str
        finalized["final_row"]["ckpt_paths"] = ckpt_paths_str
        finalized["summary_payload"]["ckpt_paths"] = ckpt_paths
        if solver == "jinit":
            selected_jinit_ckpts = ckpt_paths[:-1]
            selected_jinit_stages = list(range(1, len(selected_jinit_ckpts) + 1))
            finalized["final_row"]["selected_jinit_ckpts"] = json.dumps(selected_jinit_ckpts)
            finalized["final_row"]["selected_jinit_stages"] = json.dumps(selected_jinit_stages)
            finalized["summary_payload"]["selected_jinit_ckpts"] = selected_jinit_ckpts
            finalized["summary_payload"]["selected_jinit_stages"] = selected_jinit_stages

    preview_data = preview_collector.export()
    artifact_paths = _write_recovery_artifacts(
        metrics_dir=metrics_dir,
        images_dir=images_dir,
        solver=solver,
        lambda_=("none" if lambda_ == "" else lambda_),
        init_method=init_method,
        preview_data=preview_data,
        observation_metrics={
            "psnr": finalized["final_row"]["observation_psnr"],
            "ssim": finalized["final_row"]["observation_ssim"],
        },
        init_metrics={
            "psnr": finalized["final_row"]["init_psnr"],
            "ssim": finalized["final_row"]["init_ssim"],
        },
        stage_rows=finalized["stage_rows"],
        final_row=finalized["final_row"],
        summary_payload=finalized["summary_payload"],
    )

    result = {
        "solver": solver,
        "stage_rows": finalized["stage_rows"],
        "final_row": finalized["final_row"],
        "summary_payload": finalized["summary_payload"],
        "artifact_paths": artifact_paths,
    }
    if ckpt_paths:
        result["ckpt_paths"] = ckpt_paths
    write_json(status_path, dict(complete=True, requested_samples=expected, num_samples=metrics.total_samples))
    return result
