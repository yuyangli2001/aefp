"""Train the original AE and select auxiliary checkpoints on clean validation data."""

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.experiment_io import BatchFieldView, eval_metric, read_csv, write_csv
from utils.checkpoint_paths import direct_ckpt_path, jinit_cache_ckpt_path, jinit_cache_dir, jinit_cache_meta_path, jinit_cache_metrics_path, jinit_selected_dir, jinit_selected_ckpt_path, jinit_summary_path
from utils.configuration import normalize_model_cfg, normalize_training_cfg

from models.loading import build_ae_model, load_model_state

log = logging.getLogger(__name__)


# Manage clean training, cached candidates and selected auxiliary checkpoints.
class MultiStageCAETrainer:
    def __init__(
        self,
        *,
        dataset_cfg: Dict[str, Any],
        clean_train_loader,
        ckpt_dir: str,
        model_cfg: Optional[Dict[str, Any]],
        training_cfg: Optional[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ):
        self.clean_train_loader = clean_train_loader
        self.model_cfg = normalize_model_cfg(model_cfg)
        self.training_cfg = normalize_training_cfg(training_cfg)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model_cfg["params"]["in_channels"] = dataset_cfg["channels"]
        self.model_cfg["params"]["input_size"] = dataset_cfg["size"]

        self.direct_stage_ckpt = direct_ckpt_path(self.ckpt_dir)

    def _build_model(self):
        return build_ae_model(self.model_cfg).to(self.device)

    def _fit_model(
        self,
        model: torch.nn.Module,
        *,
        train_loader,
        ckpt_path: Path,
        epochs: int,
        shrinkage_weight: float,
    ):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "Training AE for %d epochs (lr=%s, shrinkage_weight=%s) -> %s",
            int(epochs),
            float(self.training_cfg["learning_rate"]),
            float(shrinkage_weight),
            ckpt_path,
        )
        model.fit(
            train_loader,
            device=self.device,
            learning_rate=float(self.training_cfg["learning_rate"]),
            epochs=int(epochs),
            verbose=True,
            shrinkage_weight=float(shrinkage_weight),
            save_ckpt_path=str(ckpt_path),
        )

    def _load_model_from_ckpt(self, ckpt_path: Path, freeze: bool = True):
        with torch.random.fork_rng():
            model = self._build_model()
            load_model_state(model, ckpt_path, self.device)
        model.eval()
        return model.requires_grad_(False) if freeze else model

    def _copy_checkpoint(self, source: Path, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def _require_clean_eval_loader(self, clean_eval_loader):
        if clean_eval_loader is None:
            raise ValueError("jinit training requires clean_eval_loader for PSNR-drop selection.")
        return clean_eval_loader

    def _eval_model(self, model, *, eval_img_loader, test_num_samples: Optional[int]):
        ssim, psnr = eval_metric(
            model, eval_img_loader, device=self.device, test_num_samples=test_num_samples,
            evaluation=self.selection_evaluation(),
        )
        return {"ssim": ssim, "psnr": psnr}

    def _ensure_direct_branch(self, *, train_loader, force_retrain: bool) -> Path:
        if self.direct_stage_ckpt.is_file() and not force_retrain:
            log.info("Using original AE checkpoint: %s", self.direct_stage_ckpt)
            return self.direct_stage_ckpt

        model = self._build_model()
        self._fit_model(
            model,
            train_loader=train_loader,
            ckpt_path=self.direct_stage_ckpt,
            epochs=self.training_cfg["original"]["epochs"],
            shrinkage_weight=0.0,
        )
        return self.direct_stage_ckpt

    def _clear_jinit_cache_tree(self):
        cache_root = self.ckpt_dir / "jinit"
        if cache_root.exists():
            if cache_root.is_symlink() or not cache_root.resolve().is_relative_to(self.ckpt_dir.resolve()):
                raise ValueError("Refusing to clear an external JInit cache directory.")
            shutil.rmtree(cache_root)

    # Adapt canonical settings to the existing cache metadata schema.
    def _cache_training_settings(self):
        aux = self.training_cfg["auxiliary"]
        return {"psnr_drops": list(aux["psnr_drops"]),
                "start_shrinkage_weight": aux["jacobian_weight_init"],
                "growth_factor": aux["jacobian_weight_multiplier"],
                "block_epochs": aux["epochs_per_candidate"], "max_candidates": aux["max_candidates"],
                "l1_weight": 0.0, "learning_rate": self.training_cfg["learning_rate"]}

    def _write_jinit_cache_meta(self):
        path = jinit_cache_meta_path(self.ckpt_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._cache_training_settings() | {"evaluation": self.selection_evaluation()}
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Reuse the validation and precision settings recorded with the candidates.
    def selection_evaluation(self):
        path = jinit_cache_meta_path(self.ckpt_dir)
        if hasattr(self, "_selection_evaluation"):
            return self._selection_evaluation
        evaluation = json.loads(path.read_text(encoding="utf-8")).get("evaluation") if path.is_file() else None
        return self._evaluation_protocol(evaluation)

    @staticmethod
    def _evaluation_protocol(evaluation):
        fields = {"split", "scope", "num_samples", "batch_size", "cudnn_allow_tf32", "matmul_allow_tf32"}
        return {key: value for key, value in evaluation.items() if key in fields} if evaluation else None

    def _load_jinit_cache_rows(self) -> List[Dict[str, Any]]:
        rows = read_csv(jinit_cache_metrics_path(self.ckpt_dir))
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            normalized.append(
                {
                    "stage": int(row["stage"]),
                    "shrinkage_weight": float(row["shrinkage_weight"]),
                    "l1_weight": float(row["l1_weight"]),
                    "clean_psnr": float(row["clean_psnr"]),
                    "clean_ssim": float(row["clean_ssim"]),
                    "direct_psnr": float(row["direct_psnr"]),
                    "psnr_drop_actual": float(row["psnr_drop_actual"]),
                }
            )
        normalized.sort(key=lambda item: int(item["stage"]))
        if [row["stage"] for row in normalized] != list(range(1, len(normalized) + 1)):
            raise ValueError("JInit cache stages must be contiguous and unique, starting at 1.")
        if any(not math.isfinite(value) for row in normalized for value in row.values()):
            raise ValueError("JInit cache metrics must be finite.")
        if normalized:
            path = jinit_cache_meta_path(self.ckpt_dir)
            if not path.is_file():
                raise FileNotFoundError(f"Missing JInit cache settings: {path}")
            previous = json.loads(path.read_text(encoding="utf-8"))
            current = self._cache_training_settings()
            fields = ("start_shrinkage_weight", "growth_factor", "block_epochs", "l1_weight")
            if any(previous.get(key) != current[key] for key in fields):
                raise ValueError("JInit cache training settings differ; use a separate checkpoint directory.")
            if "learning_rate" in previous and previous["learning_rate"] != self.training_cfg["learning_rate"]:
                raise ValueError("JInit cache learning_rate differs from this request.")
            for row in normalized:
                if not self._jinit_cache_stage_path(row["stage"]).is_file():
                    raise FileNotFoundError(f"Missing JInit cache stage {row['stage']}.")
        return normalized

    def _write_jinit_cache_rows(self, rows: List[Dict[str, Any]]):
        write_csv(jinit_cache_metrics_path(self.ckpt_dir), rows)

    def _jinit_cache_stage_path(self, stage_idx: int) -> Path:
        return jinit_cache_ckpt_path(self.ckpt_dir, stage_idx)

    def _select_jinit_cached_row(
        self,
        rows: List[Dict[str, Any]],
        *,
        psnr_drop: float,
    ) -> Optional[Dict[str, Any]]:
        selected_row: Optional[Dict[str, Any]] = None
        for row in rows:
            if float(row["psnr_drop_actual"]) <= psnr_drop:
                selected_row = row
                continue
            break
        return selected_row

    def _has_jinit_cached_overflow(
        self,
        rows: List[Dict[str, Any]],
        *,
        psnr_drop: float,
    ) -> bool:
        return any(float(row["psnr_drop_actual"]) > psnr_drop for row in rows)

    def _materialize_jinit_selection_group(
        self,
        *,
        selected_rows: List[Dict[str, Any]],
        direct_ckpt: Path,
        direct_psnr: float,
        psnr_drops: List[float],
        selected_dir: Path,
        summary_json: Path,
    ):
        selections = []
        selected_ckpts = []
        for idx, (psnr_drop, selected_row) in enumerate(zip(psnr_drops, selected_rows), start=1):
            source_stage = int(selected_row["stage"])
            source_ckpt = self._jinit_cache_stage_path(source_stage)
            if not source_ckpt.is_file():
                raise FileNotFoundError(
                    f"Missing jinit cache checkpoint for stage {source_stage}: {source_ckpt}"
                )
            selected_ckpt = selected_dir / f"stage_{idx}.pth"
            self._copy_checkpoint(source_ckpt, selected_ckpt)
            selected_ckpts.append(str(selected_ckpt))
            selections.append(
                {
                    "requested_psnr_drop": float(psnr_drop),
                    "selected_stage": source_stage,
                    "selected_shrinkage_weight": float(selected_row["shrinkage_weight"]),
                    "selected_clean_psnr": float(selected_row["clean_psnr"]),
                    "selected_clean_ssim": float(selected_row["clean_ssim"]),
                    "psnr_drop_actual": float(selected_row["psnr_drop_actual"]),
                    "selected_ckpt_path": str(selected_ckpt),
                    "source_stage": source_stage,
                    "source_cache_ckpt_path": str(source_ckpt),
                }
            )
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                {
                    "requested_psnr_drops": list(psnr_drops),
                    "selected_stages": [int(row["stage"]) for row in selected_rows],
                    "selected_jinit_ckpts": selected_ckpts,
                    "direct_psnr": direct_psnr,
                    "direct_ckpt_path": str(direct_ckpt),
                    "selections": selections,
                    "evaluation": self.selection_evaluation(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _prepare_jinit(
        self,
        *,
        train_loader,
        force_retrain: bool,
        clean_eval_loader,
    ) -> List[str]:
        clean_eval_loader = self._require_clean_eval_loader(clean_eval_loader)
        self.__dict__.pop("_selection_evaluation", None)
        had_direct_ckpt = self.direct_stage_ckpt.is_file()
        if force_retrain or not had_direct_ckpt:
            self._clear_jinit_cache_tree()

        direct_ckpt = self._ensure_direct_branch(train_loader=train_loader, force_retrain=force_retrain)
        psnr_drops = self.training_cfg["auxiliary"]["psnr_drops"]
        selected_dir = jinit_selected_dir(self.ckpt_dir, psnr_drops)
        selected_ckpts = [jinit_selected_ckpt_path(self.ckpt_dir, psnr_drops, idx) for idx in (1, 2)]
        summary_json = jinit_summary_path(self.ckpt_dir, psnr_drops)
        summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.is_file() else {}
        evaluation = self.selection_evaluation() or self._evaluation_protocol(summary.get("evaluation"))
        full_val = bool(evaluation and evaluation.get("scope") == "full_val" and evaluation.get("split") == "val")
        selection_current = full_val and self._evaluation_protocol(summary.get("evaluation")) == evaluation
        if all(path.is_file() for path in selected_ckpts) and not force_retrain and selection_current:
            self._selection_evaluation = evaluation
            log.info("Using existing full-validation auxiliary checkpoints: %s", selected_ckpts)
            return [str(path) for path in selected_ckpts] + [str(direct_ckpt)]

        cache_rows = self._load_jinit_cache_rows()
        if not cache_rows and any(path.is_file() for path in selected_ckpts) and not full_val:
            raise ValueError("Existing auxiliary checkpoints need full-validation candidate records before reselection.")
        self._selection_evaluation = dict(evaluation or {}, split="val", scope="full_val",
                                          num_samples=evaluation.get("num_samples") if full_val else None)
        if full_val and cache_rows:
            direct_psnr = float(cache_rows[0]["direct_psnr"])
        else:
            direct_model = self._load_model_from_ckpt(direct_ckpt, freeze=True)
            direct_psnr = float(self._eval_model(direct_model, eval_img_loader=clean_eval_loader,
                                                test_num_samples=None)["psnr"])
            # Re-evaluate cached weights on full validation; never retrain them for a protocol change.
            if cache_rows:
                summary_json.unlink(missing_ok=True)
                refreshed = []
                for row in cache_rows:
                    model = self._load_model_from_ckpt(self._jinit_cache_stage_path(row["stage"]), freeze=True)
                    metrics = self._eval_model(model, eval_img_loader=clean_eval_loader, test_num_samples=None)
                    refreshed.append(dict(row, clean_psnr=float(metrics["psnr"]), clean_ssim=float(metrics["ssim"]),
                                          direct_psnr=direct_psnr, psnr_drop_actual=direct_psnr - float(metrics["psnr"])))
                cache_rows = refreshed
                self._write_jinit_cache_rows(cache_rows)
                self._write_jinit_cache_meta()

        jinit_cfg = dict(self.training_cfg["auxiliary"])
        current_sw = float(jinit_cfg["jacobian_weight_init"])
        growth_factor = float(jinit_cfg["jacobian_weight_multiplier"])
        block_epochs = int(jinit_cfg["epochs_per_candidate"])
        max_candidates = int(jinit_cfg["max_candidates"])
        max_psnr_drop = max(psnr_drops)
        selected_rows = [self._select_jinit_cached_row(cache_rows, psnr_drop=value) for value in psnr_drops]
        for psnr_drop, selected_row in zip(psnr_drops, selected_rows):
            if selected_row is None and self._has_jinit_cached_overflow(cache_rows, psnr_drop=psnr_drop):
                raise RuntimeError(
                    f"No valid jinit checkpoint found for psnr_drop={psnr_drop}. "
                    "The first cached candidate already exceeded the threshold."
                )
        if all(selected_rows) and self._has_jinit_cached_overflow(cache_rows, psnr_drop=max_psnr_drop):
            self._materialize_jinit_selection_group(
                selected_rows=selected_rows,
                direct_ckpt=direct_ckpt,
                direct_psnr=direct_psnr,
                psnr_drops=psnr_drops,
                selected_dir=selected_dir,
                summary_json=summary_json,
            )
            return [str(path) for path in selected_ckpts] + [str(direct_ckpt)]

        if cache_rows:
            last_row = cache_rows[-1]
            current_stage = int(last_row["stage"])
            current_sw = float(last_row["shrinkage_weight"]) * growth_factor
            model = self._load_model_from_ckpt(self._jinit_cache_stage_path(current_stage), freeze=False)
        else:
            current_stage = 0
            current_sw = float(jinit_cfg["jacobian_weight_init"])
            model = self._build_model()
            load_model_state(model, direct_ckpt, self.device)

        cache_root = jinit_cache_dir(self.ckpt_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        self._write_jinit_cache_meta()

        while current_stage < max_candidates:
            candidate_idx = current_stage + 1
            candidate_ckpt = self._jinit_cache_stage_path(candidate_idx)
            if candidate_ckpt.exists():
                raise FileExistsError(f"Unrecorded JInit candidate already exists: {candidate_ckpt}")
            self._fit_model(
                model,
                train_loader=train_loader,
                ckpt_path=candidate_ckpt,
                epochs=block_epochs,
                shrinkage_weight=current_sw,
            )
            candidate_metrics = self._eval_model(
                model,
                eval_img_loader=clean_eval_loader,
                test_num_samples=None,
            )
            drop_actual = float(direct_psnr - candidate_metrics["psnr"])
            is_valid = drop_actual <= max_psnr_drop
            row = {
                "stage": candidate_idx,
                "shrinkage_weight": current_sw,
                "l1_weight": 0.0,
                "clean_psnr": float(candidate_metrics["psnr"]),
                "clean_ssim": float(candidate_metrics["ssim"]),
                "direct_psnr": direct_psnr,
                "psnr_drop_actual": drop_actual,
            }
            cache_rows.append(row)
            self._write_jinit_cache_rows(cache_rows)
            if is_valid:
                current_stage = candidate_idx
                current_sw *= growth_factor
                continue
            break

        selected_rows = [self._select_jinit_cached_row(cache_rows, psnr_drop=value) for value in psnr_drops]
        for psnr_drop, selected_row in zip(psnr_drops, selected_rows):
            if selected_row is None:
                raise RuntimeError(
                    f"No valid jinit checkpoint found for psnr_drop={psnr_drop}. "
                    "The first candidate already exceeded the threshold."
                )

        self._materialize_jinit_selection_group(
            selected_rows=selected_rows,
            direct_ckpt=direct_ckpt,
            direct_psnr=direct_psnr,
            psnr_drops=psnr_drops,
            selected_dir=selected_dir,
            summary_json=summary_json,
        )
        return [str(path) for path in selected_ckpts] + [str(direct_ckpt)]

    # Auxiliary selection always consumes the entire supplied clean validation loader.
    def prepare(self, *, clean_eval_loader=None) -> List[str]:
        return self._prepare_jinit(
            train_loader=BatchFieldView(self.clean_train_loader, "clean", "label"),
            force_retrain=bool(self.training_cfg.get("force_retrain", False)),
            clean_eval_loader=clean_eval_loader,
        )

    def prepare_direct(self) -> str:
        """Prepare only the original AE shared by direct, feedforward and JInit."""
        train_loader = BatchFieldView(self.clean_train_loader, "clean", "label")
        ckpt_path = self._ensure_direct_branch(
            train_loader=train_loader,
            force_retrain=bool(self.training_cfg.get("force_retrain", False)),
        )
        return str(ckpt_path)
