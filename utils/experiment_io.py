"""Read metrics, save previews, and evaluate autoencoder reconstructions."""

import csv
from contextlib import contextmanager
import random
import math
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from utils.utils_metrics import avg_psnr_pytorch, avg_ssim_pytorch
from utils.utils_plot import grid_images


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def grid_shape(num_images: int):
    side = max(1, int(ceil(sqrt(num_images))))
    return side, side


def save_grid(images: torch.Tensor, save_path: Path, title: str, limit: int):
    import matplotlib.pyplot as plt
    save_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(limit, images.size(0))
    figure = grid_images(
        images[:count].cpu(),
        grid_shape=grid_shape(count),
        show=False,
        title=title,
        save_path=str(save_path),
    )
    # Release each saved preview before the next dataset or recovery stage.
    plt.close(figure)


def _as_image_batch(batch) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Empty batch tuple/list is not a valid image batch.")
        image_batch = batch[0]
        if not torch.is_tensor(image_batch):
            raise TypeError(f"Expected image tensor in batch[0], got {type(image_batch)}")
        return image_batch
    if isinstance(batch, dict):
        for key in ("image", "input", "clean", "degraded", "target"):
            if key in batch:
                image_batch = batch[key]
                if not torch.is_tensor(image_batch):
                    raise TypeError(f"Expected tensor for batch['{key}'], got {type(image_batch)}")
                return image_batch
    raise TypeError(f"Unsupported image batch type: {type(batch)}")


def collect_preview(eval_img_loader, num_samples: int):
    image_batches = []
    total = 0
    for batch in eval_img_loader:
        images = _as_image_batch(batch)
        image_batches.append(images)
        total += images.size(0)
        if total >= num_samples:
            break
    if not image_batches:
        raise ValueError("eval_img_loader produced no batches.")
    return torch.cat(image_batches, dim=0)[:num_samples]


class BatchFieldView:
    def __init__(self, loader, *field_names: str):
        if not field_names:
            raise ValueError("BatchFieldView requires at least one field name.")
        self.loader = loader
        self.field_names = tuple(field_names)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            values = tuple(batch[field_name] for field_name in self.field_names)
            yield values[0] if len(values) == 1 else values


def combine_stage_metrics(
    clean_stage_metrics: List[Dict[str, Any]],
    degraded_stage_metrics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(clean_stage_metrics) != len(degraded_stage_metrics):
        raise ValueError("clean_stage_metrics and degraded_stage_metrics must have the same length.")

    combined = []
    for clean_metric, degraded_metric in zip(clean_stage_metrics, degraded_stage_metrics):
        if int(clean_metric["stage"]) != int(degraded_metric["stage"]):
            raise ValueError("Stage indices do not match between clean and degraded metrics.")
        combined.append(
            {
                "stage": int(clean_metric["stage"]),
                "shrinkage_weight": float(clean_metric["shrinkage_weight"]),
                "l1_weight": float(clean_metric["l1_weight"]),
                "clean_psnr": float(clean_metric["psnr"]),
                "clean_ssim": float(clean_metric["ssim"]),
                "degraded_psnr": float(degraded_metric["psnr"]),
                "degraded_ssim": float(degraded_metric["ssim"]),
                "ckpt_path": clean_metric["ckpt_path"],
            }
        )
    return combined


# Validation must not consume the random stream used by the next training stage.
@contextmanager
def evaluation_scope(loader, model):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    raw_loader = loader
    while isinstance(raw_loader, BatchFieldView):
        raw_loader = raw_loader.loader
    generator = getattr(raw_loader, "generator", None)
    generator_state = generator.get_state() if generator is not None else None
    training = model.training
    with torch.random.fork_rng():
        try:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if generator is not None:
                generator.manual_seed(0)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if generator is not None:
                generator.set_state(generator_state)
            model.train(training)


def eval_metric(ae_model, eval_img_loader, *, device: torch.device,
                test_num_samples: Optional[int], evaluation=None):
    if test_num_samples is not None and test_num_samples <= 0:
        raise ValueError("test_num_samples must be positive or None.")
    evaluation = evaluation or {}
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = evaluation.get("matmul_allow_tf32", matmul_tf32)
        with evaluation_scope(eval_img_loader, ae_model), torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled, benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=evaluation.get("cudnn_allow_tf32", torch.backends.cudnn.allow_tf32),
        ):
            return _eval_metric(ae_model, eval_img_loader, device=device, test_num_samples=test_num_samples)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def _eval_metric(ae_model, eval_img_loader, *, device, test_num_samples):
    ae_model.to(device).eval()
    total_ssim = 0.0
    total_psnr = 0.0
    total_samples = 0

    for batch in eval_img_loader:
        input_imgs = _as_image_batch(batch).to(device)

        batch_size = input_imgs.size(0)
        if test_num_samples is not None:
            remain = test_num_samples - total_samples
            if remain <= 0:
                break
            if batch_size > remain:
                input_imgs = input_imgs[:remain]
                batch_size = remain

        with torch.no_grad():
            recon = ae_model.get_recon(input_imgs)
        if recon.shape != input_imgs.shape or not torch.isfinite(recon).all():
            raise ValueError("AE evaluation output must be finite and match the input shape.")

        total_ssim += to_float(avg_ssim_pytorch(input_imgs, recon)) * batch_size
        total_psnr += to_float(avg_psnr_pytorch(input_imgs, recon)) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The AE evaluation loader is empty.")
    if not all(math.isfinite(value) for value in (total_ssim, total_psnr)):
        raise FloatingPointError("AE evaluation metrics must be finite.")
    return total_ssim / total_samples, total_psnr / total_samples
