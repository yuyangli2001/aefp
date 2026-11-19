"""Load clean images, apply degradations, and assemble paired batches."""


from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite
from operator import index as integer_index
from pathlib import Path
from typing import Any, Dict, Optional, Union, Sequence
import random

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from PIL import Image

from utils.degradation import (
    BaseDegradationConfig, CleanConfig, DeblurConfig, DegradationConfig,
    InpaintingConfig, SparseConfig, SUPPORTED_DEGRADATION_MODES, SUPPORTED_INPAINTING_MASK_MODES,
    build_degradation_config,
)


# Supported datasets, degradation modes and preprocessing defaults.

SUPPORTED_DATASETS = {"mnist", "cifar10", "celeba", "celeba-hq-256"}
LOCAL_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_TRANSFORM_SPECS = {
    "mnist": {
        "default_size": 28,
        "crop_size": None,
    },
    "cifar10": {
        "default_size": 32,
        "crop_size": None,
    },
    "celeba": {
        "default_size": 128,
        "crop_size": 178,
    },
    "celeba-hq-256": {
        "default_size": 256,
        "crop_size": None,
    },
}


# Clean datasets contain no degradation logic.
# Keep reading and partitioning separate from the restoration operators.

class LocalImageDataset(Dataset):
    """Read sorted RGB images from images/ or an explicit train/val/test directory."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Any] = None,
    ):
        self.root = Path(root)
        self.split = str(split).strip().lower()
        self.transform = transform

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Supported: ['train', 'val', 'test']")

        self.samples = self._resolve_samples()
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No image files found under: {self.root}")

    @staticmethod
    def _list_images(directory: Path) -> list[Path]:
        if not directory.exists() or not directory.is_dir():
            return []
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in LOCAL_IMAGE_SUFFIXES
        )

    def _resolve_samples(self) -> list[Path]:
        split_dirs = any((self.root / name).is_dir() for name in ("train", "val", "test"))
        if split_dirs and self._list_images(self.root / "images"):
            raise ValueError("Use either images/ or split directories, not both.")
        split_dir = self.root / self.split
        split_samples = self._list_images(split_dir)
        if split_samples:
            return split_samples
        if split_dirs:
            raise FileNotFoundError(f"Missing or empty local split: {split_dir}")

        image_samples = self._list_images(self.root / "images")
        if image_samples:
            return image_samples

        raise FileNotFoundError(
            "Expected local images under either `images/` or `train|val|test/` directories "
            f"inside: {self.root}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, None]:
        image_path = self.samples[int(idx)]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, None


class BaseImageDatasetFactory:
    """Read clean datasets, apply transforms and construct contiguous splits."""
    # Default val_size=None uses a 9:1 train/val split, or 9:1:2 for unsplit data; explicit val_size keeps the test fraction and takes validation from the training tail, with a nondefault-split notice.
    def __init__(
        self,
        dataset_name: str,
        root: str | Path,
        image_size: Optional[int] = None,
        val_size: Optional[int] = None,
        subset_size: Optional[Union[int, float, str]] = None,
    ):
        self.dataset_name = dataset_name.strip().lower()
        self.root = Path(root)
        self.image_size = image_size
        self.val_size = None if val_size is None else integer_index(val_size)
        if self.val_size is not None and self.val_size <= 0:
            raise ValueError(f"`val_size` must be positive when provided, got {val_size}")
        self.subset_size = subset_size
        self._manual_val_size_logged = False
        self._manual_subset_size_logged = False

    def _resolve_subset_len(self, total: int) -> Optional[int]:
        if self.subset_size is None:
            return None

        raw_value = self.subset_size
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            raw_value = float(raw_value) if any(ch in raw_value for ch in ".eE") else int(raw_value)
        if not isfinite(raw_value):
            raise ValueError("subset_size must be finite.")

        if isinstance(raw_value, float) and raw_value <= 1.0:
            if raw_value <= 0.0:
                raise ValueError(f"`subset_size` ratio must be in (0, 1], got {self.subset_size}")
            subset_n = max(1, int(total * raw_value))
        else:
            if isinstance(raw_value, float) and not isclose(raw_value, round(raw_value)):
                raise ValueError(
                    f"`subset_size` as sample count must be an integer, got {self.subset_size}"
                )
            subset_n = int(round(raw_value))
            if subset_n <= 0:
                raise ValueError(f"`subset_size` must be positive when provided, got {self.subset_size}")

        if subset_n > total:
            raise ValueError(f"`subset_size` is too large: subset_size={self.subset_size}, total={total}")
        return subset_n

    def _maybe_subset_train_dataset(self, ds: Dataset, split: str) -> Dataset:
        if split != "train":
            return ds

        subset_n = self._resolve_subset_len(len(ds))
        if subset_n is None or subset_n == len(ds):
            return ds

        if not self._manual_subset_size_logged:
            print(
                "[INFO] Non-default training subset: "
                f"dataset={self.dataset_name}, train_total={len(ds)}, "
                f"train_subset={subset_n}, requested_subset_size={self.subset_size}"
            )
            self._manual_subset_size_logged = True
        return Subset(ds, list(range(subset_n)))

    def _split_train_val_dataset(self, ds: Dataset, split: str) -> Dataset:
        total = len(ds)
        if total < 2:
            raise ValueError(f"Need at least 2 samples to build train/val split, got {total}")

        if self.val_size is None:
            val_n = max(1, int(round(total * 0.1)))
        else:
            val_n = int(self.val_size)

        if val_n >= total:
            raise ValueError(
                f"`val_size` is too large for dataset `{self.dataset_name}`: "
                f"total={total}, val_size={val_n}"
            )

        train_n = total - val_n
        if self.val_size is not None and not self._manual_val_size_logged:
            print(
                "[INFO] Non-default validation split: "
                f"dataset={self.dataset_name}, total_train={total}, "
                f"train={train_n}, val={val_n}, requested_val_size={self.val_size}"
            )
            self._manual_val_size_logged = True
        indices = list(range(total))
        subset_indices = indices[:train_n] if split == "train" else indices[train_n:]
        return Subset(ds, subset_indices)

    def _split_train_val_test_dataset(self, ds: Dataset, split: str) -> Dataset:
        total = len(ds)
        if total < 3:
            raise ValueError(f"Need at least 3 samples to build train/val/test split, got {total}")

        test_n = max(1, int(round(total / 6.0)))
        max_val_n = total - test_n - 1
        if max_val_n <= 0:
            raise ValueError(
                f"Dataset is too small for 9:1:2 split: total={total}, reserved_test={test_n}"
            )

        if self.val_size is None:
            val_n = max(1, int(round(total / 12.0)))
            val_n = min(val_n, max_val_n)
        else:
            val_n = int(self.val_size)
            if val_n > max_val_n:
                raise ValueError(
                    f"`val_size` is too large for dataset `{self.dataset_name}`: "
                    f"total={total}, reserved_test={test_n}, val_size={val_n}"
                )

        train_n = total - val_n - test_n
        if train_n <= 0:
            raise ValueError(
                f"Invalid split sizes for dataset `{self.dataset_name}`: "
                f"total={total}, train={train_n}, val={val_n}, test={test_n}"
            )

        if self.val_size is not None and not self._manual_val_size_logged:
            print(
                "[INFO] Non-default validation split: "
                f"dataset={self.dataset_name}, total={total}, "
                f"train={train_n}, val={val_n}, test={test_n}, "
                f"requested_val_size={self.val_size}"
            )
            self._manual_val_size_logged = True

        if split == "train":
            subset_indices = list(range(train_n))
        elif split == "val":
            subset_indices = list(range(train_n, train_n + val_n))
        else:
            subset_indices = list(range(train_n + val_n, total))
        return Subset(ds, subset_indices)

    def _has_local_split_dirs(self, dataset_root: Path) -> bool:
        return any((dataset_root / name).is_dir() for name in ("train", "val", "test"))

    def _build_local_image_dataset(self, dataset_root: Path, split: str, transform: Any) -> Dataset:
        has_split_dirs = self._has_local_split_dirs(dataset_root)

        if has_split_dirs:
            if split == "test":
                test_dir = dataset_root / "test"
                if test_dir.is_dir():
                    return LocalImageDataset(root=dataset_root, split="test", transform=transform)

            train_dir = dataset_root / "train"
            val_dir = dataset_root / "val"
            if self.val_size is None and split == "val" and val_dir.is_dir():
                return LocalImageDataset(root=dataset_root, split="val", transform=transform)
            if self.val_size is None and split == "train" and train_dir.is_dir() and val_dir.is_dir():
                return LocalImageDataset(root=dataset_root, split="train", transform=transform)
            if train_dir.is_dir() and split in {"train", "val"}:
                train_ds = LocalImageDataset(root=dataset_root, split="train", transform=transform)
                return self._split_train_val_dataset(train_ds, split)

        image_ds = LocalImageDataset(root=dataset_root, split=split, transform=transform)
        return self._split_train_val_test_dataset(image_ds, split)

    def _resolve_dataset_root(self) -> Path:
        root_name = self.root.name.lower()
        dataset_root = self.root if root_name == self.dataset_name else (self.root / self.dataset_name)

        if self.dataset_name != "cifar10":
            return dataset_root

        candidates = [dataset_root]
        if self.root not in candidates:
            candidates.append(self.root)
        if self.root.parent not in candidates:
            candidates.append(self.root.parent)

        for candidate in candidates:
            if (candidate / "cifar-10-batches-py").exists():
                return candidate

        return dataset_root

    def build(self, split: str) -> Dataset:
        """Return clean images and labels; unsplit local images use contiguous 9:1:2 partitions."""
        split = str(split).strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Supported: ['train', 'val', 'test']")
        if self.dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}. "
                f"Supported: {sorted(SUPPORTED_DATASETS)}"
            )

        from torchvision import datasets

        transform = build_base_transform(self.dataset_name, self.image_size)
        dataset_root = self._resolve_dataset_root()
        dataset_root.mkdir(parents=True, exist_ok=True)

        if self.dataset_name == "mnist":
            if split == "test":
                return datasets.MNIST(
                    root=str(dataset_root),
                    train=False,
                    download=True,
                    transform=transform,
                )

            ds: Dataset = datasets.MNIST(
                root=str(dataset_root),
                train=True,
                download=True,
                transform=transform,
            )
            return self._maybe_subset_train_dataset(self._split_train_val_dataset(ds, split), split)

        if self.dataset_name == "cifar10":
            if split == "test":
                return datasets.CIFAR10(
                    root=str(dataset_root),
                    train=False,
                    download=True,
                    transform=transform,
                )

            ds = datasets.CIFAR10(
                root=str(dataset_root),
                train=True,
                download=True,
                transform=transform,
            )
            return self._maybe_subset_train_dataset(self._split_train_val_dataset(ds, split), split)

        if self.dataset_name == "celeba":
            if split == "test":
                return datasets.CelebA(
                    root=str(dataset_root),
                    split="test",
                    download=True,
                    transform=transform,
                )

            ds = datasets.CelebA(
                root=str(dataset_root),
                split="train",
                download=True,
                transform=transform,
            )
            return self._maybe_subset_train_dataset(self._split_train_val_dataset(ds, split), split)

        return self._maybe_subset_train_dataset(self._build_local_image_dataset(dataset_root, split, transform), split)


def build_base_transform(dataset_name: str, image_size: Optional[int]):
    """Apply the dataset crop, fixed square resize and conversion to [0,1] tensors."""
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported: {sorted(SUPPORTED_DATASETS)}"
        )

    from torchvision import transforms

    spec = DATASET_TRANSFORM_SPECS[dataset_name]
    tfms = []

    crop_size = spec["crop_size"]
    if crop_size is not None:
        tfms.append(transforms.CenterCrop(crop_size))

    default_size = spec["default_size"]
    resize_size = default_size if image_size is None else integer_index(image_size)
    if resize_size is not None:
        if resize_size <= 0:
            raise ValueError(f"`image_size` must be positive, got {image_size}")
        tfms.append(transforms.Resize((resize_size, resize_size)))

    tfms.append(transforms.ToTensor())
    return transforms.Compose(tfms)


# Operators map clean images to observations and solver parameters.
# Each inverse problem supplies its operator here.

class BaseDegradation:
    """Map one CHW clean image and its split-local index to an observation and params."""
    def __init__(self, cfg: BaseDegradationConfig):
        self.cfg = cfg

    def apply(self, clean: Tensor, idx: int) -> tuple[Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def get_shared_params(self) -> Dict[str, Any]:
        """Return fixed operator parameters shared across samples."""
        return {}

    @staticmethod
    def _make_generator(seed: int, device: torch.device) -> torch.Generator:
        gen_device = device if device.type == "cuda" else "cpu"
        g = torch.Generator(device=gen_device)
        g.manual_seed(int(seed))
        return g

    def _rand(
        self,
        shape: tuple[int, ...],
        *,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.cfg.deterministic:
            g = self._make_generator(seed=seed, device=device)
            return torch.rand(shape, generator=g, device=device, dtype=dtype)
        return torch.rand(shape, device=device, dtype=dtype)

    def _randn(
        self,
        shape: tuple[int, ...],
        *,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.cfg.deterministic:
            g = self._make_generator(seed=seed, device=device)
            return torch.randn(shape, generator=g, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype)


class CleanDegradation(BaseDegradation):
    """Clone the clean image without extra operator parameters."""
    def apply(self, clean: Tensor, idx: int) -> tuple[Tensor, Dict[str, Any]]:
        return clean.clone(), {}


class SparseDegradation(BaseDegradation):
    """Use seed + index for deterministic corruption, or the worker RNG during training."""
    def __init__(self, cfg: BaseDegradationConfig):
        super().__init__(cfg)
        if not isinstance(cfg, SparseConfig):
            raise TypeError(f"SparseDegradation expects SparseConfig, got {type(cfg)}")

    def apply(self, clean: Tensor, idx: int) -> tuple[Tensor, Dict[str, Any]]:
        if clean.dim() != 3:
            raise ValueError(f"`clean` must be Tensor[C,H,W], got shape {tuple(clean.shape)}")

        assert isinstance(self.cfg, SparseConfig)
        ratio = float(self.cfg.ratio)
        if ratio <= 0.0:
            return clean.clone(), {}
        ratio = min(1.0, ratio)

        seed = int(self.cfg.seed) + int(idx)
        shape = clean.shape if bool(self.cfg.per_channel) else (1, clean.shape[1], clean.shape[2])

        data_min = torch.as_tensor(float(self.cfg.data_min), device=clean.device, dtype=clean.dtype)
        data_max = torch.as_tensor(float(self.cfg.data_max), device=clean.device, dtype=clean.dtype)
        mode = str(self.cfg.value_mode).strip().lower()

        if mode in {"salt_pepper", "salt-and-pepper", "saltpepper"}:
            u = self._rand(shape, seed=seed, device=clean.device, dtype=clean.dtype)
            salt_mask = u < (ratio / 2.0)
            pepper_mask = u > (1.0 - ratio / 2.0)
            corruption_mask = salt_mask | pepper_mask
            degraded = torch.where(salt_mask, data_max, clean)
            degraded = torch.where(pepper_mask, data_min, degraded)
        elif mode in {"random_extreme", "extreme"}:
            corruption_mask = self._rand(shape, seed=seed, device=clean.device, dtype=clean.dtype) < ratio
            high_mask = self._rand(
                shape,
                seed=seed + 10_000_019,
                device=clean.device,
                dtype=clean.dtype,
            ) < 0.5
            extreme_val = torch.where(high_mask, data_max, data_min)
            degraded = torch.where(corruption_mask, extreme_val, clean)
        elif mode in {"signed_impulse", "impulse"}:
            corruption_mask = self._rand(shape, seed=seed, device=clean.device, dtype=clean.dtype) < ratio
            sign = torch.where(
                self._rand(
                    shape,
                    seed=seed + 20_000_033,
                    device=clean.device,
                    dtype=clean.dtype,
                ) < 0.5,
                torch.as_tensor(-1.0, device=clean.device, dtype=clean.dtype),
                torch.as_tensor(1.0, device=clean.device, dtype=clean.dtype),
            )
            amplitude = torch.as_tensor(
                max(abs(float(self.cfg.data_min)), abs(float(self.cfg.data_max))),
                device=clean.device,
                dtype=clean.dtype,
            )
            degraded = torch.where(corruption_mask, clean + sign * amplitude, clean)
            degraded = torch.clamp(degraded, min=data_min.item(), max=data_max.item())
        else:
            raise ValueError(f"Unsupported sparse value_mode: {self.cfg.value_mode}")

        return degraded, {}


class InpaintingDegradation(BaseDegradation):
    """Return (1 - missing_mask) * clean and its missing_mask parameter."""
    def __init__(self, cfg: BaseDegradationConfig):
        super().__init__(cfg)
        if not isinstance(cfg, InpaintingConfig):
            raise TypeError(
                f"InpaintingDegradation expects InpaintingConfig, got {type(cfg)}"
            )

    def _build_fixed_mask(self, clean: Tensor) -> Tensor:
        assert isinstance(self.cfg, InpaintingConfig)
        missing_mask = self.cfg.missing_mask
        if missing_mask is None:
            raise RuntimeError("missing_mask is None")
        if not torch.is_tensor(missing_mask):
            missing_mask = torch.as_tensor(missing_mask)
        missing_mask = missing_mask.to(device=clean.device, dtype=clean.dtype)

        if missing_mask.dim() == 2:
            missing_mask = missing_mask.unsqueeze(0)  # [1,H,W]
        elif missing_mask.dim() == 4 and missing_mask.shape[0] == 1:
            missing_mask = missing_mask.squeeze(0)  # [C,H,W]

        if missing_mask.dim() != 3:
            raise ValueError(
                "Fixed missing_mask must have shape [H,W]/[C,H,W]/[1,C,H,W], "
                f"got {tuple(missing_mask.shape)}"
            )
        if missing_mask.shape[-2:] != clean.shape[-2:]:
            raise ValueError(
                f"missing_mask spatial shape {tuple(missing_mask.shape[-2:])} "
                f"!= image shape {tuple(clean.shape[-2:])}"
            )
        if missing_mask.shape[0] not in {1, clean.shape[0]}:
            raise ValueError(
                f"missing_mask channels must be 1 or {clean.shape[0]}, got {missing_mask.shape[0]}"
            )
        if not torch.logical_or(missing_mask == 0, missing_mask == 1).all():
            raise ValueError("missing_mask must be binary: 1 for missing and 0 for observed.")
        return missing_mask

    def _build_random_mask(self, clean: Tensor, idx: int) -> Tensor:
        assert isinstance(self.cfg, InpaintingConfig)
        ratio = float(self.cfg.mask_ratio)
        mode = str(self.cfg.mask_mode).strip().lower()

        if mode in {"bernoulli", "random"}:
            shape = (
                clean.shape if bool(self.cfg.per_channel) else (1, clean.shape[1], clean.shape[2])
            )
            seed = int(self.cfg.seed) + int(idx)
            return (
                self._rand(shape, seed=seed, device=clean.device, dtype=clean.dtype) < ratio
            ).to(clean.dtype)

        h, w = clean.shape[-2], clean.shape[-1]
        if mode in {"center", "center_box"}:
            missing_area = max(0.0, min(1.0, ratio))
            missing_mask = torch.zeros((1, h, w), device=clean.device, dtype=clean.dtype)
            if missing_area >= 1.0:
                missing_mask.fill_(1.0)
            elif missing_area > 0.0:
                box_h = max(1, int(round(h * missing_area ** 0.5)))
                box_w = max(1, int(round(w * missing_area ** 0.5)))
                top = max(0, (h - box_h) // 2)
                left = max(0, (w - box_w) // 2)
                missing_mask[:, top : top + box_h, left : left + box_w] = 1.0
            return (
                missing_mask
                if not self.cfg.per_channel
                else missing_mask.expand(clean.shape[0], -1, -1).clone()
            )

        raise ValueError(
            f"Unsupported inpainting mask_mode: {self.cfg.mask_mode}. "
            f"Supported: {sorted(SUPPORTED_INPAINTING_MASK_MODES)}"
        )

    def _build_mask(self, clean: Tensor, idx: int) -> Tensor:
        if not isinstance(clean, Tensor) or clean.dim() != 3:
            raise ValueError(
                f"`clean` must be Tensor[C,H,W], got {type(clean)} with shape {getattr(clean, 'shape', None)}"
            )
        if isinstance(self.cfg, InpaintingConfig) and self.cfg.missing_mask is not None:
            return self._build_fixed_mask(clean)
        return self._build_random_mask(clean, idx)

    def apply(self, clean: Tensor, idx: int) -> tuple[Tensor, Dict[str, Any]]:
        missing_mask = self._build_mask(clean, idx)
        keep_mask = 1.0 - missing_mask
        try:
            degraded = keep_mask * clean
        except RuntimeError as exc:
            raise ValueError(
                f"missing_mask shape {tuple(missing_mask.shape)} "
                f"is not broadcastable to clean shape {tuple(clean.shape)}"
            ) from exc
        return degraded, {"missing_mask": missing_mask}


class DeblurDegradation(BaseDegradation):
    """Apply depthwise blur and optional noise; expose the fixed kernel in shared params."""
    def __init__(self, cfg: BaseDegradationConfig):
        super().__init__(cfg)
        if not isinstance(cfg, DeblurConfig):
            raise TypeError(f"DeblurDegradation expects DeblurConfig, got {type(cfg)}")
        self._kernel_2d = self._init_kernel(cfg)

    @staticmethod
    def _normalize_kernel_2d(kernel_2d: Tensor) -> Tensor:
        if kernel_2d.dim() != 2:
            raise ValueError(f"kernel must be 2D, got shape {tuple(kernel_2d.shape)}")
        if kernel_2d.numel() == 0 or not torch.isfinite(kernel_2d).all():
            raise ValueError("kernel must be nonempty and finite")
        s = kernel_2d.sum()
        if not torch.isfinite(s):
            raise ValueError("kernel sum must be finite")
        if float(torch.abs(s)) < 1e-12:
            raise ValueError("kernel sum is zero, cannot normalize")
        return kernel_2d / s

    def _gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        center = (kernel_size - 1) / 2.0
        ax = torch.arange(kernel_size, device=device, dtype=dtype) - center
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel_2d = self._normalize_kernel_2d(kernel)
        kh, kw = int(kernel_2d.shape[0]), int(kernel_2d.shape[1])
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(
                f"`deblur.kernel` spatial size must be odd, got ({kh}, {kw})"
            )
        return kernel_2d

    def _init_kernel(self, cfg: DeblurConfig) -> Tensor:
        if cfg.kernel is None:
            if int(cfg.kernel_size) % 2 == 0:
                raise ValueError(
                    f"`deblur.kernel_size` must be odd, got {cfg.kernel_size}"
                )
            return self._gaussian_kernel(
                kernel_size=int(cfg.kernel_size),
                sigma=float(cfg.sigma),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        kernel = cfg.kernel if torch.is_tensor(cfg.kernel) else torch.as_tensor(cfg.kernel)
        kernel = kernel.detach().float().cpu()

        if kernel.dim() == 4:
            if kernel.shape[0] == 1 and kernel.shape[1] == 1:
                kernel = kernel[0, 0]
            else:
                raise ValueError(
                    f"Expected kernel [1,1,k,k] for 4D kernel, got {tuple(kernel.shape)}"
                )
        elif kernel.dim() == 3:
            if kernel.shape[0] == 1:
                kernel = kernel[0]
            else:
                raise ValueError(
                    f"Expected kernel [1,k,k] for 3D kernel, got {tuple(kernel.shape)}"
                )
        elif kernel.dim() != 2:
            raise ValueError(f"kernel must be 2D/3D/4D, got {tuple(kernel.shape)}")

        kernel_2d = self._normalize_kernel_2d(kernel)
        kh, kw = int(kernel_2d.shape[0]), int(kernel_2d.shape[1])
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(
                f"`deblur.kernel` spatial size must be odd, got ({kh}, {kw})"
            )
        return kernel_2d

    @staticmethod
    def _to_group_kernel(
        kernel_2d: Tensor,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        k = kernel_2d.to(device=device, dtype=dtype).view(
            1, 1, kernel_2d.shape[0], kernel_2d.shape[1]
        )
        return k.repeat(channels, 1, 1, 1)

    def _pad(self, x: Tensor, pad_h: int, pad_w: int) -> Tensor:
        assert isinstance(self.cfg, DeblurConfig)
        mode = str(self.cfg.padding_mode).strip().lower()
        if mode not in {"reflect", "replicate", "circular", "constant"}:
            raise ValueError(f"Unsupported blur padding_mode: {self.cfg.padding_mode}")
        pad = (pad_w, pad_w, pad_h, pad_h)
        if mode == "constant":
            return F.pad(x, pad, mode=mode, value=0.0)
        return F.pad(x, pad, mode=mode)

    def apply(self, clean: Tensor, idx: int) -> tuple[Tensor, Dict[str, Any]]:
        if clean.dim() != 3:
            raise ValueError(f"`clean` must be Tensor[C,H,W], got shape {tuple(clean.shape)}")
        c, _, _ = clean.shape
        x = clean.unsqueeze(0)

        kh, kw = int(self._kernel_2d.shape[0]), int(self._kernel_2d.shape[1])
        pad_h, pad_w = kh // 2, kw // 2
        x_pad = self._pad(x, pad_h=pad_h, pad_w=pad_w)

        kernel = self._to_group_kernel(
            self._kernel_2d,
            channels=c,
            device=x.device,
            dtype=x.dtype,
        )
        degraded = F.conv2d(x_pad, kernel, bias=None, stride=1, padding=0, groups=c).squeeze(0)

        if float(self.cfg.obs_noise_std) > 0.0:
            noise = self._randn(
                degraded.shape,
                seed=int(self.cfg.seed) + int(idx),
                device=degraded.device,
                dtype=degraded.dtype,
            )
            degraded = degraded + float(self.cfg.obs_noise_std) * noise

        return degraded, {}

    def get_shared_params(self) -> Dict[str, Any]:
        assert isinstance(self.cfg, DeblurConfig)
        kernel = self._kernel_2d.view(
            1, 1, self._kernel_2d.shape[0], self._kernel_2d.shape[1]
        ).clone()
        return {
            "kernel": kernel,
            "padding_mode": self.cfg.padding_mode,
            "obs_noise_std": float(self.cfg.obs_noise_std),
        }


def build_degradation(cfg: BaseDegradationConfig) -> BaseDegradation:
    """Construct the operator matching the validated degradation mode."""
    cfg = build_degradation_config(cfg)
    mode = str(cfg.mode).strip().lower()
    if mode == "clean":
        return CleanDegradation(cfg)
    if mode == "inpainting":
        return InpaintingDegradation(cfg)
    if mode == "deblur":
        return DeblurDegradation(cfg)
    if mode == "sparse":
        return SparseDegradation(cfg)
    raise ValueError(
        f"Unsupported degradation mode: {mode}. Supported: {sorted(SUPPORTED_DEGRADATION_MODES)}"
    )


# Paired samples combine clean datasets with their degradation.
# DataLoaders read this common sample dictionary.

class DegradationPairDataset(Dataset):
    """Combine clean images, labels, split-local indices and operator parameters."""

    def __init__(
        self,
        base_dataset: Dataset,
        degradation: BaseDegradation,
        dataset_name: str,
        split: str,
    ):
        self.base_dataset = base_dataset
        self.degradation = degradation
        self.dataset_name = dataset_name
        self.split = split

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Read one image and label, apply its degradation and assemble the sample dictionary."""
        sample = self.base_dataset[idx]
        label_or_attr: Any = None

        if isinstance(sample, (tuple, list)):
            if len(sample) == 0:
                raise ValueError("base_dataset returned an empty tuple/list")
            clean = sample[0]
            if len(sample) > 1:
                label_or_attr = sample[1]
        elif isinstance(sample, dict):
            if "clean" in sample:
                clean = sample["clean"]
            elif "image" in sample:
                clean = sample["image"]
            elif "img" in sample:
                clean = sample["img"]
            else:
                raise KeyError("base_dataset dict sample must contain one of: clean/image/img")
            label_or_attr = sample.get("label", sample.get("attr"))
        else:
            clean = sample

        if not torch.is_tensor(clean):
            clean = torch.as_tensor(clean)
        if clean.ndim != 3 or clean.numel() == 0 or not clean.is_floating_point() or not torch.isfinite(clean).all():
            raise ValueError("Clean images must be finite, nonempty floating-point CHW tensors.")
        if clean.min() < 0 or clean.max() > 1:
            raise ValueError("Clean images must be in [0, 1].")

        degraded, sample_params = self.degradation.apply(clean, idx)
        if sample_params is None:
            sample_params = {}
        if not isinstance(sample_params, dict):
            raise TypeError(
                f"`degradation.apply` must return params as dict, got {type(sample_params)}"
            )
        shared_params = self.degradation.get_shared_params()
        if shared_params:
            if not isinstance(shared_params, dict):
                raise TypeError(
                    f"`degradation.get_shared_params` must return dict, got {type(shared_params)}"
                )
            merged_params = dict(shared_params)
            merged_params.update(sample_params)
            sample_params = merged_params

        mode = getattr(getattr(self.degradation, "cfg", None), "mode", "unknown")
        return {
            "clean": clean,
            "degraded": degraded,
            "index": int(idx),
            "label": label_or_attr,
            "params": sample_params,
            "meta": {
                "dataset": self.dataset_name,
                "split": self.split,
                "mode": mode,
            },
        }


# Batch assembly stacks sample-wise data and retains shared parameters.
# Keep operator parameter shapes explicit.

def degradation_collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack images and sample-wise parameters; retain shared operator parameters once."""
    if len(batch) == 0:
        raise ValueError("batch is empty")

    def _as_tensor(x: Any) -> Tensor:
        return x if torch.is_tensor(x) else torch.as_tensor(x)

    clean = torch.stack([_as_tensor(s["clean"]) for s in batch], dim=0)
    degraded = torch.stack([_as_tensor(s["degraded"]) for s in batch], dim=0)
    index = torch.as_tensor([int(s["index"]) for s in batch], dtype=torch.long)

    labels = [s.get("label", None) for s in batch]
    if all(lbl is None for lbl in labels):
        label_out: Any = None
    elif all(torch.is_tensor(lbl) for lbl in labels):
        try:
            label_out = torch.stack(labels, dim=0)
        except RuntimeError:
            label_out = labels
    elif all(isinstance(lbl, (int, float, bool)) for lbl in labels):
        label_out = torch.as_tensor(labels)
    else:
        label_out = labels

    params_out: Dict[str, Any] = {}
    param_keys = set()
    for sample in batch:
        p = sample.get("params", {})
        if isinstance(p, dict):
            param_keys.update(p.keys())

    for key in sorted(param_keys):
        vals = [sample.get("params", {}).get(key, None) for sample in batch]
        present = [v for v in vals if v is not None]
        if not present:
            continue

        if len(present) != len(vals):
            params_out[key] = vals
            continue

        if all(torch.is_tensor(v) for v in present):
            first = present[0]
            if key == "missing_mask":
                params_out[key] = torch.stack(present, dim=0)
                continue
            all_same = all(v.shape == first.shape and torch.equal(v, first) for v in present[1:])
            if all_same:
                params_out[key] = first
                continue
            try:
                params_out[key] = torch.stack(present, dim=0)
            except RuntimeError:
                params_out[key] = present
            continue

        try:
            all_same = all(v == present[0] for v in present[1:])
        except Exception:
            all_same = False
        params_out[key] = present[0] if all_same else present

    first_meta = batch[0].get("meta", {}) or {}
    meta = {
        "dataset": first_meta.get("dataset"),
        "split": first_meta.get("split"),
        "mode": first_meta.get("mode"),
    }

    return {
        "clean": clean,
        "degraded": degraded,
        "index": index,
        "label": label_out,
        "params": params_out,
        "meta": meta,
    }


# Public loader entry point.

def get_degradation_dataloader(
    dataset: str,
    split: str,
    degradation_cfg: DegradationConfig,
    batch_size: int,
    num_workers: int,
    *,
    root: Optional[str | Path] = None,
    image_size: Optional[int] = None,
    val_size: Optional[int] = None,
    subset_size: Optional[Union[int, float, str]] = None,
    sample_indices: Optional[Sequence[int]] = None,
    num_samples: Optional[int] = None,
    shuffle: Optional[bool] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Build a paired loader: training shuffles, validation/test use fixed per-index noise."""
    dataset = str(dataset).strip().lower()
    split = str(split).strip().lower()

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported: {sorted(SUPPORTED_DATASETS)}"
        )
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}. Supported: ['train', 'val', 'test']")
    batch_size, num_workers = integer_index(batch_size), integer_index(num_workers)
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"`num_workers` must be >= 0, got {num_workers}")

    cfg = build_degradation_config(degradation_cfg)
    cfg = set_cfg_deterministic_by_split(cfg, split)

    resolved_root = get_default_data_root(dataset) if root is None else Path(root)

    factory = BaseImageDatasetFactory(
        dataset_name=dataset,
        root=resolved_root,
        image_size=image_size,
        val_size=val_size,
        subset_size=subset_size,
    )
    base_dataset = factory.build(split=split)
    degradation = build_degradation(cfg)

    pair_dataset = DegradationPairDataset(
        base_dataset=base_dataset,
        degradation=degradation,
        dataset_name=dataset,
        split=split,
    )

    # Select paired samples after assigning split-local indices and corruption seeds.
    if num_samples is not None:
        num_samples = integer_index(num_samples)
        if num_samples < 0 or sample_indices is not None:
            raise ValueError("Use either sample_indices or a nonnegative num_samples.")
        sample_indices = range(min(num_samples, len(pair_dataset)))
    if sample_indices is not None:
        selected = [integer_index(index) for index in sample_indices]
        if any(index < 0 or index >= len(pair_dataset) for index in selected):
            raise IndexError(f"sample_indices must be within [0, {len(pair_dataset) - 1}]")
        pair_dataset = Subset(pair_dataset, selected)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        pair_dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=degradation_collate_fn,
        worker_init_fn=worker_init_fn if int(num_workers) > 0 else None,
        drop_last=bool(drop_last),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers and int(num_workers) > 0),
        generator=generator,
    )


# Project-relative paths and deterministic worker setup.

def get_default_data_root(dataset_name: str) -> Path:
    """Resolve datasets/<name> relative to this project."""
    name = str(dataset_name).strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported: {sorted(SUPPORTED_DATASETS)}")
    return Path(__file__).resolve().parent.parent / "datasets" / name


def set_cfg_deterministic_by_split(
    cfg: BaseDegradationConfig,
    split: str,
) -> BaseDegradationConfig:
    """Return a configuration with deterministic noise on validation/test only."""
    split_norm = str(split).strip().lower()
    if split_norm not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}. Supported: ['train', 'val', 'test']")

    deterministic = split_norm in {"val", "test"}
    if cfg.deterministic == deterministic:
        return cfg
    return replace(cfg, deterministic=deterministic)


def worker_init_fn(worker_id: int):
    """Seed worker-local Python, NumPy and PyTorch random streams."""
    import numpy as np

    worker_seed = (int(torch.initial_seed()) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
