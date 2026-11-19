"""
utils_metrics.py

This module implements image quality assessment metrics: PSNR and SSIM, based on NumPy, supporting batch evaluation of multiple images.
It is suitable for scenarios where image data is unified into the following format:

Input data format:
    - Type: NumPy ndarray
    - Shape: [H, W, C] (color), [H, W] (grayscale), or flattened [H*W]
    - Range: float, normalized to [0, 1]

Advice for cross-framework data:
    - PyTorch: `.permute(0, 2, 3, 1).cpu().numpy()`
    - TensorFlow: `.numpy()`
    - OpenCV: `cv2.imread(...) / 255.`
    - PIL: `np.array(Image.open(...)) / 255.`

This module provides the following interfaces:
    - avg_psnr(original_data: np.ndarray, recon_data: np.ndarray) -> float
    - avg_ssim(original_data: np.ndarray, recon_data: np.ndarray) -> float
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


# Validate image contracts without changing the floating-point metric calculations.
def _validate_pair(original, reconstructed, dimensions):
    original, reconstructed = np.asarray(original), np.asarray(reconstructed)
    if original.shape != reconstructed.shape or original.ndim not in dimensions or original.size == 0:
        raise ValueError("Metric images must be nonempty, have matching shapes and valid dimensions.")
    for image in (original, reconstructed):
        if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all():
            raise ValueError("Metric images must contain finite floating-point values.")
    return original, reconstructed


def compute_psnr(img1, img2, max_val=1.0):
    """PSNR for a single image"""
    img1, img2 = _validate_pair(img1, img2, {1, 2, 3})
    if not np.isfinite(max_val) or max_val <= 0:
        raise ValueError("max_val must be finite and positive.")
    mse = np.mean((img1 - img2) ** 2)
    if not np.isfinite(mse):
        raise FloatingPointError("PSNR mean squared error is not finite.")
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(img1, img2, max_val=1.0, k1=0.01, k2=0.03, sigma=1.5):
    """SSIM for single-channel or color images (using Gaussian window)"""
    img1, img2 = _validate_pair(img1, img2, {1, 2, 3})
    if any(not np.isfinite(value) or value <= 0 for value in (max_val, k1, k2, sigma)):
        raise ValueError("SSIM scale, constants and sigma must be finite and positive.")
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Grayscale images and flattened vectors use one channel.
    if img1.ndim == 1 or img1.ndim == 2:
        return _ssim_per_channel(img1, img2, c1, c2, sigma)

    # Color images use HWC layout and average the channel scores.
    if img1.ndim == 3:
        ssim_vals = []
        for ch in range(img1.shape[2]):
            ssim_vals.append(_ssim_per_channel(img1[:, :, ch], img2[:, :, ch], c1, c2, sigma))
        return np.mean(ssim_vals)

    raise ValueError("Unsupported input shape for SSIM: must be [H, W, C], [H, W], or [H*W]")


def _ssim_per_channel(img1, img2, c1, c2, sigma):
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-10)
    result = np.mean(ssim_map)
    if not np.isfinite(result):
        raise FloatingPointError("SSIM is not finite.")
    return result


def avg_ssim(original_data: np.ndarray, recon_data: np.ndarray) -> float:
    """
    Calculate the average SSIM for a batch of images
    Data format:
        original_data.shape = [N, H, W, C] / [N, H, W] / [N, H*W]
        recon_data.shape should be the same
        Data range should be in [0, 1]
    """
    original_data, recon_data = _validate_pair(original_data, recon_data, {2, 3, 4})
    num_samples = original_data.shape[0]

    if original_data.min() < 0.0 or original_data.max() > 1.0:
        print("Warning: Original data is out of [0, 1] range, SSIM may be inaccurate.")

    ssim_list = []
    for i in range(num_samples):
        s = compute_ssim(original_data[i], recon_data[i])
        ssim_list.append(s)
    return float(np.mean(ssim_list))


def avg_psnr(original_data: np.ndarray, recon_data: np.ndarray) -> float:
    """
    Calculate the average PSNR for a batch of images
    Data format:
        original_data.shape = [N, H, W, C] / [N, H, W] / [N, H*W]
        recon_data.shape should be the same
        Data range should be in [0, 1]
    """
    original_data, recon_data = _validate_pair(original_data, recon_data, {2, 3, 4})
    num_samples = original_data.shape[0]

    if original_data.min() < 0.0 or original_data.max() > 1.0:
        print("Warning: Original data is out of [0, 1] range, PSNR may be inaccurate.")

    psnr_list = []
    for i in range(num_samples):
        p = compute_psnr(original_data[i], recon_data[i])
        psnr_list.append(p)
    return float(np.mean(psnr_list))

# Tensor wrappers detach before transferring images to CPU metrics.
def _numpy_batch(images):
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("Tensor metrics expect BCHW tensors.")
    return images.permute(0, 2, 3, 1).detach().cpu().numpy()


def avg_psnr_pytorch(original_data: torch.Tensor, recon_data: torch.Tensor) -> float:
    """Compute mean PSNR for detached BCHW tensors."""
    return avg_psnr(_numpy_batch(original_data), _numpy_batch(recon_data))


def avg_ssim_pytorch(original_data: torch.Tensor, recon_data: torch.Tensor) -> float:
    """Compute mean SSIM for detached BCHW tensors."""
    return avg_ssim(_numpy_batch(original_data), _numpy_batch(recon_data))
