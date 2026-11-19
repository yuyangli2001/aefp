"""Preview image batches without changing experiment tensors."""
from operator import index as integer_index

import matplotlib.pyplot as plt
import numpy as np
import torch


# Float images use [0,1], uint8 images use [0,255]; other ranges require scale.
def grid_images(imgs, grid_shape=(10, 10), show=True, title=None, save_path=None, scale=None):
    """Render NCHW or NHW images in a fixed grid and return its Figure."""
    if torch.is_tensor(imgs):
        imgs = imgs.detach().cpu().numpy()
    if not isinstance(imgs, np.ndarray):
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")
    if imgs.ndim not in (3, 4) or imgs.size == 0 or not np.isrealobj(imgs) or not np.isfinite(imgs).all():
        raise ValueError("Image batches must be nonempty, finite real NHW or NCHW arrays.")
    if imgs.ndim == 4:
        if imgs.shape[1] not in (1, 3, 4):
            raise ValueError("NCHW images must have 1, 3 or 4 channels.")
        imgs = imgs[:, 0] if imgs.shape[1] == 1 else imgs.transpose(0, 2, 3, 1)
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain rows and columns.")
    rows, cols = (integer_index(value) for value in grid_shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("Grid dimensions must be positive.")
    vmin, vmax = scale if scale is not None else ((0.0, 255.0) if imgs.dtype == np.uint8 else (0.0, 1.0))
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("scale must contain finite increasing bounds.")

    # imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()+ 1e-8)
    imgs = np.clip((imgs - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    num_images = min(rows * cols, len(imgs))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10), gridspec_kw={"wspace": 0, "hspace": 0})
    fig.subplots_adjust(top=0.9)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for index in range(num_images):
        image = imgs[index]
        axes_flat[index].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes_flat[index].axis("off")
    for index in range(num_images, rows * cols):
        axes_flat[index].axis("off")
    if title:
        fig.suptitle(title, fontsize=16, y=0.99)
    fig.subplots_adjust(left=0, right=1, top=0.97 if title else 1, bottom=0, wspace=0, hspace=0)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    if show:
        plt.draw()
        plt.pause(0.01)
    return fig
