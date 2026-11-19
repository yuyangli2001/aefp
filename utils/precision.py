"""Use full tensor precision for objective and gradient comparisons."""

from contextlib import contextmanager

import torch


# TF32 rounding can make a tiny Armijo step fail the objective comparison.
@contextmanager
def full_precision():
    cudnn = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
