"""Exact Euclidean KNN for the frozen-feature experiment."""
from __future__ import annotations
from typing import Sequence
from operator import index
import torch
import torch.nn.functional as F

class ExactKNN:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, *, device: torch.device, query_batch_size: int):
        if (features.ndim != 2 or not features.numel() or labels.ndim != 1 or len(labels) != len(features)
                or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or (labels < 0).any()):
            raise ValueError("Require nonempty matrix features and matching nonnegative integer labels.")
        if isinstance(query_batch_size, bool) or index(query_batch_size) <= 0:
            raise ValueError("query_batch_size must be a positive integer.")
        self.features = features.to(device=device, dtype=torch.float32)
        if not torch.isfinite(self.features).all():
            raise ValueError("KNN training features must be finite in float32.")
        self.labels = labels.to(device=device, dtype=torch.long)
        self.device = device
        self.query_batch_size = index(query_batch_size)
        self.num_classes = int(labels.max().item()) + 1

    @torch.inference_mode()
    def predict_many(self, query: torch.Tensor, k_values: Sequence[int]) -> dict[int, torch.Tensor]:
        if query.ndim != 2 or not len(query) or query.shape[1] != self.features.shape[1]:
            raise ValueError("Require a nonempty query matrix with the training feature dimension.")
        if any(isinstance(k, bool) for k in k_values):
            raise ValueError("Neighborhood sizes must be integers, not booleans.")
        ks = sorted({index(k) for k in k_values})
        if not ks or ks[0] <= 0 or ks[-1] > self.features.shape[0]:
            raise ValueError(f"Invalid k_values={ks} for train size {self.features.shape[0]}.")
        outputs = {k: [] for k in ks}
        train_sq = self.features.square().sum(dim=1).unsqueeze(0)
        for start in range(0, query.shape[0], self.query_batch_size):
            q = query[start : start + self.query_batch_size].to(self.device, dtype=torch.float32)
            if not torch.isfinite(q).all():
                raise ValueError("KNN query features must be finite in float32.")
            distances = (q.square().sum(dim=1, keepdim=True) + train_sq - 2.0 * q @ self.features.T).clamp_min_(0)
            if not torch.isfinite(distances).all():
                raise ValueError("Euclidean distances overflowed float32.")
            # Reuse the largest neighborhood for every requested k.
            neighbors = distances.topk(ks[-1], dim=1, largest=False, sorted=True).indices
            neighbor_labels = self.labels[neighbors]
            for k in ks:
                votes = F.one_hot(neighbor_labels[:, :k], num_classes=self.num_classes).sum(dim=1)
                outputs[k].append(votes.argmax(dim=1).cpu())
        return {k: torch.cat(parts) for k, parts in outputs.items()}
