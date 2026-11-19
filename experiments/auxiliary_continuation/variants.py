"""Frozen ablation variants, stage chains, and image-weighted metrics."""

from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from typing import Any
import torch
from methods.continuation import irae_direct
from utils.utils_metrics import avg_psnr_pytorch, avg_ssim_pytorch


DATASETS = ("mnist", "cifar10", "celeba-hq-64")
RATIOS = (0.1, 0.2, 0.3)
FULL_TEST_SIZE = {"mnist": 10000, "cifar10": 10000, "celeba-hq-64": 5000}
VARIANT_LABELS = {
    "A0": "Observation",
    "A1": "Feedforward-AE",
    "A2": "Direct-IRAE-100",
    "A3": "Direct-IRAE-300",
    "A4": "Coarse only",
    "A5": "Coarse -> Fine",
    "A6": "Coarse -> Fine -> Original",
    "A7": "Original -> Original -> Original",
    "A8": "Fine -> Original",
    "A9": "Coarse -> Original",
}
FULL_VARIANTS = tuple(VARIANT_LABELS)
TABLE_VARIANTS = tuple(variant for variant in FULL_VARIANTS if variant != "A2")
EDGE_VARIANTS = ("A3", "A6", "A7")
FIELDNAMES = (
    "dataset", "ratio", "variant", "label", "psnr", "ssim", "split",
    "num_samples", "source", "source_path", "preset_key", "model_id",
    "checkpoint_dir", "lambda",
)


def solver_schedule(config):
    return {key: config[key] for key in ("is_lambda_update", "lambda_multiplier",
        "lambda_check_interval", "backtracking_rule", "sufficient_decrease_constant")}


# Each stage resets the solver schedule and starts from the previous output.
def run_chain(observation: torch.Tensor, degradation: Mapping[str, Any], models: Sequence[Any],
              lambdas: Sequence[float], config: Mapping[str, Any], device: torch.device) -> list[torch.Tensor]:
    if not models or len(models) != len(lambdas):
        raise ValueError("Require matching nonempty model and lambda sequences.")
    schedule = solver_schedule(config)
    current, outputs = observation, []
    for model, stage_lambda in zip(models, lambdas):
        current = irae_direct(
            observation=observation, ae_model=model, degradation=dict(degradation),
            x_init=current, device=device, lambda_=stage_lambda,
            max_iteration=100, **schedule,
        )
        outputs.append(current)
    return outputs


def run_batch_variants(observation: torch.Tensor, degradation: Mapping[str, Any],
                       models: Sequence[Any], variants: set[str], config: Mapping[str, Any],
                       device: torch.device) -> dict[str, torch.Tensor]:
    """Compute only missing cells and share the A4/A5/A6 prefix execution."""
    if not variants or not variants <= set(FULL_VARIANTS):
        raise ValueError("Require a nonempty set of known ablation variants.")
    coarse, fine, original = models
    base_lambda = float(config["lambda"])
    aux_lambda = float(config["lambda_init_gamma"]) * base_lambda
    outputs: dict[str, torch.Tensor] = {}
    if "A0" in variants:
        outputs["A0"] = observation
    if "A1" in variants:
        with torch.inference_mode():
            outputs["A1"] = original.get_recon(observation).clamp(0.0, 1.0)
    for variant, iterations in (("A2", 100), ("A3", 300)):
        if variant in variants:
            outputs[variant] = irae_direct(
                observation=observation, ae_model=original, degradation=dict(degradation),
                x_init=observation, device=device, lambda_=base_lambda,
                max_iteration=iterations, **solver_schedule(config),
            )
    prefix = [name for name in ("A4", "A5", "A6") if name in variants]
    if prefix:
        count = max({"A4": 1, "A5": 2, "A6": 3}[name] for name in prefix)
        stages = run_chain(observation, degradation, (coarse, fine, original)[:count],
                           (aux_lambda, aux_lambda, base_lambda)[:count], config, device)
        for name in prefix:
            outputs[name] = stages[{"A4": 0, "A5": 1, "A6": 2}[name]]
    chains = {
        "A7": ((original, original, original), (aux_lambda, aux_lambda, base_lambda)),
        "A8": ((fine, original), (aux_lambda, base_lambda)),
        "A9": ((coarse, original), (aux_lambda, base_lambda)),
    }
    for name, (chain_models, chain_lambdas) in chains.items():
        if name in variants:
            outputs[name] = run_chain(observation, degradation, chain_models, chain_lambdas, config, device)[-1]
    return outputs


class MetricAccumulator:
    def __init__(self, variants: Sequence[str]):
        if not variants or len(set(variants)) != len(variants):
            raise ValueError("Require unique, nonempty metric variants.")
        self.sums = {name: [0.0, 0.0] for name in variants}
        self.num_samples = 0

    def update(self, clean: torch.Tensor, outputs: Mapping[str, torch.Tensor]) -> None:
        if set(outputs) != set(self.sums):
            raise ValueError("Every metric batch must contain exactly the selected variants.")
        batch_size = int(clean.shape[0])
        values = {}
        for name, reconstruction in outputs.items():
            values[name] = (float(avg_psnr_pytorch(clean, reconstruction)), float(avg_ssim_pytorch(clean, reconstruction)))
            if not all(math.isfinite(value) for value in values[name]):
                raise ValueError("Nonfinite ablation metrics.")
        for name, (psnr, ssim) in values.items():
            self.sums[name][0] += psnr * batch_size
            self.sums[name][1] += ssim * batch_size
        self.num_samples += batch_size

    def finalize(self) -> dict[str, tuple[float, float]]:
        if not self.num_samples:
            raise ValueError("Evaluation loader produced no samples.")
        return {name: (values[0] / self.num_samples, values[1] / self.num_samples)
                for name, values in self.sums.items()}


def _degradation(ratio: float) -> dict[str, Any]:
    return {"mode": "sparse", "ratio": float(ratio), "seed": 42,
            "value_mode": "salt_pepper", "per_channel": True}


def _ratio_index(ratio: float) -> int:
    for index, expected in enumerate(RATIOS):
        if abs(float(ratio) - expected) < 1e-9:
            return index
    raise ValueError(f"Unsupported ratio {ratio}; expected one of {RATIOS}.")
