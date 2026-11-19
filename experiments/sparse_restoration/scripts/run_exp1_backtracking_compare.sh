#!/usr/bin/env bash
# Compare both backtracking rules across datasets and corruption levels on one GPU.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

GPU="${GPU:-1}"
PYTHON_EXE="${PYTHON_EXE:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparse_restoration/backtracking_compare/${RUN_ID}}"
BACKTRACKING_RULES="monotone,sufficient_decrease"
SUFFICIENT_DECREASE_CONSTANT="${SUFFICIENT_DECREASE_CONSTANT:-1.0e-4}"

# Require the same original and auxiliary checkpoints for both rules.
require_checkpoint_family() {
  local root="$1"
  local relative
  for relative in \
    "jinit/psnr_drops_10p0-5p0/stage_1.pth" \
    "jinit/psnr_drops_10p0-5p0/stage_2.pth" \
    "jinit/psnr_drops_10p0-5p0/summary.json" \
    "direct/stage_1.pth"; do
    [[ -f "${root}/${relative}" ]] || {
      echo "[backtracking_compare] Missing checkpoint: ${root}/${relative}" >&2
      exit 1
    }
  done
}

MNIST_CKPT="checkpoints/image_ae/mnist/resconv_sch32-64-128-256_bps1-2-2-2_bc6_normgn_updeconv_pjinit_dep200_dl10p0_jsw1em08_jep30_jgf3p0_jmc14_jl10p0_lr0p0001_bs128"
CIFAR10_CKPT="checkpoints/image_ae/cifar10/resconv_sch32-64-128_bps1-2-2_bc20_normgn_updeconv_pjinit_dep200_dl10p0_jsw1em08_jep30_jgf3p0_jmc14_jl10p0_lr0p0001_bs64"
CELEBA_CKPT="checkpoints/image_ae/celeba-hq-64/resconv_sch32-64-128_bps1-2-2_bc20_normgn_updeconv_pjinit_dep200_dl10p0_jsw1em08_jep30_jgf3p0_jmc14_jl10p0_lr0p0001_bs64"
require_checkpoint_family "${MNIST_CKPT}"
require_checkpoint_family "${CIFAR10_CKPT}"
require_checkpoint_family "${CELEBA_CKPT}"

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' \
  "run_id=${RUN_ID}" \
  "gpu=${GPU}" \
  "split=test" \
  "num_samples=all" \
  "noise_ratios=0.1,0.2,0.3" \
  "backtracking_rules=${BACKTRACKING_RULES}" \
  "sufficient_decrease_constant=${SUFFICIENT_DECREASE_CONSTANT}" \
  > "${OUTPUT_ROOT}/run_manifest.txt"

# Preserve partial aggregate tables if a later dataset job fails.
collect_partial_results() {
  "${PYTHON_EXE}" experiments/sparse_restoration/collect.py "${OUTPUT_ROOT}" \
    --out-dir "${OUTPUT_ROOT}/aggregate" >/dev/null 2>&1 || true
}
trap collect_partial_results EXIT

echo "[backtracking_compare] GPU=${GPU} output=${OUTPUT_ROOT}"

env \
  GPU="${GPU}" PYTHON_EXE="${PYTHON_EXE}" SOLVERS=jinit DEGRADATIONS=sparse \
  INFERENCE_SPLIT=test INFERENCE_NUM_SAMPLES=null NOISE_RATIOS=0.1,0.2,0.3 \
  BACKTRACKING_RULES="${BACKTRACKING_RULES}" \
  SUFFICIENT_DECREASE_CONSTANT="${SUFFICIENT_DECREASE_CONSTANT}" \
  HYDRA_SWEEP_DIR="${OUTPUT_ROOT}/mnist" \
  MODEL_CONFIG=resae_3down MODEL_STAGE_CHANNELS='[32,64,128,256]' BOTTLENECK_CHANNELS=6 \
  PSNR_DROPS='[10.0,5.0]' LAMBDAS_0P1=0.07 LAMBDAS_0P2=0.04 LAMBDAS_0P3=0.04 \
  bash experiments/sparse_restoration/scripts/run_exp1_mnist_sweep.sh

env \
  GPU="${GPU}" PYTHON_EXE="${PYTHON_EXE}" SOLVERS=jinit DEGRADATIONS=sparse \
  INFERENCE_SPLIT=test INFERENCE_NUM_SAMPLES=null NOISE_RATIOS=0.1,0.2,0.3 \
  BACKTRACKING_RULES="${BACKTRACKING_RULES}" \
  SUFFICIENT_DECREASE_CONSTANT="${SUFFICIENT_DECREASE_CONSTANT}" \
  HYDRA_SWEEP_DIR="${OUTPUT_ROOT}/cifar10" \
  MODEL_CONFIG=resae_2down BOTTLENECK_CHANNELS=20 PSNR_DROPS='[10.0,5.0]' \
  LAMBDAS_0P1=0.04 LAMBDAS_0P2=0.01 LAMBDAS_0P3=0.01 \
  bash experiments/sparse_restoration/scripts/run_exp1_cifar10.sh

env \
  GPU="${GPU}" PYTHON_EXE="${PYTHON_EXE}" SOLVERS=jinit DEGRADATIONS=sparse \
  INFERENCE_SPLIT=test INFERENCE_NUM_SAMPLES=null NOISE_RATIOS=0.1,0.2,0.3 \
  BACKTRACKING_RULES="${BACKTRACKING_RULES}" \
  SUFFICIENT_DECREASE_CONSTANT="${SUFFICIENT_DECREASE_CONSTANT}" \
  HYDRA_SWEEP_DIR="${OUTPUT_ROOT}/celeba-hq-64" \
  MODEL_CONFIG=resae_2down BOTTLENECK_CHANNELS=20 PSNR_DROPS='[10.0,5.0]' \
  LAMBDAS_0P1=0.05 LAMBDAS_0P2=0.03 LAMBDAS_0P3=0.02 \
  bash experiments/sparse_restoration/scripts/run_exp1_celeba.sh

trap - EXIT
"${PYTHON_EXE}" experiments/sparse_restoration/collect.py "${OUTPUT_ROOT}" \
  --out-dir "${OUTPUT_ROOT}/aggregate" --top-k 2 --require-backtracking-pairs 9
echo "[backtracking_compare] Comparison: ${OUTPUT_ROOT}/aggregate/backtracking_comparison.csv"
