#!/usr/bin/env bash
#! GPU=0 bash experiments/sparse_restoration/scripts/run_exp1_cifar10.sh

# Current top-level config:
# Historical best results in the JInit summary:
# - cifar10, noise=0.1: PSNR=35.7061, SSIM=0.976701, lambda=0.03, bc=32, psnr_drops=[10.0,5.0]
# - cifar10, noise=0.2: PSNR=32.2013, SSIM=0.955575, lambda=0.01, bc=24, psnr_drops=[10.0,5.0]
# - cifar10, noise=0.3: PSNR=29.0561, SSIM=0.919483, lambda=0.01, bc=16, psnr_drops=[10.0,5.0]

# Fix Config:
# Recommended shared prior-network training settings for natural images:
# - model=resae_2down, bc=20, psnr_drops = [10.0, 5.0], upsample_mode = deconv;
# Validation-selected inference lambdas: noise=0.1 lambda = 0.04; noise=0.2 lambda = 0.01; noise=0.3 lambda = 0.01;
set -euo pipefail
# Resolve project paths before passing relative scripts and checkpoint roots to Python.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
# Note: LAMBDAS_0P1/0P2/0P3 map to the 1st/2nd/3rd items in NOISE_RATIOS.
# If you reorder NOISE_RATIOS, update the three LAMBDAS_0P* entries accordingly.
DATASETS="${DATASETS:-cifar10}"

SOLVERS="${SOLVERS:-jinit}" # direct, jinit, feedforward
INIT_METHODS="${INIT_METHODS:-observation}"

DEGRADATIONS="${DEGRADATIONS:-sparse}"
NOISE_RATIOS="${NOISE_RATIOS:-0.1,0.2,0.3}"
MASK_RATIOS="${MASK_RATIOS:-}"
BLUR_SIGMAS="${BLUR_SIGMAS:-1.0}"

# Default lambdas follow the 1st/2nd/3rd NOISE_RATIOS item and can still be overridden as CSV values.
LAMBDAS_0P1="${LAMBDAS_0P1:-0.04}"
LAMBDAS_0P2="${LAMBDAS_0P2:-0.01}"
LAMBDAS_0P3="${LAMBDAS_0P3:-0.01}"


# Training hyperparameters that define the AE checkpoints used in the sweep.
MODEL_CONFIG="${MODEL_CONFIG:-resae_2down}"
BOTTLENECK_CHANNELS="${BOTTLENECK_CHANNELS:-20}" # default 2down bottleneck sweep
UPSAMPLE_MODES="${UPSAMPLE_MODES:-deconv}" # ResConvAE: deconv, nearest_conv
PSNR_DROPS="${PSNR_DROPS:-[10.0,5.0]}"
TRAIN_SUBSET_SIZE="${TRAIN_SUBSET_SIZE:-}"
GPU="${GPU:-0}"

# Default final target is full-test via INFERENCE_SPLIT=test and INFERENCE_NUM_SAMPLES=null.
INFERENCE_SPLIT="${INFERENCE_SPLIT:-test}"
INFERENCE_NUM_SAMPLES="${INFERENCE_NUM_SAMPLES:-null}"
LAMBDA_INIT_GAMMAS="${LAMBDA_INIT_GAMMAS:-1.0}"
LAMBDA_MULTIPLIERS="${LAMBDA_MULTIPLIERS:-0.8}"
BACKTRACKING_RULES="${BACKTRACKING_RULES:-}"
SUFFICIENT_DECREASE_CONSTANT="${SUFFICIENT_DECREASE_CONSTANT:-}"
HYDRA_SWEEP_DIR="${HYDRA_SWEEP_DIR:-}"

PYTHON_EXE="${PYTHON_EXE:-python}"

warn() {
  echo "[run_exp1_cifar10] WARNING: $*" >&2
}

die() {
  echo "[run_exp1_cifar10] ERROR: $*" >&2
  exit 1
}

single_csv_value() {
  local csv="$1"
  local result=""
  local item
  IFS=',' read -r -a _values <<< "${csv}"
  for item in "${_values[@]}"; do
    item="${item// /}"
    [[ -z "${item}" ]] && continue
    if [[ -n "${result}" && "${item}" != "${result}" ]]; then
      die "This script expects a single degradation choice per call, got ${csv}."
    fi
    result="${item}"
  done
  [[ -n "${result}" ]] || die "Empty CSV value is not allowed."
  printf '%s' "${result}"
}

canonical_degradation_choice() {
  local choice="$1"
  case "${choice}" in
    sparse) printf 'sparse' ;;
    inpainting|block_inpainting) printf 'inpainting' ;;
    deblurring) printf 'deblur' ;;
    *) die "Unsupported DEGRADATIONS=${choice}. Expected sparse, inpainting, block_inpainting, or deblurring." ;;
  esac
}

default_mask_ratios() {
  local choice="$1"
  if [[ -n "${MASK_RATIOS}" ]]; then
    printf '%s' "${MASK_RATIOS}"
    return
  fi
  if [[ "${choice}" == "block_inpainting" ]]; then
    printf '0.16'
  else
    printf '0.5'
  fi
}

ensure_non_sparse_init_methods_supported() {
  local item
  IFS=',' read -r -a _init_values <<< "${INIT_METHODS}"
  for item in "${_init_values[@]}"; do
    item="${item// /}"
    [[ -z "${item}" ]] && continue
    if [[ "${item}" != "observation" ]]; then
      die "Non-sparse degradations currently support only INIT_METHODS=observation, got ${INIT_METHODS}."
    fi
  done
}

lambda_list_for_slot() {
  local slot="$1"
  local var_name lambda_values
  case "${slot}" in
    1) var_name="LAMBDAS_0P1" ;;
    2) var_name="LAMBDAS_0P2" ;;
    3) var_name="LAMBDAS_0P3" ;;
    *) die "Only 3 lambda slots are supported, got slot=${slot}." ;;
  esac
  lambda_values="${!var_name}"
  lambda_values="${lambda_values// /}"
  [[ -n "${lambda_values}" ]] || die "${var_name} is empty."
  printf '%s' "${lambda_values}"
}

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

# Split mixed solvers so their parameter sets and output directories stay distinct.
if [[ "${SOLVERS}" == *,* ]]; then
  IFS=',' read -r -a _separate_solvers <<< "${SOLVERS}"
  for _solver in "${_separate_solvers[@]}"; do
    _solver="${_solver// /}"
    case "${_solver}" in
      direct|jinit|feedforward) ;;
      *) die "Unsupported SOLVERS entry: ${_solver}." ;;
    esac
  done
  _solver_sweep_root="${HYDRA_SWEEP_DIR:-outputs/sparse_restoration/multirun/$(date +%Y-%m-%d/%H-%M-%S)}"
  for _solver in "${_separate_solvers[@]}"; do
    _solver="${_solver// /}"
    SOLVERS="${_solver}" HYDRA_SWEEP_DIR="${_solver_sweep_root}/${_solver}" bash "${BASH_SOURCE[0]}"
  done
  exit 0
fi

is_feedforward_only=0
if [[ "${SOLVERS// /}" == "feedforward" ]]; then
  is_feedforward_only=1
fi

degradation_choice="$(single_csv_value "${DEGRADATIONS}")"
degradation_mode="$(canonical_degradation_choice "${degradation_choice}")"

run_case() {
  local label="$1"
  local lambdas="$2"
  local case_tag="${label//=/_}"
  case_tag="${case_tag//./p}"
  shift 2
  local args=(
    "experiments/sparse_restoration/run.py"
    "-m"
    "dataset=${DATASETS}"
    "model=${MODEL_CONFIG}"
    "degradation=${DEGRADATIONS}"
    "inference=${SOLVERS}"
    "inference.split=${INFERENCE_SPLIT}"
    "inference.num_samples=${INFERENCE_NUM_SAMPLES}"
    "model.params.bottleneck_channels=${BOTTLENECK_CHANNELS}"
    "model.params.upsample_mode=${UPSAMPLE_MODES}"
  )
  local extra_arg
  for extra_arg in "$@"; do
    args+=("${extra_arg}")
  done
  if [[ -n "${HYDRA_SWEEP_DIR}" ]]; then
    args+=("hydra.sweep.dir=${HYDRA_SWEEP_DIR}/${case_tag}")
  fi
  if [[ ${is_feedforward_only} -eq 0 ]]; then
    args+=(
      "inference.init_method=${INIT_METHODS}"
      "inference.lambda_init=${lambdas}"
      "inference.update_lambda=true"
      "inference.lambda_multiplier=${LAMBDA_MULTIPLIERS}"
    )
    if [[ "${SOLVERS}" == "jinit" ]]; then
      args+=("inference.aux_lambda_scale=${LAMBDA_INIT_GAMMAS}")
    fi
    if [[ -n "${BACKTRACKING_RULES}" ]]; then
      args+=("inference.backtracking_rule=${BACKTRACKING_RULES}")
    fi
    if [[ -n "${SUFFICIENT_DECREASE_CONSTANT}" ]]; then
      args+=("inference.sufficient_decrease_constant=${SUFFICIENT_DECREASE_CONSTANT}")
    fi
  fi

  if [[ "${SOLVERS}" == "jinit" ]]; then
    args+=("+train.auxiliary.psnr_drops=${PSNR_DROPS}")
  fi

  if [[ -n "${TRAIN_SUBSET_SIZE}" ]]; then
    args+=("+train.train_subset_size=${TRAIN_SUBSET_SIZE}")
  fi
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  if [[ ${is_feedforward_only} -eq 1 ]]; then
    echo "Running ${label} split=${INFERENCE_SPLIT} num_samples=${INFERENCE_NUM_SAMPLES} solver=feedforward"
  else
    echo "Running ${label} split=${INFERENCE_SPLIT} num_samples=${INFERENCE_NUM_SAMPLES} lambdas=${lambdas} backtracking=${BACKTRACKING_RULES:-default}"
  fi
  echo "${PYTHON_EXE} ${args[*]}"

  "${PYTHON_EXE}" "${args[@]}"
}

run_strength_sweep() {
  local label_key="$1"
  local arg_key="$2"
  local csv_values="$3"
  local fixed_lambdas="${4:-}"
  local item slot=0
  IFS=',' read -r -a _strength_values <<< "${csv_values}"
  for item in "${_strength_values[@]}"; do
    item="${item// /}"
    [[ -z "${item}" ]] && continue
    ((slot += 1))
    [[ ${slot} -le 3 ]] || die "${label_key} supports at most 3 strengths, got ${csv_values}."
    if [[ ${is_feedforward_only} -eq 1 ]]; then
      run_case "${label_key}=${item}" "" "${arg_key}=${item}"
      continue
    fi
    run_case \
      "${label_key}=${item}" \
      "${fixed_lambdas:-$(lambda_list_for_slot "${slot}")}" \
      "${arg_key}=${item}"
  done
}

if [[ "${degradation_mode}" == "sparse" ]]; then
  run_strength_sweep "noise" "degradation.params.noise_ratio" "${NOISE_RATIOS}"
elif [[ "${degradation_mode}" == "inpainting" ]]; then
  ensure_non_sparse_init_methods_supported
  run_strength_sweep "mask_ratio" "degradation.params.mask_ratio" "$(default_mask_ratios "${degradation_choice}")" ""
elif [[ "${degradation_mode}" == "deblur" ]]; then
  ensure_non_sparse_init_methods_supported
  run_strength_sweep "blur_sigma" "degradation.params.blur_sigma" "${BLUR_SIGMAS}"
else
  die "Unsupported canonical degradation mode: ${degradation_mode}"
fi

echo
echo "Sweep finished. This script call may create multiple multirun directories."
echo "To collect aggregate tables, run one of:"
echo "  ${PYTHON_EXE} experiments/sparse_restoration/collect.py outputs/sparse_restoration/multirun"
echo "  ${PYTHON_EXE} experiments/sparse_restoration/collect.py outputs/sparse_restoration/multirun/<date>/<time>"
