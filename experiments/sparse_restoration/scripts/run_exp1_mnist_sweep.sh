#!/usr/bin/env bash
#! GPU=0 bash experiments/sparse_restoration/scripts/run_exp1_mnist_sweep.sh

# Current top-level config:


# Useful Setting
# Warm: Obs init in sparse with lambda 0.1 -- model: lighter resae_3down BOTTLENECK_CHANNELS=12, shrinkage_weight_list=[1e-3,1e-4,0.0] psnr27.9
# Fixed MNIST net: resae_2down, stage_channels=[16,32,64], bottleneck_channels=8, upsample_mode=deconv, +train.jinit.psnr_drops=[14.0,7.0]; tune lambda by noise_ratio: 0.1->0.04->27.1, 0.2->0.03->23.6, 0.3->0.03->20.8



set -euo pipefail
# Resolve project paths before passing relative scripts and checkpoint roots to Python.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
# Note: LAMBDAS_0P1/0P2/0P3 map to the 1st/2nd/3rd items in NOISE_RATIOS.
# This script uses the current MNIST resae_3down family and full-test split by default.

SOLVERS="${SOLVERS:-jinit}" # direct, jinit, feedforward
DATASETS="${DATASETS:-mnist}" # mnist, celeba-hq-256
# LAMBDAS="${LAMBDAS:-0.1, 1.0, 5.0, 10.0, 20.0, 100.0, 200.0}" # ~0.1 when sparse, ~10.0 when inpainting
NOISE_RATIOS="${NOISE_RATIOS:-0.1,0.2,0.3}"
# Default lambda grids are selected by sweep slot: 1st/2nd/3rd degradation strength.
LAMBDAS_0P1="${LAMBDAS_0P1:-0.06,0.07, 0.08,0.1}"
LAMBDAS_0P2="${LAMBDAS_0P2:-0.03,0.04,0.05,0.06}"
LAMBDAS_0P3="${LAMBDAS_0P3:-0.03,0.04,0.05,0.06}"
DEGRADATIONS="${DEGRADATIONS:-sparse}"
INIT_METHODS="${INIT_METHODS:-observation}" # observation, clean (diagnostics only)
MASK_RATIOS="${MASK_RATIOS:-}"
BLUR_SIGMAS="${BLUR_SIGMAS:-1.0}"
MODEL_CONFIG="${MODEL_CONFIG:-resae_3down}"

# After checkpoints exist, run the full recovery sweep with force_retrain=false.
BOTTLENECK_CHANNELS="${BOTTLENECK_CHANNELS:-6}"
MODEL_STAGE_CHANNELS="${MODEL_STAGE_CHANNELS:-[32,64,128,256]}"
UPSAMPLE_MODES="${UPSAMPLE_MODES:-deconv}"
PSNR_DROPS="${PSNR_DROPS:-[10.0,5.0]}"
TRAIN_SUBSET_SIZE="${TRAIN_SUBSET_SIZE:-}"
GPU="${GPU:-0}"
INFERENCE_SPLIT="${INFERENCE_SPLIT:-test}"
INFERENCE_NUM_SAMPLES="${INFERENCE_NUM_SAMPLES:-null}"
LAMBDA_INIT_GAMMAS="${LAMBDA_INIT_GAMMAS:-1.0}"
LAMBDA_MULTIPLIERS="${LAMBDA_MULTIPLIERS:-0.8}"
BACKTRACKING_RULES="${BACKTRACKING_RULES:-}"
SUFFICIENT_DECREASE_CONSTANT="${SUFFICIENT_DECREASE_CONSTANT:-}"
HYDRA_SWEEP_DIR="${HYDRA_SWEEP_DIR:-}"

PYTHON_EXE="${PYTHON_EXE:-python}"

warn() {
  echo "[run_exp1_mnist_sweep] WARNING: $*" >&2
}

die() {
  echo "[run_exp1_mnist_sweep] ERROR: $*" >&2
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
    "model.params.stage_channels=${MODEL_STAGE_CHANNELS}"
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
