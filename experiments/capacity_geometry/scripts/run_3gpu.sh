#!/usr/bin/env bash

set -euo pipefail

# Run one independent model seed per GPU while all scientific settings come from Python defaults.
# Evaluate supplied checkpoints without starting training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="outputs/capacity_geometry/3gpu_${RUN_ID}"
LOG_ROOT="outputs/capacity_geometry/logs/3gpu_${RUN_ID}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

pids=()
for worker in 0 1 2; do
  seeds=(2 1024 2026)
  seed="${seeds[$worker]}"
  CUDA_VISIBLE_DEVICES="${worker}" "${PYTHON_EXE}" experiments/capacity_geometry/run_capacity.py \
    --eval-only \
    --model-seeds "${seed}" \
    --output-root "${RUN_ROOT}" \
    > "${LOG_ROOT}/gpu${worker}_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "At least one worker failed. Check ${LOG_ROOT}." >&2
  exit 1
fi

# Merge the three seed shards into the final cross-seed summaries and figure.
"${PYTHON_EXE}" - "${RUN_ROOT}" <<'PY'
from pathlib import Path
import sys

import pandas as pd

from experiments.capacity_geometry.run_capacity import plot_summary, summarize

root = Path(sys.argv[1])
files = sorted(root.glob("*/run_metrics.csv"))
if len(files) != 3:
    raise RuntimeError(f"Expected 3 result shards under {root}, found {len(files)}.")

rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
if len(rows) != 42:
    raise RuntimeError(f"Expected 42 run rows, found {len(rows)}.")

output = root / "merged"
output.mkdir(exist_ok=False)
summary = summarize(rows.to_dict("records"), output)
plot_summary(summary, output / "capacity_tradeoff.png")
print(f"Merged results saved to: {output}")
PY

echo "Experiment completed: ${RUN_ROOT}/merged"
