#!/usr/bin/env bash
set -euo pipefail

# Stabilize NVIDIA GPUs for long-running training jobs.
# - Enables persistence mode
# - Applies optional power limit
# - Prints final status snapshot
#
# Usage examples:
#   bash tools/gpu_stabilize.sh
#   bash tools/gpu_stabilize.sh --gpus 0,1 --power-limit 300

GPUS="all"
POWER_LIMIT=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="${2:-all}"
      shift 2
      ;;
    --power-limit)
      POWER_LIMIT="${2:-300}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. NVIDIA driver is not ready." >&2
  exit 1
fi

declare -a TARGET_GPUS=()
if [[ "$GPUS" == "all" ]]; then
  while IFS= read -r idx; do
    [[ -n "$idx" ]] && TARGET_GPUS+=("$idx")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  IFS=',' read -r -a TARGET_GPUS <<< "$GPUS"
fi

if [[ ${#TARGET_GPUS[@]} -eq 0 ]]; then
  echo "No GPU found to configure." >&2
  exit 1
fi

run_nvidia_smi() {
  if nvidia-smi "$@" >/dev/null 2>&1; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n nvidia-smi "$@" >/dev/null 2>&1
    return $?
  fi
  return 1
}

echo "[gpu_stabilize] target_gpus=${TARGET_GPUS[*]}"
echo "[gpu_stabilize] requested_power_limit=${POWER_LIMIT}W"

for g in "${TARGET_GPUS[@]}"; do
  if run_nvidia_smi -i "$g" -pm 1; then
    echo "[gpu_stabilize] GPU $g persistence mode: ON"
  else
    echo "[gpu_stabilize] WARN: failed to set persistence mode on GPU $g (need sudo/root?)"
  fi

  if run_nvidia_smi -i "$g" -pl "$POWER_LIMIT"; then
    echo "[gpu_stabilize] GPU $g power limit set to ${POWER_LIMIT}W"
  else
    echo "[gpu_stabilize] WARN: failed to set power limit on GPU $g (need sudo/root?)"
  fi
done

echo "[gpu_stabilize] status:"
nvidia-smi --query-gpu=index,name,pstate,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total,display_active --format=csv
