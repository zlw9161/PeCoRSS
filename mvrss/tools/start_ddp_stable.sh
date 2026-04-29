#!/usr/bin/env bash
set -euo pipefail

# Stable DDP launcher for TFoc on Linux.
# Goals:
# - Keep realtime logs (no conda output capture)
# - Keep correct imports (PYTHONPATH=..)
# - Enforce model variant
# - Optionally avoid display-attached GPU(s)
# - Optionally apply GPU power limit before training

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CFG="config_files/tfoc_v2_full_2x3090.json"
NPROC=2
CONDA_ENV="t-foc"
MODEL_VARIANT="v2"
AUTO_EXCLUDE_DISPLAY=1
SET_POWER_LIMIT=1
POWER_LIMIT=300
VISIBLE_GPUS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg)
      CFG="${2:?missing value for --cfg}"
      shift 2
      ;;
    --nproc)
      NPROC="${2:?missing value for --nproc}"
      shift 2
      ;;
    --env)
      CONDA_ENV="${2:?missing value for --env}"
      shift 2
      ;;
    --variant)
      MODEL_VARIANT="${2:?missing value for --variant}"
      shift 2
      ;;
    --gpus)
      VISIBLE_GPUS="${2:?missing value for --gpus}"
      shift 2
      ;;
    --power-limit)
      POWER_LIMIT="${2:?missing value for --power-limit}"
      shift 2
      ;;
    --no-power-limit)
      SET_POWER_LIMIT=0
      shift
      ;;
    --no-auto-exclude-display)
      AUTO_EXCLUDE_DISPLAY=0
      shift
      ;;
    --)
      shift
      EXTRA_ARGS="$*"
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CFG" ]]; then
  echo "Config file not found: $CFG" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. NVIDIA driver is not available." >&2
  exit 1
fi

select_visible_gpus() {
  if [[ -n "$VISIBLE_GPUS" ]]; then
    echo "$VISIBLE_GPUS"
    return
  fi

  if [[ "$AUTO_EXCLUDE_DISPLAY" -eq 0 ]]; then
    local all
    all="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
    echo "$all"
    return
  fi

  local off_list
  off_list="$(nvidia-smi --query-gpu=index,display_active --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$2); if($2=="Disabled") print $1}')"

  if [[ -z "$off_list" ]]; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -
    return
  fi

  echo "$off_list" | paste -sd, -
}

VISIBLE_GPUS="$(select_visible_gpus)"
IFS=',' read -r -a GPU_ARRAY <<< "$VISIBLE_GPUS"
if [[ "${#GPU_ARRAY[@]}" -lt "$NPROC" ]]; then
  echo "Not enough visible GPUs ($VISIBLE_GPUS) for nproc=$NPROC" >&2
  exit 1
fi

SELECTED_GPUS="$(printf "%s\n" "${GPU_ARRAY[@]}" | head -n "$NPROC" | paste -sd, -)"

if [[ "$SET_POWER_LIMIT" -eq 1 ]]; then
  echo "[start_ddp_stable] applying power limit ${POWER_LIMIT}W to GPU(s): ${SELECTED_GPUS}"
  bash tools/gpu_stabilize.sh --gpus "$SELECTED_GPUS" --power-limit "$POWER_LIMIT" || true
fi

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/ddp_v2_${TS}.log"
echo "$LOG" > logs/latest_ddp_log.txt

echo "[start_ddp_stable] root=$ROOT_DIR"
echo "[start_ddp_stable] cfg=$CFG"
echo "[start_ddp_stable] nproc=$NPROC"
echo "[start_ddp_stable] conda_env=$CONDA_ENV"
echo "[start_ddp_stable] model_variant=$MODEL_VARIANT"
echo "[start_ddp_stable] CUDA_VISIBLE_DEVICES=$SELECTED_GPUS"
echo "[start_ddp_stable] log=$LOG"
echo "[start_ddp_stable] nvidia-smi snapshot:"
nvidia-smi --query-gpu=index,name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total,display_active --format=csv

env CUDA_VISIBLE_DEVICES="$SELECTED_GPUS" PYTHONPATH=.. TFOC_MODEL_VARIANT="$MODEL_VARIANT" \
  conda run --no-capture-output -n "$CONDA_ENV" \
  torchrun --standalone --nproc_per_node "$NPROC" train_F_ddp.py --cfg "$CFG" $EXTRA_ARGS 2>&1 | tee "$LOG"
