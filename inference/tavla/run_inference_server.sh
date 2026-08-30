#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data/jhr/TA-VLA"
checkpoint_base="/data/jhr/ta_vla_checkpoints"

usage() {
  cat <<'EOF'
Usage:
  inference/tavla/run_inference_server.sh {usb|button|cucumber} [options]

Options:
  --port PORT          WebSocket port (default: 8000)
  --gpu GPU_ID         One physical GPU visible to JAX (default: 0)
  --checkpoint DIR     Override the final checkpoint directory
  --record             Record server inputs and outputs for debugging

Examples:
  inference/tavla/run_inference_server.sh usb --gpu 0 --port 8000
  inference/tavla/run_inference_server.sh button --gpu 1 --port 8001
  inference/tavla/run_inference_server.sh cucumber --gpu 0 --port 8000
EOF
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

task="$1"
shift
port=8000
gpu=0
checkpoint_dir=""
record=false

while (( $# > 0 )); do
  case "$1" in
    --port)
      port="${2:?--port requires a value}"
      shift 2
      ;;
    --gpu)
      gpu="${2:?--gpu requires a value}"
      shift 2
      ;;
    --checkpoint)
      checkpoint_dir="${2:?--checkpoint requires a value}"
      shift 2
      ;;
    --record)
      record=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${task}" in
  usb)
    config="pi0_lora_icra2027_usb_tavla_best"
    experiment="icra2027_usb_tavla_best_20260825"
    prompt="insert the USB plug into the port"
    ;;
  button)
    config="pi0_lora_icra2027_button_tavla_best"
    experiment="icra2027_button_tavla_best_20260825"
    prompt="press the button"
    ;;
  cucumber|cuccumber)
    config="pi0_lora_icra2027_cucumber_tavla_best"
    experiment="icra2027_cucumber_tavla_best_20260827"
    prompt="peel the cucumber"
    ;;
  *)
    echo "Task must be 'usb', 'button', or 'cucumber'." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "${checkpoint_dir}" ]]; then
  checkpoint_dir="${checkpoint_base}/${config}/${experiment}/29999"
fi

if [[ ! -d "${checkpoint_dir}/params" || ! -f "${checkpoint_dir}/assets/norm_stats.json" ]]; then
  echo "Checkpoint is incomplete or missing: ${checkpoint_dir}" >&2
  echo "A deployable checkpoint must contain params/ and assets/norm_stats.json." >&2
  exit 1
fi

if ! [[ "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "Invalid port: ${port}" >&2
  exit 2
fi

if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
  echo "--gpu must be one physical GPU index, got: ${gpu}" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t gpu_free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( gpu >= ${#gpu_free_mib[@]} )); then
    echo "GPU index ${gpu} does not exist; found ${#gpu_free_mib[@]} GPU(s)." >&2
    exit 2
  fi
  min_free_mib=12288
  if (( gpu_free_mib[gpu] < min_free_mib )) && [[ "${TAVLA_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
    echo "GPU ${gpu} has only ${gpu_free_mib[gpu]} MiB free; at least ${min_free_mib} MiB is required." >&2
    echo "Stop the training/other GPU job first. To bypass this guard, set TAVLA_ALLOW_BUSY_GPU=1." >&2
    exit 1
  fi
fi

if command -v ss >/dev/null 2>&1 && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
  echo "TCP port ${port} is already in use." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu}"
export TMPDIR="/data/jhr/tmp/tavla_inference"
export PYTHONPYCACHEPREFIX="/data/jhr/pycache"
export XDG_CACHE_HOME="/data/jhr/cache"
export CUDA_CACHE_PATH="/data/jhr/cuda_cache"
export JAX_COMPILATION_CACHE_DIR="/data/jhr/jax_cache"
export OPENPI_DATA_HOME="/data/jhr/openpi_cache"
export LEROBOT_HOME="/data/jhr/lerobot_home"
export HF_HOME="/data/jhr/hf_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

mkdir -p "${TMPDIR}" "${CUDA_CACHE_PATH}" "${JAX_COMPILATION_CACHE_DIR}"
cd "${repo_dir}"

args=(
  --default-prompt "${prompt}"
  --port "${port}"
)
if [[ "${record}" == true ]]; then
  args+=(--record)
fi
args+=(
  policy:checkpoint
  --policy.config "${config}"
  --policy.dir "${checkpoint_dir}"
)

echo "Starting TA-VLA policy server"
echo "  task:       ${task}"
echo "  config:     ${config}"
echo "  checkpoint: ${checkpoint_dir}"
echo "  GPU:        ${gpu}"
echo "  endpoint:   ws://0.0.0.0:${port}"

exec "${repo_dir}/.venv/bin/python" scripts/serve_policy.py "${args[@]}"
