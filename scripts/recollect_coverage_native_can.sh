#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"

echo "Authorize CAN interface setup once for both long-running trajectories."
sudo -v

keep_sudo_alive() {
  while true; do
    sleep 30
    sudo -n -v || exit
  done
}

keep_sudo_alive &
SUDO_KEEPALIVE_PID=$!
trap 'kill "${SUDO_KEEPALIVE_PID}" 2>/dev/null || true' EXIT INT TERM

CONFIGS=(
  "configs/joint_pose_coverage.yaml"
  "configs/representative_replay.yaml"
)

for config in "${CONFIGS[@]}"; do
  if ! sudo -n -v; then
    echo "sudo authorization expired; stopping before ${config}." >&2
    exit 1
  fi

  echo
  echo "Validating approved trajectory: ${config}"
  "${PYTHON_BIN}" scripts/free_space.py --config "${config}" summary

  echo
  echo "Starting fresh hardware collection: ${config}"
  echo "Trajectory commands remain 50 Hz; every complete native CAN state is saved."
  "${PYTHON_BIN}" scripts/free_space.py \
    --config "${config}" \
    collect --fresh "$@"
done
