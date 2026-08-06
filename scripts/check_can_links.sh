#!/usr/bin/env bash
set -euo pipefail

BITRATE="${CAN_BITRATE:-1000000}"
DURATION="${CANDUMP_SECONDS:-3}"
INTERFACES=("$@")

if [ "${#INTERFACES[@]}" -eq 0 ]; then
  INTERFACES=(can0 can1)
fi

if ! command -v candump >/dev/null 2>&1; then
  echo "candump not found. Install can-utils first:" >&2
  echo "  sudo apt-get install -y can-utils" >&2
  exit 1
fi

for iface in "${INTERFACES[@]}"; do
  echo
  echo "========== ${iface} =========="

  if ! ip link show "${iface}" >/dev/null 2>&1; then
    echo "ERROR: ${iface} does not exist."
    echo "Available CAN interfaces:"
    ip link show | grep -E "can[0-9]+" || true
    continue
  fi

  echo "[1/3] Configure ${iface} bitrate=${BITRATE}"
  sudo ip link set "${iface}" down 2>/dev/null || true
  sudo ip link set "${iface}" type can bitrate "${BITRATE}"
  sudo ip link set "${iface}" up

  echo "[2/3] Link status"
  ip -details link show "${iface}" | sed -n '1,10p'

  tmp_file="$(mktemp)"
  echo "[3/3] candump ${iface} for ${DURATION}s"
  set +e
  timeout "${DURATION}" candump "${iface}" >"${tmp_file}"
  set -e

  frame_count="$(wc -l <"${tmp_file}" | tr -d ' ')"
  if [ "${frame_count}" -gt 0 ]; then
    echo "OK: received ${frame_count} frame(s) on ${iface}. First frames:"
    head -10 "${tmp_file}"
  else
    echo "WARN: received 0 frames on ${iface}."
    echo "      Check arm power, CANH/CANL, termination, USB-CAN mapping, and whether this interface is connected to the arm."
  fi
  rm -f "${tmp_file}"
done
