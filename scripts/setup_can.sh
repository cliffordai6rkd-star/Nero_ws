#!/usr/bin/env bash
set -euo pipefail

BITRATE="${CAN_BITRATE:-1000000}"
INTERFACES=("$@")

if [ "${#INTERFACES[@]}" -eq 0 ]; then
  INTERFACES=(can0 can1)
fi

for iface in "${INTERFACES[@]}"; do
  echo "Configuring ${iface} bitrate=${BITRATE}"
  sudo ip link set "${iface}" down 2>/dev/null || true
  sudo ip link set "${iface}" type can bitrate "${BITRATE}"
  sudo ip link set "${iface}" up
  ip -details link show "${iface}" | sed -n '1,6p'
done
