#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ubuntu 20.04 ships Python 3.8 by default, while setup_env.py uses Python
# 3.10 syntax. Run the bootstrap script with uv's managed interpreter so the
# caller does not need to activate .venv first.
if command -v uv >/dev/null 2>&1; then
  exec uv run --python 3.10 --no-project python \
    "${ROOT_DIR}/scripts/setup_env.py" "$@"
fi

exec python3 "${ROOT_DIR}/scripts/setup_env.py" "$@"
