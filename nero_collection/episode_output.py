from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def episode_path(
    output_dir: Path,
    prefix: str,
    index: int,
    *,
    timestamp: datetime | None = None,
) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{index:04d}_{stamp}.h5"


def next_episode_index(output_dir: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_.*\.h5$")
    max_index = -1
    for path in output_dir.glob(f"{prefix}_*.h5"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1
