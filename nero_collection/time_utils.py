from __future__ import annotations

import time


def now_us() -> int:
    return time.time_ns() // 1_000


def monotonic_us() -> int:
    return time.monotonic_ns() // 1_000
