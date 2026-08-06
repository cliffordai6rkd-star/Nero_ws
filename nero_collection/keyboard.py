from __future__ import annotations

import os
import logging
import select
import sys
import termios
import tty
from types import TracebackType


log = logging.getLogger(__name__)


class TerminalKeys:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings: list[int | bytes] | None = None
        self._owns_fd = False
        self.is_tty = sys.stdin.isatty()

    def __enter__(self) -> "TerminalKeys":
        if self.is_tty:
            self._fd = sys.stdin.fileno()
            log.debug("using stdin for terminal keyboard input fd=%d", self._fd)
        else:
            try:
                self._fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
            except OSError as exc:
                log.debug("no interactive terminal keyboard is available: %s", exc)
                self._fd = None
                self.is_tty = False
                return self
            self._owns_fd = True
            self.is_tty = True
            log.info("stdin is not a TTY; using /dev/tty for interactive keyboard input")
        try:
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            if self._owns_fd and self._fd is not None:
                os.close(self._fd)
            self._fd = None
            self._owns_fd = False
            self.is_tty = False
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        if self._owns_fd and self._fd is not None:
            os.close(self._fd)
        self._fd = None
        self._owns_fd = False

    def read_key(self, timeout_s: float) -> str | None:
        if not self.is_tty or self._fd is None:
            return None
        ready, _, _ = select.select([self._fd], [], [], timeout_s)
        if not ready:
            return None
        return os.read(self._fd, 1).decode("utf-8", errors="ignore")
