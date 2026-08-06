from __future__ import annotations

import nero_collection.keyboard as keyboard


def test_terminal_keys_reads_directly_from_file_descriptor(monkeypatch) -> None:
    keys = keyboard.TerminalKeys()
    keys.is_tty = True
    keys._fd = 42
    monkeypatch.setattr(keyboard.select, "select", lambda *_args: ([42], [], []))
    monkeypatch.setattr(keyboard.os, "read", lambda fd, size: b"t")

    assert keys.read_key(0.0) == "t"
