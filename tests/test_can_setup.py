from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import nero_collection.cli as cli
from nero_collection.config import (
    ArmEndpointConfig,
    ArmPairConfig,
    CollectionConfig,
    OutputConfig,
    TeleopConfig,
)


def _config(leader: ArmEndpointConfig, follower: ArmEndpointConfig) -> CollectionConfig:
    return CollectionConfig(
        teleop=TeleopConfig(
            backend="pyagxarm",
            master_slave=(ArmPairConfig(name="main", leader=leader, follower=follower),),
        ),
        output=OutputConfig(directory=Path(".")),
    )


def test_setup_can_uses_configured_channels_and_bitrate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command, *, cwd, check):
        assert check is True
        calls.append((list(command), Path(cwd)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    config = _config(
        ArmEndpointConfig(name="master", channel="can0", bitrate=1_000_000),
        ArmEndpointConfig(name="slave", channel="can1", bitrate=1_000_000),
    )

    cli._setup_can_interfaces(config)

    assert len(calls) == 1
    command, cwd = calls[0]
    assert command[0] == cli.sys.executable
    assert command[-2:] == ["can0", "can1"]
    assert command[1].endswith("scripts/setup_can.py")
    assert command[2:4] == ["--bitrate", "1000000"]
    assert cwd == Path(cli.__file__).resolve().parents[1]


def test_setup_can_runs_bitrate_groups_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_run(command, *, cwd, check):
        calls.append((list(command), command[3]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    config = _config(
        ArmEndpointConfig(name="master", channel="can0", bitrate=1_000_000),
        ArmEndpointConfig(name="slave", channel="can1", bitrate=500_000),
    )

    cli._setup_can_interfaces(config)

    assert [(call[0][-1], call[1]) for call in calls] == [
        ("can0", "1000000"),
        ("can1", "500000"),
    ]


def test_setup_can_rejects_conflicting_bitrate_on_same_channel() -> None:
    config = _config(
        ArmEndpointConfig(name="master", channel="can0", bitrate=1_000_000),
        ArmEndpointConfig(name="slave", channel="can0", bitrate=500_000),
    )

    with pytest.raises(RuntimeError, match="Conflicting bitrates"):
        cli._setup_can_interfaces(config)
