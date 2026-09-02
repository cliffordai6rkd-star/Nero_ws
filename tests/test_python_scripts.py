from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from nero_collection import socketcan
from scripts import inspect_free_space_h5, open_rerun_recording, setup_can, setup_env


def test_setup_can_defaults_to_both_interfaces() -> None:
    args = setup_can._parse_args([])

    assert args.interfaces == ["can_master", "can_slave"]
    assert args.bitrate == 1_000_000


def test_socketcan_configures_interface_with_argument_lists(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(socketcan.subprocess, "run", fake_run)

    socketcan.configure_interface("can1", 500_000)

    assert [call[0] for call in calls] == [
        ["sudo", "ip", "link", "set", "can1", "down"],
        ["sudo", "ip", "link", "set", "can1", "type", "can", "bitrate", "500000"],
        ["sudo", "ip", "link", "set", "can1", "up"],
    ]
    assert calls[0][1]["check"] is False
    assert calls[1][1]["check"] is True
    assert calls[2][1]["check"] is True


def test_socketcan_resolves_current_interface_from_usb_serial(tmp_path: Path) -> None:
    sys_class_net = tmp_path / "sys" / "class" / "net"
    usb_device = tmp_path / "devices" / "usb1" / "1-2"
    usb_interface = usb_device / "1-2:1.0"
    usb_interface.mkdir(parents=True)
    (usb_device / "serial").write_text("MASTER-SERIAL\n", encoding="ascii")
    interface_dir = sys_class_net / "can7"
    interface_dir.mkdir(parents=True)
    (interface_dir / "device").symlink_to(usb_interface, target_is_directory=True)

    assert socketcan.interface_usb_serial("can7", sys_class_net) == "MASTER-SERIAL"
    assert (
        socketcan.interface_for_usb_serial("MASTER-SERIAL", sys_class_net)
        == "can7"
    )


def test_socketcan_rejects_missing_usb_serial(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="is not present"):
        socketcan.interface_for_usb_serial("MISSING", tmp_path)


def test_inspect_free_space_h5_finds_latest_matching_episode(tmp_path: Path) -> None:
    path = tmp_path / "episode_0003_test.h5"
    metadata = {
        "source": "free_space_coverage",
        "trajectory_name": "coverage",
        "trajectory_seed": 42,
        "trajectory_sha256": "abc123",
    }
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "factr_multimodal_episode/v7"
        h5.create_dataset("teleop/timestamp_us", data=[1, 2, 3])
        h5.create_dataset("metadata/episode_json", data=json.dumps(metadata).encode())

    selected, details = inspect_free_space_h5.find_latest_episode(tmp_path)

    assert selected == path
    assert details == {
        "format": "factr_multimodal_episode/v7",
        "samples": 3,
        "trajectory_name": "coverage",
        "trajectory_seed": 42,
        "trajectory_sha256": "abc123",
    }


def test_open_rerun_uses_current_python_environment(tmp_path: Path, monkeypatch) -> None:
    recording = tmp_path / "episode.rrd"
    recording.touch()
    calls = []

    def fake_run(command, *, check):
        calls.append((list(command), check))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(open_rerun_recording.subprocess, "run", fake_run)

    assert open_rerun_recording.main([str(recording)]) == 7
    assert calls == [
        (
            [
                open_rerun_recording.sys.executable,
                "-m",
                "rerun",
                str(recording.resolve()),
            ],
            False,
        )
    ]


def test_setup_env_defaults_to_uv_python() -> None:
    args = setup_env._parse_args([])

    assert args.venv == ".venv"
    assert args.python_version == "3.10"
