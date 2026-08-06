from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py

from nero_collection import socketcan
from scripts import inspect_free_space_h5, open_rerun_recording, setup_can, setup_env


def test_setup_can_defaults_to_both_interfaces() -> None:
    args = setup_can._parse_args([])

    assert args.interfaces == ["can0", "can1"]
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


def test_setup_env_resolves_named_environment(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command[-3:] == ["env", "list", "--json"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"envs": ["/opt/conda", "/opt/conda/envs/nero"]}),
        )

    monkeypatch.setattr(setup_env.subprocess, "run", fake_run)

    assert setup_env._find_conda_environment("conda", "nero") == Path(
        "/opt/conda/envs/nero"
    )
