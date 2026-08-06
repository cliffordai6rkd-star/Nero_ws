from __future__ import annotations

from datetime import datetime

from nero_collection.episode_output import episode_path, next_episode_index


def test_episode_output_continues_shared_numbering(tmp_path) -> None:
    (tmp_path / "episode_0000_20260724_120000.h5").touch()
    (tmp_path / "episode_0007_20260724_130000.h5").touch()
    (tmp_path / "different_0042_20260724_140000.h5").touch()
    (tmp_path / "episode_invalid.h5").touch()

    index = next_episode_index(tmp_path, "episode")
    output = episode_path(
        tmp_path,
        "episode",
        index,
        timestamp=datetime(2026, 7, 24, 15, 30, 45),
    )

    assert index == 8
    assert output == tmp_path / "episode_0008_20260724_153045.h5"
