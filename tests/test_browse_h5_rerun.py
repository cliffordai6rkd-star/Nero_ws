from __future__ import annotations

from pathlib import Path

from scripts import browse_h5_rerun


def _episode(path: Path, index: int) -> Path:
    episode = path / f"episode_{index:04d}_demo.h5"
    episode.touch()
    return episode


def test_browse_plays_next_episode_and_moves_to_trash_on_d(
    tmp_path: Path, monkeypatch
) -> None:
    first = _episode(tmp_path, 1)
    second = _episode(tmp_path, 2)
    calls: list[int] = []

    monkeypatch.setattr(
        browse_h5_rerun,
        "run_rerun",
        lambda _runs_dir, episode_index, _camera: calls.append(episode_index) or 0,
    )
    actions = iter(("1", "2", "d", "q"))
    output: list[str] = []

    assert (
        browse_h5_rerun.browse(
            tmp_path,
            input_fn=lambda _prompt: next(actions),
            output_fn=output.append,
        )
        == 0
    )
    assert calls == [1, 2]
    assert first.is_file()
    assert not second.exists()
    assert (tmp_path.parent / "trash" / second.name).is_file()
    assert any("Moved" in line and "trash" in line for line in output)


def test_discover_episodes_rejects_duplicate_indices(tmp_path: Path) -> None:
    _episode(tmp_path, 3)
    (tmp_path / "episode_0003_other.h5").touch()

    try:
        browse_h5_rerun.discover_episodes(tmp_path)
    except RuntimeError as exc:
        assert "multiple files" in str(exc)
    else:  # pragma: no cover - assertion helper for a clearer failure
        raise AssertionError("duplicate episode indices should be rejected")
