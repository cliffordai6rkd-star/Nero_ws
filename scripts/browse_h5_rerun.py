#!/usr/bin/env python3
"""Interactively browse, rerun, and archive Nero H5 episodes for review."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def discover_episodes(runs_dir: Path) -> dict[int, Path]:
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise RuntimeError(f"Runs directory does not exist: {runs_dir}")
    episodes: dict[int, Path] = {}
    for path in sorted(runs_dir.glob("episode_????_*.h5")):
        try:
            episode_index = int(path.name.split("_", 2)[1])
        except (IndexError, ValueError):
            continue
        previous = episodes.get(episode_index)
        if previous is not None and previous != path:
            raise RuntimeError(
                f"Episode {episode_index} has multiple files: {previous.name}, {path.name}"
            )
        episodes[episode_index] = path
    return episodes


def run_rerun(runs_dir: Path, episode_index: int, camera: str | None) -> int:
    script = Path(__file__).with_name("visualize_h5_rerun.py").resolve()
    command = [sys.executable, str(script), str(runs_dir), "--episode", str(episode_index)]
    if camera:
        command.extend(("--camera", camera))
    return int(subprocess.run(command, check=False).returncode)


def _trash_path(runs_dir: Path, episode_path: Path) -> Path:
    """Return a non-overwriting destination in the shared runs/trash folder."""

    trash_dir = runs_dir.parent / "trash"
    candidate = trash_dir / episode_path.name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = trash_dir / f"{episode_path.stem}_{suffix}{episode_path.suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def browse(
    runs_dir: Path,
    camera: str | None = None,
    input_fn=input,
    output_fn=print,
) -> int:
    runs_dir = runs_dir.expanduser().resolve()
    episodes = discover_episodes(runs_dir)
    if not episodes:
        output_fn(f"No episode_????_*.h5 files found in {runs_dir}")
        return 1

    output_fn(f"Rerun episode browser: {runs_dir}")
    output_fn("Enter an episode number to play, or q to quit.")
    while True:
        raw_episode = input_fn("episode> ").strip()
        if raw_episode.lower() in {"q", "quit", "exit"}:
            return 0
        try:
            episode_index = int(raw_episode)
        except ValueError:
            output_fn("Please enter a numeric episode index or q.")
            continue
        episode_path = episodes.get(episode_index)
        if episode_path is None:
            output_fn(f"Episode {episode_index} was not found in {runs_dir}")
            continue

        output_fn(f"Playing episode {episode_index}: {episode_path.name}")
        return_code = run_rerun(runs_dir, episode_index, camera)
        if return_code != 0:
            output_fn(f"Rerun exited with code {return_code}; the file remains in place.")

        while True:
            action = input_fn(
                "After playback: d=move this episode to trash, number=play next, "
                "Enter=episode menu, q=quit> "
            ).strip()
            if action.lower() in {"q", "quit", "exit"}:
                return 0
            if action.lower() == "d":
                trash_path = _trash_path(runs_dir, episode_path)
                trash_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(episode_path), str(trash_path))
                output_fn(f"Moved {episode_path} to {trash_path}")
                episodes.pop(episode_index, None)
                break
            if action == "":
                break
            try:
                next_episode = int(action)
            except ValueError:
                output_fn("Enter d, a numeric episode index, Enter, or q.")
                continue
            if next_episode not in episodes:
                output_fn(f"Episode {next_episode} was not found in {runs_dir}")
                continue
            episode_index = next_episode
            episode_path = episodes[episode_index]
            output_fn(f"Playing episode {episode_index}: {episode_path.name}")
            return_code = run_rerun(runs_dir, episode_index, camera)
            if return_code != 0:
                output_fn(
                    f"Rerun exited with code {return_code}; the file remains in place."
                )

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively browse Nero H5 episodes in Rerun."
    )
    parser.add_argument(
        "runs_dir",
        nargs="?",
        type=Path,
        help="Folder containing episode_NNNN_*.h5 files; prompted when omitted",
    )
    parser.add_argument("--camera", help="Camera group passed to visualize_h5_rerun.py")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_dir = args.runs_dir
    if runs_dir is None:
        runs_dir = Path(input("Runs folder> ").strip()).expanduser()
    return browse(runs_dir, camera=args.camera)


if __name__ == "__main__":
    raise SystemExit(main())
