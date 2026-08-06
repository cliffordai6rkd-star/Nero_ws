#!/usr/bin/env python3
"""Refresh a compact architecture snapshot for the local LeRobot workspace."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import re

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - compatibility path
    tomllib = None


EXCLUDED_DIRS = {"__pycache__", ".pytest_cache"}
KEY_DOCS = [
    "installation.mdx",
    "integrate_hardware.mdx",
    "debug_processor_pipeline.mdx",
    "processors_robots_teleop.mdx",
    "bring_your_own_policies.mdx",
    "lerobot-dataset-v3.mdx",
    "async.mdx",
]


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root containing pyproject.toml and src/lerobot.")


def list_dirs(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(
        child.name
        for child in path.iterdir()
        if child.is_dir() and child.name not in EXCLUDED_DIRS and not child.name.startswith(".")
    )


def list_files(path: Path, suffixes: tuple[str, ...]) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(
        child.name
        for child in path.iterdir()
        if child.is_file() and child.suffix in suffixes and not child.name.startswith(".")
    )


def format_bullets(items: list[str], indent: int = 0) -> list[str]:
    prefix = " " * indent + "- "
    return [f"{prefix}{item}" for item in items] if items else [" " * indent + "- none"]


def read_project_meta(repo_root: Path) -> dict[str, object]:
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    if tomllib is not None:
        data = tomllib.loads(pyproject)
        project = data.get("project", {})
        return {
            "name": project.get("name", "unknown"),
            "version": project.get("version", "unknown"),
            "requires_python": project.get("requires-python", "unknown"),
            "scripts": project.get("scripts", {}),
        }

    project_match = re.search(r"(?ms)^\[project\]\n(.*?)(?:^\[|\Z)", pyproject)
    scripts_match = re.search(r"(?ms)^\[project\.scripts\]\n(.*?)(?:^\[|\Z)", pyproject)
    project_block = project_match.group(1) if project_match else ""
    scripts_block = scripts_match.group(1) if scripts_match else ""

    def extract_string(block: str, key: str) -> str:
        match = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]+)"', block, re.MULTILINE)
        return match.group(1) if match else "unknown"

    scripts: dict[str, str] = {}
    for match in re.finditer(r'^([A-Za-z0-9_-]+)\s*=\s*"([^"]+)"', scripts_block, re.MULTILINE):
        scripts[match.group(1)] = match.group(2)

    return {
        "name": extract_string(project_block, "name"),
        "version": extract_string(project_block, "version"),
        "requires_python": extract_string(project_block, "requires-python"),
        "scripts": scripts,
    }


def build_snapshot(repo_root: Path) -> str:
    project = read_project_meta(repo_root)
    src_root = repo_root / "src" / "lerobot"
    tests_root = repo_root / "tests"
    examples_root = repo_root / "examples"
    docs_root = repo_root / "docs" / "source"

    top_level = list_dirs(src_root)
    robots = list_dirs(src_root / "robots")
    teleoperators = list_dirs(src_root / "teleoperators")
    cameras = list_dirs(src_root / "cameras")
    policies = list_dirs(src_root / "policies")
    envs = [p.removesuffix(".py") for p in list_files(src_root / "envs", (".py",)) if p != "__init__.py"]
    tests = list_dirs(tests_root)
    examples = sorted(
        str(path.relative_to(examples_root))
        for path in examples_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    docs = [name for name in KEY_DOCS if (docs_root / name).is_file()]

    lines = [
        "# LeRobot Architecture Snapshot",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Repo root: {repo_root}",
        "",
        "## Project",
        f"- name: {project['name']}",
        f"- version: {project['version']}",
        f"- requires-python: {project['requires_python']}",
        "",
        "## Console scripts",
    ]
    script_names = sorted(project["scripts"].keys()) if isinstance(project["scripts"], dict) else []
    lines.extend(format_bullets(script_names))

    lines.extend(
        [
            "",
            "## Source map",
            "- top-level packages under src/lerobot:",
        ]
    )
    lines.extend(format_bullets(top_level, indent=2))
    lines.extend(
        [
            "- robots:",
        ]
    )
    lines.extend(format_bullets(robots, indent=2))
    lines.extend(
        [
            "- teleoperators:",
        ]
    )
    lines.extend(format_bullets(teleoperators, indent=2))
    lines.extend(
        [
            "- cameras:",
        ]
    )
    lines.extend(format_bullets(cameras, indent=2))
    lines.extend(
        [
            "- policies:",
        ]
    )
    lines.extend(format_bullets(policies, indent=2))
    lines.extend(
        [
            "- env modules:",
        ]
    )
    lines.extend(format_bullets(envs, indent=2))

    lines.extend(
        [
            "",
            "## Tests",
        ]
    )
    lines.extend(format_bullets(tests))

    lines.extend(
        [
            "",
            "## Examples",
        ]
    )
    lines.extend(format_bullets(examples))

    lines.extend(
        [
            "",
            "## Key docs",
        ]
    )
    lines.extend(format_bullets(docs))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the LeRobot repository root. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to references/architecture-snapshot.md next to this script.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(script_path)
    output = (
        args.output.resolve()
        if args.output
        else script_path.parent.parent / "references" / "architecture-snapshot.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_snapshot(repo_root), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
