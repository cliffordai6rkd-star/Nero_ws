from __future__ import annotations

import subprocess


def configure_interface(interface: str, bitrate: int) -> None:
    bitrate = int(bitrate)
    if not interface:
        raise ValueError("CAN interface name must not be empty")
    if bitrate <= 0:
        raise ValueError("CAN bitrate must be positive")
    subprocess.run(
        ["sudo", "ip", "link", "set", interface, "down"],
        check=False,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["sudo", "ip", "link", "set", interface, "type", "can", "bitrate", str(bitrate)],
        check=True,
    )
    subprocess.run(
        ["sudo", "ip", "link", "set", interface, "up"],
        check=True,
    )


def interface_exists(interface: str) -> bool:
    result = subprocess.run(
        ["ip", "link", "show", interface],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def link_details(interface: str, maximum_lines: int = 10) -> str:
    result = subprocess.run(
        ["ip", "-details", "link", "show", interface],
        check=True,
        capture_output=True,
        text=True,
    )
    return "\n".join(result.stdout.splitlines()[:maximum_lines])


def capture_frames(interface: str, duration_s: float) -> list[str]:
    process = subprocess.Popen(
        ["candump", interface],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=float(duration_s))
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, _ = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
    return stdout.splitlines()
