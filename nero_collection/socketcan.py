from __future__ import annotations

from pathlib import Path
import subprocess


SYS_CLASS_NET = Path("/sys/class/net")


def interface_usb_serial(interface: str, sys_class_net: Path = SYS_CLASS_NET) -> str | None:
    """Return the USB parent serial for a SocketCAN network interface."""

    device = sys_class_net / interface / "device"
    try:
        current = device.resolve(strict=True)
    except FileNotFoundError:
        return None
    for parent in (current, *current.parents):
        serial_path = parent / "serial"
        try:
            serial = serial_path.read_text(encoding="ascii").strip()
        except (FileNotFoundError, OSError, UnicodeError):
            continue
        if serial:
            return serial
    return None


def interface_for_usb_serial(
    usb_serial: str,
    sys_class_net: Path = SYS_CLASS_NET,
) -> str:
    """Resolve one stable USB-CAN serial to its current kernel interface name."""

    expected = str(usb_serial).strip()
    if not expected:
        raise ValueError("USB-CAN serial must not be empty")
    matches = sorted(
        path.name
        for path in sys_class_net.glob("can*")
        if interface_usb_serial(path.name, sys_class_net) == expected
    )
    if not matches:
        available = {
            path.name: interface_usb_serial(path.name, sys_class_net)
            for path in sorted(sys_class_net.glob("can*"))
        }
        raise RuntimeError(
            f"USB-CAN adapter serial {expected!r} is not present; available={available}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"USB-CAN adapter serial {expected!r} matched multiple interfaces: {matches}"
        )
    return matches[0]


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
