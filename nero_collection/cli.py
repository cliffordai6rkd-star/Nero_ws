from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from nero_collection.cameras import CameraManager, CameraVisualizer
from nero_collection.fixed_rate import FixedRateTicker
from nero_collection.config import CollectionConfig, load_config
from nero_collection.episode_output import episode_path, next_episode_index
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.keyboard import TerminalKeys
from nero_collection.realtime_plot import RealtimeJointPlotter
from nero_collection.socketcan import interface_for_usb_serial
from nero_collection.teleop.master_slave import MasterSlaveTeleop

log = logging.getLogger(__name__)

_ACTIVE_TELEOP_EVENT_POLL_S = 0.001


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.backend:
        config = _with_backend(config, args.backend)
    if config.teleop.backend == "pyagxarm":
        config = _resolve_can_interfaces(config)
    if config.teleop.backend == "pyagxarm" and not args.skip_can_setup:
        _setup_can_interfaces(config)
    return run_collection(
        config=config,
        episode_limit=args.episode_limit,
        dry_run_duration_s=args.dry_run_duration,
        auto_save=args.auto_save or args.dry_run_duration is not None,
    )


def run_collection(
    config: CollectionConfig,
    episode_limit: int | None = None,
    dry_run_duration_s: float | None = None,
    auto_save: bool = False,
) -> int:
    teleop = MasterSlaveTeleop(config)
    camera_visualizer = CameraVisualizer.from_config(config.cameras)
    cameras = CameraManager.from_config(
        config.cameras,
        visualizer=camera_visualizer,
    )
    realtime_plot = RealtimeJointPlotter(
        config.realtime_plot,
        config.robot_states,
        config.dynamics_processing,
    )
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_index = next_episode_index(output_dir, config.output.prefix)
    completed = 0
    teleop_started = False

    try:
        teleop.start()
        teleop_started = True
        cameras.start()
        realtime_plot.start()
        with TerminalKeys() as keys:
            if not keys.is_tty and dry_run_duration_s is None:
                raise RuntimeError("stdin is not a TTY; use --dry-run-duration for non-interactive runs")

            while episode_limit is None or completed < episode_limit:
                buffer = EpisodeBuffer(
                    config=config,
                    arm_names=teleop.arm_names,
                    online_tau_ext=getattr(teleop, "online_tau_ext", None),
                )
                if buffer.warm_up_online_inference():
                    log.info("online tau_ext inference warmed up before bilateral control")
                _wait_for_record_start(
                    teleop,
                    keys,
                    config,
                    dry_run_duration_s,
                    cameras=cameras,
                )
                reset_online_inference = getattr(
                    buffer,
                    "reset_online_inference",
                    None,
                )
                if callable(reset_online_inference) and reset_online_inference():
                    log.info("online tau inference state reset for recording episode")
                realtime_plot.clear_history()
                teleop.enter_teleop()
                log.info("recording episode %04d; press SPACE to stop", episode_index)
                print("Recording. Press SPACE to stop.", flush=True)
                _record_episode(
                    buffer,
                    teleop,
                    cameras,
                    keys,
                    config,
                    dry_run_duration_s,
                    realtime_plot,
                )
                teleop.enter_hold()
                log.info("recorded %d teleop samples", buffer.sample_count)

                save = _wait_for_save_choice(keys, auto_save)
                if save:
                    path = episode_path(output_dir, config.output.prefix, episode_index)
                    buffer.save(path)
                    log.info("saved episode to %s", path)
                    print(f"Saved: {path}", flush=True)
                    episode_index += 1
                    completed += 1
                else:
                    log.info("discarded episode %04d", episode_index)
                    print("Discarded this episode.", flush=True)
                    completed += 1
                if config.teleop.command.reset_after_episode:
                    print("Resetting and checking both arms...", flush=True)
                    teleop.reset_to_rest()
    except KeyboardInterrupt:
        print("\nCtrl-C received; shutting down.", flush=True)
        return 130
    except Exception as exc:
        log.critical(
            "collection safety fault; stopping cameras, resetting arms, and exiting: %s",
            exc,
            exc_info=True,
        )
        print(
            f"\nSAFETY FAULT: {exc}\nStopping cameras, resetting arms, and exiting.",
            flush=True,
        )
        realtime_plot.close()
        cameras.stop()
        if teleop_started:
            try:
                teleop.emergency_reset_to_rest()
            except Exception:
                log.exception("emergency arm reset failed; continuing with disconnect and shutdown")
        return 1
    finally:
        realtime_plot.close()
        cameras.stop()
        teleop.shutdown()
    return 0


def _wait_for_record_start(
    teleop: MasterSlaveTeleop,
    keys: TerminalKeys,
    config: CollectionConfig,
    dry_run_duration_s: float | None,
    cameras: CameraManager | None = None,
) -> None:
    if dry_run_duration_s is not None:
        log.info("dry-run: auto-start recording")
        return
    print(
        "Press r to enter teleop and record, t to teleoperate without recording, or q to quit.",
        flush=True,
    )
    idle_period = 1.0 / max(config.teleop.command.idle_rate_hz, 1.0)
    unrecorded_teleop_active = False
    while True:
        start = time.monotonic()
        key = keys.read_key(0.0)
        if key in {"r", "R"}:
            return
        if key in {"t", "T"}:
            teleop.enter_unrecorded_teleop()
            unrecorded_teleop_active = True
            print("Unrecorded teleoperation active. Press r to start recording.", flush=True)
            continue
        if key in {"q", "Q", "\x03"}:
            raise KeyboardInterrupt
        teleop.idle_step()
        if cameras is not None:
            cameras.poll()
        elapsed = time.monotonic() - start
        poll_period = (
            _ACTIVE_TELEOP_EVENT_POLL_S
            if unrecorded_teleop_active
            else idle_period
        )
        if elapsed < poll_period:
            time.sleep(poll_period - elapsed)


def _record_episode(
    buffer: EpisodeBuffer,
    teleop: MasterSlaveTeleop,
    cameras: CameraManager,
    keys: TerminalKeys,
    config: CollectionConfig,
    dry_run_duration_s: float | None,
    realtime_plot: RealtimeJointPlotter,
) -> None:
    start_t = time.monotonic()
    sample_rate_hz = config.teleop.command.sample_rate_hz
    ticker = FixedRateTicker(
        sample_rate_hz,
        config.teleop.command.control_watchdog_timeout_s,
    )
    discard_initial_s = config.output.discard_initial_s
    discard_complete_logged = discard_initial_s <= 0.0
    if discard_initial_s > 0.0:
        log.info(
            "discarding initial episode data duration=%.3fs; control and filters remain active",
            discard_initial_s,
        )
    while True:
        ticker.wait("master-slave recording")
        loop_t = time.monotonic()
        key = keys.read_key(0.0)
        if key == " ":
            return
        if key in {"q", "Q", "\x03"}:
            raise KeyboardInterrupt
        if dry_run_duration_s is not None and loop_t - start_t >= dry_run_duration_s:
            return

        timestamp_us, values = teleop.teleop_step()
        store = loop_t - start_t >= discard_initial_s
        if store and not discard_complete_logged:
            log.info("initial episode discard complete; saving and visualization started")
            discard_complete_logged = True
        accepted = buffer.append_teleop(timestamp_us, values, store=store)
        if accepted is not None:
            realtime_plot.append(accepted.timestamp_us, accepted.values)
        for frame in cameras.poll():
            if store:
                buffer.append_camera(frame.camera_name, frame.timestamp_us, frame.frame)


def _wait_for_save_choice(keys: TerminalKeys, auto_save: bool) -> bool:
    if auto_save:
        log.info("auto-save enabled")
        return True
    print("Press y to save the data or n to discard it.", flush=True)
    while True:
        key = keys.read_key(0.1)
        if key in {"y", "Y"}:
            return True
        if key in {"n", "N"}:
            return False
        if key in {"q", "Q", "\x03"}:
            raise KeyboardInterrupt


def _with_backend(config: CollectionConfig, backend: str) -> CollectionConfig:
    return replace(config, teleop=replace(config.teleop, backend=backend))


def _resolve_can_interfaces(config: CollectionConfig) -> CollectionConfig:
    """Bind arm endpoints to current canX names through USB adapter serials."""

    resolved_pairs = []
    used_channels: dict[str, str] = {}
    for pair in config.teleop.master_slave:
        endpoints = []
        for logical_role, endpoint in (
            ("leader", pair.leader),
            ("follower", pair.follower),
        ):
            if endpoint.interface != "socketcan" or endpoint.usb_serial is None:
                resolved = endpoint
            else:
                channel = interface_for_usb_serial(endpoint.usb_serial)
                resolved = replace(endpoint, channel=channel)
                log.info(
                    "bound USB-CAN adapter pair=%s role=%s arm=%s serial=%s channel=%s",
                    pair.name,
                    logical_role,
                    endpoint.name,
                    endpoint.usb_serial,
                    channel,
                )
            owner = used_channels.get(resolved.channel)
            identity = f"{pair.name}:{logical_role}:{resolved.name}"
            if owner is not None:
                raise RuntimeError(
                    f"CAN channel {resolved.channel!r} is assigned to both {owner} and {identity}"
                )
            used_channels[resolved.channel] = identity
            endpoints.append(resolved)
        resolved_pairs.append(replace(pair, leader=endpoints[0], follower=endpoints[1]))
    return replace(
        config,
        teleop=replace(config.teleop, master_slave=tuple(resolved_pairs)),
    )


def _setup_can_interfaces(config: CollectionConfig) -> None:
    channel_bitrates: dict[str, int] = {}
    for pair in config.teleop.master_slave:
        for endpoint in (pair.leader, pair.follower):
            if endpoint.interface != "socketcan":
                continue
            previous_bitrate = channel_bitrates.get(endpoint.channel)
            if previous_bitrate is not None and previous_bitrate != endpoint.bitrate:
                raise RuntimeError(
                    f"Conflicting bitrates configured for {endpoint.channel}: "
                    f"{previous_bitrate} and {endpoint.bitrate}"
                )
            channel_bitrates[endpoint.channel] = endpoint.bitrate
    if not channel_bitrates:
        log.info("no SocketCAN interfaces configured; skipping automatic CAN setup")
        return

    repository_root = Path(__file__).resolve().parents[1]
    setup_script = repository_root / "scripts" / "setup_can.py"
    if not setup_script.is_file():
        raise RuntimeError(f"CAN setup script not found: {setup_script}")

    bitrate_groups: dict[int, list[str]] = {}
    for channel, bitrate in channel_bitrates.items():
        bitrate_groups.setdefault(bitrate, []).append(channel)
    for bitrate, channels in bitrate_groups.items():
        channels.sort()
        log.info(
            "configuring SocketCAN interfaces before collection channels=%s bitrate=%d",
            ",".join(channels),
            bitrate,
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(setup_script),
                    "--bitrate",
                    str(bitrate),
                    *channels,
                ],
                cwd=repository_root,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Automatic CAN setup failed for {channels} at bitrate {bitrate}"
            ) from exc


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nero master-slave teleoperation data collection.")
    parser.add_argument(
        "--config",
        default="configs/master_slave_can.yaml",
        help="Path to collection YAML config.",
    )
    parser.add_argument(
        "--backend",
        choices=("pyagxarm", "mock"),
        help="Override teleop.backend from the config.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        help="Stop after this many accepted or discarded episodes.",
    )
    parser.add_argument(
        "--dry-run-duration",
        type=float,
        help="Non-interactive run duration in seconds; automatically starts and saves.",
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Save each stopped episode without asking y/n.",
    )
    parser.add_argument(
        "--skip-can-setup",
        action="store_true",
        help="Skip the automatic scripts/setup_can.py step for preconfigured interfaces.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
