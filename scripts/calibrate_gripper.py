#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from nero_collection.config import load_config
from nero_collection.socketcan import configure_interface


def main() -> int:
    parser = argparse.ArgumentParser(description="标定 Nero AGX 夹爪零点")
    parser.add_argument("arm", choices=("leader", "follower"))
    args = parser.parse_args()

    config = load_config(ROOT / "configs/master_slave_can.yaml")
    pair = config.teleop.master_slave[0]
    endpoint = pair.leader if args.arm == "leader" else pair.follower
    print("请先停止数采脚本，并确认夹爪内没有物体。")
    configure_interface(endpoint.channel, endpoint.bitrate)

    try:
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
    except ImportError:
        from PyAgxArm import (  # type: ignore
            AgxArmFactory,
            ArmModel,
            NeroFW,
            create_agx_arm_config,
        )

    kwargs = dict(endpoint.config_kwargs)
    if endpoint.can_id is not None:
        kwargs["can_id"] = endpoint.can_id
    sdk_config = create_agx_arm_config(
        robot=ArmModel.NERO,
        comm="can",
        firmeware_version=getattr(NeroFW, endpoint.firmware),
        channel=endpoint.channel,
        interface=endpoint.interface,
        bitrate=endpoint.bitrate,
        **kwargs,
    )
    robot = AgxArmFactory.create_arm(sdk_config)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    print(f"连接 {args.arm}: {endpoint.name} ({endpoint.channel})")
    robot.connect()
    try:
        time.sleep(0.5)
        for _ in range(3):
            gripper.disable_gripper()
            time.sleep(0.2)
            status = gripper.get_gripper_status()
            if status is not None and not status.msg.foc_status.driver_enable_status:
                break
        else:
            raise RuntimeError("无法确认夹爪已失能，停止标定")

        if config.gripper.max_width_m in (0.07, 0.1):
            pendant = gripper.get_gripper_teaching_pendant_param(
                timeout=2.0,
                min_interval=0.0,
            )
            if pendant is None:
                print("警告：无法读取夹爪参数，跳过最大行程同步")
            elif pendant.msg.max_range_config != config.gripper.max_width_m:
                range_ok = gripper.set_gripper_teaching_pendant_param(
                    teaching_range_per=pendant.msg.teaching_range_per,
                    max_range_config=config.gripper.max_width_m,
                    teaching_friction=pendant.msg.teaching_friction,
                    timeout=3.0,
                )
                if not range_ok:
                    print("警告：最大行程设置未收到 ACK，继续执行零点标定")

        print("请用手轻轻将夹爪完全闭合到机械限位，不要继续用力挤压。")
        input("完全闭合后按 Enter 写入零点...")

        calibrated = False
        for attempt in range(2):
            if gripper.calibrate_gripper(timeout=5.0):
                calibrated = True
                break
            print(f"第 {attempt + 1} 次标定未收到 ACK")
            time.sleep(0.5)

        time.sleep(0.5)
        zero_status = gripper.get_gripper_status()
        feedback_is_zero = (
            zero_status is not None
            and zero_status.msg.mode == "width"
            and abs(zero_status.msg.value) <= 0.002
        )
        if not calibrated and not feedback_is_zero:
            value = None if zero_status is None else zero_status.msg.value
            raise RuntimeError(
                f"零点标定失败：未收到 ACK，反馈也未归零（value={value}）"
            )
        if calibrated:
            print("零点写入成功，已收到 ACK")
        else:
            print("未收到 ACK，但夹爪反馈已归零，按标定成功继续")

        gripper.move_gripper_m(value=0.0, force=config.gripper.force_n)
        time.sleep(0.5)
        gripper.move_gripper_m(
            value=config.gripper.max_width_m,
            force=config.gripper.force_n,
        )
        time.sleep(2.0)

        status = gripper.get_gripper_status()
        if status is not None:
            print(
                f"全开反馈={status.msg.value:.6f} m, "
                f"目标={config.gripper.max_width_m:.6f} m"
            )
        print("标定完成")
        return 0
    finally:
        try:
            gripper.disable_gripper()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
