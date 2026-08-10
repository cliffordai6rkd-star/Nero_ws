from __future__ import annotations


# V120 motor_state velocity uses motor directions, while joint positions and
# URDF dynamics use Nero joint-coordinate directions.
NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN = (
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
)


def nero_v120_joint_velocity(raw_velocity: float, joint_index: int) -> float:
    try:
        sign = NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN[joint_index]
    except IndexError as exc:
        raise IndexError(f"Nero V120 joint index is out of range: {joint_index}") from exc
    return float(raw_velocity) * sign
