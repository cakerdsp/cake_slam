#!/usr/bin/env python3
"""Convert TUM poses from the IMU reference frame to the mocap rigid-body frame.

The input trajectory is interpreted as ``T_world_imu`` and the fixed extrinsic
is applied on the right:

    T_world_mocap = T_world_imu @ T_imu_mocap

The output remains in the input trajectory's world frame. World-frame
alignment and scale correction are intentionally left to evo. An optional
timestamp offset can be added for trajectory synchronization.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
from typing import Iterable, Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]

# Robust aggregate estimated from trajectory pairs 22, 3, and 4.
# Quaternion convention is (qx, qy, qz, qw).
# This is T_imu_mocap, the inverse of the estimated T_mocap_imu.
T_IMU_MOCAP: Vector3 = (-0.00288868, -0.02437370, 0.08863227)
Q_IMU_MOCAP: Quaternion = (0.00586156, 0.00164464, 0.70093443, 0.71319974)


def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    qx, qy, qz, qw = quaternion
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError("quaternion has zero or non-finite norm")
    return qx / norm, qy / norm, qz / norm, qw / norm


def multiply_quaternions(lhs: Quaternion, rhs: Quaternion) -> Quaternion:
    """Return lhs * rhs using the (qx, qy, qz, qw) convention."""
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return normalize_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector by a normalized (qx, qy, qz, qw) quaternion."""
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def transform_pose(position: Vector3, orientation: Quaternion) -> Tuple[Vector3, Quaternion]:
    """Apply T_world_mocap = T_world_imu @ T_imu_mocap to one pose."""
    orientation = normalize_quaternion(orientation)
    lever_world = rotate_vector(orientation, T_IMU_MOCAP)
    position_mocap = tuple(position[index] + lever_world[index] for index in range(3))
    orientation_mocap = multiply_quaternions(orientation, Q_IMU_MOCAP)
    return position_mocap, orientation_mocap


def format_values(values: Iterable[float]) -> str:
    return " ".join(f"{value:.15g}" for value in values)


def transform_trajectory(
    input_path: Path,
    output_path: Path,
    timestamp_offset: Decimal = Decimal("0"),
) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                output_file.write(line if line.endswith("\n") else line + "\n")
                continue

            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(
                    f"line {line_number}: expected 8 TUM fields "
                    "(timestamp tx ty tz qx qy qz qw)"
                )

            try:
                timestamp = Decimal(fields[0]) + timestamp_offset
                numeric = [float(value) for value in fields[1:]]
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"line {line_number}: non-numeric TUM value") from exc

            position_mocap, orientation_mocap = transform_pose(
                (numeric[0], numeric[1], numeric[2]),
                (numeric[3], numeric[4], numeric[5], numeric[6]),
            )
            output_file.write(
                f"{format(timestamp, 'f')} "
                f"{format_values((*position_mocap, *orientation_mocap))}\n"
            )
            converted += 1

    return converted


def decimal_value(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("time offset must be numeric") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a TUM trajectory from the IMU reference frame to the "
            "mocap rigid-body reference frame using the calibrated extrinsic."
        )
    )
    parser.add_argument("input", type=Path, help="input T_world_imu trajectory in TUM format")
    parser.add_argument("output", type=Path, help="output T_world_mocap trajectory in TUM format")
    parser.add_argument(
        "--time-offset",
        type=decimal_value,
        default=Decimal("0"),
        metavar="SECONDS",
        help=(
            "value added to every input timestamp; use approximately 0.450 "
            "for sequence 22, 0.475 for sequence 3, or 0.300 for sequence 4"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = transform_trajectory(args.input, args.output, args.time_offset)
    print(f"Converted {count} poses: {args.input} -> {args.output}")
    print(f"Applied timestamp offset: {args.time_offset} s")


if __name__ == "__main__":
    main()
