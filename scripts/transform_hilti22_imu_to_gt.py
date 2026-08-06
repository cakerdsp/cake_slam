#!/usr/bin/env python3
"""Transform a HILTI22 TUM trajectory from the IMU origin to the GT target.

The supplied HILTI22 evaluation code applies the fixed extrinsic on the right:

    T_world_gt = T_world_imu @ T_imu_gt

This script performs only that rigid-body reference-point conversion. It does
not synchronize timestamps, align trajectories, or correct scale; those steps
can be handled later by evo.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple


# Fixed transform copied from the supplied HILTI22 evaluation script.
# Quaternion convention throughout this file is (qx, qy, qz, qw).
T_IMU_GT = (0.059, -0.00855, 0.1964)
Q_IMU_GT = (0.0, 0.0, 0.0, 1.0)


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


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
    """Rotate a 3D vector by a normalized (qx, qy, qz, qw) quaternion."""
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    vx, vy, vz = vector

    # Equivalent to q * [v, 0] * conjugate(q), expanded without allocations.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def transform_pose(position: Vector3, orientation: Quaternion) -> Tuple[Vector3, Quaternion]:
    """Apply T_world_gt = T_world_imu @ T_imu_gt to one pose."""
    orientation = normalize_quaternion(orientation)
    lever_world = rotate_vector(orientation, T_IMU_GT)
    position_gt = tuple(position[i] + lever_world[i] for i in range(3))
    orientation_gt = multiply_quaternions(orientation, Q_IMU_GT)
    return position_gt, orientation_gt


def format_values(values: Iterable[float]) -> str:
    return " ".join(f"{value:.15g}" for value in values)


def transform_trajectory(input_path: Path, output_path: Path) -> int:
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
                numeric = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"line {line_number}: non-numeric pose value") from exc

            position_gt, orientation_gt = transform_pose(
                (numeric[0], numeric[1], numeric[2]),
                (numeric[3], numeric[4], numeric[5], numeric[6]),
            )
            output_file.write(
                f"{fields[0]} {format_values((*position_gt, *orientation_gt))}\n"
            )
            converted += 1

    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a HILTI22 TUM trajectory from the IMU origin to the "
            "ground-truth target using the official evaluation extrinsic."
        )
    )
    parser.add_argument("input", type=Path, help="input IMU trajectory in TUM format")
    parser.add_argument("output", type=Path, help="output trajectory in TUM format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = transform_trajectory(args.input, args.output)
    print(f"Converted {count} poses: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
