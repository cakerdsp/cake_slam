#!/usr/bin/env python3
"""Convert M2DGR FAST-LIVO IMU/body TUM poses to GNSS antenna TUM poses."""

from __future__ import annotations

import argparse
from pathlib import Path


# M2DGR calibration:
# T_IMU_GNSS = T_IMU_LiDAR * T_LiDAR_GNSS
# T_IMU_LiDAR = [0.27255, -0.00053, 0.17954]
# T_LiDAR_GNSS = [-0.09825, 0.00582, 0.72673]
T_IMU_GNSS = (0.17430, 0.00529, 0.90627)


def rotate_by_quat_xyzw(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_gnss{input_path.suffix}")


def convert(input_path: Path, output_path: Path) -> tuple[int, int]:
    written = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 8:
                skipped += 1
                continue

            ts, px, py, pz, qx, qy, qz, qw = fields
            p_imu = (float(px), float(py), float(pz))
            q_w_imu = (float(qx), float(qy), float(qz), float(qw))
            lever_w = rotate_by_quat_xyzw(q_w_imu, T_IMU_GNSS)
            p_gnss = (
                p_imu[0] + lever_w[0],
                p_imu[1] + lever_w[1],
                p_imu[2] + lever_w[2],
            )

            dst.write(
                f"{ts} {p_gnss[0]:.9f} {p_gnss[1]:.9f} {p_gnss[2]:.9f} "
                f"{q_w_imu[0]:.9f} {q_w_imu[1]:.9f} {q_w_imu[2]:.9f} {q_w_imu[3]:.9f}\n"
            )
            written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert M2DGR IMU/body TUM trajectory to GNSS TUM trajectory.")
    parser.add_argument("input", type=Path, help="Input TUM trajectory, e.g. Log/result/m2dgr.txt")
    parser.add_argument("-o", "--output", type=Path, help="Output TUM trajectory. Defaults to <input_stem>_gnss.txt")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path)

    if not input_path.is_file():
        raise SystemExit(f"input file does not exist: {input_path}")
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = convert(input_path, output_path)
    print(f"wrote {written} rows to {output_path}")
    if skipped:
        print(f"skipped {skipped} malformed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
