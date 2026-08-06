#!/usr/bin/env python3
"""Convert mid360 IMU/body TUM poses to the mocap marker-center poses."""

from __future__ import annotations

import argparse
from pathlib import Path


# User-defined mocap origin:
# midpoint of the front left/right camera positions in the mid360 config.
# The mocap frame is assumed to be rotation-aligned with the IMU frame.
#
# camera0 Pcl: [0.017582, 0.023607, -0.037925]
# camera1 Pcl: [0.017541, 0.039816, -0.033913]
T_IMU_MOCAP = (0.0175615, 0.0317115, -0.035919)


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
    return input_path.with_name(f"{input_path.stem}_mocap.txt")


def parse_offset(value: str) -> tuple[float, float, float]:
    fields = value.replace(",", " ").split()
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("offset must contain exactly three values: x,y,z")
    try:
        return (float(fields[0]), float(fields[1]), float(fields[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("offset values must be numeric") from exc


def convert(input_path: Path, output_path: Path, t_imu_mocap: tuple[float, float, float]) -> tuple[int, int]:
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
            lever_w = rotate_by_quat_xyzw(q_w_imu, t_imu_mocap)
            p_mocap = (
                p_imu[0] + lever_w[0],
                p_imu[1] + lever_w[1],
                p_imu[2] + lever_w[2],
            )

            dst.write(
                f"{ts} {p_mocap[0]:.9f} {p_mocap[1]:.9f} {p_mocap[2]:.9f} "
                f"{q_w_imu[0]:.9f} {q_w_imu[1]:.9f} {q_w_imu[2]:.9f} {q_w_imu[3]:.9f}\n"
            )
            written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert mid360 IMU/body TUM trajectory to mocap-center TUM trajectory.")
    parser.add_argument("input", type=Path, help="Input TUM txt trajectory: timestamp x y z qx qy qz qw")
    parser.add_argument("-o", "--output", type=Path, help="Output TUM txt trajectory. Defaults to <input_stem>_mocap.txt")
    parser.add_argument(
        "--offset",
        type=parse_offset,
        default=T_IMU_MOCAP,
        help="IMU-frame translation from IMU origin to mocap origin, as 'x,y,z'.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file.")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path)

    if not input_path.is_file():
        raise SystemExit(f"input file does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = convert(input_path, output_path, args.offset)
    print(f"using T_IMU_MOCAP translation: [{args.offset[0]:.7f}, {args.offset[1]:.7f}, {args.offset[2]:.7f}]")
    print(f"wrote {written} rows to {output_path}")
    if skipped:
        print(f"skipped {skipped} malformed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
