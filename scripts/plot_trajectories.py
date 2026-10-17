#!/usr/bin/env python3
"""Interactive TUM trajectory comparison with a separate time offset per file.

Requires: numpy, matplotlib

Example:
    python3 plot_trajectories.py --ref gt.txt \
        --traj ours.txt 0.05 Ours --traj other.txt -0.12 Other \
        --align --plot-mode xy

TUM columns: timestamp tx ty tz qx qy qz qw. Positions are in meters and
timestamps/offsets in seconds. The correction is t_corrected = t_file + offset.
Files are read only. No extrinsic conversion, time-offset fitting, scale
correction, trajectory smoothing, or point rejection is performed.

With --align, estimated positions are linearly interpolated at common actual
reference timestamps, excluding large gaps in the estimates. Each trajectory
is fitted to those reference positions by a rigid SE(3) transform. All original
position samples are then plotted, transformed but not resampled. Reference
gaps are never interpolated. This association differs from evo's default
nearest-timestamp matching; the printed fit RMSE is a plotting diagnostic.

Use the Matplotlib toolbar to zoom, pan, reset, or save the current view.
Click a legend entry to hide/show its trajectory. Close the window to exit.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Track:
    path: Path
    label: str
    times: np.ndarray
    xyz: np.ndarray
    offset: float = 0.0
    gap_limit: float = 0.0


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def parser():
    p = argparse.ArgumentParser(
        description="Plot TUM trajectories with independent, user-supplied time offsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 plot_trajectories.py --ref gt.txt --traj ours.txt 0.28 Ours --align
  python3 plot_trajectories.py --ref gt.txt --traj a.txt -0.1 Method-A \\
      --traj b.txt 0 Method-B --align --plot-mode xy --output comparison

TIME CONVENTION: corrected timestamp = file timestamp + offset (seconds).
The reference timestamps and input files are unchanged.
GUI: toolbar zoom/pan/save; click legend entries to toggle curves.
Dependencies: python3 -m pip install numpy matplotlib
""",
    )
    p.add_argument("--ref", required=True, type=Path, help="ground-truth TUM file")
    p.add_argument("--traj", required=True, action="append", nargs=3,
                   metavar=("FILE", "OFFSET_SECONDS", "LABEL"),
                   help="repeat once per estimate; negative offsets are supported")
    p.add_argument("--align", action="store_true", help="rigid SE(3) fit; no scale correction")
    p.add_argument("--plot-mode", choices=("xy", "xz", "yz", "xyz"), default="xy")
    p.add_argument("--title", default="", help="optional figure title")
    p.add_argument("--output", type=Path, help="optional output prefix; writes PREFIX.pdf and PREFIX.png")
    p.add_argument("--no-show", action="store_true", help="save/check without opening a GUI")
    p.add_argument("--max-gap", type=positive_float, default=None,
                   help="gap limit in seconds; default: 5 x each file's median sampling interval")
    return p


def load_track(path, label, offset, gap_limit):
    data = np.loadtxt(path, comments="#", encoding="utf-8-sig", ndmin=2)
    if data.shape[1] != 8 or len(data) < 2:
        raise ValueError(f"{path}: expected >=2 rows and 8 TUM columns, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    # Retain absolute timestamps until a shared epoch is subtracted in run().
    t = data[:, 0].copy()
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing (check duplicates/order)")
    threshold = float(gap_limit) if gap_limit is not None else float(5 * np.median(dt))
    return Track(path, label, t, data[:, 1:4].copy(), offset, threshold)


def interpolation_mask(track, query):
    """Allow observed endpoints; disallow extrapolation or interpolation over gaps."""
    t = track.times
    right = np.searchsorted(t, query, side="left")
    clipped = np.clip(right, 0, len(t) - 1)
    exact = np.abs(t[clipped] - query) <= 1e-9
    interior = (right > 0) & (right < len(t))
    i = np.clip(right, 1, len(t) - 1)
    short_bracket = (t[i] - t[i - 1]) <= track.gap_limit
    return (query >= t[0]) & (query <= t[-1]) & (exact | (interior & short_bracket))


def fit_se3(source, target):
    if len(source) < 3:
        raise ValueError("SE(3) alignment needs at least 3 common reference timestamps")
    xm, ym = source.mean(axis=0), target.mean(axis=0)
    xc, yc = source - xm, target - ym
    if np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError("SE(3) alignment is ambiguous: common positions are stationary or collinear")
    u, _, vt = np.linalg.svd(xc.T @ yc)
    rotation = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    translation = ym - rotation @ xm
    error = source @ rotation.T + translation - target
    rmse = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    return rotation, translation, rmse


def align_tracks(reference, tracks):
    valid = np.ones(len(reference.times), dtype=bool)
    for track in tracks:
        mask = interpolation_mask(track, reference.times)
        print(f"{track.label}: {int(mask.sum())} usable reference timestamps")
        if mask.sum() < 3:
            raise ValueError(
                f"{track.label}: insufficient overlap after offset {track.offset:+.6f} s. "
                "Check timestamp units, time ranges, offset sign, and --max-gap."
            )
        valid &= mask
    if valid.sum() < 3:
        raise ValueError("Fewer than 3 reference timestamps are shared by ALL estimates")
    query, target = reference.times[valid], reference.xyz[valid]
    print(f"SE(3): {len(query)} common reference timestamps; no scale correction")
    for track in tracks:
        source = np.column_stack([
            np.interp(query, track.times, track.xyz[:, j]) for j in range(3)
        ])
        rotation, translation, rmse = fit_se3(source, target)
        track.xyz = track.xyz @ rotation.T + translation
        print(f"  {track.label}: position fit RMSE = {rmse:.6f} m")


def line_coordinates(track):
    """Insert NaNs at time gaps without discarding any measured positions."""
    breaks = np.flatnonzero(np.diff(track.times) > track.gap_limit) + 1
    xyz = np.insert(track.xyz, breaks, np.nan, axis=0)
    return xyz, len(breaks)


def style_for(label, index):
    palette = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    key = label.lower().replace("-", "").replace("_", "").replace(" ", "")
    known = {
        "ours1cam": ("#E69F00", "--"), "ours2cam": ("#D55E00", "-"),
        "omnilivo1cam": ("#56B4E9", "--"), "omnilivo2cam": ("#0072B2", "-"),
        "omni1cam": ("#56B4E9", "--"), "omni2cam": ("#0072B2", "-"),
        "ourslio": ("#009E73", ":"),
    }
    return known.get(key, (palette[index % len(palette)], ["-", "--", "-.", ":"][(index // len(palette)) % 4]))


def plot(reference, tracks, args):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "toolbar": "toolbar2",
    })
    is_3d = args.plot_mode == "xyz"
    fig = plt.figure(figsize=(7.0, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    dimensions = ["xyz".index(c) for c in args.plot_mode]
    lines = []
    for index, track in enumerate([reference] + tracks):
        xyz, gaps = line_coordinates(track)
        color, style = ("#303030", "--") if index == 0 else style_for(track.label, index - 1)
        line, = ax.plot(*[xyz[:, j] for j in dimensions], color=color, linestyle=style,
                        linewidth=1.15, label=track.label, zorder=5 if index == 0 else 3)
        lines.append(line)
        if gaps:
            print(f"{track.label}: {gaps} line break(s), gap threshold {track.gap_limit:.6f} s")
    ax.set_xlabel(f"{args.plot_mode[0]} (m)")
    ax.set_ylabel(f"{args.plot_mode[1]} (m)")
    if is_3d:
        ax.set_zlabel("z (m)")
        xyz = np.vstack([t.xyz for t in [reference] + tracks])
        spans = np.ptp(xyz, axis=0)
        # Set matching box and data spans for equal units on the three axes.
        spans = np.maximum(spans, max(float(spans.max()), 1e-6) * .05)
        centers = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        for setter, center, span in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centers, spans):
            setter(center - .54 * span, center + .54 * span)
        ax.set_box_aspect(spans)
    else:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#E6E6E6", linewidth=.5)
        ax.set_axisbelow(True)
        ax.margins(.06)
    if args.title:
        ax.set_title(args.title)
    legend = ax.legend(handles=lines, loc="upper center", bbox_to_anchor=(.5, 1.15),
                       ncol=min(3, len(lines)), frameon=False, handlelength=2.6)
    # Clicking a legend sample or label hides/shows the corresponding curve.
    mapping = {}
    for sample, text, line in zip(legend.get_lines(), legend.get_texts(), lines):
        sample.set_picker(5)
        text.set_picker(True)
        mapping[sample] = (line, sample, text)
        mapping[text] = (line, sample, text)

    def on_pick(event):
        if event.artist in mapping:
            line, sample, text = mapping[event.artist]
            visible = not line.get_visible()
            line.set_visible(visible)
            sample.set_alpha(1 if visible else .25)
            text.set_alpha(1 if visible else .25)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    if args.output:
        prefix = args.output
        if prefix.suffix.lower() in (".pdf", ".png"):
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            path = Path(str(prefix) + "." + extension)
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved: {path.resolve()}")
    if not args.no_show:
        print("Toolbar: zoom / pan / reset / save. Click legend labels to toggle curves.")
        plt.show()
    plt.close(fig)


def run(args):
    reference = load_track(args.ref, "Ground truth", 0., args.max_gap)
    tracks, labels = [], {reference.label}
    for file, offset_text, label in args.traj:
        offset = float(offset_text)
        if not np.isfinite(offset):
            raise ValueError(f"{label}: time offset must be finite")
        if label in labels:
            raise ValueError(f"Duplicate legend label: {label}")
        labels.add(label)
        tracks.append(load_track(Path(file), label, offset, args.max_gap))
    # Subtract the same epoch for every trajectory, not each file's own start.
    # This retains their relative timing while avoiding arithmetic on 1e9-scale times.
    epoch = reference.times[0]
    for track in [reference] + tracks:
        track.times = track.times - epoch + track.offset
        print(f"{track.label}: {len(track.times)} poses, offset {track.offset:+.6f} s, "
              f"relative range [{track.times[0]:.6f}, {track.times[-1]:.6f}] s")
    if args.align:
        align_tracks(reference, tracks)
    plot(reference, tracks, args)


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (OSError, ValueError, ImportError) as exc:
        p.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
