#!/usr/bin/env python3
"""View memory CSVs without third-party Python packages (also works on NUC)."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MIB = 1024.0 * 1024.0


def find_run(path):
    path = Path(path).expanduser()
    if (path / "memory_samples.csv").is_file():
        return path
    candidates = list(path.glob("*/memory_samples.csv"))
    if not candidates:
        raise ValueError("No memory_samples.csv found in {} or its run folders".format(path))
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def grouped_rows(run, level, prefix):
    samples = {}
    with (run / "memory_samples.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["component"]
            sample_id = int(row["sample_id"])
            item = samples.setdefault(sample_id, {
                "wall": float(row["wall_elapsed_sec"]),
                "final": row.get("is_final", "0") == "1", "values": defaultdict(float)})
            if name == "__accounted_total__" or (prefix and not name.startswith(prefix)):
                continue
            key = ".".join(name.split(".")[:level]) if level else name
            item["values"][key] += float(row["accounted_estimate_bytes"])
    ordered = [samples[key] for key in sorted(samples)]
    summary_file = run / "memory_summary.csv"
    if ordered and summary_file.is_file():
        with summary_file.open(newline="", encoding="utf-8") as stream:
            ordered[-1]["final"] = any(row.get("is_final") == "1" for row in csv.DictReader(stream))
    keys = sorted({key for item in ordered for key in item["values"]})
    duration = ordered[-1]["wall"] - ordered[0]["wall"] if ordered else 0.0
    for key in keys:
        values = [item["values"].get(key, 0.0) for item in ordered]
        integral = sum(values[i] * max(0.0, ordered[i + 1]["wall"] - ordered[i]["wall"])
                       for i in range(len(values) - 1))
        yield (key, values[-1], max(values), sum(values) / len(values),
               integral / duration if duration > 0.0 else None,
               values[-1] if ordered[-1]["final"] else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="One run directory, or the parent Log/memory directory")
    parser.add_argument("--group-level", type=int, default=1,
                        help="Group path depth; 1=major groups, 2=subgroups, 0=all detail rows")
    parser.add_argument("--prefix", default="", help="Only component names beginning with this prefix")
    args = parser.parse_args()
    if args.group_level < 0:
        parser.error("--group-level must be non-negative")
    try:
        run = find_run(args.directory)
        rows = list(grouped_rows(run, args.group_level, args.prefix))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Memory report: {}\n".format(error))
    print("Run: {}".format(run))
    print("Accounted capacity estimate (MiB); process RSS is reported separately.")
    print("{:58s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
        "Component", "Current", "SamplePeak", "Mean", "TimeMean", "Final"))
    for row in sorted(rows, key=lambda r: r[2], reverse=True):
        values = ["{:.3f}".format(v / MIB) if v is not None else "-" for v in row[1:]]
        print("{:58s} {}".format(row[0], " ".join("{:>11s}".format(v) for v in values)))
    if not rows:
        print("No matching component samples yet.")
    # Keep the kernel process observations visible without conflating them with
    # the accounted object capacities above. No allocator/RSS subtraction.
    process_file = run / "process_samples.csv"
    if process_file.is_file():
        with process_file.open(newline="", encoding="utf-8") as stream:
            process = list(csv.DictReader(stream))
        if process:
            rss = [float(r["VmRSS_bytes"]) for r in process if r.get("VmRSS_bytes")]
            hwm = [float(r["VmHWM_bytes"]) for r in process if r.get("VmHWM_bytes")]
            last = process[-1]
            current = float(last["VmRSS_bytes"]) / MIB if last.get("VmRSS_bytes") else None
            print("\nProcess: RSS={} MiB; sampled RSS peak={} MiB; kernel VmHWM={} MiB".format(
                "{:.3f}".format(current) if current is not None else "unavailable",
                "{:.3f}".format(max(rss) / MIB) if rss else "unavailable",
                "{:.3f}".format(max(hwm) / MIB) if hwm else "unavailable"))
    process_summary = run / "process_summary.csv"
    if process_summary.is_file():
        with process_summary.open(newline="", encoding="utf-8") as stream:
            metrics = defaultdict(dict)
            for row in csv.DictReader(stream):
                metrics[row["metric"]][row["statistic"]] = row
        if metrics:
            print("\n{:24s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
                "Process metric", "Current", "ObsPeak", "Mean", "TimeMean", "Final"))
            for name, stats in metrics.items():
                rendered = []
                for stat in ["current", "sampled_peak" if name != "VmHWM" else "kernel_high_water_max",
                             "arithmetic_mean", "time_weighted_mean", "final"]:
                    row = stats.get(stat, {})
                    value = row.get("value", "")
                    divisor = MIB if row.get("unit") == "bytes" else 1.0
                    rendered.append("{:.3f}".format(float(value) / divisor) if value else "-")
                unit = "ms" if name == "collector_ms" else "MiB"
                print("{:24s} {}".format(name + " (" + unit + ")",
                                         " ".join("{:>11s}".format(v) for v in rendered)))
    print("Full automatic summaries: memory_summary.csv and process_summary.csv")


if __name__ == "__main__":
    main()
