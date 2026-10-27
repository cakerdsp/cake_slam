"""CSV viewer regression checks; no ROS, C++ compilation or dataset required."""

import csv
import importlib.util
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "view_memory_stats.py"
SPEC = importlib.util.spec_from_file_location("view_memory_stats", SCRIPT)
VIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEW)


class MemoryViewerTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures in the test workspace rather than a sandbox's system
        # TEMP directory; validate the exact cleanup target before registering it.
        fixture_root = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(prefix=".memory_stats_test_", dir=str(fixture_root))
        self.run_dir = Path(self.temp.name).resolve()
        self.assertEqual(self.run_dir.parent, fixture_root)
        self.addCleanup(self.temp.cleanup)

    def write_samples(self, samples, finalized=False):
        fields = ["sample_id", "wall_elapsed_sec", "is_final", "component", "accounted_estimate_bytes"]
        with (self.run_dir / "memory_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for sample_id, wall, values in samples:
                for name, amount in dict(values, __accounted_total__=sum(values.values())).items():
                    writer.writerow(dict(sample_id=sample_id, wall_elapsed_sec=wall, is_final=0,
                                         component=name, accounted_estimate_bytes=amount))
        with (self.run_dir / "memory_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["component", "is_final"])
            writer.writeheader()
            writer.writerow(dict(component="__accounted_total__", is_final=int(finalized)))

    def test_group_peak_is_peak_of_simultaneous_sum(self):
        self.write_samples([(0, 0, {"vio.a": 100, "vio.b": 0}),
                            (1, 2, {"vio.a": 0, "vio.b": 50})])
        row = list(VIEW.grouped_rows(self.run_dir, 1, ""))[0]
        self.assertEqual(row[:4], ("vio", 50, 100, 75))
        self.assertEqual(row[4], 100)

    def test_prefix_mean_includes_samples_before_first_appearance(self):
        self.write_samples([(0, 0, {}), (1, 2, {"vio.new": 30}), (2, 5, {"vio.new": 90})])
        row = list(VIEW.grouped_rows(self.run_dir, 0, "vio.new"))[0]
        self.assertEqual(row[1:4], (90, 90, 40))
        self.assertEqual(row[4], 18)

    def test_duplicate_final_is_read_from_summary(self):
        self.write_samples([(0, 0, {"lidar.map": 10}), (1, 1, {"lidar.map": 20})], finalized=True)
        row = list(VIEW.grouped_rows(self.run_dir, 1, ""))[0]
        self.assertEqual(row[3], 15)
        self.assertEqual(row[5], 20)

    def test_empty_prefix_result_and_one_sample_time_mean(self):
        self.write_samples([(0, 0, {"input.queue": 25})])
        self.assertEqual(list(VIEW.grouped_rows(self.run_dir, 1, "unknown")), [])
        row = list(VIEW.grouped_rows(self.run_dir, 0, ""))[0]
        self.assertIsNone(row[4])
        self.assertIsNone(row[5])

    def test_cli_displays_process_summary_and_preserves_missing_values(self):
        self.write_samples([(0, 0, {"vio.map": 1048576})], finalized=True)
        with (self.run_dir / "process_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", "unit", "statistic", "value"])
            writer.writeheader()
            writer.writerow(dict(metric="VmRSS", unit="bytes", statistic="current", value=1048576))
            writer.writerow(dict(metric="VmRSS", unit="bytes", statistic="time_weighted_mean", value=""))
            writer.writerow(dict(metric="collector_ms", unit="ms", statistic="current", value=2.5))
        output = io.StringIO()
        with mock.patch.object(sys, "argv", [str(SCRIPT), str(self.run_dir)]), contextlib.redirect_stdout(output):
            VIEW.main()
        self.assertIn("VmRSS (MiB)", output.getvalue())
        self.assertIn("collector_ms (ms)", output.getvalue())
        self.assertIn("2.500", output.getvalue())


if __name__ == "__main__":
    unittest.main()
