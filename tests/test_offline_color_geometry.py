"""Offline geometry checks; no ROS process or C++ compilation is involved."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import offline_color_unify as offline


class CameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        offline.dependencies()

    def camera(self, **extra):
        values = dict(model="Pinhole", width=640, height=480, fx=450, fy=451, cx=320, cy=240)
        values.update(extra)
        return offline.Camera(values, (480, 640, 3))

    def test_projection_round_trip_and_tangent_chart(self):
        pixels = np.array([[320, 240], [180, 120], [450, 350], [530, 70]], dtype=float)
        cameras = [self.camera(d0=-0.06, d1=0.01, d2=0.001, d3=-0.001),
                   self.camera(model="EquidistantCamera", k1=-0.04, k2=0.004),
                   self.camera(model="MEI", xi=1.2, k1=-0.02, k2=0.001, p1=0.001)]
        for camera in cameras:
            np.testing.assert_allclose(camera.project(camera.unproject(pixels)), pixels, atol=1e-6, rtol=0)
            chart = offline.Chart(camera, pixels[0], 300.)
            np.testing.assert_allclose(chart.to_raw(chart.to_chart(pixels)), pixels, atol=1e-6, rtol=0)

    def test_wrong_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "resolution"):
            offline.Camera(dict(model="Pinhole", width=640, height=480), (240, 320, 3))

    def test_raw_jacobian_has_expected_scale(self):
        chart = offline.Chart(self.camera(), np.array([320., 240.]), 300.)
        jacobian = offline.raw_jacobian(chart, np.eye(3), np.zeros(2))
        np.testing.assert_allclose(jacobian, np.diag([450/300, 451/300]), atol=1e-8, rtol=0)

    def test_default_gray_is_exact_opencv_conversion(self):
        bgr = np.random.default_rng(5).integers(0, 256, (30, 40, 3), dtype=np.uint8)
        expected = offline.cv2.cvtColor(bgr, offline.cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(offline.fixed_gray(bgr, offline.DEFAULT_WEIGHTS), expected)

    def test_bidirectional_fit_and_independent_holdout(self):
        points = np.array([(x, y) for y in range(180, 281, 20) for x in range(260, 381, 20)], float)
        matches = np.column_stack((points, points + [3., -2.]))
        camera = self.camera()
        result = offline.local_geometry(matches, len(matches)//2, camera, camera, 300., 160.)
        source_chart, target_chart, forward = result[:3]
        predicted = target_chart.to_raw(offline.transform(forward, source_chart.to_chart(points)))
        np.testing.assert_allclose(predicted, points+[3., -2.], atol=0.2, rtol=0)
        center = np.argmin(np.linalg.norm(points-[320., 240.], axis=1))
        corrupted = np.column_stack((points, points))
        # At an optical-center chart, target y/x ordering is known exactly.
        # Move only held-out x coordinates; preserve their spatial ordering and
        # the chart seed, so RANSAC cannot silently remove these test points.
        heldout = np.arange(0, len(points), 3)
        heldout = heldout[heldout != center]
        corrupted[heldout, 2] += 10
        with self.assertRaises(ValueError):
            offline.local_geometry(corrupted, center, camera, camera, 300., 160.)

    def test_four_spread_matches_can_share_a_geometry_group(self):
        points = np.array([[270, 190], [370, 190], [370, 290], [270, 290]], float)
        matches = np.column_stack((points, points + [3., -2.]))
        camera = self.camera()
        groups, counts = offline.discover_groups(matches, camera, camera, 300., 160.)
        self.assertEqual(counts["geometry_groups"], 1)
        self.assertEqual(groups[0]["indices"], [0, 1, 2, 3])
        self.assertLess(groups[0]["loo_max_raw_px"], 1e-8)

    def test_grouping_separates_incompatible_motions(self):
        first = np.array([[235, 180], [280, 178], [286, 227], [232, 235], [260, 204]], float)
        second = first + [130., 40.]
        source = np.vstack((first, second))
        target = np.vstack((first + [3., -2.], second + [-9., 24.]))
        matches = np.column_stack((source, target))
        camera = self.camera()
        groups, _ = offline.discover_groups(matches, camera, camera, 300., 160.)
        supports = [set(group["indices"]) for group in groups]
        self.assertIn(set(range(5)), supports)
        self.assertIn(set(range(5, 10)), supports)
        self.assertNotIn(set(range(10)), supports)

    def test_group_leave_one_out_rejects_inconsistent_or_collinear_points(self):
        camera = self.camera()
        chart = offline.Chart(camera, [camera.cx, camera.cy], 300.)
        points = np.array([[270, 190], [370, 190], [370, 290], [270, 290]], float)
        target = points.copy()
        target[-1] += [15., 10.]
        matches = np.column_stack((points, target))
        result = offline.validate_group(range(4), chart.to_chart(points), chart.to_chart(target),
                                        matches, chart, chart)
        self.assertIsNone(result)
        line = np.array([[270, 190], [290, 210], [310, 230], [330, 250]], float)
        groups, _ = offline.discover_groups(np.column_stack((line, line)), camera, camera, 300., 160.)
        self.assertFalse(groups)

    def test_one_group_yields_disjoint_color_blocks_and_checks_prediction_spread(self):
        camera = self.camera()
        points = np.array([[250, 170], [390, 170], [390, 310], [250, 310]], float)
        matches = np.column_stack((points, points))
        image = np.full((480, 640, 3), [70, 100, 140], np.uint8)
        args = SimpleNamespace(virtual_focal=300., context=160., max_regions=6, max_candidates=300)
        regions, _, counts, groups = offline.collect_grouped_regions(
            image, image, camera, camera, matches, args)
        self.assertEqual(len(regions), 6)
        self.assertEqual(counts["geometry_groups"], 1)
        self.assertEqual(groups[0]["color_regions"], list(range(6)))
        for i, region in enumerate(regions):
            self.assertGreater(len(region["rgb"]), 0)
            for other in regions[:i]:
                for key in ("source_grid", "reference_grid"):
                    self.assertFalse(offline.box_overlap(region[key]["box"], other[key]["box"]))
        geometry_groups, _ = offline.discover_groups(matches, camera, camera, 300., 160.)
        group = geometry_groups[0]
        inaccurate = np.eye(3)
        inaccurate[0, 2] = 10.
        with self.assertRaisesRegex(ValueError, "sampling_prediction_spread"):
            offline.sample_region(image, image, group["geometry"], center=[0., 0.],
                                  uncertainty_models=[inaccurate])


if __name__ == "__main__":
    unittest.main()
