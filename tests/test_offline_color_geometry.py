"""Offline geometry checks; no ROS process or C++ compilation is involved."""

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
