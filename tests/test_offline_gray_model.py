"""Known-answer tests matching the online fixed_gray_weights_test cases."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from offline_gray_model import DEFAULT_WEIGHTS, fit_weights


def samples(weights, gain=1.0):
    # The same LCG and explicit R,G,B order as the C++ known-answer fixtures.
    state, colors = 1234567, []
    for _ in range(12 * 24):
        color = []
        for _ in range(3):
            state = (state * 1664525 + 1013904223) & 0xffffffff
            color.append(30.0 + ((state >> 8) % 111))
        colors.append(color)
    rgb = np.array(colors)
    return rgb, gain * (rgb @ weights), np.repeat(np.arange(12), 24)


class FixedGrayTests(unittest.TestCase):
    def test_recovers_known_weights_and_separates_overall_gain(self):
        expected = np.array([0.16, 0.31, 0.53])
        for gain in (0.5, 1., 1.6):
            fit = fit_weights(*samples(expected, gain))
            self.assertTrue(fit.accepted, fit.reason)
            self.assertEqual(fit.reason, "validated_channel_weights")
            np.testing.assert_allclose(fit.weights, expected, atol=1e-9, rtol=0)
            self.assertLess(fit.validation_rmse, 1e-8)
            self.assertAlmostEqual(fit.beta.sum(), gain, places=10)

    def test_unchanged_default_remains_exact(self):
        fit = fit_weights(*samples(DEFAULT_WEIGHTS, 1.4))
        self.assertTrue(fit.accepted)
        self.assertEqual(fit.reason, "default_weights_supported")
        np.testing.assert_array_equal(fit.weights, DEFAULT_WEIGHTS)

    def test_exact_small_correction_is_not_suppressed(self):
        expected = DEFAULT_WEIGHTS + [0.002, -0.001, -0.001]
        fit = fit_weights(*samples(expected))
        self.assertEqual(fit.reason, "validated_channel_weights")
        np.testing.assert_allclose(fit.weights, expected, atol=1e-9, rtol=0)

    def test_negative_coefficients_are_rejected_not_clipped(self):
        fit = fit_weights(*samples(np.array([-0.2, 0.7, 0.5])))
        self.assertFalse(fit.accepted)
        self.assertEqual(fit.reason, "negative_or_invalid_weights")
        np.testing.assert_array_equal(fit.weights, DEFAULT_WEIGHTS)

    def test_whole_regions_are_held_out_and_never_split(self):
        rgb, y, ids = samples(np.array([0.16, 0.31, 0.53]))
        fit = fit_weights(rgb, y, ids)
        self.assertEqual(fit.validation_regions, [2, 5, 8, 11])
        self.assertFalse(set(fit.training_regions) & set(fit.validation_regions))
        bad = fit_weights(rgb, y, np.zeros(len(ids), int))
        self.assertEqual(bad.reason, "insufficient_independent_regions")

    def test_neutral_colors_do_not_establish_channel_rank(self):
        rgb, y, ids = samples(DEFAULT_WEIGHTS)
        rgb[:] = rgb[:, 0, None]
        fit = fit_weights(rgb, rgb[:, 0], ids)
        self.assertFalse(fit.accepted)
        self.assertEqual(fit.reason, "insufficient_color_rank_or_target_contrast")

    def test_training_outliers_and_incompatible_unseen_regions(self):
        expected = np.array([0.16, 0.31, 0.53])
        rgb, y, ids = samples(expected, 1.2)
        corrupt = (ids % 3 != 2) & (np.arange(len(ids)) % 7 == 0)
        y[corrupt] += 60
        fit = fit_weights(rgb, y, ids)
        self.assertTrue(fit.accepted, fit.reason)
        self.assertLess(np.linalg.norm(fit.weights-expected), 0.02)
        rgb, y, ids = samples(expected)
        y[ids % 3 == 2] = 255-y[ids % 3 == 2]
        fit = fit_weights(rgb, y, ids)
        self.assertFalse(fit.accepted)
        self.assertEqual(fit.reason, "held_out_mapping_mismatch")

    def test_nonfinite_and_out_of_range_inputs(self):
        rgb, y, ids = samples(DEFAULT_WEIGHTS)
        rgb[0, 0] = np.nan
        self.assertEqual(fit_weights(rgb, y, ids).reason, "non_finite_sample")
        rgb[0, 0] = 256
        self.assertEqual(fit_weights(rgb, y, ids).reason, "sample_outside_8bit_range")


if __name__ == "__main__":
    unittest.main()
