"""Offline NumPy translation of include/fixed_gray_weights.h (RGB order).

Keep the online estimator unchanged. This module copies its region partition,
region-balanced Huber IRLS, rank tests, scale separation and acceptance gates.
It is a numerical translation, not a claim of bit-for-bit Eigen equivalence.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


DEFAULT_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


@dataclass
class WeightFit:
    accepted: bool = False
    reason: str = ""
    weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    candidate_weights: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    baseline_scale: Optional[float] = None
    validation_rmse: Optional[float] = None
    candidate_validation_rmse: Optional[float] = None
    baseline_rmse: Optional[float] = None
    training_regions: list = field(default_factory=list)
    validation_regions: list = field(default_factory=list)

    def as_dict(self):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(self).items()}


def weighted_median(values, weights):
    # std::pair sorting in the online implementation breaks value ties by weight.
    order = np.lexsort((weights, values))
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, 0.5 * weights.sum(), side="left")
    return float(values[order[min(index, len(order) - 1)]])


def solve_weighted(design, target, weights):
    if (design.ndim != 2 or len(design) < design.shape[1] or not len(design)
            or len(design) != len(target) or len(target) != len(weights)
            or not np.isfinite(design).all() or not np.isfinite(target).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        return None
    root = np.sqrt(weights)
    a, b = design * root[:, None], target * root
    try:
        u, singular, vt = np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if (not np.isfinite(singular).all() or not len(singular)
            or singular[0] <= 1e-10 or singular[-1] < 0.01 * singular[0]):
        return None
    result = vt.T @ ((u.T @ b) / singular)
    return result if np.isfinite(result).all() else None


def robust_fit(design, target, region_weights):
    coefficients = solve_weighted(design, target, region_weights)
    if coefficients is None:
        return None
    for _ in range(12):
        residual = design @ coefficients - target
        if not np.isfinite(residual).all():
            return None
        center = weighted_median(residual, region_weights)
        sigma = 1.4826 * weighted_median(np.abs(residual - center), region_weights)
        delta = 1.5 * max(sigma, 1.0 / 255.0)
        weights = region_weights.copy()
        outside = np.abs(residual) > delta
        weights[outside] *= delta / np.abs(residual[outside])
        next_coefficients = solve_weighted(design, target, weights)
        if next_coefficients is None:
            return None
        change = np.linalg.norm(next_coefficients - coefficients)
        coefficients = next_coefficients
        if change <= 1e-11 * (1.0 + np.linalg.norm(coefficients)):
            break
    return coefficients


def fit_weights(rgb, target, region_ids):
    """Fit only fixed normalized weights; temporary gains are diagnostics.

    Inputs are RGB and reference-gray local means in 0..255, plus an integer
    physical-region ID for every row. Whole regions, not random pixels, are
    held out. Callers must provide disjoint image/filter supports per region.
    """
    result = WeightFit()

    def reject(reason):
        result.reason = reason
        return result

    rgb = np.asarray(rgb, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    region_ids = np.asarray(region_ids)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or target.shape != (len(rgb),) or region_ids.shape != (len(rgb),):
        return reject("invalid_sample_shapes")
    if not np.issubdtype(region_ids.dtype, np.integer):
        return reject("region_ids_must_be_integers")
    if len(rgb) < 64:
        return reject("insufficient_samples")
    if not np.isfinite(rgb).all() or not np.isfinite(target).all():
        return reject("non_finite_sample")
    if np.any((rgb < 0) | (rgb > 255)) or np.any((target < 0) | (target > 255)):
        return reject("sample_outside_8bit_range")
    unique = np.unique(region_ids)
    if len(unique) < 6:
        return reject("insufficient_independent_regions")
    result.validation_regions = unique[2::3].tolist()
    result.training_regions = [int(r) for i, r in enumerate(unique) if i % 3 != 2]
    validation = np.isin(region_ids, result.validation_regions)
    training = ~validation
    if training.sum() < 32 or validation.sum() < 16:
        return reject("insufficient_partition_samples")
    region_weights = np.empty(len(rgb), dtype=np.float64)
    for region in unique:
        mask = region_ids == region
        region_weights[mask] = 1.0 / mask.sum()
    x, y = rgb / 255.0, target / 255.0
    for selected in (training, validation):
        xs, ys, ws = x[selected], y[selected], region_weights[selected]
        if solve_weighted(xs, ys, ws) is None:
            return reject("insufficient_color_rank_or_target_contrast")
        mean = np.dot(ws, ys) / ws.sum()
        if np.dot(ws, (ys - mean) ** 2) / ws.sum() < (4.0 / 255.0) ** 2:
            return reject("insufficient_color_rank_or_target_contrast")
    beta = robust_fit(x[training], y[training], region_weights[training])
    if beta is None:
        return reject("ill_conditioned_robust_fit")
    result.beta = beta
    scale = float(beta.sum())
    if not np.isfinite(scale) or scale <= 1e-10:
        return reject("non_positive_temporary_scale")
    candidate = beta / scale
    result.candidate_weights = candidate
    if not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1)):
        return reject("negative_or_invalid_weights")
    baseline = robust_fit((x[training] @ DEFAULT_WEIGHTS)[:, None],
                          y[training], region_weights[training])
    if baseline is None or baseline[0] <= 0:
        return reject("invalid_default_baseline")
    result.baseline_scale = float(baseline[0])

    def rmse(coefficients):
        errors = x[validation] @ coefficients - y[validation]
        return float(255.0 * np.sqrt(np.dot(region_weights[validation], errors ** 2)
                                     / region_weights[validation].sum()))

    result.candidate_validation_rmse = rmse(beta)
    result.validation_rmse = result.candidate_validation_rmse
    result.baseline_rmse = rmse(baseline[0] * DEFAULT_WEIGHTS)
    if not np.isfinite([result.validation_rmse, result.baseline_rmse]).all():
        return reject("non_finite_validation")
    improvement = max(0.5, 0.05 * result.baseline_rmse)
    exact = result.validation_rmse <= 1e-7 < result.baseline_rmse
    if result.validation_rmse <= 5.0 and (exact or result.baseline_rmse - result.validation_rmse >= improvement):
        result.weights = candidate
        result.accepted = True
        result.reason = "validated_channel_weights"
    elif result.baseline_rmse <= 5.0:
        result.validation_rmse = result.baseline_rmse
        result.accepted = True
        result.reason = "default_weights_supported"
    else:
        result.reason = ("held_out_mapping_mismatch" if result.validation_rmse > 5.0
                         else "insufficient_held_out_improvement")
    return result
