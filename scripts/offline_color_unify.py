#!/usr/bin/env python3
"""Inspect fixed cross-camera RGB-to-gray weights from two original images.

The weight estimator is translated from include/fixed_gray_weights.h. Sampling
reuses the online 16x16 common-tangent grid, footprint/preblur, 5x5 statistics,
two-way geometry checks and region-level validation. The image-only frontend
uses full-image SIFT and local tangent affine fits WITHOUT LiDAR plane priors.
It is explicitly not a replay of color_gray_init::estimate(snapshot).

Example:
  python scripts/offline_color_unify.py --reference camera0.png --source camera1.png \
      --camera-yaml config/camera_M2DGR_multi_cam.yaml \
      --reference-camera camera0 --source-camera camera1 --output Log/color_check
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from offline_gray_model import DEFAULT_WEIGHTS, fit_weights


def dependencies():
    global cv2, yaml, plt
    try:
        import cv2
        import yaml
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install scripts/requirements_offline_color.txt: " + str(error)) from error
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})


def read_bgr(filename):
    image = cv2.imdecode(np.frombuffer(Path(filename).read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected original 8-bit three-channel image: " + str(filename))
    return image


def write_png(filename, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed: " + str(filename))
    Path(filename).write_bytes(encoded.tobytes())


def fixed_gray(bgr, weights):
    if np.array_equal(weights, DEFAULT_WEIGHTS):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Same RGB order and nearest-integer, saturating conversion as the C++ LUT.
    return np.clip(np.rint(bgr[..., ::-1].astype(float) @ weights), 0, 255).astype(np.uint8)


class Camera:
    """Projection conventions of the repository's Pinhole/Equidistant/MEI models."""

    def __init__(self, values, shape):
        self.values = values
        self.model = str(values.get("model", "")).lower()
        scale = float(values.get("scale", 1.0))
        self.width, self.height = int(float(values["width"]) * scale), int(float(values["height"]) * scale)
        if shape[:2] != (self.height, self.width):
            raise ValueError("Image resolution {} does not match calibration {}x{}; use original images".format(
                shape[:2], self.width, self.height))
        self.fx, self.fy, self.cx, self.cy = [float(values[k]) * scale for k in ("fx", "fy", "cx", "cy")]
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("Invalid camera intrinsics")
        self.k = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)
        if self.model in ("equidistantcamera", "equidistant", "kannalabrandt", "kannala-brandt"):
            self.kind = "equidistant"
            self.distortion = np.array([float(values.get("k" + str(i), 0)) for i in range(1, 5)])
        elif self.model in ("pinhole", "pinholecamera"):
            self.kind = "pinhole"
            self.distortion = np.array([float(values.get("d" + str(i), 0)) for i in range(5)])
        elif self.model in ("mei", "meicamera"):
            self.kind = "mei"
            self.xi = float(values["xi"])
            if not np.isfinite(self.xi) or self.xi < 0:
                raise ValueError("Invalid MEI xi")
            self.distortion = np.array([float(values.get(k, 0)) for k in ("k1", "k2", "p1", "p2", "k3")])
        else:
            raise ValueError("Unsupported camera model: " + self.model)

    def unproject(self, pixels):
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        xy = (pixels - [self.cx, self.cy]) / [self.fx, self.fy]
        if self.kind == "equidistant":
            radius = np.linalg.norm(xy, axis=1)
            theta = radius.copy()
            k1, k2, k3, k4 = self.distortion
            for _ in range(15):
                t2 = theta ** 2
                f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - radius
                derivative = 1 + t2 * (3*k1 + t2 * (5*k2 + t2 * (7*k3 + t2 * 9*k4)))
                theta -= f / np.where(np.abs(derivative) > 1e-12, derivative, np.nan)
            factor = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 1e-12)
            rays = np.column_stack((xy * factor[:, None], np.cos(theta)))
        else:
            points = pixels.reshape(-1, 1, 2)
            if hasattr(cv2, "undistortPointsIter"):
                xy = cv2.undistortPointsIter(points, self.k, self.distortion, None, None,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-12)).reshape(-1, 2)
            else:
                xy = cv2.undistortPoints(points, self.k, self.distortion).reshape(-1, 2)
            if self.kind == "mei":
                r2 = np.sum(xy ** 2, axis=1)
                term = 1 + (1 - self.xi ** 2) * r2
                factor = (self.xi + np.sqrt(np.where(term >= 0, term, np.nan))) / (1 + r2)
                rays = np.column_stack((factor[:, None] * xy, factor - self.xi))
            else:
                rays = np.column_stack((xy, np.ones(len(xy))))
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def project(self, rays):
        rays = np.asarray(rays, dtype=float).reshape(-1, 3)
        x, y, z = rays.T
        if self.kind == "equidistant":
            radius = np.hypot(x, y)
            theta = np.arctan2(radius, z)
            k1, k2, k3, k4 = self.distortion
            t2 = theta ** 2
            theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            factor = np.divide(theta_d, radius, out=np.ones_like(radius), where=radius > 1e-12)
            xy = np.column_stack((factor * x, factor * y))
        else:
            denominator = z.copy()
            if self.kind == "mei":
                norm = np.linalg.norm(rays, axis=1)
                denominator = z + self.xi * norm
                limit = self.xi if self.xi <= 1 else 1 / self.xi
                denominator = np.where(z > -limit * norm + 1e-6, denominator, np.nan)
            else:
                denominator = np.where(z > 1e-9, denominator, np.nan)
            mx, my = x / denominator, y / denominator
            k1, k2, p1, p2, k3 = self.distortion
            r2 = mx * mx + my * my
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            xy = np.column_stack((mx * radial + 2*p1*mx*my + p2*(r2 + 2*mx*mx),
                                  my * radial + p1*(r2 + 2*my*my) + 2*p2*mx*my))
        return xy * [self.fx, self.fy] + [self.cx, self.cy]


class Chart:
    def __init__(self, camera, center_pixel, focal):
        self.camera, self.focal = camera, focal
        z = camera.unproject(np.asarray(center_pixel)[None, :])[0]
        x = np.array([1., 0., 0.]) - z[0] * z
        if np.linalg.norm(x) < 0.1:
            x = np.array([0., 1., 0.]) - z[1] * z
        x /= np.linalg.norm(x)
        self.rotation = np.vstack((x, np.cross(z, x), z))
        if not np.isfinite(self.rotation).all():
            raise ValueError("invalid_chart_center")

    def to_chart(self, pixels):
        rays = self.camera.unproject(pixels) @ self.rotation.T
        return self.focal * rays[:, :2] / np.where(rays[:, 2:] > 1e-8, rays[:, 2:], np.nan)

    def to_raw(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
        rays = np.column_stack((coordinates / self.focal, np.ones(len(coordinates)))) @ self.rotation
        return self.camera.project(rays)


def transform(h, coordinates):
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    value = np.column_stack((coordinates, np.ones(len(coordinates)))) @ h.T
    return value[:, :2] / np.where(np.abs(value[:, 2:]) > 1e-9, value[:, 2:], np.nan)


def raw_jacobian(chart, h, q):
    result = np.empty((2, 2))
    for axis in range(2):
        delta = np.eye(2)[axis] * 0.25
        values = chart.to_raw(transform(h, [q + delta, q - delta]))
        result[:, axis] = (values[0] - values[1]) / 0.5
    if not np.isfinite(result).all() or abs(np.linalg.det(result)) <= 1e-9:
        raise ValueError("invalid_raw_jacobian")
    return result


def sift_matches(source, reference, max_features):
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=max_features)
    else:
        raise ValueError("SIFT unavailable; use a separate Python environment with requirements_offline_color.txt")
    keys_s, desc_s = detector.detectAndCompute(fixed_gray(source, DEFAULT_WEIGHTS), None)
    keys_r, desc_r = detector.detectAndCompute(fixed_gray(reference, DEFAULT_WEIGHTS), None)
    counts = {"source_keypoints": len(keys_s), "reference_keypoints": len(keys_r)}
    if desc_s is None or desc_r is None or min(len(keys_s), len(keys_r)) < 2:
        return np.empty((0, 4)), counts
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, backward = matcher.knnMatch(desc_s, desc_r, k=2), matcher.knnMatch(desc_r, desc_s, k=2)
    pairs = []
    for pair in forward:
        if len(pair) < 2 or pair[0].distance >= 0.75 * pair[1].distance:
            continue
        good = pair[0]
        reverse = backward[good.trainIdx]
        if len(reverse) < 2 or reverse[0].trainIdx != good.queryIdx or reverse[0].distance >= 0.75 * reverse[1].distance:
            continue
        value = np.array(keys_s[good.queryIdx].pt + keys_r[good.trainIdx].pt)
        if any(np.linalg.norm(value[:2] - old[:2]) < 2 or np.linalg.norm(value[2:] - old[2:]) < 2 for old in pairs):
            continue
        pairs.append(value)
    counts["unique_mutual_matches"] = len(pairs)
    return np.asarray(pairs).reshape(-1, 4), counts


def span_ok(points):
    if len(points) < 3:
        return False
    centered = points - points.mean(axis=0)
    scatter = centered.T @ centered
    trace = np.trace(scatter)
    return np.isfinite(scatter).all() and trace > 1e-9 and np.linalg.det(scatter) > 1e-9 * trace ** 2


def deduplicate_matches(matches):
    kept = []
    for pair in matches:
        if not any(np.linalg.norm(pair[:2]-old[:2]) < 2 or np.linalg.norm(pair[2:]-old[2:]) < 2 for old in kept):
            kept.append(pair)
    return np.asarray(kept, dtype=float).reshape(-1, 4)


def local_geometry(matches, seed, source_camera, reference_camera, focal, context):
    source_chart = Chart(source_camera, matches[seed, :2], focal)
    reference_chart = Chart(reference_camera, matches[seed, 2:], focal)
    a, b = source_chart.to_chart(matches[:, :2]), reference_chart.to_chart(matches[:, 2:])
    selected = np.flatnonzero(np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        & (np.max(np.abs(a), axis=1) <= context / 2)
        & (np.max(np.abs(b), axis=1) <= context))
    if len(selected) < 6:
        raise ValueError("local_matches_lt_6")
    # Same target-y/target-x ordering and every-third geometry holdout as C++.
    selected = selected[np.lexsort((b[selected, 0], b[selected, 1]))]
    a, b = a[selected], b[selected]
    training = np.arange(len(selected)) % 3 != 0
    heldout = ~training
    if min(training.sum(), heldout.sum()) < 2 or training.sum() < 4 or not span_ok(a[training]) or not span_ok(b[training]):
        raise ValueError("geometry_span_or_partition")
    fit_args = dict(method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=1000, confidence=0.995, refineIters=10)
    forward, inliers_f = cv2.estimateAffine2D(a[training], b[training], **fit_args)
    reverse, inliers_b = cv2.estimateAffine2D(b[training], a[training], **fit_args)
    if forward is None or reverse is None:
        raise ValueError("ransac_no_affine")
    forward, reverse = np.vstack((forward, [0., 0., 1.])), np.vstack((reverse, [0., 0., 1.]))
    if np.linalg.det(forward[:2, :2]) <= 0 or np.linalg.det(reverse[:2, :2]) <= 0:
        raise ValueError("invalid_affine_orientation")
    errors_f = np.linalg.norm(reference_chart.to_raw(transform(forward, a)) - matches[selected, 2:], axis=1)
    errors_b = np.linalg.norm(source_chart.to_raw(transform(reverse, b)) - matches[selected, :2], axis=1)
    if not np.isfinite(errors_f).all() or not np.isfinite(errors_b).all():
        raise ValueError("invalid_geometry_projection")
    if max(errors_f[heldout].max(), errors_b[heldout].max()) > 2:
        raise ValueError("holdout_error_gt_2px")
    if np.mean(0.5 * (errors_f[heldout] + errors_b[heldout])) > 1.25:
        raise ValueError("holdout_mean_gt_1.25px")
    support = np.zeros(len(selected), dtype=bool)
    support[training] = (inliers_f.ravel() > 0) & (inliers_b.ravel() > 0)
    support &= (errors_f <= 2) & (errors_b <= 2)
    if support.sum() < 4 or not span_ok(a[support]) or not span_ok(b[support]):
        raise ValueError("bidirectional_support_lt_4_or_degenerate")
    hull = cv2.convexHull(b[support].astype(np.float32))
    box = cv2.minAreaRect(hull)[1]
    if cv2.contourArea(hull) < 100 or min(box) < 8 or max(box) > 8 * min(box):
        raise ValueError("insufficient_hull_support")
    return source_chart, reference_chart, forward, reverse, hull, selected[support | heldout]


def box_overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def prepare_raw_grid(image, raw, minification):
    sigma = 0.8 * math.sqrt(max(0., minification ** 2 - 1))
    if not np.isfinite(sigma) or sigma > 6:
        raise ValueError("prefilter_sigma_gt_6")
    halo = int(math.ceil(3 * sigma)) + 2
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite_sampling_projection")
    low = np.floor(raw.min(axis=0)).astype(int) - halo
    high = np.ceil(raw.max(axis=0)).astype(int) + halo + 2
    if min(low) < 0 or high[0] > image.shape[1] or high[1] > image.shape[0]:
        raise ValueError("filter_halo_outside_image")
    box = tuple(int(v) for v in (low[0], low[1], high[0], high[1]))
    crop = image[low[1]:high[1], low[0]:high[0]]
    local = (raw - low).reshape(16, 16, 2).astype(np.float32)
    unclipped = cv2.inRange(crop, np.array([2, 2, 2], np.uint8), np.array([253, 253, 253], np.uint8))
    unclipped = cv2.erode(unclipped, np.ones((2*halo+1, 2*halo+1), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # Four bilinear taps must all be supported; checking only nearest is unsafe.
    xy = np.floor(local.reshape(-1, 2)).astype(int)
    supported = np.ones(len(xy), dtype=bool)
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        supported &= unclipped[xy[:, 1]+dy, xy[:, 0]+dx] != 0
    return {"box": box, "crop": crop, "map": local, "sigma": sigma,
            "supported": supported.reshape(16, 16)}


def resample(grid, image):
    image = image.astype(np.float32)
    if grid["sigma"] > 0.05:
        size = 2 * int(math.ceil(3 * grid["sigma"])) + 1
        image = cv2.GaussianBlur(image, (size, size), grid["sigma"], borderType=cv2.BORDER_REFLECT_101)
    return cv2.remap(image, grid["map"], None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def sample_region(source, reference, geometry):
    source_chart, reference_chart, forward, reverse, hull, support = geometry
    moments = cv2.moments(hull)
    center = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
    inverse = np.linalg.inv(forward)
    ja, jb = raw_jacobian(source_chart, inverse, center), raw_jacobian(reference_chart, np.eye(3), center)
    sa, sb = np.linalg.svd(ja, compute_uv=False), np.linalg.svd(jb, compute_uv=False)
    if min(sa[-1], sb[-1]) <= 1e-9 or max(sa[0]/sa[-1], sb[0]/sb[-1]) > 2:
        raise ValueError("footprint_anisotropy_gt_2")
    step = max(1/sa[-1], 1/sb[-1])
    gx, gy = np.meshgrid(np.arange(16), np.arange(16))
    q = center + (np.column_stack((gx.ravel(), gy.ravel())) - 7.5) * step
    for corner in (0, 15, 240, 255):
        a, b = raw_jacobian(source_chart, inverse, q[corner]), raw_jacobian(reference_chart, np.eye(3), q[corner])
        if np.linalg.norm(a-ja) > 0.2*np.linalg.norm(ja) or np.linalg.norm(b-jb) > 0.2*np.linalg.norm(jb):
            raise ValueError("jacobian_variation_gt_20pct")
    raw_s = source_chart.to_raw(transform(inverse, q))
    raw_r = reference_chart.to_raw(q)
    cycle = source_chart.to_raw(transform(reverse, q))
    valid = np.array([cv2.pointPolygonTest(hull, tuple(v), False) >= 0 for v in q])
    valid &= np.linalg.norm(cycle-raw_s, axis=1) <= 1.5
    grid_s = prepare_raw_grid(source, raw_s, sa[0]*step)
    grid_r = prepare_raw_grid(reference, raw_r, sb[0]*step)
    valid = valid.reshape(16, 16) & grid_s["supported"] & grid_r["supported"]
    bgr = resample(grid_s, grid_s["crop"])
    reference_patch = resample(grid_r, fixed_gray(grid_r["crop"], DEFAULT_WEIGHTS))
    default_patch = resample(grid_s, fixed_gray(grid_s["crop"], DEFAULT_WEIGHTS))
    rgb_mean = cv2.GaussianBlur(bgr, (5, 5), 1.0)[..., ::-1]
    rgb_square = cv2.GaussianBlur(bgr*bgr, (5, 5), 1.0)[..., ::-1]
    gray_mean = cv2.GaussianBlur(reference_patch, (5, 5), 1.0)
    gray_square = cv2.GaussianBlur(reference_patch**2, (5, 5), 1.0)
    mask = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    mask &= (gray_mean > 4) & (gray_mean < 251) & (gray_square-gray_mean**2 <= 36)
    mask &= np.all((rgb_mean > 4) & (rgb_mean < 251) & (rgb_square-rgb_mean**2 <= 36), axis=2)
    mask &= np.isfinite(gray_mean) & np.isfinite(rgb_mean).all(axis=2)
    if not mask.any():
        raise ValueError("no_valid_color_samples")
    return {"source_grid": grid_s, "reference_grid": grid_r, "rgb": rgb_mean[mask].astype(float),
            "target": gray_mean[mask].astype(float), "mask": mask, "valid": valid,
            "reference_patch": reference_patch, "default_patch": default_patch,
            "support_indices": support, "step": float(step)}


def collect_regions(source, reference, source_camera, reference_camera, matches, args):
    regions, rejected = [], Counter()
    for seed in range(min(len(matches), args.max_candidates)):
        if len(regions) >= args.max_regions:
            break
        try:
            geometry = local_geometry(matches, seed, source_camera, reference_camera, args.virtual_focal, args.context)
            region = sample_region(source, reference, geometry)
            if any(box_overlap(region["source_grid"]["box"], old["source_grid"]["box"])
                   or box_overlap(region["reference_grid"]["box"], old["reference_grid"]["box"]) for old in regions):
                raise ValueError("overlapping_filter_support")
            # IDs denote disjoint raw filter footprints in both images.
            region["id"] = len(regions)
            regions.append(region)
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            rejected[str(error).splitlines()[0]] += 1
    return regions, rejected


def save_figure(figure, out, name, synthetic=False):
    if synthetic:
        figure.text(0.99, 0.005, "SYNTHETIC TEST INPUTS", ha="right", va="bottom", fontsize=8)
    figure.savefig(out / (name + ".png"), dpi=180)
    figure.savefig(out / (name + ".svg"))
    plt.close(figure)


def measure_gray_patches(regions, weights, validation_regions):
    before, after = [], []
    for region in regions:
        region["fixed_patch"] = resample(region["source_grid"], fixed_gray(region["source_grid"]["crop"], weights))
        if region["id"] in validation_regions and region["valid"].any():
            valid = region["valid"]
            before.append(float(np.mean((region["default_patch"]-region["reference_patch"])[valid]**2)))
            after.append(float(np.mean((region["fixed_patch"]-region["reference_patch"])[valid]**2)))
    return {"held_out_native_gray_patch_rmse_default": float(np.sqrt(np.mean(before))) if before else None,
            "held_out_native_gray_patch_rmse_fixed": float(np.sqrt(np.mean(after))) if after else None,
            "native_gray_patch_metric": "sqrt of mean per-region MSE; no gain or offset fitted"}


def plot_results(source, reference, matches, regions, weights, report, out):
    gray_r, gray_s, gray_fixed = fixed_gray(reference, DEFAULT_WEIGHTS), fixed_gray(source, DEFAULT_WEIGHTS), fixed_gray(source, weights)
    write_png(out / "reference_gray.png", gray_r)
    write_png(out / "source_gray_default.png", gray_s)
    write_png(out / "source_gray_fixed.png", gray_fixed)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    panels = [(reference[..., ::-1], "Reference RGB"), (source[..., ::-1], "Source RGB"),
              (None, "Applied RGB weights"), (gray_r, "Reference default gray"),
              (gray_s, "Source default gray"), (gray_fixed, "Source fixed-weight gray")]
    for ax, (image, title) in zip(axes.flat, panels):
        ax.set_title(title)
        if image is not None:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        else:
            x = np.arange(3)
            ax.bar(x-0.18, DEFAULT_WEIGHTS, 0.36, label="Default", color="#777777")
            ax.bar(x+0.18, weights, 0.36, label="Applied", color="#0072B2")
            ax.set_xticks(x, ["R", "G", "B"])
            ax.set_ylim(0, 1)
            ax.legend()
    fig.suptitle(report["status"] + " | image-only geometry; no LiDAR plane prior", fontsize=11)
    save_figure(fig, out, "overview", report["synthetic_test_inputs"])
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    height, width = max(len(source), len(reference)), source.shape[1]
    canvas = np.full((height, width+reference.shape[1], 3), 255, np.uint8)
    canvas[:len(source), :width], canvas[:len(reference), width:] = source[..., ::-1], reference[..., ::-1]
    ax.imshow(canvas)
    verified = sorted({int(i) for region in regions for i in region["support_indices"]})
    for i in verified[:160]:
        u, v, x, y = matches[i]
        ax.plot([u, x+width], [v, y], color="#0072B2", alpha=0.45, linewidth=0.6)
    for region in regions:
        for key, offset in (("source_grid", 0), ("reference_grid", width)):
            x0, y0, x1, y1 = region[key]["box"]
            ax.add_patch(plt.Rectangle((x0+offset, y0), x1-x0, y1-y0, fill=False, edgecolor="#D55E00", linewidth=1))
            ax.text(x0+offset, y0, str(region["id"]), color="#D55E00", fontsize=8)
    ax.set_title("Locally checked correspondences and disjoint sampling supports")
    ax.axis("off")
    save_figure(fig, out, "correspondences", report["synthetic_test_inputs"])
    if not regions:
        return
    errors = []
    for region in regions:
        for name in ("default_patch", "fixed_patch"):
            errors.extend(np.abs(region[name]-region["reference_patch"])[region["valid"]].tolist())
    limit = max(1., max(errors))
    fig, axes = plt.subplots(len(regions), 5, figsize=(10, max(2.2, 1.7*len(regions))),
                             squeeze=False, layout="constrained")
    for row, region in enumerate(regions):
        held = region["id"] in report["validation_regions"]
        titles = ["Reference gray", "Default gray", "Fixed gray", "Default absolute error", "Fixed absolute error"]
        images = [region["reference_patch"], region["default_patch"], region["fixed_patch"],
                  np.abs(region["default_patch"]-region["reference_patch"]),
                  np.abs(region["fixed_patch"]-region["reference_patch"])]
        for col, (ax, image) in enumerate(zip(axes[row], images)):
            image = np.where(region["valid"], image, np.nan)
            im = ax.imshow(image, cmap="gray" if col < 3 else "magma", vmin=0, vmax=255 if col < 3 else limit,
                           interpolation="nearest")
            if row == 0:
                ax.set_title(titles[col])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("Region {}\n{}".format(region["id"], "held out" if held else "training"))
            if col == 4:
                fig.colorbar(im, ax=axes[row, 3:], shrink=0.65, label="Gray levels")
    fig.suptitle("Aligned native 8-bit gray patches; NO exposure-scale correction", fontsize=11)
    save_figure(fig, out, "patches_gray_only", report["synthetic_test_inputs"])
    rgb = np.concatenate([region["rgb"] for region in regions])
    target = np.concatenate([region["target"] for region in regions])
    ids = np.concatenate([np.full(len(region["rgb"]), region["id"], int) for region in regions])
    valid = np.isin(ids, report["validation_regions"])
    if valid.any() and report.get("baseline_scale") is not None and report.get("beta") is not None:
        baseline = (rgb @ DEFAULT_WEIGHTS) * report["baseline_scale"]
        learned = rgb @ np.asarray(report["beta"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        for ax, prediction, title in zip(axes, (baseline, learned),
            ("Default weights + training-fitted scale", "Candidate weights + training-fitted scale")):
            for region in report["validation_regions"]:
                keep = ids == region
                ax.scatter(target[keep], prediction[keep], s=9, alpha=0.6, label="Region " + str(region))
            ax.plot([0, 255], [0, 255], "k--", linewidth=0.8)
            ax.set(xlim=(0, 255), ylim=(0, 255), xlabel="Reference gray", ylabel="Predicted gray", title=title)
            ax.set_aspect("equal")
        axes[0].legend(fontsize=7)
        fig.suptitle("Held-out regions; scales fitted on training regions only (diagnostic)")
        save_figure(fig, out, "validation", report["synthetic_test_inputs"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path, help="Anchor camera original color image")
    parser.add_argument("--source", required=True, type=Path, help="Camera whose fixed gray weights are estimated")
    parser.add_argument("--camera-yaml", required=True, type=Path)
    parser.add_argument("--reference-camera", default="camera0")
    parser.add_argument("--source-camera", default="camera1")
    parser.add_argument("--virtual-focal", type=float, default=300.)
    parser.add_argument("--context", type=float, default=160., help="Local tangent matching context, virtual pixels")
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--matches-csv", type=Path, help="Optional checked source_u,source_v,reference_u,reference_v rows")
    parser.add_argument("--evaluate-weights", type=Path, help="Apply a previous report's weights WITHOUT refitting")
    parser.add_argument("--synthetic-input", action="store_true", help="Label generated TEST inputs in every figure")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not np.isfinite([args.virtual_focal, args.context]).all() or min(args.virtual_focal, args.context) <= 0:
        parser.error("focal and context must be finite and positive")
    if args.max_regions < 1 or args.max_candidates < 1 or args.max_features < 6:
        parser.error("invalid extraction budgets")
    dependencies()
    cv2.setRNGSeed(7)
    source, reference = read_bgr(args.source), read_bgr(args.reference)
    with args.camera_yaml.open(encoding="utf-8-sig") as stream:
        calibration = yaml.safe_load(stream)
    source_camera = Camera(calibration[args.source_camera], source.shape)
    reference_camera = Camera(calibration[args.reference_camera], reference.shape)
    out = args.output or Path(__file__).resolve().parents[1] / "Log" / "offline_color" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    if args.matches_csv:
        with args.matches_csv.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        matches = np.array([[float(row[k]) for k in ("source_u", "source_v", "reference_u", "reference_v")]
                            for row in rows], dtype=float).reshape(-1, 4)
        counts = {"provided_matches": len(matches)}
        if not np.isfinite(matches).all():
            raise ValueError("non-finite provided matches")
        matches = deduplicate_matches(matches)
        counts["unique_provided_matches"] = len(matches)
    else:
        matches, counts = sift_matches(source, reference, args.max_features)
    regions, rejections = collect_regions(source, reference, source_camera, reference_camera, matches, args)
    rgb = np.concatenate([region["rgb"] for region in regions]) if regions else np.empty((0, 3))
    target = np.concatenate([region["target"] for region in regions]) if regions else np.empty(0)
    ids = np.concatenate([np.full(len(region["rgb"]), region["id"], int) for region in regions]) if regions else np.empty(0, int)
    if args.evaluate_weights:
        with args.evaluate_weights.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("accepted") is not True:
            raise ValueError("Only accepted previous weights can be evaluated")
        if (previous.get("source_camera") != args.source_camera
                or previous.get("reference_camera") != args.reference_camera):
            raise ValueError("Previous weights refer to a different camera pair")
        weights = np.asarray(previous["weights"], dtype=float)
        if weights.shape != (3,) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)) or abs(weights.sum()-1) > 1e-6:
            raise ValueError("Invalid fixed weights")
        report = {"accepted": None, "evaluation_completed": bool(len(rgb)),
                  "reason": "fixed_previous_weights_no_refit", "weights": weights.tolist(),
                  "training_regions": [], "validation_regions": list(range(len(regions)))}
        if len(rgb):
            report["raw_default_rmse"] = float(np.sqrt(np.mean((rgb @ DEFAULT_WEIGHTS-target)**2)))
            report["raw_fixed_rmse"] = float(np.sqrt(np.mean((rgb @ weights-target)**2)))
    else:
        fit = fit_weights(rgb, target, ids)
        weights, report = fit.weights, fit.as_dict()
    status = ("FIXED-WEIGHT EVALUATION; NO REFIT" if args.evaluate_weights
              else "ACCEPTED" if report["accepted"] else "REJECTED: " + report["reason"])
    if args.evaluate_weights and not report["evaluation_completed"]:
        status = "NO VALID SAMPLES FOR FIXED-WEIGHT EVALUATION"
    report.update({"status": status,
                   "geometry_mode": "image_only_local_tangent_affine_no_lidar_prior",
                   "source_camera": args.source_camera, "reference_camera": args.reference_camera,
                   "regions": len(regions), "samples": len(rgb), "matching": counts,
                   "rejections": dict(rejections), "virtual_focal": args.virtual_focal,
                   "synthetic_test_inputs": args.synthetic_input,
                   "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
                   "calibration_models": {"source": calibration[args.source_camera],
                                          "reference": calibration[args.reference_camera]},
                   "inputs": {"source": str(args.source.resolve()), "reference": str(args.reference.resolve()),
                              "camera_yaml": str(args.camera_yaml.resolve())},
                   "input_sha256": {"source": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                                    "reference": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                                    "camera_yaml": hashlib.sha256(args.camera_yaml.read_bytes()).hexdigest()},
                   "notes": ["RGB coefficient order. No ESIKF/exposure/extrinsic parameters are modified.",
                             "Local affine support is an image-only geometric approximation, not a verified LiDAR plane.",
                             "The online LiDAR-prior +/-25% residual-shape gate cannot be reproduced without the prior.",
                             "Native fixed-gray figures have no exposure correction; regression scales are diagnostic only."]})
    if args.synthetic_input:
        report["status"] = "SYNTHETIC TEST | " + report["status"]
    report.update(measure_gray_patches(regions, weights, report["validation_regions"]))
    if len(rgb):
        heldout = np.isin(ids, report["validation_regions"])
        if heldout.any():
            report["held_out_raw_default_mean_rmse"] = float(np.sqrt(np.mean((rgb[heldout] @ DEFAULT_WEIGHTS-target[heldout])**2)))
            report["held_out_raw_fixed_mean_rmse"] = float(np.sqrt(np.mean((rgb[heldout] @ weights-target[heldout])**2)))
    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["region", "partition", "R", "G", "B", "reference_gray"])
        for color, value, region in zip(rgb, target, ids):
            writer.writerow([int(region), "validation" if int(region) in report["validation_regions"] else "training", *color, value])
    with (out / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    plot_results(source, reference, matches, regions, weights, report, out)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print("Output:", out.resolve())
    successful = report["evaluation_completed"] if args.evaluate_weights else report["accepted"]
    return 0 if successful else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("Offline color check:", error, file=sys.stderr)
        raise SystemExit(1)
