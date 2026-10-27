#!/usr/bin/env python3
"""Preview ALL saved color samples without installing or validating weights.

Reads an offline_color_unify output directory. The original unconstrained
Huber fit is reported alongside a nonnegative-constrained diagnostic fit.
The latter solves the same RGB linear model; it is not an accepted calibration.
"""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

import offline_color_unify as offline
from offline_gray_model import DEFAULT_WEIGHTS, robust_fit, weighted_median


def solve_nonnegative(design, target, weights):
    """Exact weighted NNLS for three coefficients by enumerating active sets."""
    root = np.sqrt(weights)
    best = np.zeros(3)
    loss = float(np.dot(weights, target*target))
    for count in (1, 2, 3):
        for active in itertools.combinations(range(3), count):
            a, b = design[:, active]*root[:, None], target*root
            coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
            if np.any(coefficients < 0):
                continue
            candidate = np.zeros(3)
            candidate[list(active)] = coefficients
            error = design@candidate-target
            score = float(np.dot(weights, error*error))
            if score < loss:
                best, loss = candidate, score
    return best


def fit_nonnegative(design, target, region_weights):
    coefficients = solve_nonnegative(design, target, region_weights)
    for _ in range(12):
        residual = design@coefficients-target
        center = weighted_median(residual, region_weights)
        sigma = 1.4826*weighted_median(np.abs(residual-center), region_weights)
        delta = 1.5*max(sigma, 1./255.)
        weights = region_weights.copy()
        outside = np.abs(residual) > delta
        weights[outside] *= delta/np.abs(residual[outside])
        candidate = solve_nonnegative(design, target, weights)
        change = np.linalg.norm(candidate-coefficients)
        coefficients = candidate
        if change <= 1e-11*(1.+np.linalg.norm(coefficients)):
            break
    return coefficients


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    offline.dependencies()
    report = json.loads((args.run_dir/"report.json").read_text(encoding="utf-8"))
    with (args.run_dir/"samples.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(row[k]) for k in ("R", "G", "B")] for row in rows])/255.
    y = np.array([float(row["reference_gray"]) for row in rows])/255.
    ids = np.array([int(row["region"]) for row in rows])
    if not len(rows):
        raise ValueError("No saved RGB samples")
    region_weights = np.array([1./np.count_nonzero(ids == i) for i in ids])
    read_time = time.perf_counter()-start
    start = time.perf_counter()
    original_beta = robust_fit(x, y, region_weights)
    beta = fit_nonnegative(x, y, region_weights)
    if beta.sum() <= 1e-10:
        raise ValueError("Nonpositive diagnostic fit")
    gain, weights = float(beta.sum()), beta/beta.sum()
    baseline_gain = float(robust_fit((x@DEFAULT_WEIGHTS)[:, None], y, region_weights)[0])
    singular = np.linalg.svd(x*np.sqrt(region_weights)[:, None], compute_uv=False)

    def rmse(predicted):
        return float(255.*np.sqrt(np.dot(region_weights, (predicted-y)**2)/region_weights.sum()))

    metrics = {"default_weights_only": rmse(x@DEFAULT_WEIGHTS),
               "preview_weights_only": rmse(x@weights),
               "default_weights_plus_gain": rmse(baseline_gain*(x@DEFAULT_WEIGHTS)),
               "preview_weights_plus_gain": rmse(x@beta)}
    fit_time = time.perf_counter()-start
    out = args.output or args.run_dir/"preview"
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preview output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = offline.read_bgr(report["inputs"]["source"])
    reference = offline.read_bgr(report["inputs"]["reference"])
    reference_gray = offline.fixed_gray(reference, DEFAULT_WEIGHTS)
    default_gray = offline.fixed_gray(source, DEFAULT_WEIGHTS)
    preview_gray = offline.fixed_gray(source, weights)
    # Scaled diagnostic image uses beta BEFORE integer rounding. The w-only
    # image uses the production 8-bit grayscale conversion, with no gain.
    scaled_default = np.clip(np.rint(baseline_gain*default_gray.astype(float)), 0, 255).astype(np.uint8)
    scaled_preview = np.clip(np.rint(gain*(source[..., ::-1].astype(float)@weights)), 0, 255).astype(np.uint8)
    for name, image in (("reference_gray", reference_gray), ("source_default", default_gray),
                        ("source_preview_weights_only", preview_gray),
                        ("source_default_plus_gain", scaled_default),
                        ("source_preview_plus_gain", scaled_preview)):
        offline.write_png(out/(name+".png"), image)
    for name, images, titles, subtitle in (
        ("preview_weights_only", (reference_gray, default_gray, preview_gray),
         ("Reference pinhole gray", "Fisheye: default weights", "Fisheye: preview weights ONLY"),
         "RGB weights = [{:.4f}, {:.4f}, {:.4f}]; NO brightness gain".format(*weights)),
        ("preview_with_gain", (reference_gray, scaled_default, scaled_preview),
         ("Reference pinhole gray", "Default weights + gain {:.3f}".format(baseline_gain),
          "Preview weights + gain {:.3f}".format(gain)),
         "EXTRA brightness compensation shown separately; gains fitted to the same 66 samples".replace("66", str(len(rows))))):
        fig, axes = offline.plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle("DIAGNOSTIC PREVIEW - ALL SAMPLES USED FOR FITTING\n"+subtitle, fontsize=11)
        offline.save_figure(fig, out, name)
    plot_time = time.perf_counter()-start
    result = {"status": "PREVIEW_ONLY_NOT_VALIDATED_NOT_INSTALLED", "input_run": str(args.run_dir.resolve()),
              "samples": len(rows), "regions": len(np.unique(ids)), "validation_samples": 0,
              "original_unconstrained_beta": original_beta.tolist() if original_beta is not None else None,
              "original_unconstrained_weights": (original_beta/original_beta.sum()).tolist() if original_beta is not None else None,
              "preview_method": "region-balanced Huber IRLS with nonnegative active-set solves",
              "preview_weights": weights.tolist(), "preview_temporary_gain": gain,
              "default_temporary_gain": baseline_gain,
              "weighted_design_singular_values": singular.tolist(),
              "sample_mean_region_balanced_rmse_gray_levels": metrics,
              "timing_seconds": {"imports_and_read": read_time, "weight_fits": fit_time, "images_and_plots": plot_time},
              "notes": ["All errors are training errors; no held-out validation.",
                        "Gains are visual diagnostics only; no SLAM parameters or calibration are modified.",
                        "The original unconstrained result and the constrained preview are different fits."]}
    (out/"preview_report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
