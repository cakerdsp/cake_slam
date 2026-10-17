#ifndef VIRTUAL_SAMPLING_H_
#define VIRTUAL_SAMPLING_H_

#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>

namespace virtual_sampling
{

// Geometric sampling constants, independent of estimator P and measurement R.
constexpr double kMinPitch = 0.25;
constexpr double kMaxPitch = 4.0;
constexpr double kPixelSigmaRaw = 0.5;
constexpr double kNyquistAttenuation = 0.1;
constexpr double kLogScaleCoverage = 0.34657359027997265471;  // log(sqrt(2))

struct Plan
{
  double pitch = 1.0;
  // Additional Gaussian sigma in each original image's pixel coordinates.
  double sigma_ref = 0.0;
  double sigma_cur = 0.0;
  // Diagnostics only: these are not measurement reliability weights.
  double anisotropy_ref = 1.0;
  double anisotropy_cur = 1.0;
  bool valid = false;
};

namespace detail
{

struct MatrixMetrics
{
  double minimum = 0.0;
  double maximum = 0.0;
  Eigen::Matrix2d inverse = Eigen::Matrix2d::Zero();
};

inline bool matrixMetrics(const Eigen::Matrix2d &matrix, MatrixMetrics &metrics)
{
  if (!matrix.allFinite()) return false;
  const double scale = matrix.cwiseAbs().maxCoeff();
  if (!(scale > 0.0)) return false;

  // Normalize before decomposition so unusually large/small coordinate units
  // do not overflow the singular-value calculation.
  const Eigen::Matrix2d normalized = matrix / scale;
  const Eigen::JacobiSVD<Eigen::Matrix2d> svd(
      normalized, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector2d singular = svd.singularValues();
  if (!singular.allFinite() || !(singular[0] > 0.0) ||
      singular[1] <= 1.0e-8 * singular[0])
    return false;

  metrics.maximum = scale * singular[0];
  metrics.minimum = scale * singular[1];
  metrics.inverse =
      (svd.matrixV() * singular.cwiseInverse().asDiagonal() *
       svd.matrixU().transpose()) / scale;
  return metrics.minimum > 0.0 && std::isfinite(metrics.minimum) &&
         std::isfinite(metrics.maximum) && metrics.inverse.allFinite();
}

inline double additionalSigma(double total_sigma)
{
  // Avoid squaring a potentially large finite sigma. Roundoff at the native
  // footprint floor must not produce a negative variance.
  if (total_sigma <= kPixelSigmaRaw) return 0.0;
  const double ratio = kPixelSigmaRaw / total_sigma;
  return total_sigma * std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
}

inline double logRangeDifference(double reference_range, double current_range)
{
  if (!(reference_range > 0.0) || !(current_range > 0.0) ||
      !std::isfinite(reference_range) || !std::isfinite(current_range))
    return std::numeric_limits<double>::infinity();
  // Taking separate logarithms avoids overflow/underflow in the ratio.
  return std::abs(std::log(reference_range) - std::log(current_range));
}

}  // namespace detail

// A_cur_ref maps reference canonical virtual offsets to current canonical
// virtual offsets. D_ref/D_cur map their own virtual offsets to raw pixels.
// The common q coordinates are the current canonical virtual coordinates:
// B_ref = D_ref * A_cur_ref^-1, B_cur = D_cur.
//
// The Gaussian pixel-footprint approximation gives
// C_i = sigma0^2 * B_i^-1 * B_i^-T. Choose h=clamp(d,0.25,4),
// d=max ||B_i^-1||_2, and a common target sigma
// tau=max(sigma0*d, h*sqrt(2*log(1/epsilon))/pi).
// A scalar raw-image blur with total sigma=tau*sigma_max(B_i) guarantees
// a pulled-back footprint >= tau^2 I. The two footprints match for isotropic
// local warps; anisotropic warps are conservatively overblurred, not exactly
// matched. Raw sigmas are not clipped, since clipping would break this bound.
// This local model cannot undo aliasing already present in the source image.
// Freeze the plan, masks, and corresponding gradients during an inner solve.
inline Plan makePlan(const Eigen::Matrix2d &raw_from_ref_virtual,
                     const Eigen::Matrix2d &raw_from_cur_virtual,
                     const Eigen::Matrix2d &A_cur_ref)
{
  Plan result;
  detail::MatrixMetrics affine, reference, current;
  if (!detail::matrixMetrics(A_cur_ref, affine)) return result;
  const Eigen::Matrix2d raw_ref_from_common =
      raw_from_ref_virtual * affine.inverse;
  if (!detail::matrixMetrics(raw_ref_from_common, reference) ||
      !detail::matrixMetrics(raw_from_cur_virtual, current))
    return result;

  const double footprint =
      std::max(1.0 / reference.minimum, 1.0 / current.minimum);
  if (!std::isfinite(footprint)) return result;
  const double pitch = std::clamp(footprint, kMinPitch, kMaxPitch);
  const double pi = 3.14159265358979323846;
  const double target_sigma =
      std::max(kPixelSigmaRaw * footprint,
               pitch * std::sqrt(2.0 * std::log(1.0 / kNyquistAttenuation)) / pi);
  const double total_ref = target_sigma * reference.maximum;
  const double total_cur = target_sigma * current.maximum;
  if (!std::isfinite(total_ref) || !std::isfinite(total_cur)) return result;

  result.pitch = pitch;
  result.sigma_ref = detail::additionalSigma(total_ref);
  result.sigma_cur = detail::additionalSigma(total_cur);
  result.anisotropy_ref = reference.maximum / reference.minimum;
  result.anisotropy_cur = current.maximum / current.minimum;
  result.valid = std::isfinite(result.sigma_ref) && std::isfinite(result.sigma_cur) &&
                 std::isfinite(result.anisotropy_ref) && std::isfinite(result.anisotropy_cur);
  return result;
}

// Range ratio is only a symmetric reference-coverage heuristic, not a test
// of whether a visual measurement is static, correct, or informative.
inline bool distanceScaleCovered(double reference_range, double current_range)
{
  return detail::logRangeDifference(reference_range, current_range) <=
         kLogScaleCoverage + 1.0e-12;
}

// Nonnegative ranking cost: 0 at equal range, 1 at a sqrt(2) range ratio.
// Invalid ranges return infinity; this value is not a fusion weight.
inline double scalePenalty(double reference_range, double current_range)
{
  return detail::logRangeDifference(reference_range, current_range) / kLogScaleCoverage;
}

}  // namespace virtual_sampling

#endif  // VIRTUAL_SAMPLING_H_
