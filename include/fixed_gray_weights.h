#ifndef CAKE_SLAM_FIXED_GRAY_WEIGHTS_H
#define CAKE_SLAM_FIXED_GRAY_WEIGHTS_H

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace color_gray_init
{

// This module uses RGB order, even when the caller's cv::Mat uses BGR order.
inline Eigen::Vector3d defaultWeights()
{
  return Eigen::Vector3d(0.299, 0.587, 0.114);
}

struct ColorSample
{
  Eigen::Vector3d rgb;
  double target;
  int region;
};

struct WeightFit
{
  bool accepted = false;
  Eigen::Vector3d weights = defaultWeights();
  // Gray-level units, 0..255. Validation includes the TRAINING-fitted scalar;
  // it tests relative channel response, not equality of the installed grays.
  double validation_rmse = std::numeric_limits<double>::infinity();
  double baseline_rmse = std::numeric_limits<double>::infinity();
  std::string reason;
};

namespace fixed_gray_detail
{

// Conservative engineering acceptance thresholds, not statistical guarantees.
// Regions must be distinct physical supports supplied by the caller, not
// arbitrary labels assigned to pixels from one patch. Filter halos must not
// overlap between training and validation supports.
constexpr int kMinSamples = 64;
constexpr int kMinRegions = 6;
constexpr int kMinTrainingSamples = 32;
constexpr int kMinValidationSamples = 16;
constexpr double kMinSingularRatio = 0.01;
constexpr double kMinTargetStdDev = 4.0 / 255.0;
constexpr double kMaxValidationRmse = 5.0;
constexpr double kMinAbsoluteImprovement = 0.5;
constexpr double kMinRelativeImprovement = 0.05;
constexpr double kNumericallyExactRmse = 1.0e-7;
constexpr int kMaxIrlsIterations = 12;

// The main project's ARM Release flags include -ffast-math. A bit-level
// boundary check must not be optimized away under finite-math assumptions.
inline bool finiteNumber(double value)
{
  static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                    std::numeric_limits<double>::is_iec559,
                "fixed gray fitting requires IEEE-754 binary64 doubles");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <typename Derived>
inline bool allFinite(const Eigen::MatrixBase<Derived> &values)
{
  for (Eigen::Index index = 0; index < values.size(); ++index)
    if (!finiteNumber(values.derived().coeff(index))) return false;
  return true;
}

struct SampleBlock
{
  Eigen::MatrixXd rgb;
  Eigen::VectorXd target;
  Eigen::VectorXd region_weights;
};

inline double weightedMedian(const Eigen::VectorXd &values,
                             const Eigen::VectorXd &weights)
{
  std::vector<std::pair<double, double>> entries;
  entries.reserve(static_cast<std::size_t>(values.size()));
  for (Eigen::Index index = 0; index < values.size(); ++index)
    entries.emplace_back(values[index], weights[index]);
  std::sort(entries.begin(), entries.end(),
            [](const std::pair<double, double> &a,
               const std::pair<double, double> &b) { return a.first < b.first; });
  const double midpoint = 0.5 * weights.sum();
  double cumulative = 0.0;
  for (const auto &entry : entries)
  {
    cumulative += entry.second;
    if (cumulative >= midpoint) return entry.first;
  }
  return entries.back().first;
}

inline bool solveWeighted(const Eigen::MatrixXd &design,
                          const Eigen::VectorXd &target,
                          const Eigen::VectorXd &weights,
                          Eigen::VectorXd &coefficients)
{
  if (design.rows() < design.cols() || design.rows() == 0 ||
      design.rows() != target.size() || target.size() != weights.size() ||
      !allFinite(design) || !allFinite(target) || !allFinite(weights) ||
      (weights.array() <= 0.0).any())
    return false;

  const Eigen::VectorXd square_root_weights = weights.array().sqrt().matrix();
  Eigen::MatrixXd weighted_design = design;
  weighted_design.array().colwise() *= square_root_weights.array();
  const Eigen::VectorXd weighted_target =
      target.array() * square_root_weights.array();
  const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(
      weighted_design, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd singular_values = decomposition.singularValues();
  if (!allFinite(singular_values) || singular_values.size() == 0 ||
      singular_values[0] <= 1.0e-10 ||
      singular_values[singular_values.size() - 1] <
          kMinSingularRatio * singular_values[0])
    return false;

  coefficients = decomposition.solve(weighted_target);
  return allFinite(coefficients);
}

inline bool robustFit(const Eigen::MatrixXd &design,
                      const Eigen::VectorXd &target,
                      const Eigen::VectorXd &region_weights,
                      Eigen::VectorXd &coefficients)
{
  if (!solveWeighted(design, target, region_weights, coefficients)) return false;
  for (int iteration = 0; iteration < kMaxIrlsIterations; ++iteration)
  {
    const Eigen::VectorXd residual = design * coefficients - target;
    if (!allFinite(residual)) return false;
    const double residual_center = weightedMedian(residual, region_weights);
    const Eigen::VectorXd deviation =
        (residual.array() - residual_center).abs().matrix();
    const double sigma = 1.4826 * weightedMedian(deviation, region_weights);
    // A one-gray-level floor avoids singular influence scales for noiseless
    // or quantized observations. It does not regularize the coefficients.
    const double huber_delta = 1.5 * std::max(sigma, 1.0 / 255.0);
    Eigen::VectorXd weights = region_weights;
    for (Eigen::Index index = 0; index < residual.size(); ++index)
    {
      const double magnitude = std::abs(residual[index]);
      if (magnitude > huber_delta) weights[index] *= huber_delta / magnitude;
    }

    Eigen::VectorXd next;
    // Rechecking rank after weighting prevents an effectively disconnected
    // or uninformative sample set from being accepted by the robust solver.
    if (!solveWeighted(design, target, weights, next)) return false;
    const double change = (next - coefficients).norm();
    coefficients = next;
    if (change <= 1.0e-11 * (1.0 + coefficients.norm())) break;
  }
  return true;
}

inline bool informativeBlock(const SampleBlock &block)
{
  Eigen::VectorXd unused;
  if (!solveWeighted(block.rgb, block.target, block.region_weights, unused))
    return false;
  const double weight_sum = block.region_weights.sum();
  const double mean = block.region_weights.dot(block.target) / weight_sum;
  const double variance =
      (block.region_weights.array() * (block.target.array() - mean).square()).sum() /
      weight_sum;
  return finiteNumber(variance) &&
         variance >= kMinTargetStdDev * kMinTargetStdDev;
}

inline double validationRmse(const SampleBlock &block,
                             const Eigen::Vector3d &coefficients)
{
  const Eigen::VectorXd error = block.rgb * coefficients - block.target;
  const double mean_squared =
      (block.region_weights.array() * error.array().square()).sum() /
      block.region_weights.sum();
  if (!allFinite(error) || !finiteNumber(mean_squared))
    return std::numeric_limits<double>::infinity();
  return 255.0 * std::sqrt(std::max(0.0, mean_squared));
}

} // namespace fixed_gray_detail

// Inputs are RGB/target values in 0..255 units from a single image group.
// The temporary model target ~= beta.dot(rgb) deliberately includes overall
// brightness. Only beta / beta.sum() is returned; no gain is retained and no
// exposure state, covariance, or runtime reference is modified here.
//
// The solver's exact recovery requires a noiseless linear relationship,
// positive coefficient sum, admissible weights, and sufficient color rank.
// The acceptance policy separately requires independent validation regions;
// engineering improvement gates apply to noisy fits. Unknown ISP nonlinearities
// and noisy RGB on both sides do not acquire an unbiasedness guarantee here.
inline WeightFit fitWeights(const std::vector<ColorSample> &samples)
{
  using namespace fixed_gray_detail;
  WeightFit result;
  if (samples.size() < static_cast<std::size_t>(kMinSamples))
  {
    result.reason = "insufficient_samples";
    return result;
  }

  std::map<int, std::vector<std::size_t>> regions;
  for (std::size_t index = 0; index < samples.size(); ++index)
  {
    const ColorSample &sample = samples[index];
    if (!allFinite(sample.rgb) || !finiteNumber(sample.target))
    {
      result.reason = "non_finite_sample";
      return result;
    }
    if ((sample.rgb.array() < 0.0).any() || (sample.rgb.array() > 255.0).any() ||
        sample.target < 0.0 || sample.target > 255.0)
    {
      result.reason = "sample_outside_8bit_range";
      return result;
    }
    regions[sample.region].push_back(index);
  }
  if (regions.size() < static_cast<std::size_t>(kMinRegions))
  {
    result.reason = "insufficient_independent_regions";
    return result;
  }

  std::vector<std::size_t> training_indices;
  std::vector<std::size_t> validation_indices;
  std::size_t region_number = 0;
  for (const auto &region : regions)
  {
    // Hold out whole supports, never random pixels from the same support.
    std::vector<std::size_t> &destination =
        (region_number++ % 3 == 2) ? validation_indices : training_indices;
    destination.insert(destination.end(), region.second.begin(), region.second.end());
  }
  if (training_indices.size() < static_cast<std::size_t>(kMinTrainingSamples) ||
      validation_indices.size() < static_cast<std::size_t>(kMinValidationSamples))
  {
    result.reason = "insufficient_partition_samples";
    return result;
  }

  const auto make_block = [&](const std::vector<std::size_t> &indices) {
    SampleBlock block;
    block.rgb.resize(static_cast<Eigen::Index>(indices.size()), 3);
    block.target.resize(static_cast<Eigen::Index>(indices.size()));
    block.region_weights.resize(static_cast<Eigen::Index>(indices.size()));
    for (std::size_t row = 0; row < indices.size(); ++row)
    {
      const ColorSample &sample = samples[indices[row]];
      const Eigen::Index eigen_row = static_cast<Eigen::Index>(row);
      block.rgb.row(eigen_row) = sample.rgb.transpose() / 255.0;
      block.target[eigen_row] = sample.target / 255.0;
      // Equal aggregate influence per physical region, regardless of how
      // densely the caller sampled that region.
      block.region_weights[eigen_row] =
          1.0 / static_cast<double>(regions.at(sample.region).size());
    }
    return block;
  };
  const SampleBlock training = make_block(training_indices);
  const SampleBlock validation = make_block(validation_indices);
  if (!informativeBlock(training) || !informativeBlock(validation))
  {
    result.reason = "insufficient_color_rank_or_target_contrast";
    return result;
  }

  Eigen::VectorXd fitted;
  if (!robustFit(training.rgb, training.target, training.region_weights, fitted))
  {
    result.reason = "ill_conditioned_robust_fit";
    return result;
  }
  const Eigen::Vector3d beta = fitted;
  const double scale = beta.sum();
  if (!finiteNumber(scale) || scale <= 1.0e-10)
  {
    result.reason = "non_positive_temporary_scale";
    return result;
  }
  const Eigen::Vector3d normalized = beta / scale;
  // Reject negative solutions rather than clipping coefficients and silently
  // changing the fitted model. Accepted weights form a convex RGB combination.
  if (!allFinite(normalized) || (normalized.array() < 0.0).any() ||
      (normalized.array() > 1.0).any())
  {
    result.reason = "negative_or_invalid_weights";
    return result;
  }

  const Eigen::MatrixXd baseline_design = training.rgb * defaultWeights();
  Eigen::VectorXd baseline_scalar;
  if (!robustFit(baseline_design, training.target, training.region_weights,
                 baseline_scalar) || baseline_scalar[0] <= 0.0)
  {
    result.reason = "invalid_default_baseline";
    return result;
  }
  result.validation_rmse = validationRmse(validation, beta);
  result.baseline_rmse =
      validationRmse(validation, baseline_scalar[0] * defaultWeights());
  if (!finiteNumber(result.validation_rmse) || !finiteNumber(result.baseline_rmse))
  {
    result.reason = "non_finite_validation";
    return result;
  }

  const double required_improvement =
      std::max(kMinAbsoluteImprovement,
               kMinRelativeImprovement * result.baseline_rmse);
  // Do not introduce a deliberate bias in the ideal noiseless case by
  // suppressing an exactly recovered, independently verified small change.
  // The epsilon only distinguishes numerical zero at the regression scale.
  const bool exact_model_improvement =
      result.validation_rmse <= kNumericallyExactRmse &&
      result.baseline_rmse > kNumericallyExactRmse;
  if (result.validation_rmse <= kMaxValidationRmse &&
      (exact_model_improvement ||
       result.baseline_rmse - result.validation_rmse >= required_improvement))
  {
    result.weights = normalized;
    result.accepted = true;
    result.reason = "validated_channel_weights";
    return result;
  }
  if (result.baseline_rmse <= kMaxValidationRmse)
  {
    // Sufficient independent evidence already supports the unchanged channel
    // weights; avoid installing an unsubstantiated small adjustment.
    result.validation_rmse = result.baseline_rmse;
    result.accepted = true;
    result.reason = "default_weights_supported";
    return result;
  }
  result.reason = result.validation_rmse > kMaxValidationRmse
                      ? "held_out_mapping_mismatch"
                      : "insufficient_held_out_improvement";
  return result;
}

} // namespace color_gray_init

#endif // CAKE_SLAM_FIXED_GRAY_WEIGHTS_H
