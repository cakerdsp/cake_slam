#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace visual_patch_score
{
using PoseMatrix = Eigen::Matrix<double, 6, 6>;
using PoseProjection = Eigen::Matrix<double, 2, 6>;

struct Settings
{
  double regularization_ratio = 0.1;

  bool valid() const
  {
    return std::isfinite(regularization_ratio) && regularization_ratio > 0.0;
  }
};

struct GeometrySummary
{
  // Raw H^T R^-1 H, before any directional backend weighting. Rotation is a
  // right perturbation of world_from_body; translation is in the world frame.
  PoseMatrix information = PoseMatrix::Zero();
  // Same correspondences and inverse-variance weights as information.
  // Store sums; divide both forms by weight_sum when constructing the metric.
  PoseMatrix displacement_metric = PoseMatrix::Zero();
  double weight_sum = 0.0;
  Eigen::Matrix3d world_from_body = Eigen::Matrix3d::Identity();
  Eigen::Vector3d body_position = Eigen::Vector3d::Zero();
  double stamp = -1.0;
  int count = 0;
  bool valid = false;
};

struct DirectionalWeight
{
  // Contravariant weighting in the original pose coordinates, C f(N) C^T.
  // Invalid geometry must use the frontend's Shi-Tomasi fallback, not I6.
  PoseMatrix weight = PoseMatrix::Zero();
  double lambda_min = 0.0;
  double lambda_max = 0.0;
  double alpha = 0.0;
  double metric_rcond = 0.0;
  const char *status = "unavailable";
  bool valid = false;
};

inline Eigen::Matrix3d crossMatrix(const Eigen::Vector3d &point)
{
  Eigen::Matrix3d matrix;
  matrix << 0.0, -point.z(), point.y(),
            point.z(), 0.0, -point.x(),
            -point.y(), point.x(), 0.0;
  return matrix;
}

// Represent the same infinitesimal rigid motion at the current body origin.
// The lower-left block accounts for the change of rotation centre. Omitting
// it makes rotational/translational coupling depend on the snapshot position.
inline PoseMatrix currentToSnapshot(const GeometrySummary &summary,
                                    const Eigen::Matrix3d &world_from_body,
                                    const Eigen::Vector3d &body_position)
{
  PoseMatrix current_to_snapshot = PoseMatrix::Identity();
  current_to_snapshot.topLeftCorner<3, 3>() = summary.world_from_body.transpose() * world_from_body;
  current_to_snapshot.bottomLeftCorner<3, 3>() =
      crossMatrix(body_position - summary.body_position) * world_from_body;
  return current_to_snapshot;
}

inline void accumulateDisplacementMetric(GeometrySummary &summary,
                                         const Eigen::Vector3d &point_body,
                                         const Eigen::Matrix3d &world_from_body,
                                         double inverse_variance)
{
  Eigen::Matrix<double, 3, 6> point_jacobian;
  point_jacobian.leftCols<3>() = -world_from_body * crossMatrix(point_body);
  point_jacobian.rightCols<3>().setIdentity();
  summary.displacement_metric.noalias() +=
      inverse_variance * point_jacobian.transpose() * point_jacobian;
  summary.weight_sum += inverse_variance;
}

inline DirectionalWeight directionalWeight(const GeometrySummary &summary,
                                           const Eigen::Matrix3d &world_from_body,
                                           const Eigen::Vector3d &body_position,
                                           const Settings &settings,
                                           bool use_directional_weight = true)
{
  DirectionalWeight out;
  if (!summary.valid || summary.count < 30) { out.status = "insufficient_geometry"; return out; }
  if (!settings.valid() || !std::isfinite(summary.weight_sum) || summary.weight_sum <= 0.0 ||
      !summary.information.allFinite() || !summary.world_from_body.allFinite() ||
      !summary.displacement_metric.allFinite() || !summary.body_position.allFinite() ||
      !world_from_body.allFinite() || !body_position.allFinite())
  { out.status = "nonfinite_geometry"; return out; }
  const PoseMatrix transform = currentToSnapshot(summary, world_from_body, body_position);
  const PoseMatrix raw_metric = transform.transpose() * summary.displacement_metric * transform / summary.weight_sum;
  const PoseMatrix metric = 0.5 * (raw_metric + raw_metric.transpose());
  const PoseMatrix information = transform.transpose() * summary.information * transform / summary.weight_sum;
  if (!metric.allFinite() || !information.allFinite() || (metric.diagonal().array() <= 0.0).any())
  { out.status = "invalid_metric"; return out; }

  // B_i=[-R[p_i]x,I], so M_tt=mu*I (mu=1 for the weighted mean).
  // Center at the weighted point centroid before whitening. The rotational
  // Schur complement is the point-set inertia about its centroid; its spectrum
  // detects near-collinearity without mixing metres and radians or hiding a
  // small rotational axis behind per-column normalization.
  const double mu = metric.bottomRightCorner<3, 3>().trace() / 3.0;
  if (!std::isfinite(mu) || mu <= 0.0 ||
      (metric.bottomRightCorner<3, 3>() - mu * Eigen::Matrix3d::Identity()).norm() > 1.0e-8 * mu)
  { out.status = "invalid_metric"; return out; }
  const Eigen::Matrix3d cross_block = metric.topRightCorner<3, 3>();
  const Eigen::Matrix3d inertia = metric.topLeftCorner<3, 3>() - cross_block * cross_block.transpose() / mu;
  if (!inertia.allFinite()) { out.status = "invalid_metric"; return out; }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> metric_eigen(0.5 * (inertia + inertia.transpose()));
  if (metric_eigen.info() != Eigen::Success || !metric_eigen.eigenvalues().allFinite())
  { out.status = "metric_decomposition_failed"; return out; }
  if (metric_eigen.eigenvalues().maxCoeff() <= 0.0)
  { out.status = "singular_metric"; return out; }
  out.metric_rcond = metric_eigen.eigenvalues().minCoeff() / metric_eigen.eigenvalues().maxCoeff();
  if (!std::isfinite(out.metric_rcond) || out.metric_rcond <= 1.0e-10)
  { out.status = "singular_metric"; return out; }
  const Eigen::Vector3d inverse_roots = metric_eigen.eigenvalues().array().sqrt().inverse().matrix();
  PoseMatrix whitening = PoseMatrix::Zero();
  whitening.topLeftCorner<3, 3>() = metric_eigen.eigenvectors() * inverse_roots.asDiagonal();
  whitening.bottomLeftCorner<3, 3>() = -cross_block.transpose() * whitening.topLeftCorner<3, 3>() / mu;
  whitening.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() / std::sqrt(mu);
  // C^T M C = I. The spectrum of C^T H C solves H v = lambda M v.
  const PoseMatrix normalized = whitening.transpose() * information * whitening;
  if (!normalized.allFinite()) { out.status = "invalid_spectrum"; return out; }
  Eigen::SelfAdjointEigenSolver<PoseMatrix> eigen(0.5 * (normalized + normalized.transpose()));
  if (eigen.info() != Eigen::Success || !eigen.eigenvalues().allFinite() ||
      eigen.eigenvalues().minCoeff() < -1.0e-6 || eigen.eigenvalues().maxCoeff() > 1.0 + 1.0e-6)
  { out.status = "invalid_spectrum"; return out; }
  const Eigen::Matrix<double, 6, 1> values = eigen.eigenvalues().array().max(0.0).min(1.0).matrix();
  const double largest = values.maxCoeff();
  out.alpha = settings.regularization_ratio * largest;
  if (!std::isfinite(out.alpha)) { out.status = "invalid_regularization"; return out; }
  Eigen::Matrix<double, 6, 1> weights = Eigen::Matrix<double, 6, 1>::Ones();
  if (use_directional_weight && largest > 0.0)
    weights = (settings.regularization_ratio /
               (values.array() / largest + settings.regularization_ratio)).matrix();
  const PoseMatrix directions = whitening * eigen.eigenvectors();
  out.weight = directions * weights.asDiagonal() * directions.transpose();
  out.lambda_min = values.minCoeff();
  out.lambda_max = largest;
  out.valid = out.weight.allFinite();
  out.status = out.valid ? "ok" : "nonfinite_weight";
  if (!out.valid) out.weight.setZero();
  return out;
}

inline bool fresh(const GeometrySummary &summary, double stamp)
{
  return summary.valid && summary.stamp >= 0.0 && std::isfinite(stamp) && std::isfinite(summary.stamp) &&
         stamp - summary.stamp >= -0.05 && stamp - summary.stamp <= 0.5;
}

inline PoseProjection poseProjection(const Eigen::Matrix<double, 2, 3> &projection,
                                     const Eigen::Vector3d &point_camera,
                                     const Eigen::Matrix3d &rotation_derivative,
                                     const Eigen::Matrix3d &origin_rotation_derivative,
                                     const Eigen::Matrix3d &translation_derivative)
{
  PoseProjection jacobian;
  jacobian.leftCols<3>() = projection *
      (crossMatrix(point_camera) * rotation_derivative - origin_rotation_derivative);
  jacobian.rightCols<3>() = -projection * translation_derivative;
  return jacobian;
}

// Same 8x8 central differences and image accesses as the existing Shi-Tomasi
// scan. G = mean(g g^T), where g uses the 1/2 central-difference factor.
// Stride is in bytes; no OpenCV dependency is needed by the math tests.
inline Eigen::Matrix2d gradientTensor8x8(const std::uint8_t *data, size_t stride,
                                        int width, int height, int u, int v)
{
  Eigen::Matrix2d tensor = Eigen::Matrix2d::Zero();
  if (u - 4 < 1 || u + 4 >= width - 1 || v - 4 < 1 || v + 4 >= height - 1) return tensor;
  float xx = 0, yy = 0, xy = 0;
  for (int y = v - 4; y < v + 4; ++y)
  {
    const auto *left = data + y * stride + u - 5;
    const auto *right = data + y * stride + u - 3;
    const auto *top = data + (y - 1) * stride + u - 4;
    const auto *bottom = data + (y + 1) * stride + u - 4;
    for (int x = 0; x < 8; ++x)
    {
      const float dx = right[x] - left[x], dy = bottom[x] - top[x];
      xx += dx * dx; yy += dy * dy; xy += dx * dy;
    }
  }
  tensor << xx / 256.0, xy / 256.0, xy / 256.0, yy / 256.0;
  return tensor;
}

// The pose Jacobian is unscaled; weight maps metric-normalized modes back to
// its original coordinates. Do not multiply this Jacobian by a separate D.
// This is photometric sensitivity, not independent posterior information.
inline double sensitivity(const Eigen::Matrix2d &tensor, const PoseProjection &projection,
                           const PoseMatrix &weight, double inverse_exposure,
                           double fixed_noise_variance)
{
  if (!std::isfinite(fixed_noise_variance) || fixed_noise_variance <= 0) return 0.0;
  const double value = inverse_exposure * inverse_exposure / fixed_noise_variance *
                      (tensor * (projection * weight * projection.transpose())).trace();
  return std::isfinite(value) ? std::max(0.0, value) : 0.0;
}
} // namespace visual_patch_score
