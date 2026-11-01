#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace visual_patch_score
{
struct GeometrySample
{
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  double distance = std::numeric_limits<double>::infinity();
  bool valid = false;
};

// Called only during the primary voxel traversal, before covariance gating.
// A geometric distance cutoff bounds the proxy's association error; no P or
// P-gated neighbor fallback is used to select these normals.
inline void consider(GeometrySample *sample, const Eigen::Vector3d &normal, double distance)
{
  if (!sample || !normal.allFinite() || !std::isfinite(distance) || distance < 0 ||
      distance >= 0.3 || distance >= sample->distance) return;
  const double length = normal.norm();
  if (length < 1.0e-8) return;
  sample->normal = normal / length;
  sample->distance = distance;
  sample->valid = true;
}

struct GeometrySummary
{
  Eigen::Matrix3d weight = Eigen::Matrix3d::Identity();
  double stamp = -1.0;
  double aggregate_seconds = 0.0;
  int count = 0;
  bool valid = false;
};

inline GeometrySummary summarize(const std::vector<GeometrySample> &samples)
{
  GeometrySummary out;
  Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
  for (const auto &sample : samples)
    if (sample.valid) { normal_matrix.noalias() += sample.normal * sample.normal.transpose(); ++out.count; }
  if (out.count < 30) return out;
  normal_matrix /= out.count;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(normal_matrix);
  if (eigen.info() != Eigen::Success || !eigen.eigenvalues().allFinite()) return out;
  // alpha * (mean(n n^T) + alpha I)^-1, fixed alpha=0.1.
  const Eigen::Vector3d weights = (0.1 / (eigen.eigenvalues().array().max(0.0) + 0.1)).matrix();
  out.weight = eigen.eigenvectors() * weights.asDiagonal() * eigen.eigenvectors().transpose();
  out.valid = true;
  return out;
}

inline bool fresh(const GeometrySummary &summary, double stamp)
{
  return summary.valid && std::isfinite(stamp) && std::isfinite(summary.stamp) &&
         stamp - summary.stamp >= -0.05 && stamp - summary.stamp <= 0.5;
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

// W is expressed in camera coordinates. This is translational photometric
// sensitivity, not success probability or independent posterior information.
inline double sensitivity(const Eigen::Matrix2d &tensor, const Eigen::Matrix<double, 2, 3> &projection,
                           const Eigen::Matrix3d &weight_camera, double inverse_exposure,
                           double fixed_noise_variance)
{
  if (!std::isfinite(fixed_noise_variance) || fixed_noise_variance <= 0) return 0.0;
  const double value = inverse_exposure * inverse_exposure / fixed_noise_variance *
                      (tensor * (projection * weight_camera * projection.transpose())).trace();
  return std::isfinite(value) ? std::max(0.0, value) : 0.0;
}
} // namespace visual_patch_score
