#ifndef CAKE_POSE_AFFINE_WARP_H_
#define CAKE_POSE_AFFINE_WARP_H_

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstddef>

namespace pose_affine_warp
{

using PoseJacobian = Eigen::Matrix<double, 2, 6>;

// Snapshot once per observation. The three world points are intersections of
// the original reference rays with the fixed reference plane. The denominator
// is the original finite-secant span, not a newly estimated patch scale.
struct Geometry
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::array<Eigen::Vector3d, 3> world_points = {
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::Matrix3d frozen_camera_from_body = Eigen::Matrix3d::Identity();
  Eigen::Vector3d frozen_camera_translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d frozen_world_shift = Eigen::Vector3d::Zero();
  Eigen::Matrix3d frozen_virtual_from_camera = Eigen::Matrix3d::Identity();
  double half_span = 4.0;
};

struct Evaluation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix2d affine = Eigen::Matrix2d::Zero();
  std::array<PoseJacobian, 2> column_jacobians = {
      PoseJacobian::Zero(), PoseJacobian::Zero()};
};

inline Eigen::Matrix3d skew(const Eigen::Vector3d &value)
{
  Eigen::Matrix3d result;
  result << 0.0, -value.z(), value.y(),
      value.z(), 0.0, -value.x(),
      -value.y(), value.x(), 0.0;
  return result;
}

// Pose columns are [right SO(3) increment, additive world translation].
// frozen_camera_from_body includes the frozen exposure-time rotation:
// C0 = R_ci * Exp(gyro * dt)^T; frozen_world_shift = velocity * dt.
// All snapshot quantities remain fixed while only R_wb and p_wb vary.
// On failure, output is left unchanged.
inline bool evaluate(const Geometry &geometry, const Eigen::Matrix3d &R_wb,
                     const Eigen::Vector3d &p_wb, double focal,
                     const Eigen::Vector2d &principal, double min_z,
                     Evaluation &output)
{
  if (!R_wb.allFinite() || !p_wb.allFinite() || !principal.allFinite() ||
      !geometry.frozen_camera_from_body.allFinite() ||
      !geometry.frozen_camera_translation.allFinite() ||
      !geometry.frozen_world_shift.allFinite() ||
      !geometry.frozen_virtual_from_camera.allFinite() ||
      !std::isfinite(geometry.half_span) || !(geometry.half_span > 0.0) ||
      !std::isfinite(focal) || !(focal > 0.0) ||
      !std::isfinite(min_z) || min_z < 0.0)
    return false;

  const Eigen::Matrix3d body_from_world = R_wb.transpose();
  const Eigen::Matrix3d virtual_from_body =
      geometry.frozen_virtual_from_camera * geometry.frozen_camera_from_body;
  const Eigen::Matrix3d translation_jacobian =
      -virtual_from_body * body_from_world;
  std::array<Eigen::Vector2d, 3> projections;
  std::array<PoseJacobian, 3> projection_jacobians;

  for (std::size_t index = 0; index < geometry.world_points.size(); ++index)
  {
    if (!geometry.world_points[index].allFinite()) return false;
    const Eigen::Vector3d point_body = body_from_world *
        (geometry.world_points[index] - p_wb - geometry.frozen_world_shift);
    const Eigen::Vector3d point_virtual = geometry.frozen_virtual_from_camera *
        (geometry.frozen_camera_from_body * point_body +
         geometry.frozen_camera_translation);
    if (!point_virtual.allFinite() || !(point_virtual.z() > min_z)) return false;

    const double inverse_z = 1.0 / point_virtual.z();
    const double focal_inverse_z = focal * inverse_z;
    projections[index] = focal_inverse_z * point_virtual.head<2>() + principal;
    Eigen::Matrix<double, 2, 3> projection;
    projection << focal_inverse_z, 0.0,
        -focal_inverse_z * point_virtual.x() * inverse_z,
        0.0, focal_inverse_z,
        -focal_inverse_z * point_virtual.y() * inverse_z;
    Eigen::Matrix<double, 3, 6> point_pose;
    point_pose.leftCols<3>() = virtual_from_body * skew(point_body);
    point_pose.rightCols<3>() = translation_jacobian;
    projection_jacobians[index].noalias() = projection * point_pose;
    if (!projections[index].allFinite() ||
        !projection_jacobians[index].allFinite())
      return false;
  }

  Evaluation result;
  for (std::size_t column = 0; column < result.column_jacobians.size(); ++column)
  {
    result.affine.col(column) =
        (projections[column + 1] - projections[0]) / geometry.half_span;
    result.column_jacobians[column] =
        (projection_jacobians[column + 1] - projection_jacobians[0]) /
        geometry.half_span;
    if (!result.column_jacobians[column].allFinite()) return false;
  }
  if (!result.affine.allFinite()) return false;
  output = result;
  return true;
}

}  // namespace pose_affine_warp

#endif  // CAKE_POSE_AFFINE_WARP_H_
