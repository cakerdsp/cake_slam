#ifndef CAKE_SLAM_COLOR_GRAY_INITIALIZER_H
#define CAKE_SLAM_COLOR_GRAY_INITIALIZER_H

#include <Eigen/Core>
#include <atomic>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace color_gray_init
{

// The caller owns immutable image copies and camera-model lifetimes. Callbacks
// must not capture a live estimator, PerCameraData, or map by reference.
struct CameraSnapshot
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  cv::Mat bgr;
  Eigen::Matrix3d R_c_w = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_c_w = Eigen::Vector3d::Zero();
  std::function<bool(const Eigen::Vector3d &, Eigen::Vector2d &)> project;
  std::function<Eigen::Vector3d(const Eigen::Vector2d &)> unproject;
  double virtual_focal = 300.0;
};

struct PlaneSeed
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_w = Eigen::Vector3d::Zero();
  // Estimated planar candidate support in metres, centred at point_w. This
  // statistical extent is not a visibility or complete-surface guarantee.
  double radius = 0.0;
};

struct Snapshot
{
  std::vector<CameraSnapshot> cameras;
  std::vector<PlaneSeed> planes;
};

struct CameraReport
{
  bool calibrated = false;
  int parent_camera = -1;
  int regions = 0;
  int samples = 0;
  double validation_rmse = std::numeric_limits<double>::quiet_NaN();
  double baseline_rmse = std::numeric_limits<double>::quiet_NaN();
  std::string reason;
};

struct Result
{
  bool accepted = false;
  // Camera indices are never compacted. No partial result may be installed.
  std::vector<Eigen::Vector3d> weights;
  std::vector<CameraReport> cameras;
  std::string reason;
};

Result estimate(const Snapshot &snapshot, const std::atomic<bool> &cancel);

} // namespace color_gray_init

#endif
