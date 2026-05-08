#pragma once

#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "camodocal/camera_models/CameraFactory.h"

#include "cake_slam/common_lib.h"
#include "cake_slam/config.h"
#include "cake_slam/lidar_visual_types.h"

namespace cake_slam {

/**
 * @brief Projects LiDAR points into the current image and keeps VINS-trackable seeds.
 *
 * Inputs are current-frame LiDAR points [m, lidar frame], the latest body pose
 * T_WB, a mono image, an optional VINS-style valid-domain mask, and cam0 model.
 * Output candidates carry pixel coordinates, inverse-depth priors, and world
 * anchors consumed directly by FeatureTracker/FeatureManager.
 */
class LidarVisualSelector
{
public:
  /** @brief Load visual thresholds and LiDAR covariance model parameters. */
  void Configure(const Config &config);
  /** @brief Set extrinsics R_I_L/t_I_L and R_C_L/t_C_L. Translations are [m]. */
  void SetExtrinsics(const Eigen::Matrix3d &R_I_L, const Eigen::Vector3d &t_I_L,
                     const Eigen::Matrix3d &R_C_L, const Eigen::Vector3d &t_C_L);

  /**
   * @brief Select LiDAR-projected visual seeds for one image.
   * @param cloud_lidar Current LiDAR cloud [m], lidar frame.
   * @param state Latest body pose in world frame.
   * @param gray Mono image used for local Shi-Tomasi score.
   * @param valid_mask Optional CV_8UC1 undistorted valid-domain mask.
   * @param camera cam0 projection model.
   */
  std::vector<LidarVisualCandidate> Select(
      const PointCloudXYZI::Ptr &cloud_lidar,
      const StatesGroup &state,
      const cv::Mat &gray,
      const cv::Mat &valid_mask,
      const camodocal::CameraConstPtr &camera);

  const LidarVisualSelectorStats &lastStats() const { return last_stats_; }

private:
  struct ProjectedCandidate
  {
    LidarVisualCandidate candidate;
  };

  bool inValidDomain(const cv::Mat &valid_mask, const cv::Point2f &pixel) const;
  bool textureAccepted(const cv::Mat &gray, const cv::Point2f &pixel, double *score) const;
  bool maskAccepts(const std::vector<LidarVisualCandidate> &selected, const cv::Point2f &pixel) const;
  double inverseDepthVariance(const Eigen::Matrix3d &cov_lidar, double depth) const;

  VisionConfig vision_;
  double lidar_range_noise_ = 0.05;
  double lidar_beam_noise_ = 0.02;
  Eigen::Matrix3d R_I_L_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_I_L_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_C_L_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_C_L_ = Eigen::Vector3d::Zero();
  LidarVisualSelectorStats last_stats_;
};

} // namespace cake_slam
