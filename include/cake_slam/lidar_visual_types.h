#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace cake_slam {

/**
 * @brief Whitened SE(3) pose prior exported by the LiDAR-inertial IESKF.
 *
 * Physical convention:
 * - R_WB: body/IMU orientation in world frame.
 * - p_WB: body/IMU position in world frame [m].
 * - sqrt_information: residual whitening matrix for [rad, rad, rad, m, m, m].
 */
struct LioPosePrior
{
  bool valid = false;
  double timestamp = 0.0; ///< [s], ROS time.
  Eigen::Matrix3d R_WB = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_WB = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> sqrt_information =
      Eigen::Matrix<double, 6, 6>::Identity();
};

/**
 * @brief LiDAR point projected to the host image and accepted as a visual seed.
 *
 * The candidate is still a frontend object. Only pixel, inverse-depth prior,
 * world anchor, texture score, and mask radius survive into FeatureTracker.
 */
struct LidarVisualCandidate
{
  cv::Point2f pixel = cv::Point2f(-1.0f, -1.0f);
  double depth = -1.0;          ///< [m], current camera frame z.
  double inv_depth = -1.0;      ///< [1/m], VINS inverse-depth parameter initial value.
  double inv_depth_var = 1.0;   ///< [1/m^2], prior variance after 3D covariance propagation.
  Eigen::Vector3d P_W_init = Eigen::Vector3d::Zero(); ///< [m], world frame anchor.
  double shi_tomasi_score = 0.0;
  double mask_radius = 0.0;     ///< [pixel], VINS-style spatial suppression radius.
};

/**
 * @brief Inverse-depth prior carried by a tracked visual feature.
 *
 * The FeatureManager reads only these fields when creating feature depth
 * variables and prior factors.
 */
struct LidarDepthPrior
{
  bool valid = false;
  double depth = -1.0;        ///< [m], host camera frame.
  double inv_depth = -1.0;    ///< [1/m].
  double inv_depth_var = 1.0; ///< [1/m^2].
  Eigen::Vector3d P_W_init = Eigen::Vector3d::Zero(); ///< [m], world frame anchor.
};

/**
 * @brief Frontend selector counters for static diagnostics.
 */
struct LidarVisualSelectorStats
{
  int input_points = 0;
  int positive_depth = 0;
  int in_image = 0;
  int zbuffer_kept = 0;
  int texture_kept = 0;
  int mask_kept = 0;
};

inline LidarDepthPrior MakeDepthPrior(const LidarVisualCandidate &candidate)
{
  LidarDepthPrior prior;
  prior.valid = candidate.depth > 0.0 && candidate.inv_depth > 0.0;
  prior.depth = candidate.depth;
  prior.inv_depth = candidate.inv_depth;
  prior.inv_depth_var = candidate.inv_depth_var;
  prior.P_W_init = candidate.P_W_init;
  return prior;
}

} // namespace cake_slam
