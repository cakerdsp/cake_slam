#pragma once

// 模块功能：LiDAR-视觉融合相关数据类型定义，
// 提供位姿先验、候选点与深度先验等结构体以支持前端与因子构建。

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace cake_slam {

class LocalVoxelMap;

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
 * @brief Full body-state prior exported by the LiDAR-inertial IESKF.
 *
 * The state is already in the IMU/body frame and world frame convention used
 * by StatesGroup. Do not apply LiDAR/camera extrinsics before consuming it in
 * the visual backend.
 */
struct LioFullStatePrior
{
  bool valid = false;
  double timestamp = 0.0; ///< [s], ROS time.
  Eigen::Matrix3d R_WB = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_WB = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_WB = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 15, 15> sqrt_information =
      Eigen::Matrix<double, 15, 15>::Identity();
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
  int source = 1;               ///< 0 local voxel map, 1 current scan.
  bool depth_prior_allowed = true; ///< False for occlusion-only projected scan cells.
  bool lio_gate = true;         ///< True for LiDAR-seeded tracks that should use LIO reprojection gating.
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
  int source = 0;             ///< 0 local voxel map, 1 scan projection.
  bool lio_gate = false;      ///< Whether the frontend should gate this track by LIO reprojection.
  double quality = 0.0;       ///< Larger is better, diagnostic only.
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
  int source_mode = 1;
  int occlusion_reject = 0;
  int visual_depth_cells = 0;
  double total_ms = 0.0;
  double map_collect_ms = 0.0;
  double project_ms = 0.0;
  double occlusion_ms = 0.0;
  double texture_ms = 0.0;
  double select_ms = 0.0;
};

struct LidarDepthCell
{
  bool valid = false;
  cv::Point2f pixel = cv::Point2f(-1.0f, -1.0f);
  double depth = -1.0;
  double inv_depth_var = 1.0;
  Eigen::Vector3d P_W_init = Eigen::Vector3d::Zero();
  int source = 0;
  bool depth_prior_allowed = true;
  double quality = 0.0;
};

/**
 * @brief Per-image sparse depth lookup frame for the optical-flow frontend.
 *
 * It stores the projected z-buffer used for cheap pixel-neighborhood depth
 * lookup and, when enabled, enough pose/extrinsic state for voxel raycasting
 * stable visual tracks against LocalVoxelMap.
 */
struct LidarDepthFrame
{
  bool valid = false;
  bool visual_feature_depth_prior_enable = false;
  bool voxel_raycast_enable = false;
  int source_mode = 2; ///< 0 local voxel map, 1 scan, 2 off.
  int rows = 0;
  int cols = 0;
  int cell_size = 8;
  int cell_rows = 0;
  int cell_cols = 0;
  int min_track_cnt = 2;
  int max_depth_update_features = 250;
  int max_raycast_features = 120;
  int max_raycast_steps = 160;
  double max_lidar_depth_ratio = 0.6;
  double min_depth = 0.2;
  double max_depth = 80.0;
  double search_radius = 8.0;
  double z_buffer_depth_tolerance = 0.3;
  double lidar_depth_std = 0.10;
  double min_inv_depth_var = 1e-6;
  double raycast_step = 0.5;
  double raycast_min_cos = 0.15;
  double raycast_plane_radius_scale = 3.0;
  double prior_update_var_ratio = 0.7;

  Eigen::Matrix3d R_WB = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_WB = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_I_C = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_I_C = Eigen::Vector3d::Zero();

  std::vector<LidarDepthCell> cells;
  std::vector<LidarDepthCell> occlusion_cells;
  std::shared_ptr<const LocalVoxelMap> local_map;
};

inline LidarDepthPrior MakeDepthPrior(const LidarVisualCandidate &candidate)
{
  LidarDepthPrior prior;
  prior.valid = candidate.depth > 0.0 && candidate.inv_depth > 0.0;
  prior.depth = candidate.depth;
  prior.inv_depth = candidate.inv_depth;
  prior.inv_depth_var = candidate.inv_depth_var;
  prior.P_W_init = candidate.P_W_init;
  prior.source = candidate.source;
  prior.lio_gate = candidate.lio_gate;
  prior.quality = candidate.shi_tomasi_score;
  return prior;
}

} // namespace cake_slam
