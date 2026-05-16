#pragma once

// 模块功能：LIO 核心处理接口，组织 IMU 传播、点云去畸变、
// 体素地图匹配与状态更新等关键流程。

#include <memory>
#include <vector>

#include <pcl/filters/voxel_grid.h>

#include "cake_slam/common_lib.h"
#include "cake_slam/config.h"
#include "cake_slam/imu_processor.h"
#include "cake_slam/voxel_map.h"

namespace cake_slam {

/**
 * @brief FAST-LIVO2-style LiDAR-inertial odometry core.
 *
 * Inputs are already synchronized LiDAR/IMU packets from SlamNode. The class
 * owns IMU propagation/undistortion, voxel-map registration, downsampling, and
 * the latest IESKF state/covariance. It intentionally does not subscribe to ROS
 * topics or convert ROS messages.
 */
class LioCore
{
public:
  /** @brief Allocate processing modules and point-cloud buffers. */
  LioCore();

  /**
   * @brief Configure IMU propagation, LiDAR-body extrinsics, voxel map, and filters.
   * @param config Unified YAML configuration. Translations are [m].
   */
  void Configure(const Config &config);

  /**
   * @brief Run one LIO update on a synchronized LiDAR/IMU packet.
   * @param meas Point cloud in LiDAR/body frame [m] plus IMU samples [m/s^2, rad/s].
   */
  void ProcessMeasurement(FusionMeasureGroup &meas);

  /** @brief Return the latest IESKF state and 19x19 covariance. */
  const StatesGroup &GetState() const;
  /** @brief Synchronize the LIO state from the fused main state. */
  void SetState(const StatesGroup &state);
  /** @brief Return the undistorted point cloud [m], LiDAR/body frame. */
  PointCloudXYZI::Ptr GetUndistortedCloud() const;
  /** @brief Return the downsampled point cloud [m], LiDAR/body frame. */
  PointCloudXYZI::Ptr GetDownsampledCloud() const;
  /** @brief Return the downsampled point cloud [m], world frame. */
  PointCloudXYZI::Ptr GetDownsampledWorldCloud() const;
  /** @brief Return point-to-plane residual inputs used by the latest LIO solve. */
  const std::vector<PointToPlane> &GetEffectPoints() const;

private:
  /** @brief Propagate IMU and undistort the current LiDAR scan. */
  void ProcessImu(FusionMeasureGroup &meas);
  /** @brief Run voxel-map registration and map update. */
  void ProcessLio();
  /** @brief Downsample undistorted points and transform them to world frame. */
  void Downsample();

  // IMU 初始化、传播和去畸变模块。
  ImuProcessPtr imu_proc_;
  // 体素地图构建、配准和滑窗管理模块。
  VoxelMapManagerPtr voxel_manager_;

  // 当前帧优化后的状态。
  StatesGroup state_;
  // 当前帧仅通过传播得到的预测状态，用作迭代初值。
  StatesGroup state_propagat_;
  // LiDAR 到 IMU/body 的平移外参。
  V3D extT_ = V3D::Zero();
  // LiDAR 到 IMU/body 的旋转外参。
  M3D extR_ = M3D::Identity();

  // 地图是否已用第一帧数据完成初始化。
  bool map_inited_ = false;

  // PCL 体素下采样滤波器。
  pcl::VoxelGrid<PointType> downsample_filter_;

  // 当前缓存的测量组。
  FusionMeasureGroup meas_;
  // 去畸变后的原始点云。
  PointCloudXYZI::Ptr feats_undistort_;
  // 下采样后、仍位于机体系的点云。
  PointCloudXYZI::Ptr feats_down_body_;
  // 下采样后、转换到世界系的点云。
  PointCloudXYZI::Ptr feats_down_world_;
};

} // namespace cake_slam
