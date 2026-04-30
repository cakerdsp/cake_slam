#pragma once

#include <memory>
#include <vector>

#include <pcl/filters/voxel_grid.h>

#include "cake_slam/common_lib.h"
#include "cake_slam/config.h"
#include "cake_slam/imu_processor.h"
#include "cake_slam/lidar_preprocess.h"
#include "cake_slam/voxel_map.h"

namespace cake_slam {

class LioCore
{
public:
  // 构造函数：只完成对象分配，不读取配置，也不启动处理流程。
  LioCore();

  // 根据统一配置初始化 LiDAR 预处理、IMU 处理器和体素地图管理器。
  // 输入要求：
  // 1. config 中的外参长度需与约定匹配；
  // 2. lidar / imu / map 字段应已由上层正确加载。
  void Configure(const Config &config);

  // 处理一组已经完成时间同步的 LiDAR + IMU 测量。
  // 输入要求：
  // 1. meas.measures 不能为空；
  // 2. 点云逐点时间应已在预处理环节保留下来，以便去畸变；
  // 3. IMU 队列必须按时间升序排列。
  void ProcessMeasurement(FusionMeasureGroup &meas);

  // 返回 LIO 内部的完整状态。
  const StatesGroup &GetState() const;
  // 用外部主状态覆盖 LIO 内部状态，供 LIO/VIO 顺序融合时同步初值。
  void SetState(const StatesGroup &state);
  // 返回去畸变后的点云，坐标系为 LiDAR/body 系。
  PointCloudXYZI::Ptr GetUndistortedCloud() const;
  // 返回下采样后的点云，通常用于后续配准。
  PointCloudXYZI::Ptr GetDownsampledCloud() const;
  // 返回下采样后的世界系点云。
  PointCloudXYZI::Ptr GetDownsampledWorldCloud() const;
  // 返回本次 LIO 中参与点面残差的有效点。
  const std::vector<PointToPlane> &GetEffectPoints() const;

private:
  // 使用 IMU 对当前点云进行传播和去畸变。
  void ProcessImu(FusionMeasureGroup &meas);
  // 执行基于体素地图的状态估计和地图更新。
  void ProcessLio();
  // 对去畸变点云做体素下采样，并同步生成世界系点云。
  void Downsample();

  // LiDAR 原始点云预处理器。
  PreprocessPtr preprocess_;
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
