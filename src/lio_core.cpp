// 模块功能：LIO 核心流程实现，
// 组织 IMU 传播、点云处理与地图匹配更新。

#include "cake_slam/lio_core.h"

#include <cmath>

namespace cake_slam {

// 构造函数只做对象与缓存分配，避免把配置依赖硬编码在构造阶段。
LioCore::LioCore()
{
  imu_proc_.reset(new ImuProcess());
  feats_undistort_.reset(new PointCloudXYZI());
  feats_down_body_.reset(new PointCloudXYZI());
  feats_down_world_.reset(new PointCloudXYZI());
}

// 配置 LIO 核心模块的外参与地图/IMU参数。
void LioCore::Configure(const Config &config)
{
  // 1. Read LiDAR-to-body(IMU) extrinsics, matching the FAST-LIVO2 convention.
  if (config.extrinsic.lidar_T.size() >= 3) {
    extT_ << config.extrinsic.lidar_T[0], config.extrinsic.lidar_T[1], config.extrinsic.lidar_T[2];
  }
  if (config.extrinsic.lidar_R.size() >= 9) {
    extR_ << config.extrinsic.lidar_R[0], config.extrinsic.lidar_R[1], config.extrinsic.lidar_R[2],
             config.extrinsic.lidar_R[3], config.extrinsic.lidar_R[4], config.extrinsic.lidar_R[5],
             config.extrinsic.lidar_R[6], config.extrinsic.lidar_R[7], config.extrinsic.lidar_R[8];
  }

  // 2. Configure IMU propagation and initialization policy.
  imu_proc_->set_extrinsic(extT_, extR_);
  imu_proc_->set_gyr_cov_scale(V3D(config.imu.gyr_cov, config.imu.gyr_cov, config.imu.gyr_cov));
  imu_proc_->set_acc_cov_scale(V3D(config.imu.acc_cov, config.imu.acc_cov, config.imu.acc_cov));
  imu_proc_->set_imu_init_frame_num(config.imu.imu_int_frame);
  if (!config.imu.enable) {
    imu_proc_->disable_imu();
  }
  if (!config.imu.gravity_est) {
    imu_proc_->disable_gravity_est();
  }
  if (!config.imu.bias_est) {
    imu_proc_->disable_bias_est();
  }

  // 3. Ensure a local voxel map exists. SlamNode normally owns and injects it,
  // but this fallback keeps LioCore usable in isolation.
  if (!local_map_) {
    local_map_.reset(new LocalVoxelMap());
  }
  if (!local_map_->IsConfigured()) {
    local_map_->Configure(config, extR_, extT_);
  }

  // 4. Configure body-frame point-cloud downsampling resolution [m].
  downsample_filter_.setLeafSize(config.lidar.filter_size_surf, config.lidar.filter_size_surf, config.lidar.filter_size_surf);
}

void LioCore::SetLocalMap(const LocalVoxelMapPtr &local_map)
{
  local_map_ = local_map;
}

// LIO 主入口：处理一次同步测量包并更新状态。
void LioCore::ProcessMeasurement(FusionMeasureGroup &meas)
{
  // 当前函数是 LIO 的主入口：
  // 1. 缓存测量；
  // 2. 标记为 LIO 更新；
  // 3. 先做 IMU 去畸变，再做地图匹配。
  meas_ = meas;
  meas_.lio_vio_flg = LIO;
  if (meas_.measures.empty()) {
    return;
  }
  ProcessImu(meas_);
  ProcessLio();
}

// 运行 IMU 传播与点云去畸变步骤。
void LioCore::ProcessImu(FusionMeasureGroup &meas)
{
  // Process2 会原位更新 state_，并把去畸变后的点云写入 feats_undistort_。
  imu_proc_->Process2(meas, state_, feats_undistort_);
  state_propagat_ = state_;
  if (local_map_) {
    local_map_->SetFrameState(state_);
    local_map_->SetUndistortedCloud(feats_undistort_);
  }
}

// 下采样并生成世界系点云用于匹配。
void LioCore::Downsample()
{
  // 1. 先在 LiDAR/body 系下做体素下采样。
  downsample_filter_.setInputCloud(feats_undistort_);
  downsample_filter_.filter(*feats_down_body_);
  if (!local_map_) {
    return;
  }

  // 2. 再把下采样结果变换到世界系，供地图匹配与更新使用。
  local_map_->TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, feats_down_world_);
  local_map_->SetDownsampledClouds(feats_down_body_, feats_down_world_);
}

// 地图匹配与状态更新的核心流程。
void LioCore::ProcessLio()
{
  // 没有点云就直接返回，避免后续匹配器处理空输入。
  if (!local_map_ || !feats_undistort_ || feats_undistort_->empty()) {
    return;
  }

  Downsample();

  // 第一帧只负责建图，不进行常规匹配更新。
  if (!map_inited_) {
    map_inited_ = true;
    local_map_->BuildInitialMap();
  }

  // 常规流程：状态估计 -> 回写状态 -> 地图更新 -> 视需要滑动地图。
  local_map_->EstimateState(state_propagat_);
  state_ = local_map_->State();
  local_map_->UpdateFromLatestFrame();
  local_map_->SlideIfNeeded();
}

const StatesGroup &LioCore::GetState() const
{
  return state_;
}

bool LioCore::IsImuInitialized() const
{
  return !imu_proc_ || imu_proc_->IsInitialized();
}

void LioCore::SetState(const StatesGroup &state)
{
  state_ = state;
  state_propagat_ = state;
  if (local_map_) {
    local_map_->SetFrameState(state_);
  }
}

PointCloudXYZI::Ptr LioCore::GetUndistortedCloud() const
{
  return feats_undistort_;
}

PointCloudXYZI::Ptr LioCore::GetDownsampledCloud() const
{
  return feats_down_body_;
}

PointCloudXYZI::Ptr LioCore::GetDownsampledWorldCloud() const
{
  return feats_down_world_;
}

const std::vector<PointToPlane> &LioCore::GetEffectPoints() const
{
  static const std::vector<PointToPlane> empty_points;
  return local_map_ ? local_map_->LatestResiduals() : empty_points;
}

} // namespace cake_slam
