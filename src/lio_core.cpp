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

  // 3. Build the voxel-map configuration.
  VoxelMapConfig map_cfg;
  map_cfg.max_layer_ = config.map.max_layer;
  map_cfg.max_voxel_size_ = config.map.voxel_size;
  map_cfg.planner_threshold_ = config.map.min_eigen_value;
  map_cfg.sigma_num_ = config.map.sigma_num;
  map_cfg.beam_err_ = config.map.beam_err;
  map_cfg.dept_err_ = config.map.dept_err;
  map_cfg.layer_init_num_.assign(config.map.layer_init_num.begin(), config.map.layer_init_num.end());
  map_cfg.max_points_num_ = config.map.max_points_num;
  map_cfg.max_iterations_ = config.map.min_iterations;
  map_cfg.map_sliding_en = config.map.sliding_enable;
  map_cfg.half_map_size = config.map.half_map_size;
  map_cfg.sliding_thresh = config.map.sliding_thresh;
  map_cfg.is_pub_plane_map_ = false;

  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  voxel_manager_.reset(new VoxelMapManager(map_cfg, voxel_map));
  voxel_manager_->extT_ = extT_;
  voxel_manager_->extR_ = extR_;

  // 4. Configure body-frame point-cloud downsampling resolution [m].
  downsample_filter_.setLeafSize(config.lidar.filter_size_surf, config.lidar.filter_size_surf, config.lidar.filter_size_surf);
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
  voxel_manager_->state_ = state_;
  voxel_manager_->feats_undistort_ = feats_undistort_;
}

// 下采样并生成世界系点云用于匹配。
void LioCore::Downsample()
{
  // 1. 先在 LiDAR/body 系下做体素下采样。
  downsample_filter_.setInputCloud(feats_undistort_);
  downsample_filter_.filter(*feats_down_body_);
  voxel_manager_->feats_down_body_ = feats_down_body_;

  // 2. 再把下采样结果变换到世界系，供地图匹配与更新使用。
  voxel_manager_->TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, feats_down_world_);
  voxel_manager_->feats_down_world_ = feats_down_world_;
  voxel_manager_->feats_down_size_ = feats_down_body_->points.size();
}

// 地图匹配与状态更新的核心流程。
void LioCore::ProcessLio()
{
  // 没有点云就直接返回，避免后续匹配器处理空输入。
  if (!feats_undistort_ || feats_undistort_->empty()) {
    return;
  }

  Downsample();

  // 第一帧只负责建图，不进行常规匹配更新。
  if (!map_inited_) {
    map_inited_ = true;
    voxel_manager_->BuildVoxelMap();
  }

  // 常规流程：状态估计 -> 回写状态 -> 地图更新 -> 视需要滑动地图。
  voxel_manager_->StateEstimation(state_propagat_);
  state_ = voxel_manager_->state_;
  voxel_manager_->UpdateVoxelMap(voxel_manager_->pv_list_);

  if (voxel_manager_->config_setting_.map_sliding_en) {
    voxel_manager_->mapSliding();
  }
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
  if (voxel_manager_) {
    voxel_manager_->state_ = state_;
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
  return voxel_manager_->ptpl_list_;
}

} // namespace cake_slam
