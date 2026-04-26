#include "cake_slam/lio_core.h"

#include <cmath>

namespace cake_slam {

LioCore::LioCore()
{
  preprocess_.reset(new Preprocess());
  imu_proc_.reset(new ImuProcess());
  feats_undistort_.reset(new PointCloudXYZI());
  feats_down_body_.reset(new PointCloudXYZI());
  feats_down_world_.reset(new PointCloudXYZI());
}

void LioCore::Configure(const Config &config)
{
  preprocess_->set(config.lidar.feature_extract, config.lidar.type, config.lidar.blind, config.lidar.point_filter_num);
  preprocess_->N_SCANS = config.lidar.scan_line;
  preprocess_->SCAN_RATE = config.lidar.scan_rate;
  preprocess_->blind = config.lidar.blind;
  preprocess_->blind_sqr = config.lidar.blind * config.lidar.blind;

  if (!config.extrinsic.lidar_T.empty()) {
    extT_ << config.extrinsic.lidar_T[0], config.extrinsic.lidar_T[1], config.extrinsic.lidar_T[2];
  }
  if (!config.extrinsic.lidar_R.empty()) {
    extR_ << config.extrinsic.lidar_R[0], config.extrinsic.lidar_R[1], config.extrinsic.lidar_R[2],
             config.extrinsic.lidar_R[3], config.extrinsic.lidar_R[4], config.extrinsic.lidar_R[5],
             config.extrinsic.lidar_R[6], config.extrinsic.lidar_R[7], config.extrinsic.lidar_R[8];
  }

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

  downsample_filter_.setLeafSize(config.lidar.filter_size_surf, config.lidar.filter_size_surf, config.lidar.filter_size_surf);
}

void LioCore::ProcessMeasurement(LidarMeasureGroup &meas)
{
  meas_ = meas;
  meas_.lio_vio_flg = LIO;
  if (meas_.measures.empty()) {
    return;
  }
  ProcessImu(meas_);
  ProcessLio();
}

void LioCore::ProcessImu(LidarMeasureGroup &meas)
{
  imu_proc_->Process2(meas, state_, feats_undistort_);
  state_propagat_ = state_;
  voxel_manager_->state_ = state_;
  voxel_manager_->feats_undistort_ = feats_undistort_;
}

void LioCore::Downsample()
{
  downsample_filter_.setInputCloud(feats_undistort_);
  downsample_filter_.filter(*feats_down_body_);
  voxel_manager_->feats_down_body_ = feats_down_body_;

  voxel_manager_->TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, feats_down_world_);
  voxel_manager_->feats_down_world_ = feats_down_world_;
  voxel_manager_->feats_down_size_ = feats_down_body_->points.size();
}

void LioCore::ProcessLio()
{
  if (!feats_undistort_ || feats_undistort_->empty()) {
    return;
  }

  Downsample();

  if (!map_inited_) {
    map_inited_ = true;
    voxel_manager_->BuildVoxelMap();
  }

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

PointCloudXYZI::Ptr LioCore::GetUndistortedCloud() const
{
  return feats_undistort_;
}

PointCloudXYZI::Ptr LioCore::GetDownsampledCloud() const
{
  return feats_down_body_;
}

} // namespace cake_slam
