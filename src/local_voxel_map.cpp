#include "cake_slam/local_voxel_map.h"

namespace cake_slam {

void LocalVoxelMap::Configure(const Config &config, const M3D &lidar_to_body_R, const V3D &lidar_to_body_t)
{
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
  voxel_manager_->extR_ = lidar_to_body_R;
  voxel_manager_->extT_ = lidar_to_body_t;
  configured_ = true;
}

void LocalVoxelMap::SetFrameState(const StatesGroup &state)
{
  if (voxel_manager_) {
    voxel_manager_->state_ = state;
  }
}

void LocalVoxelMap::SetUndistortedCloud(const PointCloudXYZI::Ptr &cloud)
{
  if (voxel_manager_) {
    voxel_manager_->feats_undistort_ = cloud;
  }
}

void LocalVoxelMap::SetDownsampledClouds(const PointCloudXYZI::Ptr &body_cloud,
                                         const PointCloudXYZI::Ptr &world_cloud)
{
  if (!voxel_manager_) {
    return;
  }
  voxel_manager_->feats_down_body_ = body_cloud;
  voxel_manager_->feats_down_world_ = world_cloud;
  voxel_manager_->feats_down_size_ = body_cloud ? static_cast<int>(body_cloud->points.size()) : 0;
}

void LocalVoxelMap::TransformLidar(const M3D &rot, const V3D &pos,
                                   const PointCloudXYZI::Ptr &input_cloud,
                                   PointCloudXYZI::Ptr &world_cloud)
{
  if (voxel_manager_) {
    voxel_manager_->TransformLidar(rot, pos, input_cloud, world_cloud);
  }
}

void LocalVoxelMap::BuildInitialMap()
{
  if (voxel_manager_) {
    voxel_manager_->BuildVoxelMap();
  }
}

void LocalVoxelMap::EstimateState(StatesGroup &state_propagat)
{
  if (voxel_manager_) {
    voxel_manager_->StateEstimation(state_propagat);
  }
}

const StatesGroup &LocalVoxelMap::State() const
{
  static const StatesGroup empty_state;
  return voxel_manager_ ? voxel_manager_->state_ : empty_state;
}

void LocalVoxelMap::UpdateFromLatestFrame()
{
  if (voxel_manager_) {
    UpdateWithPoints(voxel_manager_->pv_list_);
  }
}

void LocalVoxelMap::UpdateWithPoints(const std::vector<pointWithVar> &points)
{
  if (voxel_manager_) {
    voxel_manager_->UpdateVoxelMap(points);
  }
}

void LocalVoxelMap::SlideIfNeeded()
{
  if (voxel_manager_ && SlidingEnabled()) {
    voxel_manager_->mapSliding();
  }
}

bool LocalVoxelMap::SlidingEnabled() const
{
  return voxel_manager_ && voxel_manager_->config_setting_.map_sliding_en;
}

void LocalVoxelMap::BuildRegistrationResiduals(std::vector<pointWithVar> &points,
                                               std::vector<PointToPlane> &residuals)
{
  if (voxel_manager_) {
    voxel_manager_->BuildResidualListOMP(points, residuals);
  } else {
    residuals.clear();
  }
}

const std::vector<PointToPlane> &LocalVoxelMap::LatestResiduals() const
{
  static const std::vector<PointToPlane> empty_residuals;
  return voxel_manager_ ? voxel_manager_->ptpl_list_ : empty_residuals;
}

const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &LocalVoxelMap::VoxelMap() const
{
  static const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> empty_map;
  return voxel_manager_ ? voxel_manager_->voxel_map_ : empty_map;
}

} // namespace cake_slam
