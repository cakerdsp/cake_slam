#include "cake_slam/local_voxel_map.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>

namespace cake_slam {
namespace {

VOXEL_LOCATION voxelKeyFromPoint(const Eigen::Vector3d &point, double voxel_size)
{
  return VOXEL_LOCATION(static_cast<int64_t>(std::floor(point.x() / voxel_size)),
                        static_cast<int64_t>(std::floor(point.y() / voxel_size)),
                        static_cast<int64_t>(std::floor(point.z() / voxel_size)));
}

} // namespace

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

bool LocalVoxelMap::UpdateFromLatestFrameWithState(const StatesGroup &state,
                                                   const PointCloudXYZI::Ptr &world_cloud)
{
  if (!voxel_manager_ || !voxel_manager_->feats_down_body_ ||
      voxel_manager_->feats_down_body_->empty()) {
    return false;
  }

  PointCloudXYZI::Ptr update_world_cloud = world_cloud;
  if (!update_world_cloud) {
    update_world_cloud.reset(new PointCloudXYZI());
  }
  TransformLidar(state.rot_end, state.pos_end, voxel_manager_->feats_down_body_, update_world_cloud);
  voxel_manager_->feats_down_world_ = update_world_cloud;
  voxel_manager_->feats_down_size_ = static_cast<int>(voxel_manager_->feats_down_body_->points.size());

  std::vector<pointWithVar> points;
  points.resize(voxel_manager_->feats_down_body_->points.size());
  const M3D rot_var = state.cov.block<3, 3>(0, 0);
  const M3D t_var = state.cov.block<3, 3>(3, 3);
  for (size_t i = 0; i < voxel_manager_->feats_down_body_->points.size(); ++i) {
    const PointType &point = voxel_manager_->feats_down_body_->points[i];
    V3D point_lidar(point.x, point.y, point.z);
    if (point_lidar.z() == 0.0) {
      point_lidar.z() = 0.001;
    }

    M3D body_cov;
    calcBodyCov(point_lidar,
                voxel_manager_->config_setting_.dept_err_,
                voxel_manager_->config_setting_.beam_err_,
                body_cov);
    const V3D point_body = voxel_manager_->extR_ * point_lidar + voxel_manager_->extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_body);

    pointWithVar &pv = points[i];
    pv.point_b << point.x, point.y, point.z;
    pv.point_w << update_world_cloud->points[i].x,
                  update_world_cloud->points[i].y,
                  update_world_cloud->points[i].z;
    pv.body_var = body_cov;
    pv.var = state.rot_end * body_cov * state.rot_end.transpose() +
             (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
  }

  voxel_manager_->state_ = state;
  voxel_manager_->position_last_ = state.pos_end;
  voxel_manager_->pv_list_.swap(points);
  UpdateWithPoints(voxel_manager_->pv_list_);
  return true;
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

double LocalVoxelMap::VoxelSize() const
{
  return voxel_manager_ ? voxel_manager_->config_setting_.max_voxel_size_ : 0.5;
}

bool LocalVoxelMap::Raycast(const Eigen::Vector3d &origin_w,
                            const Eigen::Vector3d &direction_w,
                            const RaycastOptions &options,
                            RaycastHit &hit) const
{
  hit = RaycastHit();
  if (!voxel_manager_ || voxel_manager_->voxel_map_.empty() ||
      direction_w.norm() < 1e-9 || options.max_depth <= options.min_depth) {
    return false;
  }

  const double voxel_size = std::max(1e-3, voxel_manager_->config_setting_.max_voxel_size_);
  const double step = std::max(0.1, options.step > 0.0 ? options.step : voxel_size);
  const int max_steps = std::max(1, options.max_steps);
  const Eigen::Vector3d dir = direction_w.normalized();

  VOXEL_LOCATION last_key(std::numeric_limits<int64_t>::max(),
                          std::numeric_limits<int64_t>::max(),
                          std::numeric_limits<int64_t>::max());
  int steps = 0;
  for (double range = std::max(0.0, options.min_depth);
       range <= options.max_depth && steps < max_steps;
       range += step, ++steps) {
    const Eigen::Vector3d probe = origin_w + dir * range;
    const VOXEL_LOCATION key = voxelKeyFromPoint(probe, voxel_size);
    if (key == last_key) {
      continue;
    }
    last_key = key;

    auto it = voxel_manager_->voxel_map_.find(key);
    if (it == voxel_manager_->voxel_map_.end() || !it->second) {
      continue;
    }

    VoxelOctoTree *octo = it->second->find_correspond(probe);
    if (!octo || !octo->plane_ptr_) {
      continue;
    }

    const VoxelPlane &plane = *octo->plane_ptr_;
    if (plane.is_plane_ && plane.is_init_) {
      const double denom = plane.normal_.dot(dir);
      const double incidence = std::abs(denom);
      if (incidence >= options.min_cos) {
        const double hit_range = plane.normal_.dot(plane.center_ - origin_w) / denom;
        if (hit_range >= options.min_depth && hit_range <= options.max_depth) {
          const Eigen::Vector3d hit_point = origin_w + dir * hit_range;
          const Eigen::Vector3d plane_delta = hit_point - plane.center_;
          const Eigen::Vector3d tangent_delta = plane_delta - plane.normal_ * plane.normal_.dot(plane_delta);
          const double accepted_radius =
              std::max(0.25 * voxel_size, options.plane_radius_scale * std::max(plane.radius_, 0.05f));
          if (tangent_delta.norm() <= accepted_radius) {
            hit.valid = true;
            hit.point_w = hit_point;
            hit.normal_w = plane.normal_;
            hit.plane_var = plane.plane_var_;
            hit.range = hit_range;
            hit.plane_var_scalar = std::max(0.0, plane.plane_var_.block<3, 3>(0, 0).trace());
            hit.incidence_cos = incidence;
            hit.from_plane = true;
            return true;
          }
        }
      }
    }

    if (!options.allow_point_fallback || octo->temp_points_.empty()) {
      continue;
    }
    double best_point_range = std::numeric_limits<double>::infinity();
    const pointWithVar *best_point = nullptr;
    const double max_perp = std::max(0.25, 0.5 * voxel_size);
    const int max_points = std::min<int>(static_cast<int>(octo->temp_points_.size()), 16);
    for (int i = 0; i < max_points; ++i) {
      const pointWithVar &pv = octo->temp_points_[i];
      const Eigen::Vector3d delta = pv.point_w - origin_w;
      const double point_range = delta.dot(dir);
      if (point_range < options.min_depth || point_range > options.max_depth ||
          point_range >= best_point_range) {
        continue;
      }
      const double perp = (delta - dir * point_range).norm();
      if (perp <= max_perp) {
        best_point = &pv;
        best_point_range = point_range;
      }
    }
    if (best_point) {
      hit.valid = true;
      hit.point_w = best_point->point_w;
      hit.normal_w = Eigen::Vector3d::Zero();
      hit.range = best_point_range;
      hit.plane_var_scalar = std::max(0.0, best_point->var.trace());
      hit.incidence_cos = 1.0;
      hit.from_plane = false;
      return true;
    }
  }
  return false;
}

} // namespace cake_slam
