#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "cake_slam/common_lib.h"
#include "cake_slam/config.h"
#include "cake_slam/voxel_map.h"

namespace cake_slam {

/**
 * @brief Owns the local voxel map independently from the LIO frontend.
 *
 * The first version keeps the FAST-LIVO2 VoxelMapManager implementation intact,
 * but moves ownership and all map access behind this class. LIO can query and
 * update it through explicit methods, while future local BA code can reuse the
 * same map object without reaching into LioCore.
 */
class LocalVoxelMap
{
public:
  struct RaycastOptions
  {
    double min_depth = 0.2;
    double max_depth = 80.0;
    double step = 0.5;
    double min_cos = 0.15;
    double plane_radius_scale = 3.0;
    int max_steps = 160;
    bool allow_point_fallback = true;
  };

  struct RaycastHit
  {
    bool valid = false;
    Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_w = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 6> plane_var = Eigen::Matrix<double, 6, 6>::Zero();
    double range = -1.0;
    double plane_var_scalar = 0.0;
    double incidence_cos = 0.0;
    bool from_plane = false;
  };

  LocalVoxelMap() = default;

  void Configure(const Config &config, const M3D &lidar_to_body_R, const V3D &lidar_to_body_t);
  bool IsConfigured() const { return configured_; }

  void SetFrameState(const StatesGroup &state);
  void SetUndistortedCloud(const PointCloudXYZI::Ptr &cloud);
  void SetDownsampledClouds(const PointCloudXYZI::Ptr &body_cloud,
                            const PointCloudXYZI::Ptr &world_cloud);

  void TransformLidar(const M3D &rot, const V3D &pos,
                      const PointCloudXYZI::Ptr &input_cloud,
                      PointCloudXYZI::Ptr &world_cloud);

  void BuildInitialMap();
  void EstimateState(StatesGroup &state_propagat);
  const StatesGroup &State() const;

  void UpdateFromLatestFrame();
  void UpdateWithPoints(const std::vector<pointWithVar> &points);
  void SlideIfNeeded();
  bool SlidingEnabled() const;

  void BuildRegistrationResiduals(std::vector<pointWithVar> &points,
                                  std::vector<PointToPlane> &residuals);
  const std::vector<PointToPlane> &LatestResiduals() const;

  const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &VoxelMap() const;
  double VoxelSize() const;
  bool Raycast(const Eigen::Vector3d &origin_w,
               const Eigen::Vector3d &direction_w,
               const RaycastOptions &options,
               RaycastHit &hit) const;

private:
  VoxelMapManagerPtr voxel_manager_;
  bool configured_ = false;
};

using LocalVoxelMapPtr = std::shared_ptr<LocalVoxelMap>;

} // namespace cake_slam
