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

private:
  VoxelMapManagerPtr voxel_manager_;
  bool configured_ = false;
};

using LocalVoxelMapPtr = std::shared_ptr<LocalVoxelMap>;

} // namespace cake_slam
