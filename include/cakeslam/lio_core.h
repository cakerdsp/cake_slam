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
  LioCore();

  void Configure(const Config &config);

  // Process one synchronized lidar+imu measurement group.
  void ProcessMeasurement(LidarMeasureGroup &meas);

  const StatesGroup &GetState() const;
  PointCloudXYZI::Ptr GetUndistortedCloud() const;
  PointCloudXYZI::Ptr GetDownsampledCloud() const;

private:
  void ProcessImu(LidarMeasureGroup &meas);
  void ProcessLio();
  void Downsample();

  PreprocessPtr preprocess_;
  ImuProcessPtr imu_proc_;
  VoxelMapManagerPtr voxel_manager_;

  StatesGroup state_;
  StatesGroup state_propagat_;
  V3D extT_ = V3D::Zero();
  M3D extR_ = M3D::Identity();

  bool map_inited_ = false;

  pcl::VoxelGrid<PointType> downsample_filter_;

  LidarMeasureGroup meas_;
  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;
};

} // namespace cake_slam
