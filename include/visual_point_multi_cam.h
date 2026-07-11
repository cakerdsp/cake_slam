/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIVO_POINT_MULTI_CAM_H_
#define LIVO_POINT_MULTI_CAM_H_

#include <boost/noncopyable.hpp>
#include <cstdint>
#include "common_lib_multi_cam.h"
#include "frame_multi_cam.h"

class Feature;

/// A visual map point on the surface of the scene.
class VisualPoint : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class State
  {
    SEED = 0,
    CONFIRMED,
    SUSPECT,
    RETIRED
  };

  Vector3d pos_;                //!< 3d pos of the point in the world coordinate frame.
  Vector3d normal_;             //!< Surface normal at point.
  Matrix3d normal_information_; //!< Inverse covariance matrix of normal estimation.
  Vector3d previous_normal_;    //!< Last updated normal vector.
  list<Feature *> obs_;         //!< Reference patches which observe the point.
  Eigen::Matrix3d covariance_;  //!< Covariance of the point.
  bool is_converged_;           //!< True if the point is converged.
  bool is_normal_initialized_;  //!< True if the normal is initialized.
  bool has_ref_patch_;          //!< True if the point has a reference patch.
  Feature *ref_patch;           //!< Reference patch of the point.
  std::vector<Feature *> ref_patch_by_camera_;
  std::vector<uint8_t> has_ref_patch_by_camera_;
  int runtime_support_track_count_ = 0;
  int runtime_support_dump_id_ = -1;
  State state_;
  uint64_t point_id_;
  int64_t map_voxel_x_;
  int64_t map_voxel_y_;
  int64_t map_voxel_z_;
  int64_t surface_voxel_x_;
  int64_t surface_voxel_y_;
  int64_t surface_voxel_z_;
  int surface_plane_id_;
  uint64_t surface_revision_;
  bool surface_valid_;
  double geometry_chi2_;
  Eigen::Matrix<double, 6, 6> accumulated_pose_information_;
  std::vector<Vector3d> view_samples_;
  int last_visible_frame_;
  int last_success_frame_;
  int independent_test_count_;
  int accepted_test_count_;
  int rejected_test_count_;
  bool pending_delete_;
  VisualPoint *challenger_of_;

  VisualPoint(const Vector3d &pos);
  ~VisualPoint();
  void findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const;
  void ensureCameraCount(int num_cameras);
  Feature *referencePatch(int camera_id, bool cross_camera_reference) const;
  bool hasUsableReference(int camera_id, bool cross_camera_reference, bool allow_candidate) const;
  void deleteNonRefPatchFeatures();
  void deleteFeatureRef(Feature *ftr);
  void addFrameRef(Feature *ftr);
  bool getCloseViewObs(const Vector3d &pos, Feature *&obs, const Vector2d &cur_px, int camera_id = -1) const;
};

#endif // LIVO_POINT_H_
