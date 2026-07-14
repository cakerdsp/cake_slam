/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIVO_FEATURE_MULTI_CAM_H_
#define LIVO_FEATURE_MULTI_CAM_H_

#include "visual_point_multi_cam.h"
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>

// A salient image region tracked using either a raw image patch or a local
// virtual pinhole patch. The existing raw-camera fields keep their semantics.
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class RefState
  {
    CANDIDATE = 0,
    VALIDATED,
    RETIRED
  };

  enum FeatureType
  {
    CORNER,
    EDGELET
  };
  int id_;
  int camera_id_;
  double timestamp_;
  double raw_timestamp_;
  double corrected_timestamp_;
  double capture_timestamp_;
  double td_used_;
  double exposure_time_offset_;
  int time_offset_group_;
  FeatureType type_;
  cv::Mat img_;  //!< Non-virtual mode: raw reference grayscale image.
                 //!< Virtual fisheye mode: immutable local virtual support image
                 //!< generated once when this Feature is created.
                 //!< Type CV_32FC1, invalid pixels stored as NaN.
  Vector2d px_;
  Vector3d f_;
  int level_;
  VisualPoint *point_;
  Vector2d grad_;
  SE3<double> T_f_w_;            //!< Raw fisheye pose: {}^C T_W.
  M3D Rwi_ref_;                  //!< IMU pose at reference creation, {}^W R_I.
  V3D Pwi_ref_;                  //!< IMU position at reference creation, world frame.
  uint64_t extrinsic_version_;
  float *patch_; //!< Immutable core patch extracted once when this Feature is created.
                 //!< Length is always patch_size_total.
  float score_;
  float mean_;
  double inv_expo_time_;

  // Local virtual pinhole camera metadata. The virtual camera shares the raw
  // camera optical center, so {}^V T_C contains rotation only.
  SE3<double> T_v_w_;            //!< Virtual pinhole pose: {}^V T_W.
  Matrix3d R_v_from_c_;          //!< {}^V R_C.
  Matrix3d R_c_from_v_;          //!< {}^C R_V = ({}^V R_C)^T.
  bool virtual_patch_valid_;
  cv::Mat virtual_source_roi_;   //!< Raw grayscale source retained until lazy support materialization.
  cv::Point virtual_source_origin_;
  bool virtual_support_materialized_;
  bool virtual_support_materialization_failed_;
  mutable std::mutex virtual_support_mutex_;

  RefState ref_state_;
  uint64_t ref_id_;
  int birth_frame_id_;
  int last_test_frame_id_;
  int last_test_camera_id_;
  int last_success_frame_id_;
  Vector3d view_direction_w_;
  double view_range_;
  Eigen::Matrix<double, 6, 6> birth_pose_cov_;
  std::array<Vector3d, 4> footprint_corners_w_;
  bool footprint_valid_;
  int surface_plane_id_;
  uint64_t surface_revision_;
  double nis_ema_;
  double last_nis_;
  double fisher_log_p_sum_;
  int independent_test_count_;
  int accepted_test_count_;
  int rejected_test_count_;
  int consecutive_reject_count_;
  Eigen::Matrix<double, 6, 6> mean_pose_information_;
  bool pending_delete_;
  
  Feature(VisualPoint *_point, float *_patch, const Vector2d &_px, const Vector3d &_f, const SE3<double> &_T_f_w, int _level,
          int _camera_id = 0, double _timestamp = 0.0, double _raw_timestamp = 0.0,
          double _corrected_timestamp = 0.0, double _td_used = 0.0,
          double _exposure_time_offset = 0.0, int _time_offset_group = 0,
          const M3D &_Rwi_ref = M3D::Identity(), const V3D &_Pwi_ref = V3D::Zero(),
          uint64_t _extrinsic_version = 0)
      : id_(-1), camera_id_(_camera_id), timestamp_(_timestamp), raw_timestamp_(_raw_timestamp),
        corrected_timestamp_(_corrected_timestamp), capture_timestamp_(_timestamp), td_used_(_td_used),
        exposure_time_offset_(_exposure_time_offset), time_offset_group_(_time_offset_group),
        type_(CORNER), px_(_px), f_(_f), level_(_level), point_(_point), T_f_w_(_T_f_w),
        Rwi_ref_(_Rwi_ref), Pwi_ref_(_Pwi_ref), extrinsic_version_(_extrinsic_version), patch_(_patch), score_(0), mean_(0),
        inv_expo_time_(0), T_v_w_(_T_f_w), R_v_from_c_(Matrix3d::Identity()), R_c_from_v_(Matrix3d::Identity()),
        virtual_patch_valid_(false), virtual_source_origin_(0, 0), virtual_support_materialized_(false),
        virtual_support_materialization_failed_(false), ref_state_(RefState::VALIDATED), ref_id_(0),
        birth_frame_id_(-1), last_test_frame_id_(-1), last_test_camera_id_(-1),
        last_success_frame_id_(-1),
        view_direction_w_(Vector3d::Zero()), view_range_(0.0),
        birth_pose_cov_(Eigen::Matrix<double, 6, 6>::Zero()), footprint_valid_(false),
        surface_plane_id_(-1), surface_revision_(0), nis_ema_(0.0),
        last_nis_(std::numeric_limits<double>::quiet_NaN()), fisher_log_p_sum_(0.0),
        independent_test_count_(0), accepted_test_count_(0), rejected_test_count_(0),
        consecutive_reject_count_(0),
        mean_pose_information_(Eigen::Matrix<double, 6, 6>::Zero()), pending_delete_(false)
  {
    for (Vector3d &corner : footprint_corners_w_) corner.setZero();
  }

  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
  
  ~Feature()
  {
    delete[] patch_;
  }
};

#endif // LIVO_FEATURE_FISHEYE_H_
