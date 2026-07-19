/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIVO_FEATURE_FISHEYE_H_
#define LIVO_FEATURE_FISHEYE_H_

#include "visual_point.h"

// A salient image region tracked using either a raw image patch or a local
// virtual pinhole patch. The existing raw-camera fields keep their semantics.
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType
  {
    CORNER,
    EDGELET
  };
  int id_;
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
  SE3d T_f_w_;            //!< Raw fisheye pose: {}^C T_W.
  float *patch_; //!< Immutable core patch extracted once when this Feature is created.
                 //!< Length is always patch_size_total.
  float score_;
  float mean_;
  double inv_expo_time_;

  // Local virtual pinhole camera metadata. The virtual camera shares the raw
  // camera optical center, so {}^V T_C contains rotation only.
  SE3d T_v_w_;            //!< Virtual pinhole pose: {}^V T_W.
  Matrix3d R_v_from_c_;          //!< {}^V R_C.
  Matrix3d R_c_from_v_;          //!< {}^C R_V = ({}^V R_C)^T.
  bool virtual_patch_valid_;
  
  Feature(VisualPoint *_point, float *_patch, const Vector2d &_px, const Vector3d &_f, const SE3d &_T_f_w, int _level)
      : type_(CORNER), px_(_px), f_(_f), level_(_level), point_(_point), T_f_w_(_T_f_w), patch_(_patch), score_(0), mean_(0),
        inv_expo_time_(0), T_v_w_(_T_f_w), R_v_from_c_(Matrix3d::Identity()), R_c_from_v_(Matrix3d::Identity()),
        virtual_patch_valid_(false)
  {
  }

  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
  
  ~Feature()
  {
    delete[] patch_;
  }
};

#endif // LIVO_FEATURE_FISHEYE_H_
