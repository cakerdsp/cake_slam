/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIVO_FRAME_MULTI_CAM_H_
#define LIVO_FRAME_MULTI_CAM_H_

#include <boost/noncopyable.hpp>
#include <vikit/abstract_camera.h>

class VisualPoint;
struct Feature;

typedef list<Feature *> Features;
typedef vector<cv::Mat> ImgPyr;

/// A frame saves the image, the associated features and the estimated pose.
class Frame : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int frame_counter_; //!< Counts the number of created frames. Used to set the unique id.
  int id_;                   //!< Unique id of the frame.
  int camera_id_;            //!< Physical camera that produced this frame.
  double timestamp_;         //!< Current capture timestamp used by this camera frame.
  double raw_timestamp_;     //!< Original sensor header timestamp before online camera time offset.
  double corrected_timestamp_; //!< raw_timestamp_ + time-offset used to enqueue this frame.
  double capture_timestamp_; //!< raw_timestamp_ + current nominal time offset + exposure offset.
  double td_used_;           //!< Time offset used when the frame entered the synchronized queue.
  double exposure_time_offset_;
  double time_offset_delta_; //!< Current nominal td minus td_used_.
  int time_offset_group_;
  vk::AbstractCamera *cam_;  //!< Camera model.
  SE3<double> T_f_w_;                //!< Transform (f)rame from (w)orld.
  SE3<double> T_f_w_prior_;          //!< Transform (f)rame from (w)orld provided by the IMU prior.
  cv::Mat img_;              //!< Image of the frame.
  Features fts_;             //!< List of features in the image.

  Frame(vk::AbstractCamera *cam, const cv::Mat &img, int frame_id = -1, int camera_id = 0,
        double timestamp = 0.0, double raw_timestamp = 0.0, double corrected_timestamp = 0.0,
        double td_used = 0.0, double exposure_time_offset = 0.0, int time_offset_group = 0);
  ~Frame();

  /// Initialize new frame and create image pyramid.
  void initFrame(const cv::Mat &img);

  /// Return number of point observations.
  inline size_t nObs() const { return fts_.size(); }

  /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).
  inline Vector2d w2c(const Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_ * xyz_w); }

  /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c) using the IMU prior pose.
  inline Vector2d w2c_prior(const Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_prior_ * xyz_w); }
  
  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Vector3d c2f(const Vector2d &px) const { return cam_->cam2world(px[0], px[1]); }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

  /// Transforms point coordinates in world-frame (w) to camera-frams (f).
  inline Vector3d w2f(const Vector3d &xyz_w) const { return T_f_w_ * xyz_w; }

  /// Transforms point from frame unit sphere (f) frame to world coordinate frame (w).
  inline Vector3d f2w(const Vector3d &f) const { return T_f_w_.inverse() * f; }

  /// Projects Point from unit sphere (f) in camera pixels (c).
  inline Vector2d f2c(const Vector3d &f) const { return cam_->world2cam(f); }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
};

typedef std::unique_ptr<Frame> FramePtr;

/// Some helper functions for the frame object.
namespace frame_utils
{

/// Creates an image pyramid of half-sampled images.
void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr);

} // namespace frame_utils

#endif // LIVO_FRAME_H_
