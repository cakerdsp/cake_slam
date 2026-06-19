/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

This alternative translation unit keeps the original Frame API while using
the extended Feature definition required by vio_fisheye.cpp. The original
src/frame.cpp is intentionally unchanged.
*/

#include <algorithm>
#include <boost/bind/bind.hpp>
#include "feature_multi_cam.h"
#include "frame_multi_cam.h"
#include "visual_point_multi_cam.h"
#include <stdexcept>
#include <vikit/math_utils.h>
#include <vikit/performance_monitor.h>
#include <vikit/vision.h>

int Frame::frame_counter_ = 0;

Frame::Frame(vk::AbstractCamera *cam, const cv::Mat &img, int frame_id, int camera_id, double timestamp)
    : id_(frame_id >= 0 ? frame_id : frame_counter_++),
      camera_id_(camera_id),
      timestamp_(timestamp),
      cam_(cam)
{
  initFrame(img);
}

Frame::~Frame()
{
  std::for_each(fts_.begin(), fts_.end(), [&](Feature *i) { delete i; });
}

void Frame::initFrame(const cv::Mat &img)
{
  if (img.empty()) throw std::runtime_error("Frame: provided image is empty");
  if (img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model");
  if (img.type() != CV_8UC1) throw std::runtime_error("Frame: provided image is not grayscale");
  img_ = img;
}

namespace frame_utils
{

void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for (int i = 1; i < n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i - 1].rows / 2, pyr[i - 1].cols / 2, CV_8U);
    vk::halfSample(pyr[i - 1], pyr[i]);
  }
}

} // namespace frame_utils
