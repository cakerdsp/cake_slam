/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

This alternative translation unit uses feature_fisheye.h. The original
src/visual_point.cpp is intentionally unchanged.
*/

#include "visual_point_multi_cam.h"
#include "feature_multi_cam.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vikit/math_utils.h>

VisualPoint::VisualPoint(const Vector3d &pos)
    : pos_(pos), previous_normal_(Vector3d::Zero()), normal_(Vector3d::Zero()),
      is_converged_(false), is_normal_initialized_(false), has_ref_patch_(false), ref_patch(nullptr),
      reuse_camera_count_(0), reuse_selected_count_(0), reuse_last_selected_frame_id_(-1),
      reuse_last_selected_camera_id_(-1), state_(State::CONFIRMED), point_id_(0),
      map_voxel_x_(0), map_voxel_y_(0), map_voxel_z_(0),
      surface_voxel_x_(0), surface_voxel_y_(0), surface_voxel_z_(0), surface_plane_id_(-1),
      surface_revision_(0), surface_valid_(false), geometry_chi2_(0.0),
      accumulated_pose_information_(Eigen::Matrix<double, 6, 6>::Zero()), last_visible_frame_(-1),
      last_success_frame_(-1), independent_test_count_(0), accepted_test_count_(0),
      rejected_test_count_(0), pending_delete_(false), challenger_of_(nullptr)
{
}

VisualPoint::~VisualPoint()
{
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it) delete *it;
  obs_.clear();
  ref_patch = nullptr;
  std::fill(ref_patch_by_camera_.begin(), ref_patch_by_camera_.end(), nullptr);
  std::fill(has_ref_patch_by_camera_.begin(), has_ref_patch_by_camera_.end(), 0);
}

void VisualPoint::ensureCameraCount(int num_cameras)
{
  if (num_cameras < 1) throw std::invalid_argument("VisualPoint requires at least one camera");
  if (static_cast<int>(ref_patch_by_camera_.size()) < num_cameras)
    ref_patch_by_camera_.resize(num_cameras, nullptr);
  if (static_cast<int>(has_ref_patch_by_camera_.size()) < num_cameras)
    has_ref_patch_by_camera_.resize(num_cameras, 0);
  if (static_cast<int>(reuse_camera_observed_.size()) < num_cameras)
    reuse_camera_observed_.resize(num_cameras, 0);
}

Feature *VisualPoint::referencePatch(int camera_id, bool cross_camera_reference) const
{
  if (cross_camera_reference)
    return has_ref_patch_ && ref_patch != nullptr && !ref_patch->pending_delete_ &&
                   ref_patch->ref_state_ != Feature::RefState::RETIRED
               ? ref_patch
               : nullptr;
  if (camera_id < 0 || camera_id >= static_cast<int>(ref_patch_by_camera_.size())) return nullptr;
  Feature *reference = has_ref_patch_by_camera_[camera_id] ? ref_patch_by_camera_[camera_id] : nullptr;
  return reference != nullptr && !reference->pending_delete_ && reference->ref_state_ != Feature::RefState::RETIRED
             ? reference
             : nullptr;
}

bool VisualPoint::hasUsableReference(int camera_id, bool cross_camera_reference, bool allow_candidate) const
{
  for (Feature *feature : obs_)
  {
    if (feature == nullptr || feature->pending_delete_ || feature->ref_state_ == Feature::RefState::RETIRED) continue;
    if (!cross_camera_reference && feature->camera_id_ != camera_id) continue;
    if (feature->ref_state_ == Feature::RefState::VALIDATED || allow_candidate) return true;
  }
  return false;
}

void VisualPoint::addFrameRef(Feature *ftr)
{
  obs_.push_front(ftr);
}

void VisualPoint::deleteFeatureRef(Feature *ftr)
{
  if (ref_patch == ftr)
  {
    ref_patch = nullptr;
    has_ref_patch_ = false;
  }
  for (size_t camera_id = 0; camera_id < ref_patch_by_camera_.size(); ++camera_id)
  {
    if (ref_patch_by_camera_[camera_id] == ftr)
    {
      ref_patch_by_camera_[camera_id] = nullptr;
      has_ref_patch_by_camera_[camera_id] = 0;
    }
  }
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    if (*it == ftr)
    {
      delete *it;
      obs_.erase(it);
      return;
    }
  }
}

bool VisualPoint::getCloseViewObs(const Vector3d &framepos, Feature *&ftr, const Vector2d &cur_px, int camera_id) const
{
  (void)cur_px;
  ftr = nullptr;
  if (obs_.empty()) return false;

  Vector3d obs_dir(framepos - pos_);
  obs_dir.normalize();
  double min_cos_angle = 0.0;
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    if (*it == nullptr || (*it)->pending_delete_ || (*it)->ref_state_ == Feature::RefState::RETIRED) continue;
    if (camera_id >= 0 && (*it)->camera_id_ != camera_id) continue;
    Vector3d dir((*it)->T_f_w_.inverse().translation() - pos_);
    dir.normalize();
    const double cos_angle = obs_dir.dot(dir);
    if (cos_angle > min_cos_angle)
    {
      min_cos_angle = cos_angle;
      ftr = *it;
    }
  }
  return ftr != nullptr && min_cos_angle >= 0.5;
}

void VisualPoint::findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const
{
  (void)framepos;
  ftr = nullptr;
  float min_score = std::numeric_limits<float>::max();
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    if (*it == nullptr || (*it)->pending_delete_ || (*it)->ref_state_ == Feature::RefState::RETIRED) continue;
    if ((*it)->score_ < min_score)
    {
      min_score = (*it)->score_;
      ftr = *it;
    }
  }
}

void VisualPoint::deleteNonRefPatchFeatures()
{
  for (auto it = obs_.begin(); it != obs_.end();)
  {
    const bool kept_by_camera = std::find(ref_patch_by_camera_.begin(), ref_patch_by_camera_.end(), *it) != ref_patch_by_camera_.end();
    if (*it != ref_patch && !kept_by_camera)
    {
      delete *it;
      it = obs_.erase(it);
    }
    else
    {
      ++it;
    }
  }
}
