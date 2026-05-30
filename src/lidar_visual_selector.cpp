// 模块功能：LiDAR-视觉候选特征选择实现，
// 将点云投影到图像并筛选可跟踪种子。

#include "cake_slam/lidar_visual_selector.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "cake_slam/voxel_map.h"

namespace cake_slam {

void LidarVisualSelector::Configure(const Config &config)
{
  vision_ = config.vision;
  if (vision_.lidar_depth_source < 0 || vision_.lidar_depth_source > 2) {
    vision_.lidar_depth_source = 1;
  }
  vision_.lidar_depth_grid_rows = std::max(1, vision_.lidar_depth_grid_rows);
  vision_.lidar_depth_grid_cols = std::max(1, vision_.lidar_depth_grid_cols);
  vision_.max_lidar_depth_ratio = std::clamp(vision_.max_lidar_depth_ratio, 0.0, 1.0);
  lidar_range_noise_ = config.map.dept_err;
  lidar_beam_noise_ = config.map.beam_err;
}

void LidarVisualSelector::SetExtrinsics(const Eigen::Matrix3d &R_I_L, const Eigen::Vector3d &t_I_L,
                                        const Eigen::Matrix3d &R_C_L, const Eigen::Vector3d &t_C_L)
{
  R_I_L_ = R_I_L;
  t_I_L_ = t_I_L;
  R_C_L_ = R_C_L;
  t_C_L_ = t_C_L;
}

std::vector<LidarVisualCandidate> LidarVisualSelector::Select(
    const PointCloudXYZI::Ptr &cloud_lidar,
    const StatesGroup &state,
    const cv::Mat &gray,
    const cv::Mat &valid_mask,
    const camodocal::CameraConstPtr &camera,
    LidarDepthFrame *depth_frame)
{
  // Selector contract:
  // 1. transform current LiDAR points into cam0;
  // 2. project with camodocal;
  // 3. keep the nearest point per image cell as a cheap z-buffer;
  // 4. reject weak-texture and invalid-mask pixels;
  // 5. spatially suppress the survivors before handing them to LK tracking.
  last_stats_ = LidarVisualSelectorStats();
  last_stats_.source_mode = 1;
  std::vector<LidarVisualCandidate> selected;
  if (!vision_.lidar_depth_enable || vision_.lidar_depth_source == 2 ||
      (!vision_.lidar_prior_feature_enable && !vision_.visual_feature_depth_prior_enable) ||
      !cloud_lidar || cloud_lidar->empty() || gray.empty() || !camera) {
    return selected;
  }

  cv::Mat gray_u8;
  if (gray.channels() == 1) {
    gray_u8 = gray;
  } else {
    cv::cvtColor(gray, gray_u8, cv::COLOR_BGR2GRAY);
  }

  const int cell_size = std::max(1, vision_.z_buffer_cell_size);
  const int cell_cols = (gray_u8.cols + cell_size - 1) / cell_size;
  const int cell_rows = (gray_u8.rows + cell_size - 1) / cell_size;
  const int cell_count = std::max(1, cell_cols * cell_rows);
  std::vector<ProjectedCandidate> zbuffer(cell_count);
  std::vector<double> zbuffer_depth(cell_count, std::numeric_limits<double>::infinity());
  std::vector<bool> zbuffer_valid(cell_count, false);

  last_stats_.input_points = static_cast<int>(cloud_lidar->size());
  const int border = std::max(2, static_cast<int>(std::round(vision_.lidar_mask_radius * 0.5)));
  initializeDepthFrame(state, gray_u8, depth_frame);

  for (const auto &pt : cloud_lidar->points) {
    const Eigen::Vector3d p_lidar(pt.x, pt.y, pt.z);
    const Eigen::Vector3d p_cam = R_C_L_ * p_lidar + t_C_L_;
    const double depth = p_cam.z();
    if (depth <= vision_.min_lidar_depth || depth >= vision_.max_lidar_depth) {
      continue;
    }
    last_stats_.positive_depth++;

    Eigen::Vector2d uv;
    camera->spaceToPlane(p_cam, uv);
    const int u = static_cast<int>(std::round(uv.x()));
    const int v = static_cast<int>(std::round(uv.y()));
    if (u < border || v < border || u >= gray_u8.cols - border || v >= gray_u8.rows - border) {
      continue;
    }
    if (!inValidDomain(valid_mask, cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y())))) {
      continue;
    }
    last_stats_.in_image++;

    const int cell_x = std::min(cell_cols - 1, std::max(0, u / cell_size));
    const int cell_y = std::min(cell_rows - 1, std::max(0, v / cell_size));
    const int cell = cell_y * cell_cols + cell_x;
    if (zbuffer_valid[cell] && depth >= zbuffer_depth[cell] - vision_.z_buffer_depth_tolerance) {
      continue;
    }

    Eigen::Matrix3d cov_lidar = Eigen::Matrix3d::Identity() *
                                vision_.lidar_depth_std * vision_.lidar_depth_std;
    Eigen::Vector3d cov_point = p_lidar;
    calcBodyCov(cov_point,
                static_cast<float>(std::max(1e-6, lidar_range_noise_)),
                static_cast<float>(std::max(1e-6, lidar_beam_noise_)),
                cov_lidar);

    LidarVisualCandidate candidate;
    candidate.pixel = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
    candidate.depth = depth;
    candidate.inv_depth = 1.0 / depth;
    candidate.inv_depth_var = inverseDepthVariance(cov_lidar, depth);
    candidate.source = 1;
    candidate.depth_prior_allowed = true;
    candidate.lio_gate = true;
    candidate.shi_tomasi_score = 0.0;
    const Eigen::Vector3d p_body = R_I_L_ * p_lidar + t_I_L_;
    candidate.P_W_init = state.rot_end * p_body + state.pos_end;
    candidate.mask_radius = vision_.lidar_mask_radius;
    updateDepthFrameCell(depth_frame, candidate);

    zbuffer[cell].candidate = candidate;
    zbuffer_depth[cell] = depth;
    zbuffer_valid[cell] = true;
  }

  if (!vision_.lidar_prior_feature_enable) {
    return selected;
  }

  std::vector<LidarVisualCandidate> texture_kept;
  for (int i = 0; i < cell_count; ++i) {
    if (!zbuffer_valid[i]) {
      continue;
    }
    last_stats_.zbuffer_kept++;
    double score = 0.0;
    if (!textureAccepted(gray_u8, zbuffer[i].candidate.pixel, &score)) {
      continue;
    }
    zbuffer[i].candidate.shi_tomasi_score = score;
    texture_kept.push_back(zbuffer[i].candidate);
  }
  last_stats_.texture_kept = static_cast<int>(texture_kept.size());

  std::sort(texture_kept.begin(), texture_kept.end(),
            [](const LidarVisualCandidate &a, const LidarVisualCandidate &b) {
              if (std::abs(a.shi_tomasi_score - b.shi_tomasi_score) > 1e-12) {
                return a.shi_tomasi_score > b.shi_tomasi_score;
              }
              return a.inv_depth_var < b.inv_depth_var;
            });

  const int max_candidates = std::max(0, vision_.max_lidar_features);
  std::vector<int> grid_counts(static_cast<size_t>(vision_.lidar_depth_grid_rows * vision_.lidar_depth_grid_cols), 0);
  selected.reserve(static_cast<size_t>(max_candidates));
  for (const auto &candidate : texture_kept) {
    if (static_cast<int>(selected.size()) >= max_candidates) {
      break;
    }
    if (!maskAccepts(selected, candidate.pixel)) {
      continue;
    }
    if (!gridAccepts(grid_counts, gray_u8.size(), candidate.pixel)) {
      continue;
    }
    selected.push_back(candidate);
  }
  last_stats_.mask_kept = static_cast<int>(selected.size());
  return selected;
}

std::vector<LidarVisualCandidate> LidarVisualSelector::SelectFromLocalMap(
    const LocalVoxelMapPtr &local_map,
    const PointCloudXYZI::Ptr &occlusion_cloud_lidar,
    const StatesGroup &state,
    const cv::Mat &gray,
    const cv::Mat &valid_mask,
    const camodocal::CameraConstPtr &camera,
    LidarDepthFrame *depth_frame)
{
  last_stats_ = LidarVisualSelectorStats();
  last_stats_.source_mode = 0;
  std::vector<LidarVisualCandidate> selected;
  if (!vision_.lidar_depth_enable || vision_.lidar_depth_source == 2 ||
      (!vision_.lidar_prior_feature_enable && !vision_.visual_feature_depth_prior_enable) ||
      !local_map || !local_map->IsConfigured() ||
      gray.empty() || !camera) {
    return selected;
  }

  cv::Mat gray_u8;
  if (gray.channels() == 1) {
    gray_u8 = gray;
  } else {
    cv::cvtColor(gray, gray_u8, cv::COLOR_BGR2GRAY);
  }
  initializeDepthFrame(state, gray_u8, depth_frame);
  if (depth_frame) {
    depth_frame->local_map = local_map;
  }

  if (occlusion_cloud_lidar && !occlusion_cloud_lidar->empty()) {
    for (const auto &pt : occlusion_cloud_lidar->points) {
      const Eigen::Vector3d p_lidar(pt.x, pt.y, pt.z);
      const Eigen::Vector3d p_cam = R_C_L_ * p_lidar + t_C_L_;
      const double depth = p_cam.z();
      if (depth <= vision_.min_lidar_depth || depth >= vision_.max_lidar_depth) {
        continue;
      }
      Eigen::Vector2d uv;
      camera->spaceToPlane(p_cam, uv);
      const cv::Point2f pixel(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
      if (!inValidDomain(valid_mask, pixel)) {
        continue;
      }
      LidarVisualCandidate occlusion;
      occlusion.pixel = pixel;
      occlusion.depth = depth;
      occlusion.inv_depth = 1.0 / depth;
      occlusion.inv_depth_var = std::max(vision_.min_inv_depth_var,
                                         (vision_.lidar_depth_std * vision_.lidar_depth_std) /
                                             std::max(1e-6, depth * depth * depth * depth));
      const Eigen::Vector3d p_body = R_I_L_ * p_lidar + t_I_L_;
      occlusion.P_W_init = state.rot_end * p_body + state.pos_end;
      occlusion.source = 1;
      occlusion.depth_prior_allowed = false;
      occlusion.lio_gate = false;
      updateDepthFrameCell(depth_frame, occlusion);
    }
  }

  if (!vision_.lidar_prior_feature_enable) {
    return selected;
  }

  cv::Mat corner_mask;
  if (!valid_mask.empty() && valid_mask.type() == CV_8UC1 &&
      valid_mask.size() == gray_u8.size()) {
    corner_mask = valid_mask;
  }

  const int max_corners = std::max(vision_.max_lidar_features * 4, vision_.max_lidar_features);
  std::vector<cv::Point2f> corners;
  if (max_corners > 0) {
    cv::goodFeaturesToTrack(gray_u8, corners, max_corners, 0.01,
                            std::max(4, vision_.min_dist / 2), corner_mask);
  }
  last_stats_.input_points = static_cast<int>(corners.size());

  std::vector<LidarVisualCandidate> texture_kept;
  texture_kept.reserve(corners.size());
  for (const auto &pixel : corners) {
    if (static_cast<int>(texture_kept.size()) >= std::max(vision_.max_lidar_features * 3, vision_.max_lidar_features)) {
      break;
    }
    last_stats_.map_raycast_attempts++;
    LidarVisualCandidate candidate;
    if (!candidateFromMapRaycast(local_map, state, pixel, camera, candidate)) {
      continue;
    }
    last_stats_.map_raycast_hits++;
    if (scanOccludes(depth_frame ? *depth_frame : LidarDepthFrame(), pixel, candidate.depth)) {
      last_stats_.occlusion_reject++;
      continue;
    }
    double score = 0.0;
    if (!textureAccepted(gray_u8, pixel, &score)) {
      continue;
    }
    candidate.shi_tomasi_score = score;
    candidate.mask_radius = vision_.lidar_mask_radius;
    texture_kept.push_back(candidate);
    updateDepthFrameCell(depth_frame, candidate);
  }

  last_stats_.positive_depth = last_stats_.map_raycast_hits;
  last_stats_.in_image = static_cast<int>(texture_kept.size());
  last_stats_.zbuffer_kept = depth_frame ? last_stats_.visual_depth_cells : 0;
  last_stats_.texture_kept = static_cast<int>(texture_kept.size());

  std::sort(texture_kept.begin(), texture_kept.end(),
            [](const LidarVisualCandidate &a, const LidarVisualCandidate &b) {
              if (std::abs(a.shi_tomasi_score - b.shi_tomasi_score) > 1e-12) {
                return a.shi_tomasi_score > b.shi_tomasi_score;
              }
              return a.inv_depth_var < b.inv_depth_var;
            });

  const int max_candidates = std::max(0, vision_.max_lidar_features);
  std::vector<int> grid_counts(static_cast<size_t>(vision_.lidar_depth_grid_rows * vision_.lidar_depth_grid_cols), 0);
  selected.reserve(static_cast<size_t>(max_candidates));
  for (const auto &candidate : texture_kept) {
    if (static_cast<int>(selected.size()) >= max_candidates) {
      break;
    }
    if (!maskAccepts(selected, candidate.pixel)) {
      continue;
    }
    if (!gridAccepts(grid_counts, gray_u8.size(), candidate.pixel)) {
      continue;
    }
    selected.push_back(candidate);
  }
  last_stats_.mask_kept = static_cast<int>(selected.size());
  return selected;
}

bool LidarVisualSelector::inValidDomain(const cv::Mat &valid_mask, const cv::Point2f &pixel) const
{
  const int u = static_cast<int>(std::round(pixel.x));
  const int v = static_cast<int>(std::round(pixel.y));
  if (valid_mask.empty()) {
    return true;
  }
  if (valid_mask.type() != CV_8UC1 || u < 0 || v < 0 ||
      u >= valid_mask.cols || v >= valid_mask.rows) {
    return false;
  }
  return valid_mask.at<uchar>(v, u) != 0;
}

bool LidarVisualSelector::textureAccepted(const cv::Mat &gray, const cv::Point2f &pixel, double *score) const
{
  const int u = static_cast<int>(std::round(pixel.x));
  const int v = static_cast<int>(std::round(pixel.y));
  if (u <= 1 || v <= 1 || u >= gray.cols - 2 || v >= gray.rows - 2) {
    return false;
  }

  const auto intensity = [&gray](int y, int x) -> double {
    return static_cast<double>(gray.at<uchar>(y, x));
  };
  double ixx = 0.0;
  double iyy = 0.0;
  double ixy = 0.0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const int x = u + dx;
      const int y = v + dy;
      const double ix = 0.5 * (intensity(y, x + 1) - intensity(y, x - 1));
      const double iy = 0.5 * (intensity(y + 1, x) - intensity(y - 1, x));
      ixx += ix * ix;
      iyy += iy * iy;
      ixy += ix * iy;
    }
  }
  const double trace = ixx + iyy;
  const double det_term = std::max(0.0, (ixx - iyy) * (ixx - iyy) + 4.0 * ixy * ixy);
  const double value = 0.5 * (trace - std::sqrt(det_term));
  if (score) {
    *score = value;
  }
  return vision_.shi_tomasi_min_score <= 0.0 || value >= vision_.shi_tomasi_min_score;
}

bool LidarVisualSelector::maskAccepts(const std::vector<LidarVisualCandidate> &selected, const cv::Point2f &pixel) const
{
  const double radius = std::max(1.0, vision_.lidar_mask_radius);
  const double radius_sq = radius * radius;
  for (const auto &candidate : selected) {
    const double dx = static_cast<double>(candidate.pixel.x - pixel.x);
    const double dy = static_cast<double>(candidate.pixel.y - pixel.y);
    if (dx * dx + dy * dy < radius_sq) {
      return false;
    }
  }
  return true;
}

bool LidarVisualSelector::gridAccepts(std::vector<int> &grid_counts,
                                      const cv::Size &image_size,
                                      const cv::Point2f &pixel) const
{
  if (grid_counts.empty() || image_size.width <= 0 || image_size.height <= 0) {
    return true;
  }
  const int grid_cols = std::max(1, vision_.lidar_depth_grid_cols);
  const int grid_rows = std::max(1, vision_.lidar_depth_grid_rows);
  const int gx = std::min(grid_cols - 1, std::max(0, static_cast<int>(pixel.x * grid_cols / image_size.width)));
  const int gy = std::min(grid_rows - 1, std::max(0, static_cast<int>(pixel.y * grid_rows / image_size.height)));
  const int idx = gy * grid_cols + gx;
  if (idx < 0 || idx >= static_cast<int>(grid_counts.size())) {
    return false;
  }
  const int max_per_grid = std::max(1, static_cast<int>(std::ceil(
      static_cast<double>(std::max(1, vision_.max_lidar_features)) /
      static_cast<double>(std::max(1, grid_cols * grid_rows)))));
  if (grid_counts[idx] >= max_per_grid) {
    return false;
  }
  grid_counts[idx]++;
  return true;
}

double LidarVisualSelector::inverseDepthVariance(const Eigen::Matrix3d &cov_lidar, double depth) const
{
  // Knowledge-base rule: propagate the full 3D LiDAR covariance into the camera
  // frame, then apply lambda = 1 / z:
  // sigma_lambda^2 = Sigma_cam(2,2) / z^4.
  const double depth4 = std::max(1e-6, depth * depth * depth * depth);
  const Eigen::Matrix3d cov_cam = R_C_L_ * cov_lidar * R_C_L_.transpose();
  const double sigma_z2 = std::max(1e-12, cov_cam(2, 2));
  return std::max(vision_.min_inv_depth_var, sigma_z2 / depth4);
}

double LidarVisualSelector::inverseDepthVarianceFromRaycast(const LocalVoxelMap::RaycastHit &hit, double depth) const
{
  const double depth4 = std::max(1e-6, depth * depth * depth * depth);
  const double distance_sigma = vision_.lidar_depth_std * (1.0 + 0.02 * std::max(0.0, depth));
  const double incidence_scale = 1.0 / std::max(0.1, hit.incidence_cos);
  const double plane_sigma2 = std::max(0.0, hit.plane_var_scalar);
  const double sigma_z2 = distance_sigma * distance_sigma * incidence_scale * incidence_scale + plane_sigma2;
  return std::max(vision_.min_inv_depth_var, sigma_z2 / depth4);
}

void LidarVisualSelector::initializeDepthFrame(const StatesGroup &state, const cv::Mat &gray, LidarDepthFrame *depth_frame) const
{
  if (!depth_frame) {
    return;
  }
  *depth_frame = LidarDepthFrame();
  depth_frame->valid = vision_.lidar_depth_enable && vision_.lidar_depth_source != 2;
  depth_frame->visual_feature_depth_prior_enable = vision_.visual_feature_depth_prior_enable;
  depth_frame->voxel_raycast_enable = vision_.voxel_raycast_enable;
  depth_frame->source_mode = vision_.lidar_depth_source;
  depth_frame->rows = gray.rows;
  depth_frame->cols = gray.cols;
  depth_frame->cell_size = std::max(1, vision_.z_buffer_cell_size);
  depth_frame->cell_cols = (gray.cols + depth_frame->cell_size - 1) / depth_frame->cell_size;
  depth_frame->cell_rows = (gray.rows + depth_frame->cell_size - 1) / depth_frame->cell_size;
  depth_frame->min_track_cnt = std::max(1, vision_.visual_depth_min_track_cnt);
  depth_frame->max_depth_update_features = std::max(0, vision_.max_depth_update_features);
  depth_frame->max_raycast_features = std::max(0, vision_.max_raycast_features);
  depth_frame->max_raycast_steps = std::max(1, vision_.max_raycast_steps);
  depth_frame->max_lidar_depth_ratio = std::clamp(vision_.max_lidar_depth_ratio, 0.0, 1.0);
  depth_frame->min_depth = vision_.min_lidar_depth;
  depth_frame->max_depth = vision_.max_lidar_depth;
  depth_frame->search_radius = std::max(0.0, vision_.depth_search_radius);
  depth_frame->z_buffer_depth_tolerance = vision_.z_buffer_depth_tolerance;
  depth_frame->lidar_depth_std = vision_.lidar_depth_std;
  depth_frame->min_inv_depth_var = vision_.min_inv_depth_var;
  depth_frame->raycast_step = vision_.raycast_step;
  depth_frame->raycast_min_cos = vision_.raycast_min_cos;
  depth_frame->raycast_plane_radius_scale = vision_.raycast_plane_radius_scale;
  depth_frame->prior_update_var_ratio = vision_.depth_prior_update_var_ratio;
  depth_frame->R_WB = state.rot_end;
  depth_frame->p_WB = state.pos_end;
  depth_frame->R_I_C = R_I_L_ * R_C_L_.transpose();
  depth_frame->t_I_C = t_I_L_ - depth_frame->R_I_C * t_C_L_;
  const int cell_count = std::max(1, depth_frame->cell_rows * depth_frame->cell_cols);
  depth_frame->cells.assign(static_cast<size_t>(cell_count), LidarDepthCell());
  depth_frame->occlusion_cells.assign(static_cast<size_t>(cell_count), LidarDepthCell());
}

bool LidarVisualSelector::updateDepthFrameCell(LidarDepthFrame *depth_frame, const LidarVisualCandidate &candidate)
{
  if (!depth_frame || !depth_frame->valid || depth_frame->cell_cols <= 0 || depth_frame->cell_rows <= 0 ||
      candidate.depth <= depth_frame->min_depth || candidate.depth >= depth_frame->max_depth) {
    return false;
  }
  const int u = static_cast<int>(std::round(candidate.pixel.x));
  const int v = static_cast<int>(std::round(candidate.pixel.y));
  if (u < 0 || v < 0 || u >= depth_frame->cols || v >= depth_frame->rows) {
    return false;
  }
  const int cell_x = std::min(depth_frame->cell_cols - 1, std::max(0, u / depth_frame->cell_size));
  const int cell_y = std::min(depth_frame->cell_rows - 1, std::max(0, v / depth_frame->cell_size));
  const int cell = cell_y * depth_frame->cell_cols + cell_x;
  if (cell < 0) {
    return false;
  }

  const auto write_nearest = [&](std::vector<LidarDepthCell> &cells) -> bool {
    if (cell >= static_cast<int>(cells.size())) {
      return false;
    }
    LidarDepthCell &dst = cells[static_cast<size_t>(cell)];
    if (dst.valid && candidate.depth >= dst.depth - depth_frame->z_buffer_depth_tolerance) {
      return false;
    }
    dst.valid = true;
    dst.pixel = candidate.pixel;
    dst.depth = candidate.depth;
    dst.inv_depth_var = candidate.inv_depth_var;
    dst.P_W_init = candidate.P_W_init;
    dst.source = candidate.source;
    dst.depth_prior_allowed = candidate.depth_prior_allowed;
    dst.quality = candidate.shi_tomasi_score;
    return true;
  };

  const bool occlusion_updated = write_nearest(depth_frame->occlusion_cells);
  bool prior_updated = false;
  if (candidate.depth_prior_allowed) {
    prior_updated = write_nearest(depth_frame->cells);
    if (prior_updated) {
      last_stats_.visual_depth_cells++;
    }
  }
  return occlusion_updated || prior_updated;
}

bool LidarVisualSelector::scanOccludes(const LidarDepthFrame &depth_frame, const cv::Point2f &pixel, double depth) const
{
  const std::vector<LidarDepthCell> &cells =
      !depth_frame.occlusion_cells.empty() ? depth_frame.occlusion_cells : depth_frame.cells;
  if (!depth_frame.valid || depth_frame.cell_cols <= 0 || depth_frame.cell_rows <= 0 || cells.empty()) {
    return false;
  }
  const int u = static_cast<int>(std::round(pixel.x));
  const int v = static_cast<int>(std::round(pixel.y));
  if (u < 0 || v < 0 || u >= depth_frame.cols || v >= depth_frame.rows) {
    return true;
  }
  const int cell_x = std::min(depth_frame.cell_cols - 1, std::max(0, u / depth_frame.cell_size));
  const int cell_y = std::min(depth_frame.cell_rows - 1, std::max(0, v / depth_frame.cell_size));
  const int cell = cell_y * depth_frame.cell_cols + cell_x;
  if (cell < 0 || cell >= static_cast<int>(cells.size())) {
    return true;
  }
  const auto &z = cells[static_cast<size_t>(cell)];
  return z.valid && depth > z.depth + depth_frame.z_buffer_depth_tolerance;
}

bool LidarVisualSelector::candidateFromMapRaycast(const LocalVoxelMapPtr &local_map,
                                                  const StatesGroup &state,
                                                  const cv::Point2f &pixel,
                                                  const camodocal::CameraConstPtr &camera,
                                                  LidarVisualCandidate &candidate) const
{
  if (!local_map || !camera) {
    return false;
  }
  Eigen::Vector3d bearing;
  camera->liftProjective(Eigen::Vector2d(pixel.x, pixel.y), bearing);
  if (bearing.z() <= 1e-9) {
    return false;
  }
  bearing /= bearing.z();

  const Eigen::Matrix3d R_I_C = R_I_L_ * R_C_L_.transpose();
  const Eigen::Vector3d t_I_C = t_I_L_ - R_I_C * t_C_L_;
  const Eigen::Vector3d origin_w = state.rot_end * t_I_C + state.pos_end;
  const Eigen::Vector3d dir_w = (state.rot_end * (R_I_C * bearing)).normalized();

  LocalVoxelMap::RaycastOptions options;
  options.min_depth = vision_.min_lidar_depth;
  options.max_depth = vision_.max_lidar_depth;
  options.step = vision_.raycast_step > 0.0 ? vision_.raycast_step : local_map->VoxelSize();
  options.min_cos = vision_.raycast_min_cos;
  options.plane_radius_scale = vision_.raycast_plane_radius_scale;
  options.max_steps = vision_.max_raycast_steps;
  options.allow_point_fallback = true;
  LocalVoxelMap::RaycastHit hit;
  if (!local_map->Raycast(origin_w, dir_w, options, hit) || !hit.valid) {
    return false;
  }

  const Eigen::Vector3d p_body = state.rot_end.transpose() * (hit.point_w - state.pos_end);
  const Eigen::Vector3d p_cam = R_I_C.transpose() * (p_body - t_I_C);
  const double depth = p_cam.z();
  if (depth <= vision_.min_lidar_depth || depth >= vision_.max_lidar_depth) {
    return false;
  }

  candidate.pixel = pixel;
  candidate.depth = depth;
  candidate.inv_depth = 1.0 / depth;
  candidate.inv_depth_var = inverseDepthVarianceFromRaycast(hit, depth);
  candidate.P_W_init = hit.point_w;
  candidate.source = 0;
  candidate.depth_prior_allowed = true;
  candidate.lio_gate = true;
  candidate.mask_radius = vision_.lidar_mask_radius;
  candidate.shi_tomasi_score = hit.from_plane ? 1.0 : 0.5;
  return true;
}

} // namespace cake_slam
