// 模块功能：LiDAR-视觉候选特征选择实现，
// 将点云投影到图像并筛选可跟踪种子。

#include "cake_slam/lidar_visual_selector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "cake_slam/voxel_map.h"

namespace cake_slam {
namespace {

double elapsedMs(const std::chrono::steady_clock::time_point &start)
{
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

struct MapProjectionSample
{
  Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_w = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov_w = Eigen::Matrix3d::Zero();
  double plane_var_scalar = 0.0;
  double quality = 0.0;
  bool from_plane = false;
};

Eigen::Vector3d normalizedOrZero(const Eigen::Vector3d &v)
{
  const double n = v.norm();
  return n > 1e-9 ? (v / n).eval() : Eigen::Vector3d::Zero();
}

Eigen::Vector3d fallbackPlaneAxis(const Eigen::Vector3d &normal)
{
  Eigen::Vector3d axis = std::abs(normal.x()) < 0.9
                             ? Eigen::Vector3d::UnitX()
                             : Eigen::Vector3d::UnitY();
  axis -= normal * normal.dot(axis);
  return normalizedOrZero(axis);
}

void appendSampleIfNear(std::vector<MapProjectionSample> &bucket,
                        const MapProjectionSample &sample,
                        const Eigen::Vector3d &camera_w,
                        double max_range,
                        int max_bucket_size)
{
  if (static_cast<int>(bucket.size()) >= max_bucket_size) {
    return;
  }
  const double range = (sample.point_w - camera_w).norm();
  if (range <= max_range) {
    bucket.push_back(sample);
  }
}

void appendPlaneSamples(const VoxelOctoTree *octo,
                        const Eigen::Vector3d &camera_w,
                        double max_range,
                        int max_bucket_size,
                        std::vector<MapProjectionSample> &plane_samples)
{
  if (!octo || !octo->plane_ptr_ ||
      static_cast<int>(plane_samples.size()) >= max_bucket_size) {
    return;
  }

  const VoxelPlane &plane = *octo->plane_ptr_;
  if (!plane.is_plane_ || !plane.is_init_) {
    return;
  }

  Eigen::Vector3d normal = normalizedOrZero(plane.normal_);
  if (normal.isZero(1e-9)) {
    return;
  }
  Eigen::Vector3d x_axis = normalizedOrZero(plane.x_normal_);
  if (x_axis.isZero(1e-9) || std::abs(x_axis.dot(normal)) > 0.2) {
    x_axis = fallbackPlaneAxis(normal);
  }
  Eigen::Vector3d y_axis = normalizedOrZero(plane.y_normal_);
  if (y_axis.isZero(1e-9) || std::abs(y_axis.dot(normal)) > 0.2) {
    y_axis = normalizedOrZero(normal.cross(x_axis));
  }
  if (x_axis.isZero(1e-9) || y_axis.isZero(1e-9)) {
    return;
  }

  const double node_radius = std::max(0.05, static_cast<double>(octo->quater_length_));
  const double sample_radius = 0.5 * std::min(std::max(0.05, static_cast<double>(plane.radius_)), node_radius);
  const double plane_var_scalar = std::max(0.0, std::max(
      plane.plane_var_.block<3, 3>(0, 0).trace(),
      plane.plane_var_.block<3, 3>(3, 3).trace()));

  const auto append_at = [&](const Eigen::Vector3d &p, double quality_scale) {
    MapProjectionSample sample;
    sample.point_w = p;
    sample.normal_w = normal;
    sample.plane_var_scalar = plane_var_scalar;
    sample.quality = 2.0 + quality_scale + 1e-3 * static_cast<double>(plane.points_size_);
    sample.from_plane = true;
    appendSampleIfNear(plane_samples, sample, camera_w, max_range, max_bucket_size);
  };

  append_at(plane.center_, 0.2);
  append_at(plane.center_ + sample_radius * x_axis, 0.1);
  append_at(plane.center_ - sample_radius * x_axis, 0.1);
  append_at(plane.center_ + sample_radius * y_axis, 0.1);
  append_at(plane.center_ - sample_radius * y_axis, 0.1);
}

void appendPointSamples(const VoxelOctoTree *octo,
                        const Eigen::Vector3d &camera_w,
                        double max_range,
                        int max_bucket_size,
                        bool stable_plane_points,
                        std::vector<MapProjectionSample> &bucket)
{
  if (!octo || octo->temp_points_.empty() ||
      static_cast<int>(bucket.size()) >= max_bucket_size) {
    return;
  }

  const int max_per_node = stable_plane_points ? 4 : 2;
  const int available = static_cast<int>(octo->temp_points_.size());
  const int wanted = std::min(max_per_node, available);
  const int stride = std::max(1, available / std::max(1, wanted));
  int added = 0;
  for (int i = 0; i < available && added < wanted; i += stride) {
    const pointWithVar &pv = octo->temp_points_[static_cast<size_t>(i)];
    MapProjectionSample sample;
    sample.point_w = pv.point_w;
    sample.normal_w = normalizedOrZero(pv.normal);
    sample.cov_w = pv.var;
    sample.plane_var_scalar = std::max(0.0, pv.var.trace());
    sample.quality = stable_plane_points ? 1.5 : 0.5;
    sample.from_plane = false;
    const size_t old_size = bucket.size();
    appendSampleIfNear(bucket, sample, camera_w, max_range, max_bucket_size);
    if (bucket.size() != old_size) {
      ++added;
    }
  }
}

void collectOctoProjectionSamples(const VoxelOctoTree *octo,
                                  const Eigen::Vector3d &camera_w,
                                  double max_range,
                                  int max_bucket_size,
                                  std::vector<MapProjectionSample> &plane_samples,
                                  std::vector<MapProjectionSample> &stable_points,
                                  std::vector<MapProjectionSample> &fallback_points)
{
  if (!octo) {
    return;
  }
  if (static_cast<int>(plane_samples.size()) >= max_bucket_size &&
      static_cast<int>(stable_points.size()) >= max_bucket_size &&
      static_cast<int>(fallback_points.size()) >= max_bucket_size) {
    return;
  }

  const Eigen::Vector3d voxel_center(octo->voxel_center_[0],
                                     octo->voxel_center_[1],
                                     octo->voxel_center_[2]);
  const double bound_radius = std::sqrt(3.0) * std::max(0.0, static_cast<double>(octo->quater_length_));
  if ((voxel_center - camera_w).norm() - bound_radius > max_range) {
    return;
  }

  const bool stable_plane =
      octo->plane_ptr_ && octo->plane_ptr_->is_plane_ && octo->plane_ptr_->is_init_;
  if (stable_plane) {
    appendPlaneSamples(octo, camera_w, max_range, max_bucket_size, plane_samples);
    appendPointSamples(octo, camera_w, max_range, max_bucket_size, true, stable_points);
    return;
  }

  bool has_child = false;
  if (octo->init_octo_ && octo->layer_ < octo->max_layer_) {
    for (const auto *leaf : octo->leaves_) {
      if (leaf) {
        has_child = true;
        collectOctoProjectionSamples(leaf, camera_w, max_range, max_bucket_size,
                                     plane_samples, stable_points, fallback_points);
      }
    }
  }

  if (!has_child) {
    appendPointSamples(octo, camera_w, max_range, max_bucket_size, false, fallback_points);
  }
}

} // namespace

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
  const auto t_total = std::chrono::steady_clock::now();
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

  const auto t_project = std::chrono::steady_clock::now();
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
  last_stats_.project_ms = elapsedMs(t_project);

  if (!vision_.lidar_prior_feature_enable) {
    last_stats_.total_ms = elapsedMs(t_total);
    return selected;
  }

  const auto t_texture = std::chrono::steady_clock::now();
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
  last_stats_.texture_ms = elapsedMs(t_texture);

  const auto t_select = std::chrono::steady_clock::now();
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
  last_stats_.select_ms = elapsedMs(t_select);
  last_stats_.total_ms = elapsedMs(t_total);
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
  const auto t_total = std::chrono::steady_clock::now();
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

  const auto t_occlusion = std::chrono::steady_clock::now();
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
  last_stats_.occlusion_ms = elapsedMs(t_occlusion);

  const auto t_collect = std::chrono::steady_clock::now();
  const Eigen::Matrix3d R_I_C = R_I_L_ * R_C_L_.transpose();
  const Eigen::Vector3d t_I_C = t_I_L_ - R_I_C * t_C_L_;
  const Eigen::Vector3d camera_w = state.rot_end * t_I_C + state.pos_end;

  const int sample_bucket_cap = std::max(256, std::max(1, vision_.max_lidar_features) * 8);
  std::vector<MapProjectionSample> plane_samples;
  std::vector<MapProjectionSample> stable_points;
  std::vector<MapProjectionSample> fallback_points;
  plane_samples.reserve(static_cast<size_t>(sample_bucket_cap));
  stable_points.reserve(static_cast<size_t>(sample_bucket_cap));
  fallback_points.reserve(static_cast<size_t>(sample_bucket_cap));
  std::vector<std::pair<double, const VoxelOctoTree *>> nearby_roots;
  nearby_roots.reserve(local_map->VoxelMap().size());
  for (const auto &kv : local_map->VoxelMap()) {
    if (!kv.second) {
      continue;
    }
    const VoxelOctoTree *root = kv.second;
    const Eigen::Vector3d voxel_center(root->voxel_center_[0],
                                       root->voxel_center_[1],
                                       root->voxel_center_[2]);
    const double bound_radius = std::sqrt(3.0) * std::max(0.0, static_cast<double>(root->quater_length_));
    const double near_range = (voxel_center - camera_w).norm() - bound_radius;
    if (near_range <= vision_.max_lidar_depth) {
      nearby_roots.emplace_back(std::max(0.0, near_range), root);
    }
  }
  std::sort(nearby_roots.begin(), nearby_roots.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  for (const auto &root : nearby_roots) {
    collectOctoProjectionSamples(root.second, camera_w, vision_.max_lidar_depth,
                                 sample_bucket_cap, plane_samples,
                                 stable_points, fallback_points);
    if (static_cast<int>(plane_samples.size()) >= sample_bucket_cap &&
        static_cast<int>(stable_points.size()) >= sample_bucket_cap &&
        static_cast<int>(fallback_points.size()) >= sample_bucket_cap) {
      break;
    }
  }
  last_stats_.input_points = static_cast<int>(plane_samples.size() + stable_points.size() + fallback_points.size());
  last_stats_.map_collect_ms = elapsedMs(t_collect);

  const auto t_project = std::chrono::steady_clock::now();
  const int cell_size = std::max(1, vision_.z_buffer_cell_size);
  const int cell_cols = (gray_u8.cols + cell_size - 1) / cell_size;
  const int cell_rows = (gray_u8.rows + cell_size - 1) / cell_size;
  const int cell_count = std::max(1, cell_cols * cell_rows);
  std::vector<ProjectedCandidate> zbuffer(cell_count);
  std::vector<double> zbuffer_depth(cell_count, std::numeric_limits<double>::infinity());
  std::vector<bool> zbuffer_valid(cell_count, false);
  const int border = std::max(2, static_cast<int>(std::round(vision_.lidar_mask_radius * 0.5)));
  const Eigen::Matrix3d R_C_W = R_I_C.transpose() * state.rot_end.transpose();

  const auto inv_depth_var_from_sample = [&](const MapProjectionSample &sample,
                                             double depth,
                                             double incidence) -> double {
    const double depth4 = std::max(1e-6, depth * depth * depth * depth);
    if (sample.from_plane) {
      const double distance_sigma = vision_.lidar_depth_std * (1.0 + 0.02 * std::max(0.0, depth));
      const double incidence_scale = 1.0 / std::max(0.1, incidence);
      const double sigma_z2 = distance_sigma * distance_sigma * incidence_scale * incidence_scale +
                              std::max(0.0, sample.plane_var_scalar);
      return std::max(vision_.min_inv_depth_var, sigma_z2 / depth4);
    }
    double sigma_z2 = vision_.lidar_depth_std * vision_.lidar_depth_std;
    if (sample.cov_w.trace() > 1e-12) {
      const Eigen::Matrix3d cov_cam = R_C_W * sample.cov_w * R_C_W.transpose();
      sigma_z2 = std::max(sigma_z2, cov_cam(2, 2));
    }
    return std::max(vision_.min_inv_depth_var, sigma_z2 / depth4);
  };

  const auto project_bucket = [&](const std::vector<MapProjectionSample> &samples) {
    for (const auto &sample : samples) {
      const Eigen::Vector3d p_body = state.rot_end.transpose() * (sample.point_w - state.pos_end);
      const Eigen::Vector3d p_cam = R_I_C.transpose() * (p_body - t_I_C);
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
      const cv::Point2f pixel(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
      if (!inValidDomain(valid_mask, pixel)) {
        continue;
      }
      last_stats_.in_image++;

      if (scanOccludes(depth_frame ? *depth_frame : LidarDepthFrame(), pixel, depth)) {
        last_stats_.occlusion_reject++;
        continue;
      }

      Eigen::Vector3d ray_w = sample.point_w - camera_w;
      if (ray_w.norm() <= 1e-9) {
        continue;
      }
      ray_w.normalize();
      double incidence = 1.0;
      if (!sample.normal_w.isZero(1e-9)) {
        incidence = std::abs(sample.normal_w.normalized().dot(ray_w));
      }
      if (sample.from_plane && incidence < vision_.raycast_min_cos) {
        continue;
      }

      const int cell_x = std::min(cell_cols - 1, std::max(0, u / cell_size));
      const int cell_y = std::min(cell_rows - 1, std::max(0, v / cell_size));
      const int cell = cell_y * cell_cols + cell_x;
      if (zbuffer_valid[cell] && depth >= zbuffer_depth[cell] - vision_.z_buffer_depth_tolerance) {
        continue;
      }

      LidarVisualCandidate candidate;
      candidate.pixel = pixel;
      candidate.depth = depth;
      candidate.inv_depth = 1.0 / depth;
      candidate.inv_depth_var = inv_depth_var_from_sample(sample, depth, incidence);
      candidate.P_W_init = sample.point_w;
      candidate.source = 0;
      candidate.depth_prior_allowed = true;
      candidate.lio_gate = true;
      candidate.shi_tomasi_score = sample.quality;
      candidate.mask_radius = vision_.lidar_mask_radius;

      zbuffer[cell].candidate = candidate;
      zbuffer_depth[cell] = depth;
      zbuffer_valid[cell] = true;
    }
  };

  project_bucket(plane_samples);
  project_bucket(stable_points);
  project_bucket(fallback_points);
  last_stats_.project_ms = elapsedMs(t_project);

  for (int i = 0; i < cell_count; ++i) {
    if (!zbuffer_valid[i]) {
      continue;
    }
    last_stats_.zbuffer_kept++;
    updateDepthFrameCell(depth_frame, zbuffer[i].candidate);
  }

  if (!vision_.lidar_prior_feature_enable) {
    last_stats_.total_ms = elapsedMs(t_total);
    return selected;
  }

  const auto t_texture = std::chrono::steady_clock::now();
  std::vector<LidarVisualCandidate> texture_kept;
  texture_kept.reserve(static_cast<size_t>(last_stats_.zbuffer_kept));
  for (int i = 0; i < cell_count; ++i) {
    if (!zbuffer_valid[i]) {
      continue;
    }
    double score = 0.0;
    if (!textureAccepted(gray_u8, zbuffer[i].candidate.pixel, &score)) {
      continue;
    }
    zbuffer[i].candidate.shi_tomasi_score = score + 1e-3 * zbuffer[i].candidate.shi_tomasi_score;
    texture_kept.push_back(zbuffer[i].candidate);
  }
  last_stats_.texture_kept = static_cast<int>(texture_kept.size());
  last_stats_.texture_ms = elapsedMs(t_texture);

  const auto t_select = std::chrono::steady_clock::now();
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
  last_stats_.select_ms = elapsedMs(t_select);
  last_stats_.total_ms = elapsedMs(t_total);
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
  for (int dy = -1; dy <= 1; ++dy) {
    const int cy = cell_y + dy;
    if (cy < 0 || cy >= depth_frame.cell_rows) {
      continue;
    }
    for (int dx = -1; dx <= 1; ++dx) {
      const int cx = cell_x + dx;
      if (cx < 0 || cx >= depth_frame.cell_cols) {
        continue;
      }
      const int cell = cy * depth_frame.cell_cols + cx;
      if (cell < 0 || cell >= static_cast<int>(cells.size())) {
        continue;
      }
      const auto &z = cells[static_cast<size_t>(cell)];
      if (z.valid && depth > z.depth + depth_frame.z_buffer_depth_tolerance) {
        return true;
      }
    }
  }
  return false;
}

} // namespace cake_slam
