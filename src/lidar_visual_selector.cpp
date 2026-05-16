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
    const camodocal::CameraConstPtr &camera)
{
  // Selector contract:
  // 1. transform current LiDAR points into cam0;
  // 2. project with camodocal;
  // 3. keep the nearest point per image cell as a cheap z-buffer;
  // 4. reject weak-texture and invalid-mask pixels;
  // 5. spatially suppress the survivors before handing them to LK tracking.
  last_stats_ = LidarVisualSelectorStats();
  std::vector<LidarVisualCandidate> selected;
  if (!vision_.lidar_depth_enable || !vision_.lidar_prior_feature_enable ||
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
    const Eigen::Vector3d p_body = R_I_L_ * p_lidar + t_I_L_;
    candidate.P_W_init = state.rot_end * p_body + state.pos_end;
    candidate.mask_radius = vision_.lidar_mask_radius;

    zbuffer[cell].candidate = candidate;
    zbuffer_depth[cell] = depth;
    zbuffer_valid[cell] = true;
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
  selected.reserve(static_cast<size_t>(max_candidates));
  for (const auto &candidate : texture_kept) {
    if (static_cast<int>(selected.size()) >= max_candidates) {
      break;
    }
    if (!maskAccepts(selected, candidate.pixel)) {
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

} // namespace cake_slam
