/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio_fisheye.h"

using namespace Eigen;

namespace
{
struct VirtualPatchWorkspace
{
  cv::Mat values;
  cv::Mat weight_sum;
  cv::Mat valid_mask;
  std::vector<int> touched_indices;
  std::vector<uint8_t> touched_flag;
};

extern VirtualPatchWorkspace workspace;
#ifdef _OPENMP
#pragma omp threadprivate(workspace)
#endif
VirtualPatchWorkspace workspace;

void clearVirtualPatchWorkspaceTouched(VirtualPatchWorkspace &workspace)
{
  if (workspace.touched_indices.empty()) return;

  float *values = workspace.values.ptr<float>(0);
  float *weights = workspace.weight_sum.ptr<float>(0);
  uint8_t *mask = workspace.valid_mask.ptr<uint8_t>(0);
  for (const int idx : workspace.touched_indices)
  {
    values[idx] = 0.0f;
    weights[idx] = 0.0f;
    mask[idx] = 0;
    workspace.touched_flag[idx] = 0;
  }
  workspace.touched_indices.clear();
}

void prepareVirtualPatchWorkspace(VirtualPatchWorkspace &workspace, int support_size)
{
  const int support_total = support_size * support_size;
  const bool resize_required = workspace.values.rows != support_size || workspace.values.cols != support_size ||
                               workspace.weight_sum.rows != support_size || workspace.weight_sum.cols != support_size ||
                               workspace.valid_mask.rows != support_size || workspace.valid_mask.cols != support_size ||
                               static_cast<int>(workspace.touched_flag.size()) != support_total;

  if (resize_required)
  {
    workspace.values.create(support_size, support_size, CV_32FC1);
    workspace.weight_sum.create(support_size, support_size, CV_32FC1);
    workspace.valid_mask.create(support_size, support_size, CV_8UC1);
    workspace.values.setTo(0.0f);
    workspace.weight_sum.setTo(0.0f);
    workspace.valid_mask.setTo(0);
    workspace.touched_flag.assign(support_total, 0);
    workspace.touched_indices.clear();
    return;
  }

  clearVirtualPatchWorkspaceTouched(workspace);
}
}

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
  visual_submap = nullptr;
  pinhole_cam = nullptr;
  optical_flow_triangulated_points.reset(new PointCloudXYZI());
}

VIOManager::~VIOManager()
{
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO()
{
  visual_submap = new SubSparseMap;

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  width = cam->width();
  height = cam->height();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;

  if (grid_size > 10)
  {
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  length = grid_n_width * grid_n_height;

  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float range_min = 0.1;
    float range_max = 3.0;
    float range_step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        for (float range_temp = range_min; range_temp <= range_max; range_temp += range_step)
        {
          V3D xyz = cam->cam2world(u, v);
          if (virtual_fisheye_patch_en)
          {
            if (xyz.norm() <= virtual_min_z) continue;
            xyz.normalize();
            xyz *= range_temp;
          }
          else
          {
            if (std::fabs(xyz[2]) <= virtual_min_z) continue;
            xyz *= range_temp / xyz[2];
          }
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if(colmap_output_en)
  {
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    if (pinhole_cam == nullptr)
    {
      printf("[ VIO ] COLMAP output disabled: the active camera model is not pinhole.\n");
      colmap_output_en = false;
    }
  }

  if(colmap_output_en)
  {
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  if (virtual_fisheye_patch_en)
  {
    if (virtual_focal_length <= 0.0) throw std::runtime_error("virtual_focal_length must be positive");
    if (virtual_max_search_level < 0) throw std::runtime_error("virtual_max_search_level must be non-negative");
    if (virtual_raw_window_half_size < 0) throw std::runtime_error("virtual_raw_window_half_size must be non-negative");
    if (virtual_splat_min_weight <= 0.0) throw std::runtime_error("virtual_splat_min_weight must be positive");
    if (virtual_patch_resampling_mode == "pull_exact")
    {
      virtual_patch_resampling_mode_enum = VirtualPatchResamplingMode::PULL_EXACT;
    }
    else if (virtual_patch_resampling_mode == "forward_splat")
    {
      virtual_patch_resampling_mode_enum = VirtualPatchResamplingMode::FORWARD_SPLAT;
    }
    else
    {
      throw std::runtime_error("virtual_patch_resampling_mode must be forward_splat or pull_exact");
    }

    const int max_scale = 1 << ((patch_pyrimid_level - 1) + virtual_max_search_level);
    virtual_support_radius = patch_size_half * max_scale + virtual_patch_margin + 2;
    virtual_support_size = 2 * virtual_support_radius + 1;
    virtual_support_ray_lut_.resize(virtual_support_size * virtual_support_size);
    for (int y = 0; y < virtual_support_size; ++y)
    {
      for (int x = 0; x < virtual_support_size; ++x)
      {
        V3F ray((x - virtual_support_radius) / static_cast<float>(virtual_focal_length),
                (y - virtual_support_radius) / static_cast<float>(virtual_focal_length), 1.0f);
        virtual_support_ray_lut_[y * virtual_support_size + x] = ray.normalized();
      }
    }

    core_patch_offsets_.resize(patch_size_total);
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x)
      {
        core_patch_offsets_[y * patch_size + x] = V2F(x - patch_size_half, y - patch_size_half);
      }
    }

    raw_pixel_to_unit_ray_lut_.resize(width * height);
    raw_pixel_unit_ray_valid_mask_.assign(width * height, 0);
    int valid_raw_rays = 0;
    for (int raw_y = 0; raw_y < height; ++raw_y)
    {
      for (int raw_x = 0; raw_x < width; ++raw_x)
      {
        const int raw_idx = raw_y * width + raw_x;
        if (!cam->isInFrame(V2D(raw_x, raw_y).cast<int>(), 0)) continue;
        V3D ray_c = cam->cam2world(raw_x, raw_y);
        if (!ray_c.array().isFinite().all()) continue;
        const double ray_norm = ray_c.norm();
        if (!std::isfinite(ray_norm) || ray_norm <= virtual_min_z) continue;
        ray_c /= ray_norm;
        raw_pixel_to_unit_ray_lut_[raw_idx] = ray_c.cast<float>();
        raw_pixel_unit_ray_valid_mask_[raw_idx] = 1;
        ++valid_raw_rays;
      }
    }

    printf("[ VIO ] Virtual fisheye patches enabled: focal=%.3f support=%dx%d max_search_level=%d resampling=%s raw_window_half=%d raw_lut=%d/%d\n",
           virtual_focal_length, virtual_support_size, virtual_support_size, virtual_max_search_level,
           virtual_patch_resampling_mode.c_str(), virtual_raw_window_half_size, valid_raw_rays, width * height);
  }

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

SE3<double> VIOManager::composeVirtualPose(const M3D &R_v_from_c, const SE3<double> &T_c_w) const
{
  // {}^V T_W = {}^V T_C * {}^C T_W. The virtual and raw cameras share the optical center.
  return SE3<double>(R_v_from_c * T_c_w.rotationMatrix(), R_v_from_c * T_c_w.translation());
}

bool VIOManager::buildVirtualFrameRotation(const V3D &point_in_raw_camera, M3D &R_v_from_c, M3D &R_c_from_v) const
{
  const double norm = point_in_raw_camera.norm();
  if (!std::isfinite(norm) || norm <= virtual_min_z) return false;

  const V3D z_v_in_c = point_in_raw_camera / norm;
  V3D reference_axis = V3D::UnitX();
  if (std::fabs(reference_axis.dot(z_v_in_c)) > 0.95) reference_axis = V3D::UnitY();

  V3D x_v_in_c = reference_axis - reference_axis.dot(z_v_in_c) * z_v_in_c;
  const double x_norm = x_v_in_c.norm();
  if (!std::isfinite(x_norm) || x_norm <= virtual_min_z) return false;
  x_v_in_c /= x_norm;

  V3D y_v_in_c = z_v_in_c.cross(x_v_in_c);
  const double y_norm = y_v_in_c.norm();
  if (!std::isfinite(y_norm) || y_norm <= virtual_min_z) return false;
  y_v_in_c /= y_norm;

  R_v_from_c.row(0) = x_v_in_c.transpose();
  R_v_from_c.row(1) = y_v_in_c.transpose();
  R_v_from_c.row(2) = z_v_in_c.transpose();
  R_c_from_v = R_v_from_c.transpose();
  return R_v_from_c.array().isFinite().all() && std::fabs(R_v_from_c.determinant() - 1.0) < 1e-6;
}

bool VIOManager::projectRawFisheyeIfValid(const V3D &ray_or_point_in_raw_camera, int required_border, V2D &raw_px) const
{
  if (!ray_or_point_in_raw_camera.array().isFinite().all() || ray_or_point_in_raw_camera.norm() <= virtual_min_z) return false;
  raw_px = cam->world2cam(ray_or_point_in_raw_camera);
  if (!raw_px.array().isFinite().all()) return false;
  if (raw_px[0] < required_border || raw_px[1] < required_border || raw_px[0] >= width - required_border - 1 ||
      raw_px[1] >= height - required_border - 1)
    return false;
  return cam->isInFrame(raw_px.cast<int>(), required_border);
}

bool VIOManager::buildVirtualSupportPatchPullExact(const cv::Mat &raw_img, const M3D &R_c_from_v, VirtualPatchImage &output) const
{
  if (!virtual_fisheye_patch_en || raw_img.empty() || raw_img.type() != CV_8UC1 || virtual_support_size <= 0) return false;

  output.values.create(virtual_support_size, virtual_support_size, CV_32FC1);
  output.valid_mask.create(virtual_support_size, virtual_support_size, CV_8UC1);
  output.values.setTo(std::numeric_limits<float>::quiet_NaN());
  output.valid_mask.setTo(0);
  output.R_c_from_v = R_c_from_v;
  output.R_v_from_c = R_c_from_v.transpose();

  for (int y = 0; y < virtual_support_size; ++y)
  {
    float *values = output.values.ptr<float>(y);
    uint8_t *mask = output.valid_mask.ptr<uint8_t>(y);
    for (int x = 0; x < virtual_support_size; ++x)
    {
      const V3D ray_v = virtual_support_ray_lut_[y * virtual_support_size + x].cast<double>();
      const V3D ray_c = R_c_from_v * ray_v;
      V2D raw_px;
      if (!projectRawFisheyeIfValid(ray_c, 1, raw_px)) continue;
      values[x] = static_cast<float>(vk::interpolateMat_8u(raw_img, raw_px[0], raw_px[1]));
      mask[x] = 255;
    }
  }

  output.valid = true;
  return true;
}

void VIOManager::splatRawPixelToVirtualPatch(const cv::Mat &raw_img, int raw_x, int raw_y, const M3D &R_v_from_c, cv::Mat &value_sum,
                                             cv::Mat &weight_sum) const
{
  const int raw_idx = raw_y * width + raw_x;
  if (raw_idx < 0 || raw_idx >= static_cast<int>(raw_pixel_unit_ray_valid_mask_.size()) || !raw_pixel_unit_ray_valid_mask_[raw_idx]) return;

  const V3D ray_c = raw_pixel_to_unit_ray_lut_[raw_idx].cast<double>();
  const V3D ray_v = R_v_from_c * ray_c;
  if (!ray_v.array().isFinite().all() || ray_v[2] <= virtual_min_z) return;

  const float u_v = static_cast<float>(virtual_focal_length * ray_v[0] / ray_v[2] + virtual_support_radius);
  const float v_v = static_cast<float>(virtual_focal_length * ray_v[1] / ray_v[2] + virtual_support_radius);
  const int x0 = static_cast<int>(std::floor(u_v));
  const int y0 = static_cast<int>(std::floor(v_v));
  if (x0 < 0 || y0 < 0 || x0 + 1 >= virtual_support_size || y0 + 1 >= virtual_support_size) return;

  const float dx = u_v - x0;
  const float dy = v_v - y0;
  const float w00 = (1.0f - dx) * (1.0f - dy);
  const float w10 = dx * (1.0f - dy);
  const float w01 = (1.0f - dx) * dy;
  const float w11 = dx * dy;
  const float intensity = static_cast<float>(raw_img.ptr<uint8_t>(raw_y)[raw_x]);
  float *value_row0 = value_sum.ptr<float>(y0);
  float *value_row1 = value_sum.ptr<float>(y0 + 1);
  float *weight_row0 = weight_sum.ptr<float>(y0);
  float *weight_row1 = weight_sum.ptr<float>(y0 + 1);

  value_row0[x0] += w00 * intensity;
  value_row0[x0 + 1] += w10 * intensity;
  value_row1[x0] += w01 * intensity;
  value_row1[x0 + 1] += w11 * intensity;

  weight_row0[x0] += w00;
  weight_row0[x0 + 1] += w10;
  weight_row1[x0] += w01;
  weight_row1[x0 + 1] += w11;
}

bool VIOManager::hasFullVirtualCoreCoverage(const cv::Mat &valid_mask) const
{
  if (valid_mask.empty()) return false;
  const int center = virtual_support_radius;
  for (const V2F &offset : core_patch_offsets_)
  {
    const int x = center + static_cast<int>(std::lround(offset[0]));
    const int y = center + static_cast<int>(std::lround(offset[1]));
    if (x < 0 || y < 0 || x + 1 >= valid_mask.cols || y + 1 >= valid_mask.rows) return false;
    const uint8_t *row0 = valid_mask.ptr<uint8_t>(y);
    const uint8_t *row1 = valid_mask.ptr<uint8_t>(y + 1);
    if (row0[x] == 0 || row0[x + 1] == 0 || row1[x] == 0 || row1[x + 1] == 0)
      return false;
  }
  return true;
}

bool VIOManager::buildVirtualSupportPatchForwardSplat(const cv::Mat &raw_img, const V2D &raw_center_px, const M3D &R_v_from_c,
                                                      VirtualPatchImage &output) const
{
  if (!virtual_fisheye_patch_en || raw_img.empty() || raw_img.type() != CV_8UC1 || virtual_support_size <= 0) return false;
  if (raw_pixel_to_unit_ray_lut_.size() != static_cast<size_t>(width * height) ||
      raw_pixel_unit_ray_valid_mask_.size() != static_cast<size_t>(width * height))
    return false;

  VirtualPatchWorkspace &patch_workspace = workspace;
  prepareVirtualPatchWorkspace(patch_workspace, virtual_support_size);

  output.values.create(virtual_support_size, virtual_support_size, CV_32FC1);
  output.valid_mask.create(virtual_support_size, virtual_support_size, CV_8UC1);
  output.values.setTo(std::numeric_limits<float>::quiet_NaN());
  output.valid_mask.setTo(0);
  output.R_v_from_c = R_v_from_c;
  output.R_c_from_v = R_v_from_c.transpose();
  output.valid = false;

  float *value_sum = patch_workspace.values.ptr<float>(0);
  float *weight_sum = patch_workspace.weight_sum.ptr<float>(0);
  uint8_t *workspace_mask = patch_workspace.valid_mask.ptr<uint8_t>(0);
  uint8_t *touched_flag = patch_workspace.touched_flag.data();

  auto add_splat_value = [&](int idx, float weight, float intensity) {
    if (weight == 0.0f) return;
    if (touched_flag[idx] == 0)
    {
      touched_flag[idx] = 1;
      patch_workspace.touched_indices.push_back(idx);
    }
    value_sum[idx] += weight * intensity;
    weight_sum[idx] += weight;
  };

  const int raw_center_x = static_cast<int>(std::lround(raw_center_px[0]));
  const int raw_center_y = static_cast<int>(std::lround(raw_center_px[1]));
  for (int raw_y = raw_center_y - virtual_raw_window_half_size; raw_y <= raw_center_y + virtual_raw_window_half_size; ++raw_y)
  {
    if (raw_y < 0 || raw_y >= height) continue;
    const uint8_t *raw_row = raw_img.ptr<uint8_t>(raw_y);
    for (int raw_x = raw_center_x - virtual_raw_window_half_size; raw_x <= raw_center_x + virtual_raw_window_half_size; ++raw_x)
    {
      if (raw_x < 0 || raw_x >= width) continue;
      const int raw_idx = raw_y * width + raw_x;
      if (!raw_pixel_unit_ray_valid_mask_[raw_idx]) continue;

      const V3D ray_c = raw_pixel_to_unit_ray_lut_[raw_idx].cast<double>();
      const V3D ray_v = R_v_from_c * ray_c;
      if (!ray_v.array().isFinite().all() || ray_v[2] <= virtual_min_z) continue;

      const float u_v = static_cast<float>(virtual_focal_length * ray_v[0] / ray_v[2] + virtual_support_radius);
      const float v_v = static_cast<float>(virtual_focal_length * ray_v[1] / ray_v[2] + virtual_support_radius);
      const int x0 = static_cast<int>(std::floor(u_v));
      const int y0 = static_cast<int>(std::floor(v_v));
      if (x0 < 0 || y0 < 0 || x0 + 1 >= virtual_support_size || y0 + 1 >= virtual_support_size) continue;

      const float dx = u_v - x0;
      const float dy = v_v - y0;
      const float w00 = (1.0f - dx) * (1.0f - dy);
      const float w10 = dx * (1.0f - dy);
      const float w01 = (1.0f - dx) * dy;
      const float w11 = dx * dy;
      const float intensity = static_cast<float>(raw_row[raw_x]);
      const int idx00 = y0 * virtual_support_size + x0;
      const int idx10 = idx00 + 1;
      const int idx01 = idx00 + virtual_support_size;
      const int idx11 = idx01 + 1;

      add_splat_value(idx00, w00, intensity);
      add_splat_value(idx10, w10, intensity);
      add_splat_value(idx01, w01, intensity);
      add_splat_value(idx11, w11, intensity);
    }
  }

  float *output_values = output.values.ptr<float>(0);
  uint8_t *output_mask = output.valid_mask.ptr<uint8_t>(0);
  for (const int idx : patch_workspace.touched_indices)
  {
    if (weight_sum[idx] > virtual_splat_min_weight)
    {
      output_values[idx] = value_sum[idx] / weight_sum[idx];
      output_mask[idx] = 255;
      workspace_mask[idx] = 255;
    }
  }

  const bool has_required_coverage = !virtual_splat_require_full_core_coverage || hasFullVirtualCoreCoverage(patch_workspace.valid_mask);
  clearVirtualPatchWorkspaceTouched(patch_workspace);
  if (!has_required_coverage) return false;
  output.valid = true;
  return true;
}

bool VIOManager::buildVirtualSupportPatch(const cv::Mat &raw_img, const V2D &raw_center_px, const M3D &R_v_from_c, const M3D &R_c_from_v,
                                          VirtualPatchImage &output) const
{
  if (virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::PULL_EXACT)
    return buildVirtualSupportPatchPullExact(raw_img, R_c_from_v, output);

  const double splat_start = omp_get_wtime();
  const bool splat_valid = buildVirtualSupportPatchForwardSplat(raw_img, raw_center_px, R_v_from_c, output);
  const double splat_time = omp_get_wtime() - splat_start;

  if (virtual_splat_debug_compare_pull_exact)
  {
    VirtualPatchImage pull_exact_patch;
    const double pull_start = omp_get_wtime();
    const bool pull_valid = buildVirtualSupportPatchPullExact(raw_img, R_c_from_v, pull_exact_patch);
    const double pull_time = omp_get_wtime() - pull_start;

    std::vector<float> absolute_errors;
    absolute_errors.reserve(patch_size_total);
    const V2D center(virtual_support_radius, virtual_support_radius);
    for (const V2F &offset : core_patch_offsets_)
    {
      float splat_value = 0.0f;
      float pull_value = 0.0f;
      const V2D px = center + offset.cast<double>();
      if (!splat_valid || !pull_valid ||
          !interpolateVirtualFloat(output.values, output.valid_mask, px[0], px[1], splat_value) ||
          !interpolateVirtualFloat(pull_exact_patch.values, pull_exact_patch.valid_mask, px[0], px[1], pull_value))
        continue;
      absolute_errors.push_back(std::fabs(splat_value - pull_value));
    }

    double mean_error = 0.0;
    double max_error = 0.0;
    double p95_error = 0.0;
    if (!absolute_errors.empty())
    {
      std::sort(absolute_errors.begin(), absolute_errors.end());
      max_error = absolute_errors.back();
      mean_error = std::accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) / absolute_errors.size();
      const size_t p95_index = std::min(absolute_errors.size() - 1, static_cast<size_t>(std::floor(0.95 * (absolute_errors.size() - 1))));
      p95_error = absolute_errors[p95_index];
    }

    printf("[ VIO Virtual Splat A/B ] pull_valid=%d splat_valid=%d valid_core=%zu mean_abs=%.6f max_abs=%.6f p95_abs=%.6f pull=%.6f s splat=%.6f s speedup=%.3f\n",
           pull_valid ? 1 : 0, splat_valid ? 1 : 0, absolute_errors.size(), mean_error, max_error, p95_error, pull_time, splat_time,
           splat_time > 1e-12 ? pull_time / splat_time : 0.0);
  }

  return splat_valid;
}

bool VIOManager::interpolateVirtualFloat(const cv::Mat &img, const cv::Mat &valid_mask, float u, float v, float &value) const
{
  if (!std::isfinite(u) || !std::isfinite(v) || img.empty() || valid_mask.empty()) return false;
  const int x = static_cast<int>(std::floor(u));
  const int y = static_cast<int>(std::floor(v));
  if (x < 0 || y < 0 || x + 1 >= img.cols || y + 1 >= img.rows) return false;
  if (valid_mask.at<uint8_t>(y, x) == 0 || valid_mask.at<uint8_t>(y, x + 1) == 0 ||
      valid_mask.at<uint8_t>(y + 1, x) == 0 || valid_mask.at<uint8_t>(y + 1, x + 1) == 0)
    return false;

  const float dx = u - x;
  const float dy = v - y;
  const float top = (1.0f - dx) * img.at<float>(y, x) + dx * img.at<float>(y, x + 1);
  const float bottom = (1.0f - dx) * img.at<float>(y + 1, x) + dx * img.at<float>(y + 1, x + 1);
  value = (1.0f - dy) * top + dy * bottom;
  return std::isfinite(value);
}

bool VIOManager::interpolateStoredVirtualImage(const cv::Mat &img, float u, float v, float &value) const
{
  if (!std::isfinite(u) || !std::isfinite(v) || img.empty() || img.type() != CV_32FC1) return false;

  const int x = static_cast<int>(std::floor(u));
  const int y = static_cast<int>(std::floor(v));
  if (x < 0 || y < 0 || x + 1 >= img.cols || y + 1 >= img.rows) return false;

  const float tl = img.at<float>(y, x);
  const float tr = img.at<float>(y, x + 1);
  const float bl = img.at<float>(y + 1, x);
  const float br = img.at<float>(y + 1, x + 1);
  if (!std::isfinite(tl) || !std::isfinite(tr) || !std::isfinite(bl) || !std::isfinite(br)) return false;

  const float dx = u - x;
  const float dy = v - y;
  const float top = (1.0f - dx) * tl + dx * tr;
  const float bottom = (1.0f - dx) * bl + dx * br;
  value = (1.0f - dy) * top + dy * bottom;
  return std::isfinite(value);
}

V2D VIOManager::virtualProject(const V3D &p_v) const
{
  if (!p_v.array().isFinite().all() || p_v[2] <= virtual_min_z)
    return V2D::Constant(std::numeric_limits<double>::quiet_NaN());
  return V2D(virtual_focal_length * p_v[0] / p_v[2] + virtual_support_radius,
             virtual_focal_length * p_v[1] / p_v[2] + virtual_support_radius);
}

V3D VIOManager::virtualCam2World(const V2D &px_v) const
{
  V3D ray((px_v[0] - virtual_support_radius) / virtual_focal_length,
          (px_v[1] - virtual_support_radius) / virtual_focal_length, 1.0);
  return ray.normalized();
}

void VIOManager::computeVirtualProjectionJacobian(const V3D &p_v, MD(2, 3) &J) const
{
  const double z_inv = 1.0 / p_v[2];
  const double z_inv_2 = z_inv * z_inv;
  J << virtual_focal_length * z_inv, 0.0, -virtual_focal_length * p_v[0] * z_inv_2, 0.0,
      virtual_focal_length * z_inv, -virtual_focal_length * p_v[1] * z_inv_2;
}

bool VIOManager::sampleVirtualCorePatch(const VirtualPatchImage &support, const V2D &center, int scale, float *patch) const
{
  if (!support.valid || patch == nullptr) return false;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x)
    {
      const V2F offset = core_patch_offsets_[y * patch_size + x] * static_cast<float>(scale);
      if (!interpolateVirtualFloat(support.values, support.valid_mask, center[0] + offset[0], center[1] + offset[1],
                                   patch[y * patch_size + x]))
        return false;
    }
  }
  return true;
}

bool VIOManager::sampleVirtualValueAndGradient(const VirtualPatchImage &support, const V2D &px, int scale, float &value, V2D &gradient) const
{
  float left, right, up, down;
  if (!interpolateVirtualFloat(support.values, support.valid_mask, px[0], px[1], value) ||
      !interpolateVirtualFloat(support.values, support.valid_mask, px[0] - scale, px[1], left) ||
      !interpolateVirtualFloat(support.values, support.valid_mask, px[0] + scale, px[1], right) ||
      !interpolateVirtualFloat(support.values, support.valid_mask, px[0], px[1] - scale, up) ||
      !interpolateVirtualFloat(support.values, support.valid_mask, px[0], px[1] + scale, down))
    return false;
  gradient << 0.5 * (right - left), 0.5 * (down - up);
  return true;
}

bool VIOManager::sampleStoredVirtualValueAndGradient(const cv::Mat &img, const V2D &px, int scale, float &value, V2D &gradient) const
{
  float left, right, up, down;
  if (!interpolateStoredVirtualImage(img, px[0], px[1], value) ||
      !interpolateStoredVirtualImage(img, px[0] - scale, px[1], left) ||
      !interpolateStoredVirtualImage(img, px[0] + scale, px[1], right) ||
      !interpolateStoredVirtualImage(img, px[0], px[1] - scale, up) ||
      !interpolateStoredVirtualImage(img, px[0], px[1] + scale, down))
    return false;
  gradient << 0.5 * (right - left), 0.5 * (down - up);
  return true;
}

bool VIOManager::createVirtualFeaturePatch(const cv::Mat &raw_img, const SE3<double> &T_c_w, const V3D &point_w, float *core_patch,
                                           cv::Mat &virtual_support_img, SE3<double> &T_v_w, M3D &R_v_from_c, M3D &R_c_from_v) const
{
  const V3D point_c = T_c_w * point_w;
  V2D raw_center_px;
  if (!projectRawFisheyeIfValid(point_c, 1, raw_center_px)) return false;
  if (!buildVirtualFrameRotation(point_c, R_v_from_c, R_c_from_v)) return false;
  T_v_w = composeVirtualPose(R_v_from_c, T_c_w);
  VirtualPatchImage support;
  support.T_v_w_seed = T_v_w;
  if (!buildVirtualSupportPatch(raw_img, raw_center_px, R_v_from_c, R_c_from_v, support)) return false;
  const V2D center(virtual_support_radius, virtual_support_radius);
  if (!sampleVirtualCorePatch(support, center, 1, core_patch)) return false;

  // Keep the complete immutable local virtual image for future warpAffineVirtual().
  // cv::Mat assignment keeps the underlying storage alive through reference counting.
  virtual_support_img = support.values;
  return true;
}

bool VIOManager::getWarpMatrixAffineVirtual(const V3D &xyz_ref, const SE3<double> &T_vcur_vref, int level_ref, int pyramid_level,
                                            int halfpatch_size, Matrix2d &A_cur_ref) const
{
  if (xyz_ref[2] <= virtual_min_z) return false;
  const V2D px_ref(virtual_support_radius, virtual_support_radius);
  const double offset = halfpatch_size * (1 << level_ref) * (1 << pyramid_level);
  V3D xyz_du_ref = virtualCam2World(px_ref + V2D(offset, 0.0));
  V3D xyz_dv_ref = virtualCam2World(px_ref + V2D(0.0, offset));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];

  const V3D xyz_cur = T_vcur_vref * xyz_ref;
  const V3D xyz_du_cur = T_vcur_vref * xyz_du_ref;
  const V3D xyz_dv_cur = T_vcur_vref * xyz_dv_ref;
  if (xyz_cur[2] <= virtual_min_z || xyz_du_cur[2] <= virtual_min_z || xyz_dv_cur[2] <= virtual_min_z) return false;
  const V2D px_cur = virtualProject(xyz_cur);
  const V2D px_du = virtualProject(xyz_du_cur);
  const V2D px_dv = virtualProject(xyz_dv_cur);
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
  return A_cur_ref.array().isFinite().all() && std::fabs(A_cur_ref.determinant()) > 1e-9;
}

bool VIOManager::getWarpMatrixAffineHomographyVirtual(const V3D &xyz_ref, const V3D &normal_ref, const SE3<double> &T_vcur_vref,
                                                      int level_ref, Matrix2d &A_cur_ref) const
{
  const V3D t = T_vcur_vref.inverse().translation();
  const M3D H_cur_ref =
      T_vcur_vref.rotationMatrix() * (normal_ref.dot(xyz_ref) * M3D::Identity() - t * normal_ref.transpose());
  const int halfpatch_size = 4;
  const V2D px_ref(virtual_support_radius, virtual_support_radius);
  const V3D f_ref = xyz_ref.normalized();
  const V3D f_du_ref = virtualCam2World(px_ref + V2D(halfpatch_size * (1 << level_ref), 0.0));
  const V3D f_dv_ref = virtualCam2World(px_ref + V2D(0.0, halfpatch_size * (1 << level_ref)));
  const V3D f_cur = H_cur_ref * f_ref;
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  if (f_cur[2] <= virtual_min_z || f_du_cur[2] <= virtual_min_z || f_dv_cur[2] <= virtual_min_z) return false;
  const V2D px_cur = virtualProject(f_cur);
  const V2D px_du = virtualProject(f_du_cur);
  const V2D px_dv = virtualProject(f_dv_cur);
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
  return A_cur_ref.array().isFinite().all() && std::fabs(A_cur_ref.determinant()) > 1e-9;
}

bool VIOManager::warpAffineVirtual(const Matrix2d &A_cur_ref, const cv::Mat &virtual_ref_img, int level_ref, int search_level,
                                   int pyramid_level, int halfpatch_size, float *patch) const
{
  (void)level_ref;
  if (virtual_ref_img.empty() || virtual_ref_img.type() != CV_32FC1 || std::fabs(A_cur_ref.determinant()) <= 1e-9) return false;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  const int local_patch_size = halfpatch_size * 2;
  const V2F center(virtual_support_radius, virtual_support_radius);
  for (int y = 0; y < local_patch_size; ++y)
  {
    for (int x = 0; x < local_patch_size; ++x)
    {
      V2F offset(x - halfpatch_size, y - halfpatch_size);
      offset *= static_cast<float>((1 << search_level) * (1 << pyramid_level));
      const V2F px = A_ref_cur * offset + center;
      float value;
      if (!interpolateStoredVirtualImage(virtual_ref_img, px[0], px[1], value)) return false;
      patch[patch_size_total * pyramid_level + y * local_patch_size + x] = value;
    }
  }
  return true;
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3<double> &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotationMatrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3<double> &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMapVirtual(cv::Mat img, vector<pointWithVar> &pg,
                                                    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  visual_submap->reset();
  total_points = 0;
  rejected_virtual_support_oob_ = 0;
  rejected_virtual_projection_invalid_ = 0;
  rejected_virtual_z_ = 0;
  rejected_virtual_affine_oob_ = 0;
  build_virtual_support_time_ = 0.0;
  virtual_affine_time_ = 0.0;
  virtual_candidate_select_time_ = 0.0;
  virtual_parallel_track_time_ = 0.0;
  virtual_result_collect_time_ = 0.0;
  virtual_warp_time_ = 0.0;
  virtual_current_core_time_ = 0.0;
  virtual_map_grid_count_ = 0;
  virtual_candidate_null_count_ = 0;
  virtual_candidate_normal_uninit_count_ = 0;
  virtual_candidate_projection_fail_count_ = 0;
  virtual_candidate_range_reject_count_ = 0;
  virtual_candidate_close_view_fail_count_ = 0;
  virtual_candidate_ref_missing_count_ = 0;
  virtual_candidate_ref_invalid_count_ = 0;
  virtual_candidate_count_ = 0;
  virtual_valid_track_count_ = 0;
  virtual_track_rotation_fail_count_ = 0;
  virtual_track_support_fail_count_ = 0;
  virtual_track_affine_fail_count_ = 0;
  virtual_track_warp_fail_count_ = 0;
  virtual_track_current_z_fail_count_ = 0;
  virtual_track_current_core_fail_count_ = 0;
  virtual_track_ncc_reject_count_ = 0;
  virtual_track_photometric_reject_count_ = 0;
  if (draw_rejected_points_en && virtual_fisheye_patch_en)
  {
    rejected_visual_points_for_draw_.clear();
    rejected_visual_points_for_draw_.reserve(length);
  }
  if (feat_map.empty()) return;

  const double candidate_select_start = omp_get_wtime();
  sub_feat_map.clear();

  const float voxel_size = 0.5f;
  cv::Mat range_img = cv::Mat::zeros(height, width, CV_32FC1);
  int loc_xyz[3];

  for (const auto &point : pg)
  {
    const V3D &pt_w = point.point_w;
    for (int j = 0; j < 3; ++j) loc_xyz[j] = static_cast<int>(std::floor(pt_w[j] / voxel_size));
    sub_feat_map[VOXEL_LOCATION(loc_xyz[0], loc_xyz[1], loc_xyz[2])] = 0;

    const V3D pt_c = new_frame_->w2f(pt_w);
    V2D px;
    if (!projectRawFisheyeIfValid(pt_c, 1, px)) continue;
    const int col = static_cast<int>(px[0]);
    const int row = static_cast<int>(px[1]);
    float &stored_range = range_img.at<float>(row, col);
    const float range = static_cast<float>(pt_c.norm());
    if (stored_range == 0.0f || range < stored_range) stored_range = range;
  }

  vector<VOXEL_LOCATION> delete_keys;
  for (auto &sub_voxel : sub_feat_map)
  {
    auto map_voxel = feat_map.find(sub_voxel.first);
    if (map_voxel == feat_map.end()) continue;

    bool voxel_in_fov = false;
    for (VisualPoint *pt : map_voxel->second->voxel_points)
    {
      if (pt == nullptr || pt->obs_.empty()) continue;
      V2D raw_px;
      if (!projectRawFisheyeIfValid(new_frame_->w2f(pt->pos_), 1, raw_px)) continue;
      const int grid_col = static_cast<int>(raw_px[0] / grid_size);
      const int grid_row = static_cast<int>(raw_px[1] / grid_size);
      if (grid_col < 0 || grid_col >= grid_n_width || grid_row < 0 || grid_row >= grid_n_height) continue;
      const int index = grid_row * grid_n_width + grid_col;
      voxel_in_fov = true;
      grid_num[index] = TYPE_MAP;
      const float cur_dist = static_cast<float>((new_frame_->pos() - pt->pos_).norm());
      if (cur_dist <= map_dist[index])
      {
        map_dist[index] = cur_dist;
        retrieve_voxel_points[index] = pt;
      }
    }
    if (!voxel_in_fov) delete_keys.push_back(sub_voxel.first);
  }

  if (raycast_en)
  {
    for (int i = 0; i < length; ++i)
    {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;
      for (const V3D &sample_c : rays_with_sample_points[i])
      {
        const V3D sample_w = new_frame_->f2w(sample_c);
        for (int j = 0; j < 3; ++j) loc_xyz[j] = static_cast<int>(std::floor(sample_w[j] / voxel_size));
        const VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
        if (sub_feat_map.find(sample_pos) != sub_feat_map.end()) break;

        auto visual_voxel = feat_map.find(sample_pos);
        if (visual_voxel != feat_map.end())
        {
          bool found = false;
          for (VisualPoint *pt : visual_voxel->second->voxel_points)
          {
            if (pt == nullptr || pt->obs_.empty()) continue;
            V2D raw_px;
            if (!projectRawFisheyeIfValid(new_frame_->w2f(pt->pos_), 1, raw_px)) continue;
            const int grid_col = static_cast<int>(raw_px[0] / grid_size);
            const int grid_row = static_cast<int>(raw_px[1] / grid_size);
            if (grid_col < 0 || grid_col >= grid_n_width || grid_row < 0 || grid_row >= grid_n_height) continue;
            const int index = grid_row * grid_n_width + grid_col;
            grid_num[index] = TYPE_MAP;
            const float cur_dist = static_cast<float>((new_frame_->pos() - pt->pos_).norm());
            if (cur_dist <= map_dist[index])
            {
              map_dist[index] = cur_dist;
              retrieve_voxel_points[index] = pt;
            }
            found = true;
          }
          if (found) sub_feat_map[sample_pos] = 0;
          break;
        }

        auto plane_voxel = plane_map.find(sample_pos);
        if (plane_voxel != plane_map.end())
        {
          VoxelOctoTree *octo = plane_voxel->second->find_correspond(sample_w);
          if (octo->plane_ptr_->is_plane_)
          {
            pointWithVar plane_center;
            plane_center.point_w = octo->plane_ptr_->center_;
            plane_center.normal = octo->plane_ptr_->normal_;
            visual_submap->add_from_voxel_map.push_back(plane_center);
            break;
          }
        }
      }
    }
  }
  for (const auto &key : delete_keys) sub_feat_map.erase(key);

  struct VirtualCandidate
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VisualPoint *point = nullptr;
    Feature *reference = nullptr;
    V3D point_c_seed = V3D::Zero();
    V2D current_raw_center_px = V2D::Zero();
  };
  enum VirtualRejectReason
  {
    VIRTUAL_REJECT_NONE = 0,
    VIRTUAL_REJECT_ROTATION,
    VIRTUAL_REJECT_SUPPORT_BUILD,
    VIRTUAL_REJECT_AFFINE_MATRIX,
    VIRTUAL_REJECT_REFERENCE_WARP,
    VIRTUAL_REJECT_CURRENT_Z,
    VIRTUAL_REJECT_CURRENT_CORE,
    VIRTUAL_REJECT_NCC,
    VIRTUAL_REJECT_PHOTOMETRIC
  };
  struct VirtualCandidateResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VirtualTrackPatch track;
    vector<float> warped_reference;
    float error = 0.0f;
    double inverse_reference_exposure = 0.0;
    double build_time = 0.0;
    double affine_time = 0.0;
    double warp_time = 0.0;
    double current_core_time = 0.0;
    int search_level = -1;
    int warp_fail_pyramid_level = -1;
    int rejection = 0;
    bool valid = false;
  };
  auto recordRejectedPoint = [&](const V2D &raw_px, int reason) {
    if (draw_rejected_points_en)
    {
      VIOManager::RejectedVisualPointForDraw rejected_point;
      rejected_point.px = cv::Point2f(static_cast<float>(raw_px[0]), static_cast<float>(raw_px[1]));
      rejected_point.reason = reason;
      rejected_visual_points_for_draw_.push_back(rejected_point);
    }
  };
  auto drawReasonFromVirtualReject = [](int rejection) {
    switch (rejection)
    {
    case VIRTUAL_REJECT_ROTATION:
      return VIOManager::REJECT_DRAW_ROTATION;
    case VIRTUAL_REJECT_SUPPORT_BUILD:
      return VIOManager::REJECT_DRAW_SUPPORT_BUILD;
    case VIRTUAL_REJECT_AFFINE_MATRIX:
      return VIOManager::REJECT_DRAW_AFFINE;
    case VIRTUAL_REJECT_REFERENCE_WARP:
      return VIOManager::REJECT_DRAW_WARP_REF;
    case VIRTUAL_REJECT_CURRENT_Z:
      return VIOManager::REJECT_DRAW_CURRENT_Z;
    case VIRTUAL_REJECT_CURRENT_CORE:
      return VIOManager::REJECT_DRAW_CURRENT_CORE;
    case VIRTUAL_REJECT_NCC:
      return VIOManager::REJECT_DRAW_NCC;
    case VIRTUAL_REJECT_PHOTOMETRIC:
      return VIOManager::REJECT_DRAW_PHOTOMETRIC;
    default:
      return VIOManager::REJECT_DRAW_RANGE;
    }
  };

  vector<VirtualCandidate> candidates;
  candidates.reserve(length);
  for (int i = 0; i < length; ++i)
  {
    if (grid_num[i] != TYPE_MAP) continue;
    ++virtual_map_grid_count_;
    VisualPoint *pt = retrieve_voxel_points[i];
    if (pt == nullptr)
    {
      ++virtual_candidate_null_count_;
      continue;
    }
    if (!pt->is_normal_initialized_)
    {
      ++virtual_candidate_normal_uninit_count_;
      if (draw_rejected_points_en)
      {
        V2D raw_px;
        if (projectRawFisheyeIfValid(new_frame_->w2f(pt->pos_), 1, raw_px))
          recordRejectedPoint(raw_px, REJECT_DRAW_NORMAL_UNINIT);
      }
      continue;
    }

    V2D raw_px;
    const V3D pt_c_seed = new_frame_->w2f(pt->pos_);
    if (!projectRawFisheyeIfValid(pt_c_seed, 1, raw_px))
    {
      ++virtual_candidate_projection_fail_count_;
      continue;
    }

    bool range_discontinuous = false;
    const double point_range = pt_c_seed.norm();
    const int center_col = static_cast<int>(raw_px[0]);
    const int center_row = static_cast<int>(raw_px[1]);
    for (int dv = -patch_size_half; dv <= patch_size_half && !range_discontinuous; ++dv)
    {
      for (int du = -patch_size_half; du <= patch_size_half; ++du)
      {
        if (du == 0 && dv == 0) continue;
        const int col = center_col + du;
        const int row = center_row + dv;
        if (col < 0 || col >= width || row < 0 || row >= height) continue;
        const float range = range_img.at<float>(row, col);
        if (range > 0.0f && std::fabs(point_range - range) > 0.5)
        {
          range_discontinuous = true;
          break;
        }
      }
    }
    if (range_discontinuous)
    {
      ++virtual_candidate_range_reject_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_RANGE);
      continue;
    }

    Feature *ref_ftr = nullptr;
    if (normal_en)
    {
      if (pt->has_ref_patch_ && pt->ref_patch != nullptr && pt->ref_patch->virtual_patch_valid_)
      {
        ref_ftr = pt->ref_patch;
      }
      else
      {
        float minimum_error = std::numeric_limits<float>::max();
        for (Feature *candidate_ref : pt->obs_)
        {
          if (candidate_ref == nullptr || !candidate_ref->virtual_patch_valid_) continue;
          float error = 0.0f;
          int comparisons = 0;
          for (Feature *other : pt->obs_)
          {
            if (other == nullptr || other == candidate_ref || !other->virtual_patch_valid_) continue;
            for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
            {
              const float residual = candidate_ref->patch_[patch_index] - other->patch_[patch_index];
              error += residual * residual;
            }
            ++comparisons;
          }
          if (comparisons > 0) error /= comparisons;
          if (error < minimum_error)
          {
            minimum_error = error;
            ref_ftr = candidate_ref;
          }
        }
        if (ref_ftr != nullptr)
        {
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
      }
    }
    else if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, raw_px))
    {
      ++virtual_candidate_close_view_fail_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_CLOSE_VIEW);
      continue;
    }
    if (ref_ftr == nullptr)
    {
      ++virtual_candidate_ref_missing_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_REF_MISSING);
      continue;
    }
    if (!ref_ftr->virtual_patch_valid_)
    {
      ++virtual_candidate_ref_invalid_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_REF_INVALID);
      continue;
    }

    VirtualCandidate candidate;
    candidate.point = pt;
    candidate.reference = ref_ftr;
    candidate.point_c_seed = pt_c_seed;
    candidate.current_raw_center_px = raw_px;
    candidates.push_back(candidate);
  }
  virtual_candidate_select_time_ = omp_get_wtime() - candidate_select_start;
  virtual_candidate_count_ = static_cast<int>(candidates.size());

  vector<VirtualCandidateResult> results(candidates.size());
  const double parallel_track_start = omp_get_wtime();
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index)
  {
    const VirtualCandidate &candidate = candidates[candidate_index];
    VirtualCandidateResult &result = results[candidate_index];
    VisualPoint *pt = candidate.point;
    Feature *ref_ftr = candidate.reference;

    if (!buildVirtualFrameRotation(candidate.point_c_seed, result.track.R_vcur_from_ccur_seed, result.track.R_ccur_from_vcur_seed))
    {
      result.rejection = VIRTUAL_REJECT_ROTATION;
      continue;
    }
    result.track.T_vcur_w_seed = composeVirtualPose(result.track.R_vcur_from_ccur_seed, new_frame_->T_f_w_);
    result.track.cur_support.T_v_w_seed = result.track.T_vcur_w_seed;

    const double build_start = omp_get_wtime();
    const bool supports_ok = buildVirtualSupportPatch(img, candidate.current_raw_center_px, result.track.R_vcur_from_ccur_seed,
                                                      result.track.R_ccur_from_vcur_seed, result.track.cur_support);
    result.build_time = omp_get_wtime() - build_start;
    if (!supports_ok)
    {
      result.rejection = VIRTUAL_REJECT_SUPPORT_BUILD;
      continue;
    }

    const SE3<double> T_vcur_vref = result.track.T_vcur_w_seed * ref_ftr->T_v_w_.inverse();
    const V3D point_vref = ref_ftr->T_v_w_ * pt->pos_;
    const double affine_start = omp_get_wtime();
    bool affine_ok;
    if (normal_en)
    {
      const V3D normal_vref = ref_ftr->T_v_w_.rotationMatrix() * pt->normal_;
      affine_ok = getWarpMatrixAffineHomographyVirtual(point_vref, normal_vref, T_vcur_vref, ref_ftr->level_, result.track.A_cur_ref);
    }
    else
    {
      affine_ok = getWarpMatrixAffineVirtual(point_vref, T_vcur_vref, ref_ftr->level_, 0, patch_size_half, result.track.A_cur_ref);
    }
    result.affine_time = omp_get_wtime() - affine_start;
    if (!affine_ok)
    {
      result.rejection = VIRTUAL_REJECT_AFFINE_MATRIX;
      continue;
    }
    result.track.search_level = getBestSearchLevel(result.track.A_cur_ref, virtual_max_search_level);
    result.search_level = result.track.search_level;

    const double warp_start = omp_get_wtime();
    result.warped_reference.assign(warp_len, 0.0f);
    for (int pyramid_level = 0; pyramid_level < patch_pyrimid_level; ++pyramid_level)
    {
      // [MODIFY] 使用第一次生成的参考 patch，不再每帧重构
      if (!warpAffineVirtual(result.track.A_cur_ref, ref_ftr->img_, ref_ftr->level_, result.track.search_level, pyramid_level,
                             patch_size_half, result.warped_reference.data()))
      {
        result.rejection = VIRTUAL_REJECT_REFERENCE_WARP;
        result.warp_fail_pyramid_level = pyramid_level;
        break;
      }
    }
    result.warp_time = omp_get_wtime() - warp_start;
    if (result.rejection != 0) continue;

    const double current_core_start = omp_get_wtime();
    vector<float> current_core(patch_size_total);
    const V3D point_vcur = result.track.T_vcur_w_seed * pt->pos_;
    if (point_vcur[2] <= virtual_min_z)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_Z;
      continue;
    }
    if (!sampleVirtualCorePatch(result.track.cur_support, virtualProject(point_vcur), 1, current_core.data()))
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_CORE;
      continue;
    }

    for (int k = 0; k < patch_size_total; ++k)
    {
      const double residual = ref_ftr->inv_expo_time_ * result.warped_reference[k] - state->inv_expo_time * current_core[k];
      result.error += residual * residual;
    }
    if (ncc_en && calculateNCC(result.warped_reference.data(), current_core.data(), patch_size_total) < ncc_thre)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_NCC;
      continue;
    }
    if (result.error > outlier_threshold * patch_size_total)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_PHOTOMETRIC;
      continue;
    }
    result.current_core_time = omp_get_wtime() - current_core_start;

    result.inverse_reference_exposure = ref_ftr->inv_expo_time_;
    result.track.valid = true;
    result.valid = true;
  }
  virtual_parallel_track_time_ = omp_get_wtime() - parallel_track_start;

  const double result_collect_start = omp_get_wtime();
  vector<int> virtual_search_level_count(std::max(virtual_max_search_level + 1, 1), 0);
  vector<int> virtual_warp_fail_search_level_count(std::max(virtual_max_search_level + 1, 1), 0);
  vector<int> virtual_warp_fail_pyramid_level_count(std::max(patch_pyrimid_level, 1), 0);
  for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index)
  {
    VirtualCandidateResult &result = results[candidate_index];
    build_virtual_support_time_ += result.build_time;
    virtual_affine_time_ += result.affine_time;
    virtual_warp_time_ += result.warp_time;
    virtual_current_core_time_ += result.current_core_time;
    if (result.search_level >= 0 && result.search_level < static_cast<int>(virtual_search_level_count.size()))
      ++virtual_search_level_count[result.search_level];
    if (!result.valid)
    {
      if (result.rejection != VIRTUAL_REJECT_NONE)
        recordRejectedPoint(candidates[candidate_index].current_raw_center_px, drawReasonFromVirtualReject(result.rejection));
      switch (result.rejection)
      {
      case VIRTUAL_REJECT_ROTATION:
        ++virtual_track_rotation_fail_count_;
        break;
      case VIRTUAL_REJECT_SUPPORT_BUILD:
        ++virtual_track_support_fail_count_;
        break;
      case VIRTUAL_REJECT_AFFINE_MATRIX:
        ++virtual_track_affine_fail_count_;
        break;
      case VIRTUAL_REJECT_REFERENCE_WARP:
        ++virtual_track_warp_fail_count_;
        if (result.search_level >= 0 && result.search_level < static_cast<int>(virtual_warp_fail_search_level_count.size()))
          ++virtual_warp_fail_search_level_count[result.search_level];
        if (result.warp_fail_pyramid_level >= 0 &&
            result.warp_fail_pyramid_level < static_cast<int>(virtual_warp_fail_pyramid_level_count.size()))
          ++virtual_warp_fail_pyramid_level_count[result.warp_fail_pyramid_level];
        break;
      case VIRTUAL_REJECT_CURRENT_Z:
        ++virtual_track_current_z_fail_count_;
        break;
      case VIRTUAL_REJECT_CURRENT_CORE:
        ++virtual_track_current_core_fail_count_;
        break;
      case VIRTUAL_REJECT_NCC:
        ++virtual_track_ncc_reject_count_;
        break;
      case VIRTUAL_REJECT_PHOTOMETRIC:
        ++virtual_track_photometric_reject_count_;
        break;
      default:
        break;
      }
      continue;
    }

    ++virtual_valid_track_count_;
    visual_submap->voxel_points.push_back(candidates[candidate_index].point);
    visual_submap->propa_errors.push_back(result.error);
    visual_submap->search_levels.push_back(result.track.search_level);
    visual_submap->errors.push_back(result.error);
    visual_submap->warp_patch.push_back(std::move(result.warped_reference));
    visual_submap->inv_expo_list.push_back(result.inverse_reference_exposure);
    visual_submap->virtual_track_patches.push_back(std::move(result.track));
  }
  virtual_result_collect_time_ = omp_get_wtime() - result_collect_start;

  total_points = static_cast<int>(visual_submap->voxel_points.size());
  rejected_virtual_support_oob_ = virtual_track_support_fail_count_ + virtual_track_current_core_fail_count_;
  rejected_virtual_projection_invalid_ = virtual_candidate_projection_fail_count_;
  rejected_virtual_z_ = virtual_track_rotation_fail_count_ + virtual_track_current_z_fail_count_;
  rejected_virtual_affine_oob_ = virtual_track_affine_fail_count_ + virtual_track_warp_fail_count_;

  auto formatCountVector = [](const vector<int> &counts) {
    std::string text;
    for (int i = 0; i < static_cast<int>(counts.size()); ++i)
    {
      if (!text.empty()) text += ",";
      text += std::to_string(i) + ":" + std::to_string(counts[i]);
    }
    return text;
  };
  const std::string search_level_text = formatCountVector(virtual_search_level_count);
  const std::string warp_fail_search_text = formatCountVector(virtual_warp_fail_search_level_count);
  const std::string warp_fail_pyramid_text = formatCountVector(virtual_warp_fail_pyramid_level_count);

  printf("[ VIO Virtual ] Retrieve %d/%d tracked from map_grid=%d candidates=%d\n",
         total_points, virtual_candidate_count_, virtual_map_grid_count_, virtual_candidate_count_);
  printf("[ VIO Virtual Candidate ] null=%d normal_uninit=%d proj_fail=%d range_reject=%d close_view_fail=%d ref_missing=%d ref_invalid=%d select_wall=%.6f s\n",
         virtual_candidate_null_count_, virtual_candidate_normal_uninit_count_, virtual_candidate_projection_fail_count_,
         virtual_candidate_range_reject_count_, virtual_candidate_close_view_fail_count_, virtual_candidate_ref_missing_count_,
         virtual_candidate_ref_invalid_count_, virtual_candidate_select_time_);
  printf("[ VIO Virtual Reject ] rotation=%d support_build=%d affine_matrix=%d warp_ref=%d current_z=%d current_core=%d ncc=%d photometric=%d\n",
         virtual_track_rotation_fail_count_, virtual_track_support_fail_count_, virtual_track_affine_fail_count_,
         virtual_track_warp_fail_count_, virtual_track_current_z_fail_count_, virtual_track_current_core_fail_count_,
         virtual_track_ncc_reject_count_, virtual_track_photometric_reject_count_);
  printf("[ VIO Virtual Warp ] search_level{%s} warp_fail_search{%s} warp_fail_pyramid{%s}\n",
         search_level_text.c_str(), warp_fail_search_text.c_str(), warp_fail_pyramid_text.c_str());
  printf("[ VIO Virtual Timing ] retrieve_wall=%.6f s parallel_wall=%.6f s support_sum=%.6f s affine_sum=%.6f s warp_sum=%.6f s current_core_sum=%.6f s result_collect_wall=%.6f s\n",
         virtual_candidate_select_time_ + virtual_parallel_track_time_ + virtual_result_collect_time_,
         virtual_parallel_track_time_, build_virtual_support_time_, virtual_affine_time_, virtual_warp_time_,
         virtual_current_core_time_, virtual_result_collect_time_);
  printf("[ VIO Virtual Timing Note ] *_sum accumulates per-candidate time across OpenMP workers; wall fields are real elapsed time.\n");
}

void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (virtual_fisheye_patch_en)
  {
    retrieveFromVisualSparseMapVirtual(img, pg, plane_map);
    return;
  }
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  for (int i = 0; i < pg.size(); i++)
  {
    // double t0 = omp_get_wtime();

    V3D pt_w = pg[i].point_w;

    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = new_frame_->cam_->world2cam(pt_c);

      if (new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : sub_feat_map)
  {
    VOXEL_LOCATION position = iter.first;

    // double t4 = omp_get_wtime();
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++)
      {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist[index])
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < length; i++)
    {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points[i])
      {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end())
        {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index])
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_)
            {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; }
      }
      else
      {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en)
      {
        V3D norm_vec = (ref_ftr->T_f_w_.rotationMatrix() * pt->normal_).normalized();
        
        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();
        
        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  compute_jacobian_time = update_ekf_time = 0.0;
  if (total_points == 0) return;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

void VIOManager::generateVisualMapPointsVirtual(cv::Mat img, vector<pointWithVar> &pg)
{
  if (pg.size() <= 10) return;

  auto consider_candidate = [&](const pointWithVar &candidate) {
    if (candidate.normal == V3D::Zero()) return;
    V2D raw_px;
    if (!projectRawFisheyeIfValid(new_frame_->w2f(candidate.point_w), border, raw_px)) return;
    const int grid_col = static_cast<int>(raw_px[0] / grid_size);
    const int grid_row = static_cast<int>(raw_px[1] / grid_size);
    if (grid_col < 0 || grid_col >= grid_n_width || grid_row < 0 || grid_row >= grid_n_height) return;
    const int index = grid_row * grid_n_width + grid_col;
    if (grid_num[index] == TYPE_MAP) return;

    const float score = vk::shiTomasiScore(img, raw_px[0], raw_px[1]);
    if (!std::isfinite(score)) return;
    if (score > scan_value[index])
    {
      scan_value[index] = score;
      append_voxel_points[index] = candidate;
      grid_num[index] = TYPE_POINTCLOUD;
    }
  };

  for (const auto &candidate : pg) consider_candidate(candidate);
  for (const auto &candidate : visual_submap->add_from_voxel_map) consider_candidate(candidate);

  int add = 0;
  for (int i = 0; i < length; ++i)
  {
    if (grid_num[i] != TYPE_POINTCLOUD) continue;
    const pointWithVar &pt_var = append_voxel_points[i];
    V2D raw_px;
    if (!projectRawFisheyeIfValid(new_frame_->w2f(pt_var.point_w), 1, raw_px)) continue;

    std::unique_ptr<float[]> patch(new float[patch_size_total]);
    cv::Mat virtual_support_img;
    SE3<double> T_v_w;
    M3D R_v_from_c, R_c_from_v;
    if (!createVirtualFeaturePatch(img, new_frame_->T_f_w_, pt_var.point_w, patch.get(), virtual_support_img, T_v_w, R_v_from_c,
                                   R_c_from_v))
    {
      ++rejected_virtual_support_oob_;
      continue;
    }

    VisualPoint *pt_new = new VisualPoint(pt_var.point_w);
    const V3D bearing = cam->cam2world(raw_px);
    Feature *ftr_new = new Feature(pt_new, patch.release(), raw_px, bearing, new_frame_->T_f_w_, 0);
    // [MODIFY] 使用第一次生成的参考 patch，不再每帧重构
    ftr_new->img_ = virtual_support_img;
    ftr_new->id_ = new_frame_->id_;
    ftr_new->inv_expo_time_ = state->inv_expo_time;
    ftr_new->T_v_w_ = T_v_w;
    ftr_new->R_v_from_c_ = R_v_from_c;
    ftr_new->R_c_from_v_ = R_c_from_v;
    ftr_new->virtual_patch_valid_ = true;

    pt_new->addFrameRef(ftr_new);
    pt_new->covariance_ = pt_var.var;
    pt_new->is_normal_initialized_ = true;
    const V3D dir = new_frame_->w2f(pt_var.point_w).normalized();
    const V3D normal_c = new_frame_->T_f_w_.rotationMatrix() * pt_var.normal;
    pt_new->normal_ = dir.dot(normal_c) < 0.0 ? -pt_var.normal : pt_var.normal;
    pt_new->previous_normal_ = pt_new->normal_;
    insertPointIntoVoxelMap(pt_new);
    ++add;
  }
  printf("[ VIO Virtual ] Append %d new visual map points\n", add);
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  if (virtual_fisheye_patch_en)
  {
    generateVisualMapPointsVirtual(img, pg);
    return;
  }
  if (pg.size() <= 10) return;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    V3D pt = pg[i].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }
      
      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points\n", add);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::updateVisualMapPointsVirtual(cv::Mat img)
{
  if (total_points == 0) return;

  int update_num = 0;
  const SE3<double> pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; ++i)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    {
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D raw_px;
    if (!projectRawFisheyeIfValid(new_frame_->w2f(pt->pos_), 1, raw_px)) continue;
    Feature *last_feature = pt->obs_.back();
    bool add_flag = false;
    const SE3<double> delta_pose = last_feature->T_f_w_ * pose_cur.inverse();
    const double delta_p = delta_pose.translation().norm();
    const double trace = delta_pose.rotationMatrix().trace();
    const double delta_theta = trace > 3.0 - 1e-6 ? 0.0 : std::acos(std::clamp(0.5 * (trace - 1.0), -1.0, 1.0));
    if (delta_p > 0.5 || delta_theta > 0.3 || (raw_px - last_feature->px_).norm() > 40.0) add_flag = true;

    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
    }
    if (!add_flag) continue;

    std::unique_ptr<float[]> patch(new float[patch_size_total]);
    cv::Mat virtual_support_img;
    SE3<double> T_v_w;
    M3D R_v_from_c, R_c_from_v;
    if (!createVirtualFeaturePatch(img, new_frame_->T_f_w_, pt->pos_, patch.get(), virtual_support_img, T_v_w, R_v_from_c, R_c_from_v))
    {
      ++rejected_virtual_support_oob_;
      continue;
    }

    const V3D bearing = cam->cam2world(raw_px);
    Feature *ftr_new = new Feature(pt, patch.release(), raw_px, bearing, new_frame_->T_f_w_, visual_submap->search_levels[i]);
    // [MODIFY] 使用第一次生成的参考 patch，不再每帧重构
    ftr_new->img_ = virtual_support_img;
    ftr_new->id_ = new_frame_->id_;
    ftr_new->inv_expo_time_ = state->inv_expo_time;
    ftr_new->T_v_w_ = T_v_w;
    ftr_new->R_v_from_c_ = R_v_from_c;
    ftr_new->R_c_from_v_ = R_c_from_v;
    ftr_new->virtual_patch_valid_ = true;
    pt->addFrameRef(ftr_new);
    update_flag[i] = 1;
    ++update_num;
  }
  printf("[ VIO Virtual ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (virtual_fisheye_patch_en)
  {
    updateVisualMapPointsVirtual(img);
    return;
  }
  if (total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;
    
    float *patch_temp = new float[patch_size_total];
    getImagePatch(img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta = (delta_pose.rotationMatrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotationMatrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      update_num += 1;
      update_flag[i] = 1;
      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * plane.normal_);
            // V3D pf(new_frame_->T_f_w_ * pt->pos_);
            // V3D pf_ref(pt->ref_patch->T_f_w_ * pt->pos_);
            // V3D norm_vec_ref(pt->ref_patch->T_f_w_.rotation_matrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);
            
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotationMatrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean = ref_patch_temp->mean_;
      if (abs(ref_mean) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean = (*itm)->mean_;
        if (abs(other_mean) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (virtual_fisheye_patch_en)
  {
    static bool warned = false;
    if (!warned)
    {
      printf("[ VIO Virtual ] projectPatchFromRefToCur raw-coordinate debug output is disabled in virtual patch mode.\n");
      warned = true;
    }
    return;
  }
  if (total_points == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotationMatrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatchesVirtual(int level)
{
  if (total_points == 0) return;
  const int H_DIM = total_points * patch_size_total;
  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();

  for (int i = 0; i < total_points; ++i)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr || pt->ref_patch == nullptr || !pt->ref_patch->virtual_patch_valid_) continue;
    if (i >= static_cast<int>(visual_submap->virtual_track_patches.size())) continue;

    const V3D point_vref = pt->ref_patch->T_v_w_ * pt->pos_;
    if (point_vref[2] <= virtual_min_z) continue;
    const V2D center = virtualProject(point_vref);
    MD(2, 3) Jdpi;
    computeVirtualProjectionJacobian(point_vref, Jdpi);
    const M3D R_vref_w = pt->ref_patch->T_v_w_.rotationMatrix();
    M3D p_w_hat;
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);
    const int scale = 1 << level;

    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x)
      {
        const int patch_index = y * patch_size + x;
        const V2F offset = core_patch_offsets_[patch_index] * static_cast<float>(scale);
        float value;
        V2D gradient;
        // [MODIFY] 使用第一次生成的参考 patch，不再每帧重构
        if (!sampleStoredVirtualValueAndGradient(pt->ref_patch->img_, center + offset.cast<double>(), scale, value, gradient)) continue;
        MD(1, 2) Jimg;
        Jimg << gradient[0] / scale, gradient[1] / scale;
        const MD(1, 3) JdR = Jimg * Jdpi * R_vref_w * p_w_hat;
        const MD(1, 3) Jdt = -Jimg * Jdpi * R_vref_w;
        H_sub_inv.block<1, 6>(i * patch_size_total + patch_index, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::precomputeReferencePatches(int level)
{
  if (virtual_fisheye_patch_en)
  {
    precomputeReferencePatchesVirtual(level);
    return;
  }
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points; i++)
  {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_;

    if (pt == nullptr) continue;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotationMatrix();

    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverseVirtual(cv::Mat img, int level)
{
  (void)img;
  if (total_points == 0) return;
  StatesGroup old_state = *state;
  const int H_DIM = total_points * patch_size_total;
  VectorXd z(H_DIM);
  MatrixXd H_sub(H_DIM, 6);
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  for (int iteration = 0; iteration < max_iterations; ++iteration)
  {
    const double t1 = omp_get_wtime();
    if (!has_ref_patch_cache) precomputeReferencePatchesVirtual(level);
    z.setZero();
    H_sub.setZero();

    M3D Rwi(state->rot_end);
    const V3D Pwi(state->pos_end);
    M3D P_wi_hat;
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    int n_meas = 0;
    double error = 0.0;
    for (int i = 0; i < total_points; ++i)
    {
      VisualPoint *pt = visual_submap->voxel_points[i];
      if (pt == nullptr || i >= static_cast<int>(visual_submap->virtual_track_patches.size())) continue;
      const VirtualTrackPatch &track = visual_submap->virtual_track_patches[i];
      const V3D point_c = Rcw * pt->pos_ + Pcw;
      const V3D point_v = track.R_vcur_from_ccur_seed * point_c;
      if (point_v[2] <= virtual_min_z)
      {
        ++rejected_virtual_z_;
        continue;
      }
      const V2D center = virtualProject(point_v);
      const int scale = 1 << level;

      vector<float> current_values(patch_size_total);
      bool patch_valid = true;
      for (int y = 0; y < patch_size && patch_valid; ++y)
      {
        for (int x = 0; x < patch_size; ++x)
        {
          const int patch_index = y * patch_size + x;
          const V2F offset = core_patch_offsets_[patch_index] * static_cast<float>(scale);
          if (!interpolateVirtualFloat(track.cur_support.values, track.cur_support.valid_mask, center[0] + offset[0],
                                       center[1] + offset[1], current_values[patch_index]))
          {
            patch_valid = false;
            break;
          }
        }
      }
      if (!patch_valid)
      {
        ++rejected_virtual_support_oob_;
        continue;
      }

      double patch_error = 0.0;
      const vector<float> &reference = visual_submap->warp_patch[i];
      for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
      {
        const int row = i * patch_size_total + patch_index;
        const double residual = current_values[patch_index] - reference[patch_size_total * level + patch_index];
        z(row) = residual;
        patch_error += residual * residual;
        const MD(1, 3) J_dR = H_sub_inv.block<1, 3>(row, 0);
        const MD(1, 3) J_dt = H_sub_inv.block<1, 3>(row, 3);
        const MD(1, 3) JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
        const MD(1, 3) Jdt = J_dt * Rwi;
        H_sub.block<1, 6>(row, 0) << JdR, Jdt;
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
      n_meas += patch_size_total;
    }
    if (n_meas == 0) return;
    error /= n_meas;
    compute_jacobian_time += omp_get_wtime() - t1;

    const double t3 = omp_get_wtime();
    if (error <= last_error)
    {
      old_state = *state;
      last_error = static_cast<float>(error);
      const auto H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      const MD(DIM_STATE, DIM_STATE) K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      const auto HTz = H_sub_T * z;
      const MD(DIM_STATE, 1) vec = *state_propagat - *state;
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      const MD(DIM_STATE, 1) solution =
          -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      *state += solution;
      if (solution.block<3, 1>(0, 0).norm() * 57.3f < 0.001f && solution.block<3, 1>(3, 0).norm() * 100.0f < 0.001f)
        EKF_end = true;
    }
    else
    {
      *state = old_state;
      EKF_end = true;
    }
    update_ekf_time += omp_get_wtime() - t3;
    if (EKF_end) break;
  }
}

void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (virtual_fisheye_patch_en)
  {
    updateStateInverseVirtual(img, level);
    return;
  }
  if (total_points == 0) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break; 
  }
}

void VIOManager::updateStateVirtual(cv::Mat img, int level)
{
  (void)img;
  if (total_points == 0) return;
  StatesGroup old_state = *state;
  const int H_DIM = total_points * patch_size_total;
  VectorXd z(H_DIM);
  MatrixXd H_sub(H_DIM, 7);
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  for (int iteration = 0; iteration < max_iterations; ++iteration)
  {
    const double t1 = omp_get_wtime();
    z.setZero();
    H_sub.setZero();

    M3D Rwi(state->rot_end);
    const V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();

    double error = 0.0;
    int n_meas = 0;
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for reduction(+ : error, n_meas)
#endif
    for (int i = 0; i < total_points; ++i)
    {
      VisualPoint *pt = visual_submap->voxel_points[i];
      if (pt == nullptr || i >= static_cast<int>(visual_submap->virtual_track_patches.size())) continue;
      const VirtualTrackPatch &track = visual_submap->virtual_track_patches[i];
      const V3D point_c = Rcw * pt->pos_ + Pcw;
      const V3D point_v = track.R_vcur_from_ccur_seed * point_c;
      if (point_v[2] <= virtual_min_z) continue;
      const V2D center = virtualProject(point_v);
      const int search_level = visual_submap->search_levels[i];
      const int pyramid_level = level + search_level;
      const int scale = 1 << pyramid_level;
      const double inv_scale = 1.0 / scale;

      vector<float> current_values(patch_size_total);
      vector<V2D> gradients(patch_size_total);
      bool patch_valid = true;
      for (int y = 0; y < patch_size && patch_valid; ++y)
      {
        for (int x = 0; x < patch_size; ++x)
        {
          const int patch_index = y * patch_size + x;
          const V2F offset = core_patch_offsets_[patch_index] * static_cast<float>(scale);
          if (!sampleVirtualValueAndGradient(track.cur_support, center + offset.cast<double>(), scale, current_values[patch_index],
                                             gradients[patch_index]))
          {
            patch_valid = false;
            break;
          }
        }
      }
      if (!patch_valid) continue;

      MD(2, 3) Jdpi;
      computeVirtualProjectionJacobian(point_v, Jdpi);
      M3D point_c_hat;
      point_c_hat << SKEW_SYM_MATRX(point_c);
      const vector<float> &reference = visual_submap->warp_patch[i];
      const double inv_ref_expo = visual_submap->inv_expo_list[i];
      double patch_error = 0.0;

      for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
      {
        MD(1, 2) Jimg;
        Jimg << gradients[patch_index][0], gradients[patch_index][1];
        Jimg *= state->inv_expo_time * inv_scale;
        const MD(1, 3) Jimg_Jpi_R = Jimg * Jdpi * track.R_vcur_from_ccur_seed;
        const MD(1, 3) Jdphi = Jimg_Jpi_R * point_c_hat;
        const MD(1, 3) Jdp = -Jimg_Jpi_R;
        const MD(1, 3) JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
        const MD(1, 3) Jdt = Jdp * Jdp_dt;
        const double cur_value = current_values[patch_index];
        const double residual =
            state->inv_expo_time * cur_value - inv_ref_expo * reference[patch_size_total * level + patch_index];
        const int row = i * patch_size_total + patch_index;
        z(row) = residual;
        if (exposure_estimate_en) H_sub.block<1, 7>(row, 0) << JdR, Jdt, cur_value;
        else H_sub.block<1, 6>(row, 0) << JdR, Jdt;
        patch_error += residual * residual;
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
      n_meas += patch_size_total;
    }
    if (n_meas == 0) return;
    error /= n_meas;
    compute_jacobian_time += omp_get_wtime() - t1;

    const double t3 = omp_get_wtime();
    if (error <= last_error)
    {
      old_state = *state;
      last_error = static_cast<float>(error);
      const auto H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      const MD(DIM_STATE, DIM_STATE) K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      const auto HTz = H_sub_T * z;
      const MD(DIM_STATE, 1) vec = *state_propagat - *state;
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      const MD(DIM_STATE, 1) solution =
          -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);
      *state += solution;
      if (solution.block<3, 1>(0, 0).norm() * 57.3f < 0.001f && solution.block<3, 1>(3, 0).norm() * 100.0f < 0.001f)
        EKF_end = true;
    }
    else
    {
      *state = old_state;
      EKF_end = true;
    }
    update_ekf_time += omp_get_wtime() - t3;
    if (EKF_end) break;
  }
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (virtual_fisheye_patch_en)
  {
    updateStateVirtual(img, level);
    return;
  }
  if (total_points == 0) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();
    
    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);
  
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      V2D pc = cam->world2cam(pf);

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          n_meas += 1;
          
          if (exposure_estimate_en) { H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; }
          else { H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;
    
    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Eigen::Quaterniond(Rcw).normalized().toRotationMatrix(), Pcw);  // avoid R is not orthogonal
}

void VIOManager::plotTrackedPoints()
{
  int total_points = visual_submap->voxel_points.size();
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    V2D pc;
    if (virtual_fisheye_patch_en)
    {
      if (!projectRawFisheyeIfValid(new_frame_->w2f(pt->pos_), 1, pc)) continue;
    }
    else
    {
      pc = new_frame_->w2c(pt->pos_);
    }

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  if (draw_rejected_points_en)
  {
    auto rejectedPointColor = [](int reason) {
      switch (reason)
      {
      case VIOManager::REJECT_DRAW_NORMAL_UNINIT:
        return cv::Scalar(255, 255, 255); // white
      case VIOManager::REJECT_DRAW_RANGE:
        return cv::Scalar(255, 0, 255);   // purple
      case VIOManager::REJECT_DRAW_CLOSE_VIEW:
        return cv::Scalar(0, 128, 255);   // orange
      case VIOManager::REJECT_DRAW_REF_MISSING:
        return cv::Scalar(255, 255, 0);   // cyan
      case VIOManager::REJECT_DRAW_REF_INVALID:
        return cv::Scalar(255, 128, 0);   // sky blue
      case VIOManager::REJECT_DRAW_ROTATION:
        return cv::Scalar(0, 80, 160);    // brown
      case VIOManager::REJECT_DRAW_SUPPORT_BUILD:
        return cv::Scalar(0, 255, 255);   // yellow
      case VIOManager::REJECT_DRAW_AFFINE:
        return cv::Scalar(0, 0, 160);     // dark red
      case VIOManager::REJECT_DRAW_WARP_REF:
        return cv::Scalar(0, 200, 255);   // orange-yellow
      case VIOManager::REJECT_DRAW_CURRENT_Z:
        return cv::Scalar(180, 180, 255); // light pink
      case VIOManager::REJECT_DRAW_CURRENT_CORE:
        return cv::Scalar(128, 0, 255);   // rose
      case VIOManager::REJECT_DRAW_NCC:
        return cv::Scalar(255, 0, 128);   // violet
      case VIOManager::REJECT_DRAW_PHOTOMETRIC:
        return cv::Scalar(0, 0, 255);     // red
      default:
        return cv::Scalar(0, 0, 255);
      }
    };
    for (const VIOManager::RejectedVisualPointForDraw &rejected_pt : rejected_visual_points_for_draw_)
    {
      cv::circle(img_cp, rejected_pt.px, 5, rejectedPointColor(rejected_pt.reason), -1, 8);
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::dumpDataForColmap()
{
  if (pinhole_cam == nullptr)
  {
    colmap_output_en = false;
    printf("[ VIO ] COLMAP output disabled: pinhole camera conversion is unavailable.\n");
    return;
  }
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";
  
  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(new_frame_->T_f_w_.rotationMatrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

bool VIOManager::inOpticalFlowBorder(const cv::Point2f &pt) const
{
  const int img_x = cvRound(pt.x);
  const int img_y = cvRound(pt.y);
  const int border_size = 1;
  return border_size <= img_x && img_x < width - border_size && border_size <= img_y && img_y < height - border_size;
}

V3D VIOManager::getOpticalFlowBearing(const cv::Point2f &px) const
{
  V3D bearing = cam->cam2world(px.x, px.y);
  if (bearing.norm() > 1e-12) bearing.normalize();
  return bearing;
}

void VIOManager::setOpticalFlowMask(cv::Mat &mask)
{
  mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));

  std::vector<std::tuple<int, cv::Point2f, int>> cnt_pts_id;
  cnt_pts_id.reserve(optical_flow_cur_pts.size());
  for (size_t i = 0; i < optical_flow_cur_pts.size(); i++)
  {
    cnt_pts_id.emplace_back(optical_flow_track_cnt[i], optical_flow_cur_pts[i], optical_flow_ids[i]);
  }

  std::sort(cnt_pts_id.begin(), cnt_pts_id.end(),
            [](const std::tuple<int, cv::Point2f, int> &a, const std::tuple<int, cv::Point2f, int> &b) {
              return std::get<0>(a) > std::get<0>(b);
            });

  optical_flow_cur_pts.clear();
  optical_flow_ids.clear();
  optical_flow_track_cnt.clear();

  for (const auto &it : cnt_pts_id)
  {
    const cv::Point2f &pt = std::get<1>(it);
    const int x = cvRound(pt.x);
    const int y = cvRound(pt.y);
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    if (mask.at<uchar>(y, x) == 255)
    {
      optical_flow_track_cnt.push_back(std::get<0>(it));
      optical_flow_cur_pts.push_back(pt);
      optical_flow_ids.push_back(std::get<2>(it));
      cv::circle(mask, cv::Point(x, y), optical_flow_min_dist, 0, -1);
    }
  }
}

void VIOManager::addOpticalFlowObservation(OpticalFlowTrack &track, const cv::Point2f &px, double img_time)
{
  OpticalFlowObservation obs;
  obs.frame_id = optical_flow_frame_id;
  obs.timestamp = img_time;
  obs.px = px;
  obs.bearing = getOpticalFlowBearing(px);
  obs.T_f_w = new_frame_->T_f_w_;
  track.observations.push_back(obs);

  track.history.push_back(px);
  while (static_cast<int>(track.history.size()) > optical_flow_track_history_size) track.history.pop_front();
}

bool VIOManager::triangulateOpticalFlowTrack(OpticalFlowTrack &track)
{
  const int obs_num = static_cast<int>(track.observations.size());
  if (obs_num < optical_flow_min_track_len_for_triangulation) return false;

  Eigen::MatrixXd A(obs_num * 2, 4);
  for (int i = 0; i < obs_num; i++)
  {
    const OpticalFlowObservation &obs = track.observations[i];
    Eigen::Matrix<double, 3, 4> pose;
    pose.block<3, 3>(0, 0) = obs.T_f_w.rotationMatrix();
    pose.block<3, 1>(0, 3) = obs.T_f_w.translation();
    const V3D &f = obs.bearing;
    A.row(i * 2) = f[0] * pose.row(2) - f[2] * pose.row(0);
    A.row(i * 2 + 1) = f[1] * pose.row(2) - f[2] * pose.row(1);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::Vector4d point_h = svd.matrixV().col(3);
  if (std::fabs(point_h[3]) < 1e-8) return false;

  V3D point_w = point_h.head<3>() / point_h[3];
  if (!point_w.array().isFinite().all()) return false;

  for (const auto &obs : track.observations)
  {
    V3D point_c = obs.T_f_w * point_w;
    if (!point_c.array().isFinite().all()) return false;
    if (virtual_fisheye_patch_en)
    {
      V2D raw_px;
      if (!projectRawFisheyeIfValid(point_c, 1, raw_px)) return false;
    }
    else if (point_c[2] <= 0.05)
    {
      return false;
    }
  }

  track.point_w = point_w;
  track.triangulated = true;
  track.rejected = false;
  return true;
}

void VIOManager::updateOpticalFlowPointClouds()
{
  optical_flow_triangulated_points->clear();

  optical_flow_triangulated_points->reserve(optical_flow_cur_pts.size());
  for (const int id : optical_flow_ids)
  {
    const auto track_item = optical_flow_tracks.find(id);
    if (track_item == optical_flow_tracks.end()) continue;
    const OpticalFlowTrack &track = track_item->second;
    if (!track.triangulated) continue;
    PointType point;
    point.x = track.point_w[0];
    point.y = track.point_w[1];
    point.z = track.point_w[2];
    point.intensity = static_cast<float>(track.id);
    point.normal_x = point.normal_y = point.normal_z = 0.0f;
    point.curvature = static_cast<float>(track.observations.size());
    optical_flow_triangulated_points->push_back(point);
  }
}

void VIOManager::drawOpticalFlowDebugImage(const std::vector<cv::Point2f> &rejected_pts, int prev, int tracked, int flow_back_pass,
                                           int border_pass, int mask_reject, int new_points, int final_points, int triangulated)
{
  if (img_rgb.empty()) return;
  if (img_rgb.channels() == 3)
  {
    optical_flow_debug_img = img_rgb.clone();
  }
  else
  {
    cv::cvtColor(img_rgb, optical_flow_debug_img, CV_GRAY2BGR);
  }

  for (const auto &pt : rejected_pts)
  {
    if (!inOpticalFlowBorder(pt)) continue;
    cv::drawMarker(optical_flow_debug_img, pt, cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 12, 2);
  }

  for (size_t i = 0; i < optical_flow_cur_pts.size(); i++)
  {
    const int id = optical_flow_ids[i];
    const int age = optical_flow_track_cnt[i];
    const auto track_it = optical_flow_tracks.find(id);
    const double ratio = std::min(1.0, static_cast<double>(age) / 20.0);
    cv::Scalar age_color(255.0 * (1.0 - ratio), 255.0 * ratio, 80.0 + 120.0 * (1.0 - ratio));

    if (track_it != optical_flow_tracks.end())
    {
      const OpticalFlowTrack &track = track_it->second;
      for (size_t j = 1; j < track.history.size(); j++)
      {
        cv::line(optical_flow_debug_img, track.history[j - 1], track.history[j], age_color, 2, cv::LINE_AA);
      }
      if (track.rejected && !track.triangulated)
      {
        cv::circle(optical_flow_debug_img, optical_flow_cur_pts[i], 6, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
      }
    }

    if (age <= 1)
    {
      cv::circle(optical_flow_debug_img, optical_flow_cur_pts[i], 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
      cv::circle(optical_flow_debug_img, optical_flow_cur_pts[i], 6, cv::Scalar(0, 80, 80), 2, cv::LINE_AA);
    }
    else
    {
      cv::circle(optical_flow_debug_img, optical_flow_cur_pts[i], 4, age_color, -1, cv::LINE_AA);
      cv::circle(optical_flow_debug_img, optical_flow_cur_pts[i], 6, age_color, 2, cv::LINE_AA);
    }
  }

  char text[256];
  snprintf(text, sizeof(text), "prev %d tracked %d fb %d border %d mask_rej %d new %d final %d tri %d",
           prev, tracked, flow_back_pass, border_pass, mask_reject, new_points, final_points, triangulated);
  cv::putText(optical_flow_debug_img, text, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
  cv::putText(optical_flow_debug_img, text, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

void VIOManager::processFrameOpticalFlow(cv::Mat &img, double img_time)
{
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ OpticalFlow ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }

  img_rgb = img.clone();
  img_cp = img.clone();

  cv::Mat cur_img;
  if (img.channels() == 3)
  {
    cv::cvtColor(img, cur_img, CV_BGR2GRAY);
  }
  else
  {
    cur_img = img.clone();
  }

  new_frame_.reset(new Frame(cam, cur_img));
  updateFrameState(*state);

  const int prev = static_cast<int>(optical_flow_prev_pts.size());
  int tracked = 0;
  int flow_back_pass = 0;
  int border_pass = 0;
  int mask_reject = 0;
  int new_points = 0;
  int new_triangulated = 0;
  int triangulation_reject = 0;
  std::vector<cv::Point2f> rejected_pts;

  std::vector<cv::Point2f> prev_pts = optical_flow_prev_pts;
  std::vector<int> prev_ids = optical_flow_ids;
  std::vector<int> prev_track_cnt = optical_flow_track_cnt;

  optical_flow_cur_pts.clear();
  optical_flow_ids.clear();
  optical_flow_track_cnt.clear();

  if (!optical_flow_prev_img.empty() && !prev_pts.empty())
  {
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(optical_flow_prev_img, cur_img, prev_pts, optical_flow_cur_pts, status, err, cv::Size(21, 21), 3);

    tracked = static_cast<int>(std::count(status.begin(), status.end(), static_cast<uchar>(1)));

    if (optical_flow_flow_back)
    {
      std::vector<uchar> reverse_status;
      std::vector<cv::Point2f> reverse_pts = prev_pts;
      cv::calcOpticalFlowPyrLK(cur_img, optical_flow_prev_img, optical_flow_cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
      for (size_t i = 0; i < status.size(); i++)
      {
        const double dx = prev_pts[i].x - reverse_pts[i].x;
        const double dy = prev_pts[i].y - reverse_pts[i].y;
        const double fb_dist = std::sqrt(dx * dx + dy * dy);
        if (!(status[i] && reverse_status[i] && fb_dist <= optical_flow_f_threshold))
        {
          if (i < optical_flow_cur_pts.size()) rejected_pts.push_back(optical_flow_cur_pts[i]);
          status[i] = 0;
        }
      }
    }

    flow_back_pass = static_cast<int>(std::count(status.begin(), status.end(), static_cast<uchar>(1)));

    for (size_t i = 0; i < optical_flow_cur_pts.size(); i++)
    {
      if (status[i] && !inOpticalFlowBorder(optical_flow_cur_pts[i]))
      {
        rejected_pts.push_back(optical_flow_cur_pts[i]);
        status[i] = 0;
      }
    }

    border_pass = static_cast<int>(std::count(status.begin(), status.end(), static_cast<uchar>(1)));

    auto reduce_by_status = [](auto &data, const std::vector<uchar> &status_vec) {
      int j = 0;
      for (int i = 0; i < static_cast<int>(data.size()); i++)
      {
        if (status_vec[i]) data[j++] = data[i];
      }
      data.resize(j);
    };

    optical_flow_ids = prev_ids;
    optical_flow_track_cnt = prev_track_cnt;
    reduce_by_status(prev_pts, status);
    reduce_by_status(optical_flow_cur_pts, status);
    reduce_by_status(optical_flow_ids, status);
    reduce_by_status(optical_flow_track_cnt, status);
  }

  for (auto &track_count : optical_flow_track_cnt) track_count++;

  cv::Mat mask;
  const int before_mask = static_cast<int>(optical_flow_cur_pts.size());
  setOpticalFlowMask(mask);
  mask_reject = before_mask - static_cast<int>(optical_flow_cur_pts.size());

  const int need_new_points = optical_flow_max_cnt - static_cast<int>(optical_flow_cur_pts.size());
  std::vector<cv::Point2f> n_pts;
  if (need_new_points > 0)
  {
    cv::goodFeaturesToTrack(cur_img, n_pts, need_new_points, optical_flow_quality_level, optical_flow_min_dist, mask);
  }

  new_points = static_cast<int>(n_pts.size());
  for (const auto &pt : n_pts)
  {
    optical_flow_cur_pts.push_back(pt);
    optical_flow_ids.push_back(optical_flow_next_id++);
    optical_flow_track_cnt.push_back(1);
  }

  for (size_t i = 0; i < optical_flow_cur_pts.size(); i++)
  {
    const int id = optical_flow_ids[i];
    OpticalFlowTrack &track = optical_flow_tracks[id];
    if (track.id < 0) track.id = id;
    track.age = optical_flow_track_cnt[i];
    addOpticalFlowObservation(track, optical_flow_cur_pts[i], img_time);

    if (!track.triangulated && static_cast<int>(track.observations.size()) >= optical_flow_min_track_len_for_triangulation)
    {
      if (triangulateOpticalFlowTrack(track))
      {
        new_triangulated++;
      }
      else
      {
        track.rejected = true;
        triangulation_reject++;
      }
    }
  }

  updateOpticalFlowPointClouds();
  int total_triangulated = 0;
  for (const auto &track_item : optical_flow_tracks)
  {
    if (track_item.second.triangulated) total_triangulated++;
  }
  drawOpticalFlowDebugImage(rejected_pts, prev, tracked, flow_back_pass, border_pass, mask_reject, new_points,
                            static_cast<int>(optical_flow_cur_pts.size()), static_cast<int>(optical_flow_triangulated_points->size()));

  printf(BOLDWHITE "[ OpticalFlow ] stamp=%.6f " BOLDBLUE "prev=%d " BOLDGREEN "tracked=%d "
         BOLDCYAN "flow_back_pass=%d " BOLDMAGENTA "border_pass=%d " BOLDYELLOW "mask_reject=%d "
         BOLDREDPURPLE "new_points=%d " BOLDWHITE "final=%zu " BOLDGREEN "triangulated_current=%zu "
         BOLDCYAN "triangulated_total=%d new_tri=%d " BOLDRED "rejected/outlier={lk=%d flow_back=%d border=%d triangulation=%d}\n" RESET,
         img_time, prev, tracked, flow_back_pass, border_pass, mask_reject, new_points, optical_flow_cur_pts.size(),
         optical_flow_triangulated_points->size(), total_triangulated, new_triangulated, prev - tracked, tracked - flow_back_pass,
         flow_back_pass - border_pass, triangulation_reject);

  // TODO: build optical-flow reprojection residuals and feed them into the IESKF update.

  optical_flow_prev_img = cur_img.clone();
  optical_flow_prev_pts = optical_flow_cur_pts;
  optical_flow_prev_time = img_time;
  optical_flow_frame_id++;
}

void VIOManager::processFrameFake(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time) 
{
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  return;
}
void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  
  resetGrid();

  double t1 = omp_get_wtime();

  retrieveFromVisualSparseMap(img, pg, feat_map);

  double t2 = omp_get_wtime();

  computeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();

  generateVisualMapPoints(img, pg);

  double t4 = omp_get_wtime();
  
  plotTrackedPoints();

  if (plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();
  
  if(colmap_output_en)  dumpDataForColmap();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;
 
  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;
  
  if (virtual_fisheye_patch_en)
  {
    const int virtual_candidate_reject_count = virtual_map_grid_count_ - virtual_candidate_count_;
    const int virtual_track_reject_count = virtual_candidate_count_ - virtual_valid_track_count_;
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m|                      VIO Virtual Time                       |\033[0m\n");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
    printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Map Grid Points", virtual_map_grid_count_);
    printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Virtual Candidates", virtual_candidate_count_);
    printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Tracked Virtual Points", virtual_valid_track_count_);
    printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Candidate Rejects", virtual_candidate_reject_count);
    printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Track Rejects", virtual_track_reject_count);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> candidate/select wall", virtual_candidate_select_time_);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> parallel track wall", virtual_parallel_track_time_);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> support build sum(all)", build_virtual_support_time_);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> affine matrix sum(all)", virtual_affine_time_);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> warp reference sum(all)", virtual_warp_time_);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> current core sum(all)", virtual_current_core_time_);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> result collect wall", virtual_result_collect_time_);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  }
  else
  {
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  }

  // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + " HZ";
  // cv::Point2f origin;
  // origin.x = 20;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
  // cv::imwrite("/home/chunran/Desktop/raycasting/" + std::to_string(new_frame_->id_) + ".png", img_cp);
}
