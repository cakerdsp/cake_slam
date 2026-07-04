/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio_multi_cam.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace Eigen;

namespace
{
constexpr int kRefPatchDumpCameraId = 0;
constexpr int kRefPatchDumpMaxPoints = 8;
constexpr int kRefPatchDumpMaxRefsPerPoint = 12;
constexpr int kRefPatchDumpFrameInterval = 5;
constexpr int kRefPatchDumpMaxMissedFrames = 20;
constexpr double kRefPatchDumpMaxInitialIncidenceDeg = 20.0;
constexpr double kRefPatchDumpMinAngleChangeDeg = 8.0;
constexpr double kRefPatchDumpMinStdDev = 12.0;
constexpr double kRadiansToDegrees = 57.29577951308232;
constexpr int kRefPatchDumpWarpDisplaySize = 128;
constexpr double kS2Eps = 1.0e-12;

std::string normalizeCameraModelType(std::string model)
{
  std::string normalized;
  normalized.reserve(model.size());
  for (unsigned char ch : model)
  {
    if (std::isalnum(ch)) normalized.push_back(static_cast<char>(std::tolower(ch)));
  }
  return normalized;
}

bool isPinholeJacobianModel(const std::string &model)
{
  const std::string normalized = normalizeCameraModelType(model);
  return normalized.empty() || normalized == "pinhole" || normalized == "pinholecamera";
}

bool isEquidistantJacobianModel(const std::string &model)
{
  const std::string normalized = normalizeCameraModelType(model);
  return normalized == "equidistant" || normalized == "equidistantcamera" ||
         normalized == "kannalabrandt" || normalized == "kannalabrandtcamera" ||
         normalized == "kb" || normalized == "kb4";
}

bool isMeiJacobianModel(const std::string &model)
{
  const std::string normalized = normalizeCameraModelType(model);
  return normalized == "mei" || normalized == "meicamera" || normalized == "omni" ||
         normalized == "unified" || normalized == "unifiedcamera";
}

bool isSupportedJacobianModel(const std::string &model)
{
  return isPinholeJacobianModel(model) || isEquidistantJacobianModel(model) || isMeiJacobianModel(model);
}

void computePinholeProjectionJacobianForContext(const PerCameraData &ctx, const V3D &p, MD(2, 3) &J)
{
  J.setZero();
  if (!p.array().isFinite().all() || std::fabs(p[2]) <= kS2Eps) return;
  const double z_inv = 1.0 / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = ctx.fx * z_inv;
  J(0, 2) = -ctx.fx * p[0] * z_inv_2;
  J(1, 1) = ctx.fy * z_inv;
  J(1, 2) = -ctx.fy * p[1] * z_inv_2;
}

bool computeEquidistantProjectionJacobianForContext(const PerCameraData &ctx, const V3D &p, MD(2, 3) &J)
{
  J.setZero();
  if (!p.array().isFinite().all()) return false;
  const double x = p[0];
  const double y = p[1];
  const double z = p[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double r2 = x2 + y2;
  if (r2 < 1.0e-8)
  {
    computePinholeProjectionJacobianForContext(ctx, p, J);
    return J.array().isFinite().all();
  }

  const double r = std::sqrt(r2);
  const double r_inv = 1.0 / r;
  const double r2_inv = r_inv * r_inv;
  const double theta = std::atan2(r, z);
  const double th2 = theta * theta;
  const double th4 = th2 * th2;
  const double th6 = th4 * th2;
  const double th8 = th4 * th4;

  const double f_theta = theta * (1.0 + ctx.k1 * th2 + ctx.k2 * th4 + ctx.k3 * th6 + ctx.k4 * th8);
  const double f_theta_prime = 1.0 + 3.0 * ctx.k1 * th2 + 5.0 * ctx.k2 * th4 +
                               7.0 * ctx.k3 * th6 + 9.0 * ctx.k4 * th8;
  const double rho2 = r2 + z * z;
  if (!std::isfinite(rho2) || rho2 <= kS2Eps) return false;

  const double term_A = f_theta_prime / rho2;
  const double term_G = f_theta * r_inv;
  const double term_diff = (z * term_A - term_G) * r2_inv;

  J(0, 0) = ctx.fx * (term_G + x2 * term_diff);
  J(0, 1) = ctx.fx * (x * y * term_diff);
  J(0, 2) = -ctx.fx * x * term_A;
  J(1, 0) = ctx.fy * (x * y * term_diff);
  J(1, 1) = ctx.fy * (term_G + y2 * term_diff);
  J(1, 2) = -ctx.fy * y * term_A;
  return J.array().isFinite().all();
}

bool computeMeiProjectionJacobianForContext(const PerCameraData &ctx, const V3D &p, MD(2, 3) &J)
{
  J.setZero();
  if (!p.array().isFinite().all()) return false;
  const double x = p[0];
  const double y = p[1];
  const double z = p[2];
  const double d2 = x * x + y * y + z * z;
  if (!std::isfinite(d2) || d2 <= kS2Eps) return false;

  const double d = std::sqrt(d2);
  const double d_inv = 1.0 / d;
  const double rho = z + ctx.xi * d;
  if (!std::isfinite(rho) || std::fabs(rho) <= 1.0e-8) return false;

  const double rho_inv = 1.0 / rho;
  const double rho2_inv = rho_inv * rho_inv;
  const double drho_dx = ctx.xi * x * d_inv;
  const double drho_dy = ctx.xi * y * d_inv;
  const double drho_dz = 1.0 + ctx.xi * z * d_inv;

  const double xu = x * rho_inv;
  const double yu = y * rho_inv;
  const V3D Jxu(rho_inv - x * drho_dx * rho2_inv,
                -x * drho_dy * rho2_inv,
                -x * drho_dz * rho2_inv);
  const V3D Jyu(-y * drho_dx * rho2_inv,
                rho_inv - y * drho_dy * rho2_inv,
                -y * drho_dz * rho2_inv);

  const double xu2 = xu * xu;
  const double yu2 = yu * yu;
  const double r2 = xu2 + yu2;
  const double r4 = r2 * r2;
  const double radial = 1.0 + ctx.k1 * r2 + ctx.k2 * r4;
  const double dradial_dxu = 2.0 * ctx.k1 * xu + 4.0 * ctx.k2 * xu * r2;
  const double dradial_dyu = 2.0 * ctx.k1 * yu + 4.0 * ctx.k2 * yu * r2;

  const double dxd_dxu = radial + xu * dradial_dxu + 2.0 * ctx.p1 * yu + 6.0 * ctx.p2 * xu;
  const double dxd_dyu = xu * dradial_dyu + 2.0 * ctx.p1 * xu + 2.0 * ctx.p2 * yu;
  const double dyd_dxu = yu * dradial_dxu + 2.0 * ctx.p1 * xu + 2.0 * ctx.p2 * yu;
  const double dyd_dyu = radial + yu * dradial_dyu + 6.0 * ctx.p1 * yu + 2.0 * ctx.p2 * xu;

  const V3D Jxd = dxd_dxu * Jxu + dxd_dyu * Jyu;
  const V3D Jyd = dyd_dxu * Jxu + dyd_dyu * Jyu;
  J.row(0) = ctx.fx * Jxd.transpose();
  J.row(1) = ctx.fy * Jyd.transpose();
  return J.array().isFinite().all();
}

bool computeCameraModelProjectionJacobianForContext(const PerCameraData &ctx, const V3D &p, MD(2, 3) &J)
{
  if (isPinholeJacobianModel(ctx.camera_model_type))
  {
    computePinholeProjectionJacobianForContext(ctx, p, J);
    return J.array().isFinite().all();
  }
  if (isEquidistantJacobianModel(ctx.camera_model_type))
    return computeEquidistantProjectionJacobianForContext(ctx, p, J);
  if (isMeiJacobianModel(ctx.camera_model_type))
    return computeMeiProjectionJacobianForContext(ctx, p, J);

  J.setZero();
  return false;
}
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
  Rli = M3D::Identity();
  Pli = V3D::Zero();
  optical_flow_triangulated_points.reset(new PointCloudXYZI());
}

VIOManager::~VIOManager()
{
  for (PerCameraData &ctx : cameras_)
  {
    delete ctx.visual_submap;
    ctx.visual_submap = nullptr;
    for (auto &pair : ctx.warp_map) delete pair.second;
    ctx.warp_map.clear();
  }
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::configureCameras(int num_cameras)
{
  if (num_cameras < 1) throw std::invalid_argument("VIOManager requires at least one camera");
  cameras_.clear();
  cameras_.resize(num_cameras);
  for (int camera_id = 0; camera_id < num_cameras; ++camera_id) cameras_[camera_id].camera_id = camera_id;
}

void VIOManager::setCameraCalibration(int camera_id, const std::string &topic, const std::string &camera_namespace,
                                      const std::vector<double> &R, const std::vector<double> &P)
{
  if (camera_id < 0 || camera_id >= numCameras()) throw std::out_of_range("invalid camera_id");
  if (R.size() != 9 || P.size() != 3) throw std::invalid_argument("camera Rcl/Pcl dimensions must be 9/3");
  PerCameraData &ctx = cameras_[camera_id];
  ctx.topic = topic;
  ctx.camera_namespace = camera_namespace;
  ctx.Rcl << MAT_FROM_ARRAY(R);
  ctx.Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::setCameraModelJacobianParameters(int camera_id, const std::string &model_type,
                                                  double k1, double k2, double k3, double k4,
                                                  double xi, double p1, double p2)
{
  if (camera_id < 0 || camera_id >= numCameras()) throw std::out_of_range("invalid camera_id");
  PerCameraData &ctx = cameras_[camera_id];
  ctx.camera_model_type = model_type.empty() ? "Pinhole" : model_type;
  ctx.k1 = k1;
  ctx.k2 = k2;
  ctx.k3 = k3;
  ctx.k4 = k4;
  ctx.xi = xi;
  ctx.p1 = p1;
  ctx.p2 = p2;
}
void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::updateCameraExtrinsicDerived(PerCameraData &ctx)
{
  ctx.Rci = ctx.Rcl * Rli;
  ctx.Pci = ctx.Rcl * Pli + ctx.Pcl;
  ctx.Jdphi_dR = ctx.Rci;
  const V3D Pic = -ctx.Rci.transpose() * ctx.Pci;
  M3D pic_hat;
  pic_hat << SKEW_SYM_MATRX(Pic);
  ctx.Jdp_dR = -ctx.Rci * pic_hat;
}

void VIOManager::syncCameraExtrinsicsFromState(const StatesGroup &state_value)
{
  if (state_value.num_cameras != numCameras())
    throw std::runtime_error("state/camera count mismatch while syncing camera extrinsics");
  for (PerCameraData &ctx : cameras_)
  {
    if (ctx.camera_id < 0 || ctx.camera_id >= state_value.num_cameras) continue;
    ctx.Rcl = state_value.Rcl[ctx.camera_id];
    ctx.Pcl = state_value.Pcl[ctx.camera_id];
    updateCameraExtrinsicDerived(ctx);
  }
}

bool VIOManager::isOnlineExtrinsicEnabledForCamera(int camera_id) const
{
  if (!online_extrinsic_en || camera_id < 0 || camera_id >= numCameras()) return false;
  if (online_extrinsic_camera_mask.empty()) return true;
  if (camera_id >= static_cast<int>(online_extrinsic_camera_mask.size())) return false;
  return online_extrinsic_camera_mask[camera_id] != 0;
}

void VIOManager::applyOnlineExtrinsicPriors(Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient,
                                            bool allow_rotation, bool allow_translation) const
{
  if (state == nullptr || !online_extrinsic_en) return;
  constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
  const double rot_std = online_extrinsic_prior_rot_std_deg * kDegToRad;
  const double trans_std = online_extrinsic_prior_trans_std_m;
  if (rot_std <= 0.0 || trans_std <= 0.0) return;
  const double rot_info = img_point_cov / (rot_std * rot_std);
  const double trans_info = img_point_cov / (trans_std * trans_std);

  for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
  {
    if (!isOnlineExtrinsicEnabledForCamera(camera_id)) continue;
    if (allow_rotation)
    {
      const int ridx = state->extrinsicRotIndex(camera_id);
      const M3D dR_cl = state->Rcl_prior[camera_id].transpose() * state->Rcl[camera_id];
      const V3D rot_error = Log(dR_cl);
      hessian.block<3, 3>(ridx, ridx).diagonal().array() += rot_info;
      gradient.segment<3>(ridx) += rot_info * rot_error;
    }
    if (allow_translation)
    {
      const int tidx = state->extrinsicTransIndex(camera_id);
      const V3D trans_error = state->Pcl[camera_id] - state->Pcl_prior[camera_id];
      hessian.block<3, 3>(tidx, tidx).diagonal().array() += trans_info;
      gradient.segment<3>(tidx) += trans_info * trans_error;
    }
  }
}

void VIOManager::limitOnlineExtrinsicUpdate(Eigen::VectorXd &solution, bool allow_rotation, bool allow_translation) const
{
  if (state == nullptr) return;
  constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
  const double max_rot = std::max(0.0, online_extrinsic_max_rot_update_deg) * kDegToRad;
  const double max_trans = std::max(0.0, online_extrinsic_max_trans_update_m);

  for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
  {
    const int ridx = state->extrinsicRotIndex(camera_id);
    const int tidx = state->extrinsicTransIndex(camera_id);
    const bool camera_enabled = isOnlineExtrinsicEnabledForCamera(camera_id);

    if (!camera_enabled || !allow_rotation || max_rot == 0.0)
    {
      solution.segment<3>(ridx).setZero();
    }
    else
    {
      V3D delta = solution.segment<3>(ridx);
      const double norm = delta.norm();
      if (norm > max_rot) solution.segment<3>(ridx) = delta * (max_rot / norm);
    }

    if (!camera_enabled || !allow_translation || max_trans == 0.0)
    {
      solution.segment<3>(tidx).setZero();
    }
    else
    {
      V3D delta = solution.segment<3>(tidx);
      const double norm = delta.norm();
      if (norm > max_trans) solution.segment<3>(tidx) = delta * (max_trans / norm);
    }
  }
}

void VIOManager::initializeVIO()
{
  if (cameras_.empty()) throw std::runtime_error("VIOManager cameras were not configured");
  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  if (raw_camera_model_jacobian_en && virtual_fisheye_patch_en)
  {
    printf("[ VIO ] raw_camera_model_jacobian_en is ignored because virtual_fisheye_patch_en is true.\n");
    raw_camera_model_jacobian_en = false;
  }

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

  }

  if (virtual_s2_optimize_en)
  {
    if (virtual_fisheye_patch_en)
      printf("[ VIO Virtual S2 ] enabled: S2/raw-fisheye photometric optimization branch is active.\n");
    else
      printf("[ VIO Virtual S2 ] virtual_s2_optimize_en is true but virtual_fisheye_patch_en is false; S2 branch is ignored.\n");
  }

  if (virtual_sparse_patch_en && !virtual_fisheye_patch_en)
    throw std::runtime_error("virtual_sparse_patch_en requires virtual_fisheye_patch_en");
  if (virtual_sparse_patch_en)
    printf("[ VIO Virtual Sparse ] enabled: sparse current sampling and lazy reference support\n");

  if (ref_patch_dump_en && !virtual_fisheye_patch_en)
  {
    printf("[ VIO Ref Patch Dump ] Disabled because virtual_fisheye_patch_en is false.\n");
    ref_patch_dump_en = false;
  }

  if (ref_patch_dump_en)
  {
    if (ref_patch_dump_max_candidate_skip < 0)
      throw std::runtime_error("ref_patch_dump_max_candidate_skip must be non-negative");
    if (ref_patch_dump_ncc_threshold < -1.0 || ref_patch_dump_ncc_threshold > 1.0)
      throw std::runtime_error("ref_patch_dump_ncc_threshold must be in [-1, 1]");
    ref_patch_dump_effective_seed_ =
        ref_patch_dump_random_seed >= 0 ? static_cast<unsigned int>(ref_patch_dump_random_seed) : std::random_device{}();
    ref_patch_dump_rng_.seed(ref_patch_dump_effective_seed_);
    ref_patch_dump_candidates_to_skip_ = -1;
    printf("[ VIO Ref Patch Dump ] Selection seed=%u max_candidate_skip=%d ncc_threshold=%.3f\n",
           ref_patch_dump_effective_seed_, ref_patch_dump_max_candidate_skip, ref_patch_dump_ncc_threshold);
  }

  if (colmap_output_en && numCameras() > 1)
  {
    printf("[ VIO ] COLMAP output disabled: multi-camera COLMAP is not implemented.\n");
    colmap_output_en = false;
  }

  for (PerCameraData &ctx : cameras_)
  {
    if (ctx.cam == nullptr)
      throw std::runtime_error("camera model is null for camera_id=" + std::to_string(ctx.camera_id));
    ctx.visual_submap = new SubSparseMap;
    ctx.fx = ctx.cam->fx();
    ctx.fy = ctx.cam->fy();
    ctx.cx = ctx.cam->cx();
    ctx.cy = ctx.cam->cy();
    ctx.image_resize_factor = ctx.cam->scale();
    ctx.width = ctx.cam->width();
    ctx.height = ctx.cam->height();
    if ((raw_camera_model_jacobian_en || (virtual_fisheye_patch_en && virtual_s2_optimize_en)) && !isSupportedJacobianModel(ctx.camera_model_type))
      throw std::runtime_error("unsupported camera model for projection Jacobian: camera_id=" +
                               std::to_string(ctx.camera_id) + " model=" + ctx.camera_model_type);
    if (state != nullptr && state->num_cameras == numCameras())
    {
      ctx.Rcl = state->Rcl[ctx.camera_id];
      ctx.Pcl = state->Pcl[ctx.camera_id];
    }
    updateCameraExtrinsicDerived(ctx);

    ctx.grid_size = grid_size;
    if (ctx.grid_size > 10)
    {
      ctx.grid_n_width = static_cast<int>(std::ceil(static_cast<double>(ctx.width) / ctx.grid_size));
      ctx.grid_n_height = static_cast<int>(std::ceil(static_cast<double>(ctx.height) / ctx.grid_size));
    }
    else
    {
      ctx.grid_size = std::max(1, static_cast<int>(ctx.height / grid_n_height));
      ctx.grid_n_height = static_cast<int>(std::ceil(static_cast<double>(ctx.height) / ctx.grid_size));
      ctx.grid_n_width = static_cast<int>(std::ceil(static_cast<double>(ctx.width) / ctx.grid_size));
    }
    ctx.length = ctx.grid_n_width * ctx.grid_n_height;
    ctx.grid_num.resize(ctx.length);
    ctx.map_index.resize(ctx.length);
    ctx.map_dist.resize(ctx.length);
    ctx.update_flag.resize(ctx.length);
    ctx.scan_value.resize(ctx.length);
    ctx.border_flag.assign(ctx.length, 0);
    ctx.retrieve_voxel_points.reserve(ctx.length);
    ctx.append_voxel_points.reserve(ctx.length);
    ctx.append_voxel_source_type.reserve(ctx.length);
    ctx.append_voxel_source_index.reserve(ctx.length);

    if (raycast_en)
    {
      ctx.rays_with_sample_points.reserve(ctx.length);
      for (int grid_row = 1; grid_row <= ctx.grid_n_height; ++grid_row)
      {
        for (int grid_col = 1; grid_col <= ctx.grid_n_width; ++grid_col)
        {
          const int index = (grid_row - 1) * ctx.grid_n_width + grid_col - 1;
          if (grid_row == 1 || grid_col == 1 || grid_row == ctx.grid_n_height || grid_col == ctx.grid_n_width)
            ctx.border_flag[index] = 1;
          const int u = ctx.grid_size / 2 + (grid_col - 1) * ctx.grid_size;
          const int v = ctx.grid_size / 2 + (grid_row - 1) * ctx.grid_size;
          std::vector<V3D> samples;
          for (float range = 0.1f; range <= 3.0f; range += 0.2f)
          {
            V3D xyz = ctx.cam->cam2world(u, v);
            if (virtual_fisheye_patch_en)
            {
              if (xyz.norm() <= virtual_min_z) continue;
              xyz.normalize();
              xyz *= range;
            }
            else
            {
              if (std::fabs(xyz[2]) <= virtual_min_z) continue;
              xyz *= range / xyz[2];
            }
            samples.push_back(xyz);
          }
          ctx.rays_with_sample_points.push_back(std::move(samples));
        }
      }
    }

    if (virtual_fisheye_patch_en)
    {
      ctx.raw_pixel_to_unit_ray_lut.resize(ctx.width * ctx.height);
      ctx.raw_pixel_unit_ray_valid_mask.assign(ctx.width * ctx.height, 0);
      for (int raw_y = 0; raw_y < ctx.height; ++raw_y)
      {
        for (int raw_x = 0; raw_x < ctx.width; ++raw_x)
        {
          const int raw_idx = raw_y * ctx.width + raw_x;
          if (!ctx.cam->isInFrame(V2D(raw_x, raw_y).cast<int>(), 0)) continue;
          V3D ray_c = ctx.cam->cam2world(raw_x, raw_y);
          const double ray_norm = ray_c.norm();
          if (!ray_c.array().isFinite().all() || !std::isfinite(ray_norm) || ray_norm <= virtual_min_z) continue;
          ctx.raw_pixel_to_unit_ray_lut[raw_idx] = (ray_c / ray_norm).cast<float>();
          ctx.raw_pixel_unit_ray_valid_mask[raw_idx] = 1;
        }
      }
    }

    ctx.pinhole_cam = dynamic_cast<vk::PinholeCamera *>(ctx.cam);
    printf("[ VIO ] camera_id=%d ns=%s model=%s intrinsic=%.6f %.6f %.6f %.6f size=%dx%d scale=%.3f raw_model_jacobian=%d\n",
           ctx.camera_id, ctx.camera_namespace.c_str(), ctx.camera_model_type.c_str(),
           ctx.fx, ctx.fy, ctx.cx, ctx.cy, ctx.width, ctx.height,
           ctx.image_resize_factor, raw_camera_model_jacobian_en ? 1 : 0);
    resetGrid(ctx);
  }

  if (colmap_output_en)
  {
    PerCameraData &ctx = cameras_.front();
    if (ctx.pinhole_cam == nullptr)
    {
      printf("[ VIO ] COLMAP output disabled: camera0 is not pinhole.\n");
      colmap_output_en = false;
    }
    else
    {
      fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
      fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
      fout_camera << "1 PINHOLE " << ctx.width << " " << ctx.height << " " << std::fixed << std::setprecision(6)
                  << ctx.fx << " " << ctx.fy << " " << ctx.cx << " " << ctx.cy << std::endl;
      fout_camera.close();
    }
  }
  const int state_dim = state != nullptr ? state->stateDim() : BASE_STATE_DIM + numCameras() + 6 * numCameras();
  G = Eigen::MatrixXd::Zero(state_dim, state_dim);
  H_T_H = Eigen::MatrixXd::Zero(state_dim, state_dim);
}

void VIOManager::resetGrid(PerCameraData &ctx)
{
  fill(ctx.grid_num.begin(), ctx.grid_num.end(), TYPE_UNKNOWN);
  fill(ctx.map_index.begin(), ctx.map_index.end(), 0);
  fill(ctx.map_dist.begin(), ctx.map_dist.end(), 10000.0f);
  fill(ctx.update_flag.begin(), ctx.update_flag.end(), 0);
  fill(ctx.scan_value.begin(), ctx.scan_value.end(), 0.0f);
  ctx.retrieve_voxel_points.assign(ctx.length, nullptr);
  ctx.append_voxel_points.assign(ctx.length, pointWithVar());
  ctx.append_voxel_source_type.assign(ctx.length, SOURCE_UNKNOWN);
  ctx.append_voxel_source_index.assign(ctx.length, -1);
  ctx.pending_new_points.clear();
  ctx.rejected_visual_points_for_draw.clear();
  ctx.sub_feat_map.clear();
  for (auto &pair : ctx.warp_map) delete pair.second;
  ctx.warp_map.clear();
  ctx.total_points = 0;
  if (ctx.visual_submap != nullptr) ctx.visual_submap->reset();
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

void VIOManager::computeProjectionJacobian(const PerCameraData &ctx, V3D p, MD(2, 3) & J)
{
  if (raw_camera_model_jacobian_en)
  {
    if (!computeCameraModelProjectionJacobianForContext(ctx, p, J)) J.setZero();
    return;
  }
  computePinholeProjectionJacobianForContext(ctx, p, J);
}
void VIOManager::getImagePatch(const PerCameraData &ctx, const cv::Mat &img, V2D pc, float *patch_tmp, int level)
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
    const uint8_t *img_ptr = img.data + (v_ref_i - patch_size_half * scale + x * scale) * ctx.width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * ctx.width] + w_ref_br * img_ptr[scale * ctx.width + scale];
    }
  }
}

SE3<double> VIOManager::composeVirtualPose(const M3D &R_v_from_c, const SE3<double> &T_c_w) const
{
  // {}^V T_W = {}^V T_C * {}^C T_W. The virtual and raw cameras share the optical center.
  return SE3<double>(R_v_from_c * T_c_w.rotationMatrix(), R_v_from_c * T_c_w.translation());
}

bool VIOManager::buildVirtualFrameRotation(const PerCameraData &ctx, const V3D &point_in_raw_camera,
                                           const V2D &raw_center_px, M3D &R_v_from_c, M3D &R_c_from_v) const
{
  const double norm = point_in_raw_camera.norm();
  if (!std::isfinite(norm) || norm <= virtual_min_z) return false;

  const V3D z_v_in_c = point_in_raw_camera / norm;
  const double min_z = virtual_min_z;

  auto sample_raw_ray = [&ctx, min_z](const V2D &px, V3D &ray) {
    if (!px.array().isFinite().all() || px[0] < 0.0 || px[1] < 0.0 || px[0] >= ctx.width - 1 || px[1] >= ctx.height - 1)
      return false;

    const int x0 = static_cast<int>(std::floor(px[0]));
    const int y0 = static_cast<int>(std::floor(px[1]));
    const int indices[4] = {y0 * ctx.width + x0, y0 * ctx.width + x0 + 1,
                            (y0 + 1) * ctx.width + x0, (y0 + 1) * ctx.width + x0 + 1};
    if (ctx.raw_pixel_to_unit_ray_lut.size() != static_cast<size_t>(ctx.width * ctx.height) ||
        ctx.raw_pixel_unit_ray_valid_mask.size() != static_cast<size_t>(ctx.width * ctx.height))
      return false;
    for (const int index : indices)
      if (!ctx.raw_pixel_unit_ray_valid_mask[index]) return false;

    const double dx = px[0] - x0;
    const double dy = px[1] - y0;
    ray = (1.0 - dx) * (1.0 - dy) * ctx.raw_pixel_to_unit_ray_lut[indices[0]].cast<double>() +
          dx * (1.0 - dy) * ctx.raw_pixel_to_unit_ray_lut[indices[1]].cast<double>() +
          (1.0 - dx) * dy * ctx.raw_pixel_to_unit_ray_lut[indices[2]].cast<double>() +
          dx * dy * ctx.raw_pixel_to_unit_ray_lut[indices[3]].cast<double>();

    const double ray_norm = ray.norm();
    if (!ray.array().isFinite().all() || !std::isfinite(ray_norm) || ray_norm <= min_z) return false;

    ray /= ray_norm;
    return true;
  };

  auto tangent_difference = [&](const V2D &offset, V3D &tangent) {
    V3D ray_plus, ray_minus;
    const bool plus_valid = sample_raw_ray(raw_center_px + offset, ray_plus);
    const bool minus_valid = sample_raw_ray(raw_center_px - offset, ray_minus);
    if (plus_valid && minus_valid) tangent = ray_plus - ray_minus;
    else if (plus_valid) tangent = ray_plus - z_v_in_c;
    else if (minus_valid) tangent = z_v_in_c - ray_minus;
    else return false;

    tangent -= tangent.dot(z_v_in_c) * z_v_in_c;
    const double tangent_norm = tangent.norm();
    if (!tangent.array().isFinite().all() || !std::isfinite(tangent_norm) || tangent_norm <= virtual_min_z) return false;
    tangent /= tangent_norm;
    return true;
  };

  V3D tangent_u, tangent_v;
  const bool tangent_u_valid = tangent_difference(V2D(1.0, 0.0), tangent_u);
  const bool tangent_v_valid = tangent_difference(V2D(0.0, 1.0), tangent_v);

  V3D x_v_in_c = V3D::Zero();
  if (tangent_u_valid) x_v_in_c = tangent_u;
  if (tangent_v_valid)
  {
    V3D x_from_v = tangent_v.cross(z_v_in_c);
    const double x_from_v_norm = x_from_v.norm();
    if (std::isfinite(x_from_v_norm) && x_from_v_norm > virtual_min_z)
    {
      x_from_v /= x_from_v_norm;
      if (!tangent_u_valid || x_v_in_c.dot(x_from_v) > 0.0) x_v_in_c += x_from_v;
    }
  }

  // The fixed-axis fallback is only used at invalid image-chart derivatives.
  if (x_v_in_c.norm() <= virtual_min_z)
  {
    V3D reference_axis = V3D::UnitX();
    if (std::fabs(reference_axis.dot(z_v_in_c)) > 0.95) reference_axis = V3D::UnitY();
    x_v_in_c = reference_axis - reference_axis.dot(z_v_in_c) * z_v_in_c;
  }
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

bool VIOManager::projectRawFisheyeIfValid(const PerCameraData &ctx, const V3D &ray_or_point_in_raw_camera, int required_border, V2D &raw_px) const
{
  if (!ray_or_point_in_raw_camera.array().isFinite().all() || ray_or_point_in_raw_camera.norm() <= virtual_min_z) return false;
  raw_px = ctx.cam->world2cam(ray_or_point_in_raw_camera);
  if (!raw_px.array().isFinite().all()) return false;
  if (raw_px[0] < required_border || raw_px[1] < required_border || raw_px[0] >= ctx.width - required_border - 1 ||
      raw_px[1] >= ctx.height - required_border - 1)
    return false;
  return ctx.cam->isInFrame(raw_px.cast<int>(), required_border);
}

bool VIOManager::computeS2SamplePointAndJacobian(const V3D &point_c, const M3D &R_v_from_c,
                                                  const M3D &R_c_from_v, const V2D &offset,
                                                  V3D &p_sample_c, M3D &J_sample_pc) const
{
  p_sample_c.setZero();
  J_sample_pc.setZero();
  if (!point_c.array().isFinite().all() || !offset.array().isFinite().all()) return false;
  if (!R_v_from_c.array().isFinite().all() || !R_c_from_v.array().isFinite().all()) return false;
  if (!std::isfinite(virtual_focal_length) || virtual_focal_length <= kS2Eps) return false;

  const double rho = point_c.norm();
  if (!std::isfinite(rho) || rho <= kS2Eps) return false;
  const V3D d_c = point_c / rho;
  if (!d_c.array().isFinite().all()) return false;

  const V3D d_v = R_v_from_c * d_c;
  if (!d_v.array().isFinite().all()) return false;

  const double alpha = offset.x() / virtual_focal_length;
  const double beta = offset.y() / virtual_focal_length;
  if (!std::isfinite(alpha) || !std::isfinite(beta)) return false;

  V3D q_v = d_v;
  q_v.x() += alpha;
  q_v.y() += beta;
  const double q_norm = q_v.norm();
  if (!std::isfinite(q_norm) || q_norm <= kS2Eps) return false;

  const V3D s_v = q_v / q_norm;
  if (!s_v.array().isFinite().all()) return false;
  const V3D s_c = R_c_from_v * s_v;
  if (!s_c.array().isFinite().all()) return false;

  p_sample_c = rho * s_c;
  if (!p_sample_c.array().isFinite().all()) return false;

  const M3D I = M3D::Identity();
  const M3D J_dc_pc = (I - d_c * d_c.transpose()) / rho;
  const M3D J_dv_pc = R_v_from_c * J_dc_pc;
  const M3D J_sv_q = (I - s_v * s_v.transpose()) / q_norm;
  const M3D J_sv_pc = J_sv_q * J_dv_pc;
  const M3D J_sc_pc = R_c_from_v * J_sv_pc;
  J_sample_pc = s_c * d_c.transpose() + rho * J_sc_pc;
  return J_sample_pc.array().isFinite().all();
}

bool VIOManager::projectEquidistantFisheyeWithJacobian(const PerCameraData &ctx, const V3D &p_c,
                                                        int required_border, V2D &raw_px,
                                                        MD(2, 3) &J_raw_pc) const
{
  raw_px.setZero();
  J_raw_pc.setZero();
  if (ctx.width <= 0 || ctx.height <= 0) return false;
  if (!std::isfinite(ctx.fx) || !std::isfinite(ctx.fy) || !std::isfinite(ctx.cx) || !std::isfinite(ctx.cy)) return false;

  auto project_value = [&](const V3D &p, V2D &px) -> bool {
    if (!p.array().isFinite().all()) return false;
    const double norm = p.norm();
    if (!std::isfinite(norm) || norm <= kS2Eps) return false;
    const double x = p.x();
    const double y = p.y();
    const double z = p.z();
    const double a = std::hypot(x, y);
    if (!std::isfinite(a)) return false;

    double k = 0.0;
    if (a <= kS2Eps)
    {
      if (z <= kS2Eps) return false;
      k = 1.0 / z;
    }
    else
    {
      const double theta = std::atan2(a, z);
      k = theta / a;
    }
    if (!std::isfinite(k)) return false;
    px << ctx.cx + ctx.fx * k * x, ctx.cy + ctx.fy * k * y;
    return px.array().isFinite().all();
  };

  if (!project_value(p_c, raw_px)) return false;
  const int border_req = std::max(0, required_border);
  if (raw_px[0] < border_req || raw_px[1] < border_req ||
      raw_px[0] >= ctx.width - border_req || raw_px[1] >= ctx.height - border_req)
    return false;

  const double x = p_c.x();
  const double y = p_c.y();
  const double z = p_c.z();
  const double a = std::hypot(x, y);
  const double r2 = a * a + z * z;
  if (!std::isfinite(a) || !std::isfinite(r2) || r2 <= kS2Eps) return false;

  if (a > kS2Eps)
  {
    const double theta = std::atan2(a, z);
    const double k = theta / a;
    const double dtheta_dx = z * x / (a * r2);
    const double dtheta_dy = z * y / (a * r2);
    const double dtheta_dz = -a / r2;
    const double da_dx = x / a;
    const double da_dy = y / a;
    const double a2 = a * a;
    const double dk_dx = (a * dtheta_dx - theta * da_dx) / a2;
    const double dk_dy = (a * dtheta_dy - theta * da_dy) / a2;
    const double dk_dz = dtheta_dz / a;

    J_raw_pc << ctx.fx * (k + x * dk_dx), ctx.fx * (x * dk_dy), ctx.fx * (x * dk_dz),
                ctx.fy * (y * dk_dx), ctx.fy * (k + y * dk_dy), ctx.fy * (y * dk_dz);
  }
  else
  {
    const double eps_fd = 1.0e-6 * std::max(1.0, p_c.norm());
    for (int col = 0; col < 3; ++col)
    {
      V3D step = V3D::Zero();
      step[col] = eps_fd;
      V2D uv_plus, uv_minus;
      if (!project_value(p_c + step, uv_plus) || !project_value(p_c - step, uv_minus)) return false;
      J_raw_pc.col(col) = (uv_plus - uv_minus) / (2.0 * eps_fd);
    }
  }

  return J_raw_pc.array().isFinite().all();
}

bool VIOManager::projectRawCameraWithJacobian(const PerCameraData &ctx, const V3D &p_c,
                                              int required_border, V2D &raw_px,
                                              MD(2, 3) &J_raw_pc) const
{
  if (ctx.cam == nullptr || !p_c.array().isFinite().all() || p_c.norm() <= virtual_min_z) return false;
  raw_px = ctx.cam->world2cam(p_c);
  if (!raw_px.array().isFinite().all()) return false;
  const int border_req = std::max(0, required_border);
  if (raw_px[0] < border_req || raw_px[1] < border_req ||
      raw_px[0] >= ctx.width - border_req || raw_px[1] >= ctx.height - border_req)
    return false;
  if (!ctx.cam->isInFrame(raw_px.cast<int>(), border_req)) return false;
  return computeCameraModelProjectionJacobianForContext(ctx, p_c, J_raw_pc);
}
bool VIOManager::sampleRawImageValueAndGradient(const cv::Mat &raw_img, const V2D &raw_px,
                                                int scale, float &value, V2D &gradient) const
{
  value = 0.0f;
  gradient.setZero();
  if (raw_img.empty() || raw_img.type() != CV_8UC1 || scale <= 0 || !raw_px.array().isFinite().all()) return false;

  auto sample = [&](double u, double v, float &sampled) -> bool {
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0 || v < 0.0 ||
        u >= raw_img.cols - 1 || v >= raw_img.rows - 1)
      return false;
    sampled = static_cast<float>(vk::interpolateMat_8u(raw_img, u, v));
    return std::isfinite(sampled);
  };

  if (raw_px[0] - scale < 0.0 || raw_px[1] - scale < 0.0 ||
      raw_px[0] + scale >= raw_img.cols - 1 || raw_px[1] + scale >= raw_img.rows - 1)
    return false;

  float left = 0.0f;
  float right = 0.0f;
  float up = 0.0f;
  float down = 0.0f;
  if (!sample(raw_px[0], raw_px[1], value) ||
      !sample(raw_px[0] - scale, raw_px[1], left) ||
      !sample(raw_px[0] + scale, raw_px[1], right) ||
      !sample(raw_px[0], raw_px[1] - scale, up) ||
      !sample(raw_px[0], raw_px[1] + scale, down))
    return false;

  gradient << 0.5 * (right - left), 0.5 * (down - up);
  return gradient.array().isFinite().all();
}

bool VIOManager::linearizeVirtualS2Sample(const PerCameraData &ctx, const cv::Mat &raw_img,
                                          const V3D &point_c, const VirtualTrackPatch &track,
                                          const V2D &offset, int scale, double current_exposure,
                                          float &current_value, MD(1, 3) &J_photo_center) const
{
  current_value = 0.0f;
  J_photo_center.setZero();
  if (scale <= 0 || !std::isfinite(current_exposure)) return false;

  V3D p_sample_c;
  M3D J_sample_pc;
  if (!computeS2SamplePointAndJacobian(point_c, track.R_vcur_from_ccur_seed,
                                       track.R_ccur_from_vcur_seed, offset,
                                       p_sample_c, J_sample_pc))
    return false;

  V2D raw_px;
  MD(2, 3) J_raw_sample;
  const int required_border = scale + 1;
  if (!projectRawCameraWithJacobian(ctx, p_sample_c, required_border, raw_px, J_raw_sample)) return false;

  V2D raw_gradient;
  if (!sampleRawImageValueAndGradient(raw_img, raw_px, scale, current_value, raw_gradient)) return false;

  MD(1, 2) Jimg;
  Jimg << raw_gradient[0], raw_gradient[1];
  Jimg *= current_exposure * (1.0 / scale);
  J_photo_center = Jimg * J_raw_sample * J_sample_pc;
  return J_photo_center.array().isFinite().all();
}

bool VIOManager::hasRangeDiscontinuity(const PerCameraData &ctx, const cv::Mat &range_img,
                                       const V2D &raw_px, double point_range) const
{
  if (range_img.empty() || range_img.type() != CV_32FC1 ||
      range_img.cols != ctx.width || range_img.rows != ctx.height ||
      !raw_px.array().isFinite().all() || !std::isfinite(point_range))
    return false;

  const int center_col = static_cast<int>(raw_px[0]);
  const int center_row = static_cast<int>(raw_px[1]);
  for (int dv = -patch_size_half; dv <= patch_size_half; ++dv)
  {
    for (int du = -patch_size_half; du <= patch_size_half; ++du)
    {
      if (du == 0 && dv == 0) continue;
      const int col = center_col + du;
      const int row = center_row + dv;
      if (col < 0 || col >= ctx.width || row < 0 || row >= ctx.height) continue;
      const float range = range_img.at<float>(row, col);
      if (range > 0.0f && std::fabs(point_range - range) > 0.5) return true;
    }
  }
  return false;
}

bool VIOManager::buildVirtualSupportPatchPullExact(const PerCameraData &ctx, const cv::Mat &raw_img, const M3D &R_c_from_v,
                                                    VirtualPatchImage &output, const cv::Point &raw_origin) const
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
      if (!projectRawFisheyeIfValid(ctx, ray_c, 1, raw_px)) continue;
      const double local_u = raw_px[0] - raw_origin.x;
      const double local_v = raw_px[1] - raw_origin.y;
      if (local_u < 0.0 || local_v < 0.0 || local_u >= raw_img.cols - 1 || local_v >= raw_img.rows - 1) continue;
      values[x] = static_cast<float>(vk::interpolateMat_8u(raw_img, local_u, local_v));
      mask[x] = 255;
    }
  }

  output.valid = true;
  return true;
}

void VIOManager::splatRawPixelToVirtualPatch(const PerCameraData &ctx, const cv::Mat &raw_img, int raw_x, int raw_y,
                                             const M3D &R_v_from_c, cv::Mat &value_sum,
                                             cv::Mat &weight_sum) const
{
  const int raw_idx = raw_y * ctx.width + raw_x;
  if (raw_idx < 0 || raw_idx >= static_cast<int>(ctx.raw_pixel_unit_ray_valid_mask.size()) || !ctx.raw_pixel_unit_ray_valid_mask[raw_idx]) return;

  const V3D ray_c = ctx.raw_pixel_to_unit_ray_lut[raw_idx].cast<double>();
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

bool VIOManager::buildVirtualSupportPatchForwardSplat(const PerCameraData &ctx, const cv::Mat &raw_img,
                                                       const V2D &raw_center_px, const M3D &R_v_from_c,
                                                       VirtualPatchImage &output, const cv::Point &raw_origin) const
{
  if (!virtual_fisheye_patch_en || raw_img.empty() || raw_img.type() != CV_8UC1 || virtual_support_size <= 0) return false;
  if (ctx.raw_pixel_to_unit_ray_lut.size() != static_cast<size_t>(ctx.width * ctx.height) ||
      ctx.raw_pixel_unit_ray_valid_mask.size() != static_cast<size_t>(ctx.width * ctx.height))
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
    if (raw_y < 0 || raw_y >= ctx.height) continue;
    const int local_y = raw_y - raw_origin.y;
    if (local_y < 0 || local_y >= raw_img.rows) continue;
    const uint8_t *raw_row = raw_img.ptr<uint8_t>(local_y);
    for (int raw_x = raw_center_x - virtual_raw_window_half_size; raw_x <= raw_center_x + virtual_raw_window_half_size; ++raw_x)
    {
      if (raw_x < 0 || raw_x >= ctx.width) continue;
      const int local_x = raw_x - raw_origin.x;
      if (local_x < 0 || local_x >= raw_img.cols) continue;
      const int raw_idx = raw_y * ctx.width + raw_x;
      if (!ctx.raw_pixel_unit_ray_valid_mask[raw_idx]) continue;

      const V3D ray_c = ctx.raw_pixel_to_unit_ray_lut[raw_idx].cast<double>();
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
      const float intensity = static_cast<float>(raw_row[local_x]);
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

bool VIOManager::buildVirtualSupportPatch(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_center_px,
                                          const M3D &R_v_from_c, const M3D &R_c_from_v,
                                          VirtualPatchImage &output, const cv::Point &raw_origin) const
{
  if (virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::PULL_EXACT)
    return buildVirtualSupportPatchPullExact(ctx, raw_img, R_c_from_v, output, raw_origin);

  const double splat_start = omp_get_wtime();
  const bool splat_valid = buildVirtualSupportPatchForwardSplat(ctx, raw_img, raw_center_px, R_v_from_c, output, raw_origin);
  const double splat_time = omp_get_wtime() - splat_start;

  if (virtual_splat_debug_compare_pull_exact)
  {
    VirtualPatchImage pull_exact_patch;
    const double pull_start = omp_get_wtime();
    const bool pull_valid = buildVirtualSupportPatchPullExact(ctx, raw_img, R_c_from_v, pull_exact_patch, raw_origin);
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

bool VIOManager::captureVirtualReferenceSource(const PerCameraData &ctx, const cv::Mat &raw_img,
                                               const V2D &raw_center_px, const M3D &R_c_from_v,
                                               cv::Mat &raw_roi, cv::Point &raw_origin) const
{
  raw_roi.release();
  raw_origin = cv::Point();
  if (ctx.cam == nullptr || raw_img.empty() || raw_img.type() != CV_8UC1 ||
      !raw_center_px.array().isFinite().all() || !R_c_from_v.array().isFinite().all())
    return false;

  double min_u = raw_center_px[0];
  double max_u = raw_center_px[0];
  double min_v = raw_center_px[1];
  double max_v = raw_center_px[1];
  if (virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::FORWARD_SPLAT)
  {
    min_u = std::min(min_u, std::floor(raw_center_px[0]) - virtual_raw_window_half_size);
    max_u = std::max(max_u, std::floor(raw_center_px[0]) + virtual_raw_window_half_size);
    min_v = std::min(min_v, std::floor(raw_center_px[1]) - virtual_raw_window_half_size);
    max_v = std::max(max_v, std::floor(raw_center_px[1]) + virtual_raw_window_half_size);
  }

  if (virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::PULL_EXACT ||
      virtual_splat_debug_compare_pull_exact)
  {
    auto include_virtual_pixel = [&](int x, int y) {
      const V3D ray_v = virtual_support_ray_lut_[y * virtual_support_size + x].cast<double>();
      const V3D ray_c = R_c_from_v * ray_v;
      const V2D raw_px = ctx.cam->world2cam(ray_c);
      if (!raw_px.array().isFinite().all()) return;
      min_u = std::min(min_u, raw_px[0]);
      max_u = std::max(max_u, raw_px[0]);
      min_v = std::min(min_v, raw_px[1]);
      max_v = std::max(max_v, raw_px[1]);
    };
    for (int x = 0; x < virtual_support_size; ++x)
    {
      include_virtual_pixel(x, 0);
      include_virtual_pixel(x, virtual_support_size - 1);
    }
    for (int y = 1; y + 1 < virtual_support_size; ++y)
    {
      include_virtual_pixel(0, y);
      include_virtual_pixel(virtual_support_size - 1, y);
    }
  }

  const int x0 = std::max(0, static_cast<int>(std::floor(min_u)) - 2);
  const int y0 = std::max(0, static_cast<int>(std::floor(min_v)) - 2);
  const int x1 = std::min(raw_img.cols, static_cast<int>(std::ceil(max_u)) + 3);
  const int y1 = std::min(raw_img.rows, static_cast<int>(std::ceil(max_v)) + 3);
  if (x0 >= x1 || y0 >= y1) return false;

  raw_origin = cv::Point(x0, y0);
  raw_roi = raw_img(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
  return !raw_roi.empty();
}

bool VIOManager::materializeVirtualReferenceSupport(Feature &feature, bool &materialized_now)
{
  materialized_now = false;
  std::lock_guard<std::mutex> lock(feature.virtual_support_mutex_);
  if (!feature.virtual_patch_valid_ || feature.virtual_support_materialization_failed_) return false;
  if (feature.virtual_support_materialized_)
    return !feature.img_.empty() && feature.img_.type() == CV_32FC1;
  if (feature.camera_id_ < 0 || feature.camera_id_ >= numCameras() || feature.virtual_source_roi_.empty())
  {
    feature.virtual_support_materialization_failed_ = true;
    return false;
  }

  const PerCameraData &source_ctx = cameras_[feature.camera_id_];
  VirtualPatchImage support;
  const bool ok = buildVirtualSupportPatch(source_ctx, feature.virtual_source_roi_, feature.px_,
                                           feature.R_v_from_c_, feature.R_c_from_v_, support,
                                           feature.virtual_source_origin_);
  if (!ok)
  {
    feature.virtual_support_materialization_failed_ = true;
    feature.virtual_patch_valid_ = false;
    feature.virtual_source_roi_.release();
    return false;
  }

  feature.img_ = support.values;
  feature.virtual_source_roi_.release();
  feature.virtual_support_materialized_ = true;
  materialized_now = true;
  return true;
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

bool VIOManager::sampleSparseVirtualValue(const PerCameraData &ctx, const cv::Mat &raw_img,
                                          const M3D &R_c_from_v, const V2D &px, float &value) const
{
  if (raw_img.empty() || raw_img.type() != CV_8UC1 || !px.array().isFinite().all()) return false;
  const V3D ray_c = R_c_from_v * virtualCam2World(px);
  V2D raw_px;
  if (!projectRawFisheyeIfValid(ctx, ray_c, 1, raw_px)) return false;
  value = static_cast<float>(vk::interpolateMat_8u(raw_img, raw_px[0], raw_px[1]));
  return std::isfinite(value);
}

bool VIOManager::sampleSparseVirtualCorePatch(const PerCameraData &ctx, const cv::Mat &raw_img,
                                              const M3D &R_c_from_v, const V2D &center,
                                              int scale, float *patch) const
{
  if (patch == nullptr) return false;
  for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
  {
    const V2D px = center + (core_patch_offsets_[patch_index] * static_cast<float>(scale)).cast<double>();
    if (!sampleSparseVirtualValue(ctx, raw_img, R_c_from_v, px, patch[patch_index])) return false;
  }
  return true;
}

bool VIOManager::buildSparseVirtualReferenceCorePatch(const PerCameraData &ctx, const cv::Mat &raw_img,
                                                       const V2D &raw_center_px, const M3D &R_v_from_c,
                                                       const M3D &R_c_from_v, float *patch) const
{
  if (virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::PULL_EXACT)
    return sampleSparseVirtualCorePatch(ctx, raw_img, R_c_from_v,
                                        V2D(virtual_support_radius, virtual_support_radius), 1, patch);
  if (patch == nullptr || raw_img.empty() || raw_img.type() != CV_8UC1 ||
      ctx.raw_pixel_to_unit_ray_lut.size() != static_cast<size_t>(ctx.width * ctx.height) ||
      ctx.raw_pixel_unit_ray_valid_mask.size() != static_cast<size_t>(ctx.width * ctx.height))
    return false;

  const int node_size = patch_size + 1;
  const int min_offset = -patch_size_half;
  std::vector<float> value_sum(node_size * node_size, 0.0f);
  std::vector<float> weight_sum(node_size * node_size, 0.0f);
  auto add_value = [&](int virtual_x, int virtual_y, float weight, float intensity) {
    const int local_x = virtual_x - (virtual_support_radius + min_offset);
    const int local_y = virtual_y - (virtual_support_radius + min_offset);
    if (local_x < 0 || local_y < 0 || local_x >= node_size || local_y >= node_size || weight == 0.0f) return;
    const int index = local_y * node_size + local_x;
    value_sum[index] += weight * intensity;
    weight_sum[index] += weight;
  };

  const int raw_center_x = static_cast<int>(std::lround(raw_center_px[0]));
  const int raw_center_y = static_cast<int>(std::lround(raw_center_px[1]));
  for (int raw_y = raw_center_y - virtual_raw_window_half_size;
       raw_y <= raw_center_y + virtual_raw_window_half_size; ++raw_y)
  {
    if (raw_y < 0 || raw_y >= ctx.height) continue;
    const uint8_t *raw_row = raw_img.ptr<uint8_t>(raw_y);
    for (int raw_x = raw_center_x - virtual_raw_window_half_size;
         raw_x <= raw_center_x + virtual_raw_window_half_size; ++raw_x)
    {
      if (raw_x < 0 || raw_x >= ctx.width) continue;
      const int raw_index = raw_y * ctx.width + raw_x;
      if (raw_index < 0 || raw_index >= static_cast<int>(ctx.raw_pixel_unit_ray_valid_mask.size()) ||
          !ctx.raw_pixel_unit_ray_valid_mask[raw_index])
        continue;
      const V3D ray_v = R_v_from_c * ctx.raw_pixel_to_unit_ray_lut[raw_index].cast<double>();
      if (!ray_v.array().isFinite().all() || ray_v[2] <= virtual_min_z) continue;
      const float u = static_cast<float>(virtual_focal_length * ray_v[0] / ray_v[2] + virtual_support_radius);
      const float v = static_cast<float>(virtual_focal_length * ray_v[1] / ray_v[2] + virtual_support_radius);
      const int x0 = static_cast<int>(std::floor(u));
      const int y0 = static_cast<int>(std::floor(v));
      const float dx = u - x0;
      const float dy = v - y0;
      const float intensity = static_cast<float>(raw_row[raw_x]);
      add_value(x0, y0, (1.0f - dx) * (1.0f - dy), intensity);
      add_value(x0 + 1, y0, dx * (1.0f - dy), intensity);
      add_value(x0, y0 + 1, (1.0f - dx) * dy, intensity);
      add_value(x0 + 1, y0 + 1, dx * dy, intensity);
    }
  }

  for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
  {
    const int local_x = static_cast<int>(std::lround(core_patch_offsets_[patch_index][0])) - min_offset;
    const int local_y = static_cast<int>(std::lround(core_patch_offsets_[patch_index][1])) - min_offset;
    const int indices[4] = {local_y * node_size + local_x, local_y * node_size + local_x + 1,
                            (local_y + 1) * node_size + local_x, (local_y + 1) * node_size + local_x + 1};
    for (const int index : indices)
      if (weight_sum[index] <= virtual_splat_min_weight) return false;
    patch[patch_index] = value_sum[indices[0]] / weight_sum[indices[0]];
  }
  return true;
}

bool VIOManager::sampleSparseVirtualValueAndGradient(const PerCameraData &ctx, const cv::Mat &raw_img,
                                                     const M3D &R_c_from_v, const V2D &px, int scale,
                                                     float &value, V2D &gradient) const
{
  float left, right, up, down;
  if (!sampleSparseVirtualValue(ctx, raw_img, R_c_from_v, px, value) ||
      !sampleSparseVirtualValue(ctx, raw_img, R_c_from_v, px - V2D(scale, 0.0), left) ||
      !sampleSparseVirtualValue(ctx, raw_img, R_c_from_v, px + V2D(scale, 0.0), right) ||
      !sampleSparseVirtualValue(ctx, raw_img, R_c_from_v, px - V2D(0.0, scale), up) ||
      !sampleSparseVirtualValue(ctx, raw_img, R_c_from_v, px + V2D(0.0, scale), down))
    return false;
  gradient << 0.5 * (right - left), 0.5 * (down - up);
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

bool VIOManager::createVirtualFeaturePatch(const PerCameraData &ctx, const cv::Mat &raw_img,
                                            const SE3<double> &T_c_w, const V3D &point_w,
                                            float *core_patch, cv::Mat &virtual_support_img,
                                            cv::Point &virtual_source_origin, SE3<double> &T_v_w,
                                            M3D &R_v_from_c, M3D &R_c_from_v) const
{
  virtual_support_img.release();
  virtual_source_origin = cv::Point();
  const V3D point_c = T_c_w * point_w;
  V2D raw_center_px;
  if (!projectRawFisheyeIfValid(ctx, point_c, 1, raw_center_px)) return false;
  if (!buildVirtualFrameRotation(ctx, point_c, raw_center_px, R_v_from_c, R_c_from_v)) return false;
  T_v_w = composeVirtualPose(R_v_from_c, T_c_w);
  const V2D center(virtual_support_radius, virtual_support_radius);

  if (virtual_sparse_patch_en)
  {
    if (!buildSparseVirtualReferenceCorePatch(ctx, raw_img, raw_center_px, R_v_from_c, R_c_from_v, core_patch)) return false;
    return captureVirtualReferenceSource(ctx, raw_img, raw_center_px, R_c_from_v,
                                         virtual_support_img, virtual_source_origin);
  }

  VirtualPatchImage support;
  support.T_v_w_seed = T_v_w;
  if (!buildVirtualSupportPatch(ctx, raw_img, raw_center_px, R_v_from_c, R_c_from_v, support)) return false;
  if (!sampleVirtualCorePatch(support, center, 1, core_patch)) return false;
  virtual_support_img = support.values;
  return true;
}

bool VIOManager::extractRefPatchDumpRawRoi(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_px,
                                           const M3D &R_c_from_v, cv::Mat &raw_roi, M3D &raw_to_roi) const
{
  raw_to_roi.setIdentity();
  if (ctx.cam == nullptr || raw_img.empty() || raw_img.type() != CV_8UC1 || virtual_support_size <= 0 ||
      virtual_support_ray_lut_.size() != static_cast<size_t>(virtual_support_size * virtual_support_size) ||
      !raw_px.array().isFinite().all() || !R_c_from_v.array().isFinite().all())
    return false;

  double min_u = raw_px[0];
  double max_u = raw_px[0];
  double min_v = raw_px[1];
  double max_v = raw_px[1];
  int valid_projected_points = 0;
  for (const V3F &ray_v_float : virtual_support_ray_lut_)
  {
    const V3D ray_c = R_c_from_v * ray_v_float.cast<double>();
    const V2D projected_px = ctx.cam->world2cam(ray_c);
    if (!projected_px.array().isFinite().all() ||
        projected_px[0] < -raw_img.cols || projected_px[0] > 2 * raw_img.cols ||
        projected_px[1] < -raw_img.rows || projected_px[1] > 2 * raw_img.rows)
      continue;
    min_u = std::min(min_u, projected_px[0]);
    max_u = std::max(max_u, projected_px[0]);
    min_v = std::min(min_v, projected_px[1]);
    max_v = std::max(max_v, projected_px[1]);
    ++valid_projected_points;
  }
  if (valid_projected_points < 4) return false;

  const int requested_x0 = static_cast<int>(std::floor(min_u));
  const int requested_y0 = static_cast<int>(std::floor(min_v));
  const int requested_x1 = static_cast<int>(std::ceil(max_u)) + 1;
  const int requested_y1 = static_cast<int>(std::ceil(max_v)) + 1;
  const int native_width = requested_x1 - requested_x0;
  const int native_height = requested_y1 - requested_y0;
  if (native_width <= 0 || native_height <= 0) return false;

  const int source_x0 = std::max(0, requested_x0);
  const int source_y0 = std::max(0, requested_y0);
  const int source_x1 = std::min(raw_img.cols, requested_x1);
  const int source_y1 = std::min(raw_img.rows, requested_y1);
  if (source_x0 >= source_x1 || source_y0 >= source_y1) return false;

  raw_roi = cv::Mat::zeros(native_height, native_width, CV_8UC1);
  const cv::Rect source_rect(source_x0, source_y0, source_x1 - source_x0, source_y1 - source_y0);
  const cv::Rect destination_rect(source_x0 - requested_x0, source_y0 - requested_y0,
                                  source_rect.width, source_rect.height);
  raw_img(source_rect).copyTo(raw_roi(destination_rect));

  // Preserve native raw pixels; map debug markers by removing only the crop origin.
  raw_to_roi << 1.0, 0.0, -static_cast<double>(requested_x0),
                0.0, 1.0, -static_cast<double>(requested_y0),
                0.0, 0.0, 1.0;
  return true;
}

void VIOManager::maybeInitializeRefPatchDumpProbe(const PerCameraData &ctx, const V3D &point_w, const V3D &normal_w,
                                                   const V2D &raw_px, const V3D &bearing, const float *core_patch)
{
  if (!ref_patch_dump_en || !virtual_fisheye_patch_en || ctx.camera_id != kRefPatchDumpCameraId ||
      ref_patch_dump_probe_.active || ref_patch_dump_next_point_id_ >= kRefPatchDumpMaxPoints ||
      core_patch == nullptr || ctx.new_frame == nullptr)
    return;

  const double bearing_norm = bearing.norm();
  const double normal_norm = normal_w.norm();
  if (!point_w.array().isFinite().all() || !normal_w.array().isFinite().all() ||
      !std::isfinite(bearing_norm) || bearing_norm <= 1.0e-9 || !std::isfinite(normal_norm) || normal_norm <= 1.0e-9)
    return;
  const double incidence_deg =
      std::acos(std::clamp(bearing[2] / bearing_norm, -1.0, 1.0)) * kRadiansToDegrees;
  if (incidence_deg > kRefPatchDumpMaxInitialIncidenceDeg) return;

  double sum = 0.0;
  double squared_sum = 0.0;
  for (int i = 0; i < patch_size_total; ++i)
  {
    sum += core_patch[i];
    squared_sum += core_patch[i] * core_patch[i];
  }
  const double mean = sum / patch_size_total;
  const double variance = std::max(0.0, squared_sum / patch_size_total - mean * mean);
  if (std::sqrt(variance) < kRefPatchDumpMinStdDev) return;

  if (ref_patch_dump_candidates_to_skip_ < 0)
  {
    std::uniform_int_distribution<int> skip_distribution(0, ref_patch_dump_max_candidate_skip);
    ref_patch_dump_candidates_to_skip_ = skip_distribution(ref_patch_dump_rng_);
  }
  if (ref_patch_dump_candidates_to_skip_ > 0)
  {
    --ref_patch_dump_candidates_to_skip_;
    return;
  }
  ref_patch_dump_candidates_to_skip_ = -1;

  ref_patch_dump_probe_ = RefPatchDumpProbeState();
  ref_patch_dump_probe_.active = true;
  ref_patch_dump_probe_.point_id = ref_patch_dump_next_point_id_++;
  ref_patch_dump_probe_.selected_frame_id = ctx.new_frame->id_;
  ref_patch_dump_probe_.normal_w = normal_w / normal_norm;
  const V3D point_c = ctx.new_frame->T_f_w_ * point_w;
  const V3D normal_c = ctx.new_frame->T_f_w_.rotationMatrix() * ref_patch_dump_probe_.normal_w;
  if (point_c.normalized().dot(normal_c) < 0.0)
    ref_patch_dump_probe_.normal_w = -ref_patch_dump_probe_.normal_w;
  ref_patch_dump_probe_.point_w = point_w;
  printf("[ VIO Ref Patch Dump ] Selected probe=%03d camera=0 frame=%d incidence=%.2f deg pixel=(%.1f, %.1f)\n",
         ref_patch_dump_probe_.point_id, ctx.new_frame->id_, incidence_deg, raw_px[0], raw_px[1]);
}

void VIOManager::initializeRefPatchDump()
{
  if (ref_patch_dump_initialized_) return;

  const std::filesystem::path root = std::filesystem::path(ROOT_DIR) / "Log" / "ref_patch_probe";
  std::filesystem::create_directories(root / "raw");
  std::filesystem::create_directories(root / "virtual");
  std::filesystem::create_directories(root / "raw_patch");
  std::filesystem::create_directories(root / "virtual_patch");
  std::ofstream selection(root / "selection.txt");
  selection << "random_seed=" << ref_patch_dump_effective_seed_ << "\n"
            << "max_candidate_skip=" << ref_patch_dump_max_candidate_skip << "\n"
            << "ncc_threshold=" << ref_patch_dump_ncc_threshold << "\n";
  const std::filesystem::path index_path = root / "index.csv";
  if (!std::filesystem::exists(index_path) || std::filesystem::file_size(index_path) == 0)
  {
    std::ofstream index(index_path);
    index << "point_id,ref_index,camera_id,frame_id,frames_since_selection,timestamp,incidence_deg,"
             "world_x,world_y,world_z,raw_u,raw_v,raw_path,virtual_path\n";
  }

  ref_patch_dump_initialized_ = true;
  printf("[ VIO Ref Patch Dump ] Output: %s\n", root.string().c_str());
}

void VIOManager::processRefPatchDumpProbe(PerCameraData &ctx, const cv::Mat &raw_img)
{
  if (!ref_patch_dump_en || !ref_patch_dump_probe_.active || ctx.camera_id != kRefPatchDumpCameraId ||
      ctx.new_frame == nullptr)
    return;

  auto mark_missed = [&]() {
    ++ref_patch_dump_probe_.missed_frames;
    if (ref_patch_dump_probe_.missed_frames >= kRefPatchDumpMaxMissedFrames)
    {
      printf("[ VIO Ref Patch Dump ] Probe=%03d released after %d missed frames with %d saved refs.\n",
             ref_patch_dump_probe_.point_id, ref_patch_dump_probe_.missed_frames,
             ref_patch_dump_probe_.saved_refs);
      ref_patch_dump_probe_.active = false;
    }
  };

  const V3D point_c = ctx.new_frame->w2f(ref_patch_dump_probe_.point_w);
  V2D raw_px;
  if (!projectRawFisheyeIfValid(ctx, point_c, 1, raw_px))
  {
    mark_missed();
    return;
  }

  const double point_norm = point_c.norm();
  if (!std::isfinite(point_norm) || point_norm <= 1.0e-9)
  {
    mark_missed();
    return;
  }
  if (ref_patch_dump_range_pose_valid_)
  {
    const V3D range_point_c = ref_patch_dump_range_T_f_w_ * ref_patch_dump_probe_.point_w;
    V2D range_raw_px;
    if (projectRawFisheyeIfValid(ctx, range_point_c, 1, range_raw_px) &&
        hasRangeDiscontinuity(ctx, ref_patch_dump_range_img_, range_raw_px, range_point_c.norm()))
    {
      printf("[ VIO Ref Patch Dump ] Probe=%03d stopped at frame=%d after range discontinuity, with %d saved refs.\n",
             ref_patch_dump_probe_.point_id, ctx.new_frame->id_, ref_patch_dump_probe_.saved_refs);
      ref_patch_dump_probe_.active = false;
      return;
    }
  }

  const double incidence_deg =
      std::acos(std::clamp(point_c[2] / point_norm, -1.0, 1.0)) * kRadiansToDegrees;
  const int frame_id = ctx.new_frame->id_;
  const bool first_sample = ref_patch_dump_probe_.saved_refs == 0;
  const bool interval_due = frame_id - ref_patch_dump_probe_.last_saved_frame_id >= kRefPatchDumpFrameInterval;
  const bool angle_due =
      std::fabs(incidence_deg - ref_patch_dump_probe_.last_saved_incidence_deg) >= kRefPatchDumpMinAngleChangeDeg;
  const bool save_due = first_sample || interval_due || angle_due;

  M3D R_v_from_c, R_c_from_v;
  if (!buildVirtualFrameRotation(ctx, point_c, raw_px, R_v_from_c, R_c_from_v))
  {
    mark_missed();
    return;
  }
  VirtualPatchImage support;
  if (!buildVirtualSupportPatch(ctx, raw_img, raw_px, R_v_from_c, R_c_from_v, support))
  {
    mark_missed();
    return;
  }

  const SE3<double> T_v_w = composeVirtualPose(R_v_from_c, ctx.new_frame->T_f_w_);
  cv::Mat raw_patch_display;
  cv::Mat virtual_patch_display;
  std::vector<V2D> raw_sample_pixels;
  std::vector<V2D> virtual_sample_pixels;
  double virtual_ncc = std::numeric_limits<double>::quiet_NaN();
  if (!buildRefPatchDumpWarpPatches(ctx, raw_img, raw_px, point_c, T_v_w, support.values,
                                    raw_patch_display, virtual_patch_display,
                                    raw_sample_pixels, virtual_sample_pixels, virtual_ncc))
  {
    if (std::isfinite(virtual_ncc) && virtual_ncc < ref_patch_dump_ncc_threshold)
      printf("[ VIO Ref Patch Dump ] Probe=%03d stopped at frame=%d: virtual NCC %.3f < %.3f, with %d saved refs.\n",
             ref_patch_dump_probe_.point_id, ctx.new_frame->id_, virtual_ncc,
             ref_patch_dump_ncc_threshold, ref_patch_dump_probe_.saved_refs);
    else
      printf("[ VIO Ref Patch Dump ] Probe=%03d stopped at frame=%d: debug patch warp invalid, with %d saved refs.\n",
             ref_patch_dump_probe_.point_id, ctx.new_frame->id_, ref_patch_dump_probe_.saved_refs);
    ref_patch_dump_probe_.active = false;
    return;
  }
  ref_patch_dump_probe_.missed_frames = 0;
  if (!save_due) return;

  cv::Mat raw_roi;
  M3D raw_to_roi;
  if (!extractRefPatchDumpRawRoi(ctx, raw_img, raw_px, R_c_from_v, raw_roi, raw_to_roi))
  {
    mark_missed();
    return;
  }

  const V2D virtual_px = virtualProject(T_v_w * ref_patch_dump_probe_.point_w);
  dumpRefPatchProbeObservation(ctx, raw_px, virtual_px, incidence_deg, raw_roi, raw_to_roi,
                               support.values, raw_sample_pixels, virtual_sample_pixels,
                               raw_patch_display, virtual_patch_display);
  if (ref_patch_dump_probe_.saved_refs >= kRefPatchDumpMaxRefsPerPoint)
  {
    printf("[ VIO Ref Patch Dump ] Probe=%03d completed with %d saved refs.\n",
           ref_patch_dump_probe_.point_id, ref_patch_dump_probe_.saved_refs);
    ref_patch_dump_probe_.active = false;
  }
}

bool VIOManager::buildRefPatchDumpWarpPatches(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_px,
                                               const V3D &point_c, const SE3<double> &T_v_w,
                                               const cv::Mat &virtual_support_img, cv::Mat &raw_patch_display,
                                               cv::Mat &virtual_patch_display,
                                               std::vector<V2D> &raw_sample_pixels, std::vector<V2D> &virtual_sample_pixels,
                                               double &virtual_ncc)
{
  virtual_ncc = std::numeric_limits<double>::quiet_NaN();
  raw_patch_display.release();
  virtual_patch_display.release();
  raw_sample_pixels.clear();
  virtual_sample_pixels.clear();
  if (!ref_patch_dump_probe_.active || ctx.new_frame == nullptr || ctx.cam == nullptr ||
      raw_img.empty() || raw_img.type() != CV_8UC1 || virtual_support_img.empty() ||
      virtual_support_img.type() != CV_32FC1 || !raw_px.array().isFinite().all() ||
      !point_c.array().isFinite().all() || point_c.norm() <= virtual_min_z ||
      patch_size_half <= 0 || patch_size_half * 2 != patch_size)
    return false;

  if (!ref_patch_dump_probe_.anchor_valid)
  {
    ref_patch_dump_probe_.anchor_T_c_w = ctx.new_frame->T_f_w_;
    ref_patch_dump_probe_.anchor_T_v_w = T_v_w;
    ref_patch_dump_probe_.anchor_valid = true;
  }

  auto make_display = [&](const std::vector<float> &values, cv::Mat &display) {
    if (values.size() != static_cast<size_t>(patch_size_total)) return false;
    cv::Mat patch_image(patch_size, patch_size, CV_8UC1);
    for (int y = 0; y < patch_size; ++y)
    {
      uint8_t *destination = patch_image.ptr<uint8_t>(y);
      for (int x = 0; x < patch_size; ++x)
      {
        const float value = values[y * patch_size + x];
        destination[x] = std::isfinite(value)
                             ? static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)))
                             : 0;
      }
    }
    cv::resize(patch_image, display, cv::Size(kRefPatchDumpWarpDisplaySize, kRefPatchDumpWarpDisplaySize),
               0.0, 0.0, cv::INTER_NEAREST);
    return true;
  };
  auto collect_sample_pixels = [&](const Matrix2d &A_target_source, const V2D &source_center,
                                   std::vector<V2D> &sample_pixels) {
    sample_pixels.clear();
    if (!A_target_source.array().isFinite().all() || std::fabs(A_target_source.determinant()) <= 1.0e-9)
      return false;
    const Matrix2f A_source_target = A_target_source.inverse().cast<float>();
    const V2F center = source_center.cast<float>();
    sample_pixels.reserve(patch_size_total);
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x)
      {
        const V2F offset(x - patch_size_half, y - patch_size_half);
        const V2F pixel = A_source_target * offset + center;
        sample_pixels.push_back(pixel.cast<double>());
      }
    }
    return sample_pixels.size() == static_cast<size_t>(patch_size_total);
  };

  bool raw_valid = false;
  V3D source_bearing = ctx.cam->cam2world(raw_px);
  const double bearing_norm = source_bearing.norm();
  if (source_bearing.array().isFinite().all() && std::isfinite(bearing_norm) && bearing_norm > virtual_min_z)
  {
    source_bearing /= bearing_norm;
    // Current observation is the source; the first saved observation is the fixed target.
    const SE3<double> T_canchor_csource =
        ref_patch_dump_probe_.anchor_T_c_w * ctx.new_frame->T_f_w_.inverse();
    Matrix2d A_anchor_source = Matrix2d::Zero();
    bool raw_affine_valid = false;
    if (normal_en)
    {
      V3D normal_c = ctx.new_frame->T_f_w_.rotationMatrix() * ref_patch_dump_probe_.normal_w;
      const double normal_norm = normal_c.norm();
      if (normal_c.array().isFinite().all() && std::isfinite(normal_norm) && normal_norm > virtual_min_z)
      {
        normal_c /= normal_norm;
        getWarpMatrixAffineHomography(ctx, ctx, raw_px, point_c, normal_c, T_canchor_csource, 0, A_anchor_source);
        raw_affine_valid = true;
      }
    }
    else
    {
      getWarpMatrixAffine(ctx, ctx, raw_px, source_bearing, point_c.norm(), T_canchor_csource,
                          0, 0, patch_size_half, A_anchor_source);
      raw_affine_valid = true;
    }

    if (raw_affine_valid && A_anchor_source.array().isFinite().all() && std::fabs(A_anchor_source.determinant()) > 1.0e-9)
    {
      std::vector<float> raw_patch(patch_size_total, 0.0f);
      if (warpAffine(A_anchor_source, raw_img, raw_px, 0, 0, 0, patch_size_half, raw_patch.data()))
      {
        collect_sample_pixels(A_anchor_source, raw_px, raw_sample_pixels);
        raw_valid = make_display(raw_patch, raw_patch_display);
      }
    }
  }

  bool virtual_valid = false;
  const V3D point_vsource = T_v_w * ref_patch_dump_probe_.point_w;
  const SE3<double> T_vanchor_vsource = ref_patch_dump_probe_.anchor_T_v_w * T_v_w.inverse();
  Matrix2d A_virtual_anchor_source = Matrix2d::Zero();
  bool virtual_affine_valid = false;
  if (normal_en)
  {
    V3D normal_vsource = T_v_w.rotationMatrix() * ref_patch_dump_probe_.normal_w;
    const double normal_norm = normal_vsource.norm();
    if (normal_vsource.array().isFinite().all() && std::isfinite(normal_norm) && normal_norm > virtual_min_z)
    {
      normal_vsource /= normal_norm;
      virtual_affine_valid = getWarpMatrixAffineHomographyVirtual(
          point_vsource, normal_vsource, T_vanchor_vsource, 0, A_virtual_anchor_source);
    }
  }
  else
  {
    virtual_affine_valid = getWarpMatrixAffineVirtual(
        point_vsource, T_vanchor_vsource, 0, 0, patch_size_half, A_virtual_anchor_source);
  }

  if (virtual_affine_valid && A_virtual_anchor_source.array().isFinite().all() &&
      std::fabs(A_virtual_anchor_source.determinant()) > 1.0e-9)
  {
    std::vector<float> virtual_patch(patch_size_total, 0.0f);
    if (warpAffineVirtual(A_virtual_anchor_source, virtual_support_img, 0, 0, 0,
                          patch_size_half, virtual_patch.data()))
    {
      collect_sample_pixels(A_virtual_anchor_source, V2D(virtual_support_radius, virtual_support_radius), virtual_sample_pixels);
      if (ref_patch_dump_probe_.anchor_virtual_patch.empty())
      {
        ref_patch_dump_probe_.anchor_virtual_patch = virtual_patch;
        virtual_ncc = 1.0;
      }
      else
        virtual_ncc = calculateNCC(ref_patch_dump_probe_.anchor_virtual_patch.data(),
                                   virtual_patch.data(), patch_size_total);
      if (std::isfinite(virtual_ncc) && virtual_ncc >= ref_patch_dump_ncc_threshold)
        virtual_valid = make_display(virtual_patch, virtual_patch_display);
    }
  }

  return raw_valid && virtual_valid;
}

void VIOManager::dumpRefPatchProbeObservation(const PerCameraData &ctx, const V2D &raw_px, const V2D &virtual_px,
                                               double incidence_deg, const cv::Mat &raw_roi, const M3D &raw_to_roi,
                                               const cv::Mat &virtual_support_img,
                                               const std::vector<V2D> &raw_sample_pixels,
                                               const std::vector<V2D> &virtual_sample_pixels,
                                               const cv::Mat &raw_patch_display, const cv::Mat &virtual_patch_display)
{
  if (!ref_patch_dump_probe_.active || ctx.new_frame == nullptr || raw_roi.empty() ||
      virtual_support_img.empty() || virtual_support_img.type() != CV_32FC1)
    return;

  cv::Mat virtual_gray(virtual_support_img.rows, virtual_support_img.cols, CV_8UC1);
  for (int y = 0; y < virtual_support_img.rows; ++y)
  {
    const float *source = virtual_support_img.ptr<float>(y);
    uint8_t *destination = virtual_gray.ptr<uint8_t>(y);
    for (int x = 0; x < virtual_support_img.cols; ++x)
      destination[x] = std::isfinite(source[x])
                           ? static_cast<uint8_t>(std::lround(std::clamp(source[x], 0.0f, 255.0f)))
                           : 0;
  }

  cv::Mat raw_display;
  cv::Mat virtual_display;
  cv::cvtColor(raw_roi, raw_display, CV_GRAY2BGR);
  cv::cvtColor(virtual_gray, virtual_display, CV_GRAY2BGR);
  auto draw_marker = [](cv::Mat &image, const V2D &pixel, const cv::Vec3b &color) {
    if (!pixel.array().isFinite().all()) return;
    const cv::Point center(static_cast<int>(std::lround(pixel[0])),
                           static_cast<int>(std::lround(pixel[1])));
    if (center.x < 0 || center.x >= image.cols || center.y < 0 || center.y >= image.rows) return;
    image.at<cv::Vec3b>(center.y, center.x) = color;
  };
  auto map_raw_to_roi = [&](const V2D &pixel) {
    const V3D mapped = raw_to_roi * V3D(pixel[0], pixel[1], 1.0);
    return V2D(mapped[0], mapped[1]);
  };

  const cv::Vec3b green(0, 255, 0);
  const cv::Vec3b red(0, 0, 255);
  for (const V2D &sample_pixel : raw_sample_pixels)
    draw_marker(raw_display, map_raw_to_roi(sample_pixel), green);
  for (const V2D &sample_pixel : virtual_sample_pixels)
    draw_marker(virtual_display, sample_pixel, green);

  // Draw the feature last so it remains red when it overlaps a sample pixel.
  draw_marker(raw_display, map_raw_to_roi(raw_px), red);
  draw_marker(virtual_display, virtual_px, red);

  initializeRefPatchDump();
  const std::filesystem::path root = std::filesystem::path(ROOT_DIR) / "Log" / "ref_patch_probe";
  std::ostringstream point_name;
  point_name << "point_" << std::setw(3) << std::setfill('0') << ref_patch_dump_probe_.point_id;
  const std::filesystem::path raw_dir = root / "raw" / point_name.str();
  const std::filesystem::path virtual_dir = root / "virtual" / point_name.str();
  const std::filesystem::path raw_patch_dir = root / "raw_patch" / point_name.str();
  const std::filesystem::path virtual_patch_dir = root / "virtual_patch" / point_name.str();
  std::filesystem::create_directories(raw_dir);
  std::filesystem::create_directories(virtual_dir);
  std::filesystem::create_directories(raw_patch_dir);
  std::filesystem::create_directories(virtual_patch_dir);

  std::ostringstream file_name;
  file_name << "ref_" << std::setw(3) << std::setfill('0') << ref_patch_dump_probe_.saved_refs
            << "_cam_0_frame_" << std::setw(6) << std::setfill('0') << ctx.new_frame->id_ << ".png";
  const std::filesystem::path raw_path = raw_dir / file_name.str();
  const std::filesystem::path virtual_path = virtual_dir / file_name.str();
  const std::filesystem::path raw_patch_path = raw_patch_dir / file_name.str();
  const std::filesystem::path virtual_patch_path = virtual_patch_dir / file_name.str();
  const bool raw_saved = cv::imwrite(raw_path.string(), raw_display);
  const bool virtual_saved = cv::imwrite(virtual_path.string(), virtual_display);
  if (!raw_saved || !virtual_saved)
  {
    printf("[ VIO Ref Patch Dump ] Failed to save probe=%03d ref=%03d\n",
           ref_patch_dump_probe_.point_id, ref_patch_dump_probe_.saved_refs);
    return;
  }
  bool patch_saved = true;
  if (!raw_patch_display.empty())
    patch_saved = cv::imwrite(raw_patch_path.string(), raw_patch_display) && patch_saved;
  if (!virtual_patch_display.empty())
    patch_saved = cv::imwrite(virtual_patch_path.string(), virtual_patch_display) && patch_saved;
  if (!patch_saved)
  {
    printf("[ VIO Ref Patch Dump ] Failed to save warped patch probe=%03d ref=%03d\n",
           ref_patch_dump_probe_.point_id, ref_patch_dump_probe_.saved_refs);
  }
  std::ofstream index(root / "index.csv", std::ios::app);
  index << ref_patch_dump_probe_.point_id << ',' << ref_patch_dump_probe_.saved_refs << ','
        << ctx.camera_id << ',' << ctx.new_frame->id_ << ','
        << ctx.new_frame->id_ - ref_patch_dump_probe_.selected_frame_id << ','
        << std::fixed << std::setprecision(9) << ctx.new_frame->timestamp_ << ','
        << std::setprecision(3) << incidence_deg << ','
        << ref_patch_dump_probe_.point_w[0] << ',' << ref_patch_dump_probe_.point_w[1] << ','
        << ref_patch_dump_probe_.point_w[2] << ',' << raw_px[0] << ',' << raw_px[1] << ','
        << std::filesystem::relative(raw_path, root).generic_string() << ','
        << std::filesystem::relative(virtual_path, root).generic_string() << '\n';

  ref_patch_dump_probe_.last_saved_frame_id = ctx.new_frame->id_;
  ref_patch_dump_probe_.last_saved_incidence_deg = incidence_deg;
  printf("[ VIO Ref Patch Dump ] Saved probe=%03d ref=%03d frame=%d incidence=%.2f deg\n",
         ref_patch_dump_probe_.point_id, ref_patch_dump_probe_.saved_refs,
         ctx.new_frame->id_, incidence_deg);
  ++ref_patch_dump_probe_.saved_refs;
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

void VIOManager::getWarpMatrixAffineHomography(const PerCameraData &ref_ctx, const PerCameraData &cur_ctx, const V2D &px_ref,
                                               const V3D &xyz_ref, const V3D &normal_ref, const SE3<double> &T_cur_ref,
                                               const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotationMatrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(ref_ctx.cam->cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(ref_ctx.cam->cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cur_ctx.cam->world2cam(f_cur));
  V2D px_du_cur(cur_ctx.cam->world2cam(f_du_cur));
  V2D px_dv_cur(cur_ctx.cam->world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const PerCameraData &ref_ctx, const PerCameraData &cur_ctx, const Vector2d &px_ref,
                                     const Vector3d &f_ref, const double depth_ref, const SE3<double> &T_cur_ref,
                                     const int level_ref, const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(ref_ctx.cam->cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(ref_ctx.cam->cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  const double ref_range = xyz_ref.norm();
  xyz_du_ref.normalize();
  xyz_dv_ref.normalize();
  xyz_du_ref *= ref_range;
  xyz_dv_ref *= ref_range;
  const Vector2d px_cur(cur_ctx.cam->world2cam(T_cur_ref * xyz_ref));
  const Vector2d px_du(cur_ctx.cam->world2cam(T_cur_ref * xyz_du_ref));
  const Vector2d px_dv(cur_ctx.cam->world2cam(T_cur_ref * xyz_dv_ref));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

bool VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                            const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const double debug_affine_det = A_cur_ref.determinant();
  const bool debug_warp_input_ok = !img_ref.empty() && img_ref.type() == CV_8UC1 && patch != nullptr &&
                                   halfpatch_size > 0 && search_level >= 0 && pyramid_level >= 0 &&
                                   A_cur_ref.array().isFinite().all() && std::isfinite(debug_affine_det) &&
                                   std::fabs(debug_affine_det) > 1e-9 && px_ref.array().isFinite().all();
  if (!debug_warp_input_ok)
  {
    printf("[ VIO Debug ] warpAffine reject det=%.6e finite=%d px=(%.2f,%.2f) img=%dx%d type=%d level_ref=%d search=%d pyramid=%d half=%d patch_null=%d\n",
           debug_affine_det, A_cur_ref.array().isFinite().all() ? 1 : 0, px_ref[0], px_ref[1], img_ref.cols, img_ref.rows, img_ref.type(),
           level_ref, search_level, pyramid_level, halfpatch_size, patch == nullptr ? 1 : 0);
    fflush(stdout);
    return false;
  }

  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (!A_ref_cur.array().isFinite().all())
  {
    printf("[ VIO Debug ] warpAffine reject inverse_nonfinite det=%.6e\n", debug_affine_det);
    fflush(stdout);
    return false;
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
      if (!px.array().isFinite().all()) return false;
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
  return true;
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

void VIOManager::retrieveFromVisualSparseMapVirtual(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                                    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  ctx.visual_submap->reset();
  ctx.total_points = 0;
  if (ref_patch_dump_en && ctx.camera_id == kRefPatchDumpCameraId)
  {
    ref_patch_dump_range_img_.release();
    ref_patch_dump_range_pose_valid_ = false;
  }
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
  virtual_ref_support_materialized_count_ = 0;
  virtual_ref_support_materialize_fail_count_ = 0;
  virtual_ref_support_materialize_time_ = 0.0;
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
    ctx.rejected_visual_points_for_draw.clear();
    ctx.rejected_visual_points_for_draw.reserve(ctx.length);
  }
  if (feat_map.empty()) return;

  const double candidate_select_start = omp_get_wtime();
  ctx.sub_feat_map.clear();

  const float voxel_size = 0.5f;
  cv::Mat range_img = cv::Mat::zeros(ctx.height, ctx.width, CV_32FC1);
  int loc_xyz[3];

  for (const auto &point : pg)
  {
    const V3D &pt_w = point.point_w;
    for (int j = 0; j < 3; ++j) loc_xyz[j] = static_cast<int>(std::floor(pt_w[j] / voxel_size));
    ctx.sub_feat_map[VOXEL_LOCATION(loc_xyz[0], loc_xyz[1], loc_xyz[2])] = 0;

    const V3D pt_c = ctx.new_frame->w2f(pt_w);
    V2D px;
    if (!projectRawFisheyeIfValid(ctx, pt_c, 1, px)) continue;
    const int col = static_cast<int>(px[0]);
    const int row = static_cast<int>(px[1]);
    float &stored_range = range_img.at<float>(row, col);
    const float range = static_cast<float>(pt_c.norm());
    if (stored_range == 0.0f || range < stored_range) stored_range = range;
  }

  if (ref_patch_dump_en && ctx.camera_id == kRefPatchDumpCameraId)
  {
    ref_patch_dump_range_img_ = range_img;
    ref_patch_dump_range_T_f_w_ = ctx.new_frame->T_f_w_;
    ref_patch_dump_range_pose_valid_ = true;
  }

  vector<VOXEL_LOCATION> delete_keys;
  for (auto &sub_voxel : ctx.sub_feat_map)
  {
    auto map_voxel = feat_map.find(sub_voxel.first);
    if (map_voxel == feat_map.end()) continue;

    bool voxel_in_fov = false;
    for (VisualPoint *pt : map_voxel->second->voxel_points)
    {
      if (pt == nullptr || pt->obs_.empty()) continue;
      V2D raw_px;
      if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, raw_px)) continue;
      const int grid_col = static_cast<int>(raw_px[0] / ctx.grid_size);
      const int grid_row = static_cast<int>(raw_px[1] / ctx.grid_size);
      if (grid_col < 0 || grid_col >= ctx.grid_n_width || grid_row < 0 || grid_row >= ctx.grid_n_height) continue;
      const int index = grid_row * ctx.grid_n_width + grid_col;
      voxel_in_fov = true;
      ctx.grid_num[index] = TYPE_MAP;
      const float cur_dist = static_cast<float>((ctx.new_frame->pos() - pt->pos_).norm());
      if (cur_dist <= ctx.map_dist[index])
      {
        ctx.map_dist[index] = cur_dist;
        ctx.retrieve_voxel_points[index] = pt;
      }
    }
    if (!voxel_in_fov) delete_keys.push_back(sub_voxel.first);
  }

  if (raycast_en)
  {
    for (int i = 0; i < ctx.length; ++i)
    {
      if (ctx.grid_num[i] == TYPE_MAP || ctx.border_flag[i] == 1) continue;
      for (const V3D &sample_c : ctx.rays_with_sample_points[i])
      {
        const V3D sample_w = ctx.new_frame->f2w(sample_c);
        for (int j = 0; j < 3; ++j) loc_xyz[j] = static_cast<int>(std::floor(sample_w[j] / voxel_size));
        const VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
        if (ctx.sub_feat_map.find(sample_pos) != ctx.sub_feat_map.end()) break;

        auto visual_voxel = feat_map.find(sample_pos);
        if (visual_voxel != feat_map.end())
        {
          bool found = false;
          for (VisualPoint *pt : visual_voxel->second->voxel_points)
          {
            if (pt == nullptr || pt->obs_.empty()) continue;
            V2D raw_px;
            if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, raw_px)) continue;
            const int grid_col = static_cast<int>(raw_px[0] / ctx.grid_size);
            const int grid_row = static_cast<int>(raw_px[1] / ctx.grid_size);
            if (grid_col < 0 || grid_col >= ctx.grid_n_width || grid_row < 0 || grid_row >= ctx.grid_n_height) continue;
            const int index = grid_row * ctx.grid_n_width + grid_col;
            ctx.grid_num[index] = TYPE_MAP;
            const float cur_dist = static_cast<float>((ctx.new_frame->pos() - pt->pos_).norm());
            if (cur_dist <= ctx.map_dist[index])
            {
              ctx.map_dist[index] = cur_dist;
              ctx.retrieve_voxel_points[index] = pt;
            }
            found = true;
          }
          if (found) ctx.sub_feat_map[sample_pos] = 0;
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
            ctx.visual_submap->add_from_voxel_map.push_back(plane_center);
            break;
          }
        }
      }
    }
  }
  for (const auto &key : delete_keys) ctx.sub_feat_map.erase(key);

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
    VIRTUAL_REJECT_REFERENCE_SUPPORT,
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
    double ref_materialize_time = 0.0;
    double warp_time = 0.0;
    double current_core_time = 0.0;
    int search_level = -1;
    int warp_fail_pyramid_level = -1;
    int rejection = 0;
    bool ref_materialized = false;
    bool valid = false;
  };
  auto recordRejectedPoint = [&](const V2D &raw_px, int reason) {
    if (draw_rejected_points_en)
    {
      ctx.rejected_visual_points_for_draw.emplace_back(
          cv::Point2f(static_cast<float>(raw_px[0]), static_cast<float>(raw_px[1])), reason);
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
    case VIRTUAL_REJECT_REFERENCE_SUPPORT:
      return VIOManager::REJECT_DRAW_REF_INVALID;
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
  candidates.reserve(ctx.length);
  for (int i = 0; i < ctx.length; ++i)
  {
    if (ctx.grid_num[i] != TYPE_MAP) continue;
    ++virtual_map_grid_count_;
    VisualPoint *pt = ctx.retrieve_voxel_points[i];
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
        if (projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, raw_px))
          recordRejectedPoint(raw_px, REJECT_DRAW_NORMAL_UNINIT);
      }
      continue;
    }

    V2D raw_px;
    const V3D pt_c_seed = ctx.new_frame->w2f(pt->pos_);
    if (!projectRawFisheyeIfValid(ctx, pt_c_seed, 1, raw_px))
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
        if (col < 0 || col >= ctx.width || row < 0 || row >= ctx.height) continue;
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
      pt->ensureCameraCount(numCameras());
      Feature *cached_ref = pt->referencePatch(ctx.camera_id, cross_camera_reference_en);
      if (cached_ref != nullptr && cached_ref->virtual_patch_valid_)
      {
        ref_ftr = cached_ref;
      }
      else
      {
        float minimum_error = std::numeric_limits<float>::max();
        for (Feature *candidate_ref : pt->obs_)
        {
          if (candidate_ref == nullptr || !candidate_ref->virtual_patch_valid_) continue;
          if (!cross_camera_reference_en && candidate_ref->camera_id_ != ctx.camera_id) continue;
          float error = 0.0f;
          int comparisons = 0;
          for (Feature *other : pt->obs_)
          {
            if (other == nullptr || other == candidate_ref || !other->virtual_patch_valid_) continue;
            if (!cross_camera_reference_en && other->camera_id_ != ctx.camera_id) continue;
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
          if (cross_camera_reference_en)
          {
            pt->ref_patch = ref_ftr;
            pt->has_ref_patch_ = true;
          }
          else
          {
            pt->ref_patch_by_camera_[ctx.camera_id] = ref_ftr;
            pt->has_ref_patch_by_camera_[ctx.camera_id] = 1;
          }
        }
      }
    }
    else if (!pt->getCloseViewObs(ctx.new_frame->pos(), ref_ftr, raw_px,
                                  cross_camera_reference_en ? -1 : ctx.camera_id))
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

    if (!buildVirtualFrameRotation(ctx, candidate.point_c_seed, candidate.current_raw_center_px,
                                   result.track.R_vcur_from_ccur_seed, result.track.R_ccur_from_vcur_seed))
    {
      result.rejection = VIRTUAL_REJECT_ROTATION;
      continue;
    }
    result.track.T_vcur_w_seed = composeVirtualPose(result.track.R_vcur_from_ccur_seed, ctx.new_frame->T_f_w_);
    result.track.cur_support.T_v_w_seed = result.track.T_vcur_w_seed;

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

    if (virtual_sparse_patch_en)
    {
      const double materialize_start = omp_get_wtime();
      if (!materializeVirtualReferenceSupport(*ref_ftr, result.ref_materialized))
      {
        result.ref_materialize_time = omp_get_wtime() - materialize_start;
        result.rejection = VIRTUAL_REJECT_REFERENCE_SUPPORT;
        continue;
      }
      result.ref_materialize_time = omp_get_wtime() - materialize_start;
    }

    const double warp_start = omp_get_wtime();
    result.warped_reference.assign(warp_len, 0.0f);
    for (int pyramid_level = 0; pyramid_level < patch_pyrimid_level; ++pyramid_level)
    {
      // [MODIFY] 使用第一次生成的参�?patch，不再每帧重�?
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

    if (!virtual_sparse_patch_en)
    {
      // The affine warp and warped reference depend only on geometry and the
      // immutable reference support. Reject them before constructing the more
      // expensive current virtual support.
      const double build_start = omp_get_wtime();
      const bool supports_ok = buildVirtualSupportPatch(ctx, img, candidate.current_raw_center_px,
                                                        result.track.R_vcur_from_ccur_seed,
                                                        result.track.R_ccur_from_vcur_seed,
                                                        result.track.cur_support);
      result.build_time = omp_get_wtime() - build_start;
      if (!supports_ok)
      {
        result.rejection = VIRTUAL_REJECT_SUPPORT_BUILD;
        continue;
      }
    }

    const double current_core_start = omp_get_wtime();
    vector<float> current_core(patch_size_total);
    const V3D point_vcur = result.track.T_vcur_w_seed * pt->pos_;
    if (point_vcur[2] <= virtual_min_z)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_Z;
      continue;
    }
    const bool current_core_ok = virtual_sparse_patch_en
        ? sampleSparseVirtualCorePatch(ctx, img, result.track.R_ccur_from_vcur_seed,
                                       virtualProject(point_vcur), 1, current_core.data())
        : sampleVirtualCorePatch(result.track.cur_support, virtualProject(point_vcur), 1, current_core.data());
    if (!current_core_ok)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_CORE;
      continue;
    }

    for (int k = 0; k < patch_size_total; ++k)
    {
      const double residual = ref_ftr->inv_expo_time_ * result.warped_reference[k] -
                              state->inv_expo_time[ctx.camera_id] * current_core[k];
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
    virtual_ref_support_materialize_time_ += result.ref_materialize_time;
    if (result.ref_materialized) ++virtual_ref_support_materialized_count_;
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
      case VIRTUAL_REJECT_REFERENCE_SUPPORT:
        ++virtual_ref_support_materialize_fail_count_;
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
    ctx.visual_submap->voxel_points.push_back(candidates[candidate_index].point);
    ctx.visual_submap->reference_features.push_back(candidates[candidate_index].reference);
    ctx.visual_submap->propa_errors.push_back(result.error);
    ctx.visual_submap->search_levels.push_back(result.track.search_level);
    ctx.visual_submap->warp_affines.push_back(result.track.A_cur_ref);
    ctx.visual_submap->errors.push_back(result.error);
    ctx.visual_submap->warp_patch.push_back(std::move(result.warped_reference));
    ctx.visual_submap->inv_expo_list.push_back(result.inverse_reference_exposure);
    ctx.visual_submap->virtual_track_patches.push_back(std::move(result.track));
  }
  virtual_result_collect_time_ = omp_get_wtime() - result_collect_start;

  ctx.total_points = static_cast<int>(ctx.visual_submap->voxel_points.size());
  rejected_virtual_support_oob_ = virtual_track_support_fail_count_ + virtual_track_current_core_fail_count_ + virtual_ref_support_materialize_fail_count_;
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
         ctx.total_points, virtual_candidate_count_, virtual_map_grid_count_, virtual_candidate_count_);
  printf("[ VIO Virtual Candidate ] null=%d normal_uninit=%d proj_fail=%d range_reject=%d close_view_fail=%d ref_missing=%d ref_invalid=%d select_wall=%.6f s\n",
         virtual_candidate_null_count_, virtual_candidate_normal_uninit_count_, virtual_candidate_projection_fail_count_,
         virtual_candidate_range_reject_count_, virtual_candidate_close_view_fail_count_, virtual_candidate_ref_missing_count_,
         virtual_candidate_ref_invalid_count_, virtual_candidate_select_time_);
  printf("[ VIO Virtual Reject ] rotation=%d support_build=%d affine_matrix=%d ref_support=%d warp_ref=%d current_z=%d current_core=%d ncc=%d photometric=%d\n",
         virtual_track_rotation_fail_count_, virtual_track_support_fail_count_, virtual_track_affine_fail_count_,
         virtual_ref_support_materialize_fail_count_, virtual_track_warp_fail_count_,
         virtual_track_current_z_fail_count_, virtual_track_current_core_fail_count_,
         virtual_track_ncc_reject_count_, virtual_track_photometric_reject_count_);
  printf("[ VIO Virtual Warp ] search_level{%s} warp_fail_search{%s} warp_fail_pyramid{%s}\n",
         search_level_text.c_str(), warp_fail_search_text.c_str(), warp_fail_pyramid_text.c_str());
  printf("[ VIO Virtual Timing ] retrieve_wall=%.6f s parallel_wall=%.6f s support_sum=%.6f s affine_sum=%.6f s warp_sum=%.6f s current_core_sum=%.6f s result_collect_wall=%.6f s\n",
         virtual_candidate_select_time_ + virtual_parallel_track_time_ + virtual_result_collect_time_,
         virtual_parallel_track_time_, build_virtual_support_time_, virtual_affine_time_, virtual_warp_time_,
         virtual_current_core_time_, virtual_result_collect_time_);
  printf("[ VIO Virtual Timing Note ] *_sum accumulates per-candidate time across OpenMP workers; wall fields are real elapsed time.\n");
  if (virtual_sparse_patch_en)
    printf("[ VIO Virtual Sparse ] ref_materialized=%d ref_fail=%d ref_materialize_sum=%.6f s\n",
           virtual_ref_support_materialized_count_, virtual_ref_support_materialize_fail_count_,
           virtual_ref_support_materialize_time_);
}

void VIOManager::retrieveFromVisualSparseMap(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                             const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (virtual_fisheye_patch_en)
  {
    retrieveFromVisualSparseMapVirtual(ctx, img, pg, plane_map);
    return;
  }
  if (feat_map.size() <= 0)
  {
    printf("[ VIO Debug ] retrieve skip camera_id=%d reason=empty_feat_map pg=%zu img=%dx%d\n",
           ctx.camera_id, pg.size(), img.cols, img.rows);
    fflush(stdout);
    return;
  }
  printf("[ VIO Debug ] retrieve start camera_id=%d img=%dx%d pg=%zu feat_map=%zu cross_ref=%d normal=%d raycast=%d border=%d grid=%d\n",
         ctx.camera_id, img.cols, img.rows, pg.size(), feat_map.size(), cross_camera_reference_en ? 1 : 0,
         normal_en ? 1 : 0, raycast_en ? 1 : 0, border, ctx.length);
  fflush(stdout);
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  ctx.visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  ctx.sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en)
  {
    for (auto &pair : ctx.warp_map) delete pair.second;
    ctx.warp_map.clear();
  }

  cv::Mat depth_img = cv::Mat::zeros(ctx.height, ctx.width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];
  int debug_depth_samples = 0;

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

    auto iter = ctx.sub_feat_map.find(position);
    if (iter == ctx.sub_feat_map.end()) { ctx.sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(ctx.new_frame->w2f(pt_w));

    if (pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = ctx.cam->world2cam(pt_c);

      if (ctx.cam->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[ctx.width * row + col] = depth;
        ++debug_depth_samples;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  printf("[ VIO Debug ] retrieve depth camera_id=%d depth_samples=%d sub_voxels=%zu elapsed=%.6f\n",
         ctx.camera_id, debug_depth_samples, ctx.sub_feat_map.size(), omp_get_wtime() - ts0);
  fflush(stdout);

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : ctx.sub_feat_map)
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

        V3D norm_vec(ctx.new_frame->T_f_w_.rotationMatrix() * pt->normal_);
        V3D dir(ctx.new_frame->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(ctx.new_frame->w2c(pt->pos_));
        if (ctx.cam->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / ctx.grid_size) * ctx.grid_n_width + static_cast<int>(pc[0] / ctx.grid_size);
          ctx.grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(ctx.new_frame->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= ctx.map_dist[index])
          {
            ctx.map_dist[index] = cur_dist;
            ctx.retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < ctx.length; i++)
    {
      if (ctx.grid_num[i] == TYPE_MAP || ctx.border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : ctx.rays_with_sample_points[i])
      {
        V3D sample_point_w = ctx.new_frame->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = ctx.sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != ctx.sub_feat_map.end()) break;

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

            V3D norm_vec(ctx.new_frame->T_f_w_.rotationMatrix() * pt->normal_);
            V3D dir(ctx.new_frame->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(ctx.new_frame->w2c(pt->pos_));

            if (ctx.cam->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / ctx.grid_size) * ctx.grid_n_width + static_cast<int>(pc[0] / ctx.grid_size);
              ctx.grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(ctx.new_frame->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= ctx.map_dist[index])
              {
                ctx.map_dist[index] = cur_dist;
                ctx.retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) ctx.sub_feat_map[sample_pos] = 0;
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
              ctx.visual_submap->add_from_voxel_map.push_back(plane_center);
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
    ctx.sub_feat_map.erase(key);
  }

  printf("[ VIO Debug ] retrieve map camera_id=%d sub_voxels=%zu deleted_voxels=%zu raycast=%d\n",
         ctx.camera_id, ctx.sub_feat_map.size(), DeleteKeyList.size(), raycast_en ? 1 : 0);
  fflush(stdout);
  int debug_grid_candidates = 0;
  int debug_ref_invalid = 0;
  int debug_cross_refs = 0;
  int debug_warp_attempts = 0;
  int debug_warp_logs = 0;
  int debug_affine_bad = 0;
  int debug_accepted = 0;

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < ctx.length; i++)
  {
    if (ctx.grid_num[i] == TYPE_MAP)
    {
      ++debug_grid_candidates;
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = ctx.retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(ctx.new_frame->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      V3D pt_cam(ctx.new_frame->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[ctx.width * (v + int(pc[1])) + u + int(pc[0])];

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
        pt->ensureCameraCount(numCameras());
        float phtometric_errors_min = std::numeric_limits<float>::max();
        ref_ftr = pt->referencePatch(ctx.camera_id, cross_camera_reference_en);
        if (ref_ftr == nullptr)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            if (ref_patch_temp == nullptr) continue;
            if (!cross_camera_reference_en && ref_patch_temp->camera_id_ != ctx.camera_id) continue;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if (*itm == nullptr || *itm == ref_patch_temp) continue;
              if (!cross_camera_reference_en && (*itm)->camera_id_ != ctx.camera_id) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            if (count > 0) phtometric_errors /= count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          if (ref_ftr != nullptr)
          {
            if (cross_camera_reference_en)
            {
              pt->ref_patch = ref_ftr;
              pt->has_ref_patch_ = true;
            }
            else
            {
              pt->ref_patch_by_camera_[ctx.camera_id] = ref_ftr;
              pt->has_ref_patch_by_camera_[ctx.camera_id] = 1;
            }
          }
        }
      }
      else
      {
        if (!pt->getCloseViewObs(ctx.new_frame->pos(), ref_ftr, pc,
                                 cross_camera_reference_en ? -1 : ctx.camera_id)) continue;
      }
      if (ref_ftr == nullptr || ref_ftr->camera_id_ < 0 || ref_ftr->camera_id_ >= numCameras())
      {
        ++debug_ref_invalid;
        continue;
      }
      const PerCameraData &ref_ctx = cameras_[ref_ftr->camera_id_];
      if (ref_ftr->camera_id_ != ctx.camera_id) ++debug_cross_refs;

      if (normal_en)
      {
        V3D norm_vec = (ref_ftr->T_f_w_.rotationMatrix() * pt->normal_).normalized();
        
        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();
        
        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = ctx.new_frame->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(ref_ctx, ctx, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = ctx.warp_map.find(ref_ftr);
        if (iter_warp != ctx.warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(ref_ctx, ctx, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(),
                              ctx.new_frame->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          ctx.warp_map[ref_ftr] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      const double debug_affine_det = A_cur_ref_zero.determinant();
      const bool debug_affine_ok = A_cur_ref_zero.array().isFinite().all() && std::isfinite(debug_affine_det) && std::fabs(debug_affine_det) > 1e-9;
      if (!debug_affine_ok)
      {
        ++debug_affine_bad;
        if (debug_affine_bad <= 5)
        {
          printf("[ VIO Debug ] retrieve affine suspicious camera_id=%d ref_camera_id=%d det=%.6e search_level=%d finite=%d\n",
                 ctx.camera_id, ref_ftr->camera_id_, debug_affine_det, search_level, debug_affine_ok ? 1 : 0);
          fflush(stdout);
        }
      }
      if (debug_warp_logs < 8)
      {
        printf("[ VIO Debug ] retrieve warp camera_id=%d ref_camera_id=%d search_level=%d det=%.6e finite=%d ref_px=(%.2f,%.2f) ref_img=%dx%d ref_level=%d\n",
               ctx.camera_id, ref_ftr->camera_id_, search_level, debug_affine_det, debug_affine_ok ? 1 : 0,
               ref_ftr->px_[0], ref_ftr->px_[1], ref_ftr->img_.cols, ref_ftr->img_.rows, ref_ftr->level_);
        fflush(stdout);
        ++debug_warp_logs;
      }
      ++debug_warp_attempts;

      bool warp_ok = true;
      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        if (!warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level,
                        patch_size_half, patch_wrap.data()))
        {
          warp_ok = false;
          break;
        }
      }
      if (!warp_ok) continue;

      getImagePatch(ctx, img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time[ctx.camera_id] * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time[ctx.camera_id] * patch_buffer[ind]);
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

      ctx.visual_submap->voxel_points.push_back(pt);
      ctx.visual_submap->reference_features.push_back(ref_ftr);
      ctx.visual_submap->propa_errors.push_back(error);
      ctx.visual_submap->search_levels.push_back(search_level);
      ctx.visual_submap->warp_affines.push_back(A_cur_ref_zero);
      ctx.visual_submap->errors.push_back(error);
      ctx.visual_submap->warp_patch.push_back(patch_wrap);
      ctx.visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      ++debug_accepted;

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  ctx.total_points = ctx.visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] camera_id=%d retrieve %d points from visual sparse map\n", ctx.camera_id, ctx.total_points);
  printf("[ VIO Debug ] retrieve summary camera_id=%d grid_candidates=%d ref_invalid=%d cross_refs=%d warp_attempts=%d affine_bad=%d accepted=%d elapsed=%.6f\n",
         ctx.camera_id, debug_grid_candidates, debug_ref_invalid, debug_cross_refs, debug_warp_attempts, debug_affine_bad, debug_accepted,
         omp_get_wtime() - ts0);
  fflush(stdout);
}

bool VIOManager::interpolateReferenceFeature(const Feature &reference, const V2D &px, float &value) const
{
  if (!px.array().isFinite().all() || reference.img_.empty()) return false;

  if (reference.virtual_patch_valid_)
    return interpolateStoredVirtualImage(reference.img_, static_cast<float>(px[0]), static_cast<float>(px[1]), value);

  if (reference.img_.type() != CV_8UC1 || px[0] < 0.0 || px[1] < 0.0 ||
      px[0] >= reference.img_.cols - 1 || px[1] >= reference.img_.rows - 1)
    return false;

  value = static_cast<float>(vk::interpolateMat_8u(reference.img_, px[0], px[1]));
  return std::isfinite(value);
}

bool VIOManager::computeWarpedReferenceGradient(const Feature &reference, const Matrix2d &A_cur_ref,
                                                int search_level, int level, int patch_index, V2D &gradient) const
{
  if (patch_index < 0 || patch_index >= patch_size_total || search_level < 0 || level < 0 ||
      !A_cur_ref.array().isFinite().all() || std::fabs(A_cur_ref.determinant()) <= 1e-9)
    return false;

  const int pyramid_level = level + search_level;
  if (pyramid_level >= 30) return false;
  const double scale = static_cast<double>(1 << pyramid_level);
  const Matrix2d A_ref_cur = A_cur_ref.inverse();
  if (!A_ref_cur.array().isFinite().all()) return false;

  const V2D center = reference.virtual_patch_valid_
                         ? V2D(virtual_support_radius, virtual_support_radius)
                         : reference.px_;
  const V2D patch_offset = core_patch_offsets_[patch_index].cast<double>() * scale;
  const V2D reference_px = center + A_ref_cur * patch_offset;
  const V2D reference_du = A_ref_cur * V2D(scale, 0.0);
  const V2D reference_dv = A_ref_cur * V2D(0.0, scale);

  float left = 0.0f, right = 0.0f, up = 0.0f, down = 0.0f;
  if (!interpolateReferenceFeature(reference, reference_px - reference_du, left) ||
      !interpolateReferenceFeature(reference, reference_px + reference_du, right) ||
      !interpolateReferenceFeature(reference, reference_px - reference_dv, up) ||
      !interpolateReferenceFeature(reference, reference_px + reference_dv, down))
    return false;

  gradient << 0.5 * (right - left) / scale, 0.5 * (down - up) / scale;
  return gradient.array().isFinite().all();
}

void VIOManager::buildFixedTemplateGradientCache(int level)
{
  for (PerCameraData &ctx : cameras_)
  {
    FixedTemplateGradientCache &cache = ctx.fixed_template_cache;
    cache.reset(level, ctx.total_points, patch_size_total);
    if (ctx.total_points == 0 || ctx.visual_submap == nullptr) continue;

    const size_t point_count = static_cast<size_t>(ctx.total_points);
    if (ctx.visual_submap->reference_features.size() != point_count ||
        ctx.visual_submap->warp_affines.size() != point_count ||
        ctx.visual_submap->search_levels.size() != point_count ||
        ctx.visual_submap->inv_expo_list.size() != point_count)
      throw std::runtime_error("fixed-template cache input vectors are not aligned for camera_id=" +
                               std::to_string(ctx.camera_id));

    for (int point_index = 0; point_index < ctx.total_points; ++point_index)
    {
      Feature *reference = ctx.visual_submap->reference_features[point_index];
      if (reference == nullptr) continue;
      const double reference_exposure = ctx.visual_submap->inv_expo_list[point_index];
      if (!std::isfinite(reference_exposure)) continue;

      for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
      {
        V2D gradient;
        if (!computeWarpedReferenceGradient(*reference, ctx.visual_submap->warp_affines[point_index],
                                            ctx.visual_submap->search_levels[point_index], level,
                                            patch_index, gradient))
          continue;
        const int row = point_index * patch_size_total + patch_index;
        cache.photometric_gradients.row(row) = (reference_exposure * gradient).transpose();
        cache.valid[row] = 1;
      }
    }
  }
}

void VIOManager::computeJacobianAndUpdateEKF()
{
  compute_jacobian_time = update_ekf_time = 0.0;
  int total_observations = 0;
  for (const PerCameraData &ctx : cameras_) total_observations += ctx.total_points;
  if (total_observations == 0) return;
  G = Eigen::MatrixXd::Zero(state->stateDim(), state->stateDim());
  const bool online_extrinsic_active = online_extrinsic_en &&
      frame_count >= online_extrinsic_start_frame &&
      total_observations >= online_extrinsic_min_tracks;
  const bool allow_extrinsic_rotation = online_extrinsic_active && online_extrinsic_rot_en;
  const bool allow_extrinsic_translation = online_extrinsic_active && online_extrinsic_trans_en;

  for (int level = patch_pyrimid_level - 1; level >= 0; --level)
  {
    StatesGroup old_state = *state;
    if (inverse_composition_en) buildFixedTemplateGradientCache(level);
    double last_error = std::numeric_limits<double>::max();
    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
      const double linearize_start = omp_get_wtime();
      const int state_dim = state->stateDim();
      Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_dim, state_dim);
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(state_dim);
      double error = 0.0;
      int measurement_count = 0;

      for (PerCameraData &ctx : cameras_)
      {
        if (ctx.total_points == 0 || ctx.visual_submap == nullptr || ctx.new_frame == nullptr) continue;
        const cv::Mat &img = ctx.new_frame->img_;
        const M3D Rwi = state->rot_end;
        const V3D Pwi = state->pos_end;
        const bool estimate_extrinsic = isOnlineExtrinsicEnabledForCamera(ctx.camera_id) &&
                                        (allow_extrinsic_rotation || allow_extrinsic_translation);
        ctx.Rcw = ctx.Rci * Rwi.transpose();
        ctx.Pcw = -ctx.Rci * Rwi.transpose() * Pwi + ctx.Pci;
        ctx.Jdp_dt = ctx.Rci * Rwi.transpose();
        const double current_exposure = state->inv_expo_time[ctx.camera_id];

        for (int point_index = 0; point_index < ctx.total_points; ++point_index)
        {
          VisualPoint *point = ctx.visual_submap->voxel_points[point_index];
          if (point == nullptr) continue;
          const int search_level = ctx.visual_submap->search_levels[point_index];
          const int pyramid_level = level + search_level;
          const int scale = 1 << pyramid_level;
          const double inv_scale = 1.0 / scale;
          const std::vector<float> &reference_patch = ctx.visual_submap->warp_patch[point_index];
          const double reference_exposure = ctx.visual_submap->inv_expo_list[point_index];
          double patch_error = 0.0;

          if (virtual_fisheye_patch_en)
          {
            if (point_index >= static_cast<int>(ctx.visual_submap->virtual_track_patches.size())) continue;
            const VirtualTrackPatch &track = ctx.visual_submap->virtual_track_patches[point_index];
            M3D Jpc_dRcl = M3D::Zero();
            if (estimate_extrinsic)
            {
              const V3D point_i = Rwi.transpose() * (point->pos_ - Pwi);
              const V3D point_l = Rli * point_i + Pli;
              M3D point_l_hat;
              point_l_hat << SKEW_SYM_MATRX(point_l);
              Jpc_dRcl = -ctx.Rcl * point_l_hat;
            }
            const V3D point_c = ctx.Rcw * point->pos_ + ctx.Pcw;

            if (virtual_s2_optimize_en)
            {
              if (!point_c.array().isFinite().all()) continue;
              const double point_c_norm = point_c.norm();
              if (!std::isfinite(point_c_norm) || point_c_norm <= kS2Eps) continue;
              M3D point_c_hat;
              point_c_hat << SKEW_SYM_MATRX(point_c);
              for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
              {
                const V2F offset_f = core_patch_offsets_[patch_index] * static_cast<float>(scale);
                const V2D offset = offset_f.cast<double>();
                float current_value = 0.0f;
                MD(1, 3) J_photo_center;
                if (!linearizeVirtualS2Sample(ctx, img, point_c, track, offset, scale,
                                              current_exposure, current_value, J_photo_center))
                  continue;

                const MD(1, 3) Jdphi = J_photo_center * point_c_hat;
                const MD(1, 3) Jdp = -J_photo_center;
                const MD(1, 3) JdR = Jdphi * ctx.Jdphi_dR + Jdp * ctx.Jdp_dR;
                const MD(1, 3) Jdt = Jdp * ctx.Jdp_dt;
                const double residual = current_exposure * current_value -
                                        reference_exposure * reference_patch[patch_size_total * level + patch_index];
                Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(state_dim);
                jacobian.segment<3>(0) = JdR.transpose();
                jacobian.segment<3>(3) = Jdt.transpose();
                if (exposure_estimate_en) jacobian[state->exposureIndex(ctx.camera_id)] = current_value;
                if (estimate_extrinsic)
                {
                  if (allow_extrinsic_rotation)
                    jacobian.segment<3>(state->extrinsicRotIndex(ctx.camera_id)) =
                        (J_photo_center * Jpc_dRcl).transpose();
                  if (allow_extrinsic_translation)
                    jacobian.segment<3>(state->extrinsicTransIndex(ctx.camera_id)) = J_photo_center.transpose();
                }
                hessian.noalias() += jacobian * jacobian.transpose();
                gradient.noalias() += jacobian * residual;
                patch_error += residual * residual;
                ++measurement_count;
              }
            }
            else
            {
              const V3D point_v = track.R_vcur_from_ccur_seed * point_c;
              if (point_v[2] <= virtual_min_z) continue;
              const V2D center = virtualProject(point_v);
              MD(2, 3) Jdpi;
              computeVirtualProjectionJacobian(point_v, Jdpi);
              M3D point_c_hat;
              point_c_hat << SKEW_SYM_MATRX(point_c);
              for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
              {
                const V2F offset = core_patch_offsets_[patch_index] * static_cast<float>(scale);
                const int cache_row = point_index * patch_size_total + patch_index;
                float current_value = 0.0f;
                MD(1, 2) Jimg;
                if (inverse_composition_en)
                {
                  const bool current_value_ok = virtual_sparse_patch_en
                      ? sampleSparseVirtualValue(ctx, img, track.R_ccur_from_vcur_seed,
                                                 center + offset.cast<double>(), current_value)
                      : interpolateVirtualFloat(track.cur_support.values, track.cur_support.valid_mask,
                                                center[0] + offset[0], center[1] + offset[1], current_value);
                  if (cache_row >= static_cast<int>(ctx.fixed_template_cache.valid.size()) ||
                      !ctx.fixed_template_cache.valid[cache_row] || !current_value_ok)
                    continue;
                  Jimg = ctx.fixed_template_cache.photometric_gradients.row(cache_row);
                }
                else
                {
                  V2D image_gradient;
                  const bool current_gradient_ok = virtual_sparse_patch_en
                      ? sampleSparseVirtualValueAndGradient(ctx, img, track.R_ccur_from_vcur_seed,
                                                            center + offset.cast<double>(), scale,
                                                            current_value, image_gradient)
                      : sampleVirtualValueAndGradient(track.cur_support, center + offset.cast<double>(), scale,
                                                      current_value, image_gradient);
                  if (!current_gradient_ok) continue;
                  Jimg << image_gradient[0], image_gradient[1];
                  Jimg *= current_exposure * inv_scale;
                }
                const MD(1, 3) Jimg_Jpi_R = Jimg * Jdpi * track.R_vcur_from_ccur_seed;
                const MD(1, 3) Jdphi = Jimg_Jpi_R * point_c_hat;
                const MD(1, 3) Jdp = -Jimg_Jpi_R;
                const MD(1, 3) JdR = Jdphi * ctx.Jdphi_dR + Jdp * ctx.Jdp_dR;
                const MD(1, 3) Jdt = Jdp * ctx.Jdp_dt;
                const double residual = current_exposure * current_value -
                                        reference_exposure * reference_patch[patch_size_total * level + patch_index];
                Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(state_dim);
                jacobian.segment<3>(0) = JdR.transpose();
                jacobian.segment<3>(3) = Jdt.transpose();
                if (exposure_estimate_en) jacobian[state->exposureIndex(ctx.camera_id)] = current_value;
                if (estimate_extrinsic)
                {
                  if (allow_extrinsic_rotation)
                    jacobian.segment<3>(state->extrinsicRotIndex(ctx.camera_id)) =
                        (Jimg_Jpi_R * Jpc_dRcl).transpose();
                  if (allow_extrinsic_translation)
                    jacobian.segment<3>(state->extrinsicTransIndex(ctx.camera_id)) = Jimg_Jpi_R.transpose();
                }
                hessian.noalias() += jacobian * jacobian.transpose();
                gradient.noalias() += jacobian * residual;
                patch_error += residual * residual;
                ++measurement_count;
              }
            }
          }
          else
          {
            M3D Jpc_dRcl = M3D::Zero();
            if (estimate_extrinsic)
            {
              const V3D point_i = Rwi.transpose() * (point->pos_ - Pwi);
              const V3D point_l = Rli * point_i + Pli;
              M3D point_l_hat;
              point_l_hat << SKEW_SYM_MATRX(point_l);
              Jpc_dRcl = -ctx.Rcl * point_l_hat;
            }
            const V3D point_c = ctx.Rcw * point->pos_ + ctx.Pcw;
            const V2D pixel = ctx.cam->world2cam(point_c);
            const int required_border = (patch_size_half + 1) * scale + 1;
            if (!pixel.array().isFinite().all() || !ctx.cam->isInFrame(pixel.cast<int>(), required_border)) continue;
            MD(2, 3) Jdpi;
            computeProjectionJacobian(ctx, point_c, Jdpi);
            M3D point_hat;
            point_hat << SKEW_SYM_MATRX(point_c);
            const int u_i = static_cast<int>(std::floor(pixel[0] / scale)) * scale;
            const int v_i = static_cast<int>(std::floor(pixel[1] / scale)) * scale;
            const double du = (pixel[0] - u_i) / scale;
            const double dv = (pixel[1] - v_i) / scale;
            const double w_tl = (1.0 - du) * (1.0 - dv);
            const double w_tr = du * (1.0 - dv);
            const double w_bl = (1.0 - du) * dv;
            const double w_br = du * dv;
            for (int x = 0; x < patch_size; ++x)
            {
              const uint8_t *img_ptr = img.data +
                  (v_i + x * scale - patch_size_half * scale) * ctx.width + u_i - patch_size_half * scale;
              for (int y = 0; y < patch_size; ++y, img_ptr += scale)
              {
                const int patch_index = x * patch_size + y;
                const int cache_row = point_index * patch_size_total + patch_index;
                const double current_value = w_tl * img_ptr[0] + w_tr * img_ptr[scale] +
                                             w_bl * img_ptr[ctx.width * scale] + w_br * img_ptr[ctx.width * scale + scale];
                MD(1, 2) Jimg;
                if (inverse_composition_en)
                {
                  if (cache_row >= static_cast<int>(ctx.fixed_template_cache.valid.size()) ||
                      !ctx.fixed_template_cache.valid[cache_row])
                    continue;
                  Jimg = ctx.fixed_template_cache.photometric_gradients.row(cache_row);
                }
                else
                {
                  const double grad_u = 0.5 *
                      ((w_tl * img_ptr[scale] + w_tr * img_ptr[2 * scale] + w_bl * img_ptr[ctx.width * scale + scale] +
                        w_br * img_ptr[ctx.width * scale + 2 * scale]) -
                       (w_tl * img_ptr[-scale] + w_tr * img_ptr[0] + w_bl * img_ptr[ctx.width * scale - scale] +
                        w_br * img_ptr[ctx.width * scale]));
                  const double grad_v = 0.5 *
                      ((w_tl * img_ptr[ctx.width * scale] + w_tr * img_ptr[ctx.width * scale + scale] +
                        w_bl * img_ptr[2 * ctx.width * scale] + w_br * img_ptr[2 * ctx.width * scale + scale]) -
                       (w_tl * img_ptr[-ctx.width * scale] + w_tr * img_ptr[-ctx.width * scale + scale] +
                        w_bl * img_ptr[0] + w_br * img_ptr[scale]));
                  Jimg << grad_u, grad_v;
                  Jimg *= current_exposure * inv_scale;
                }
                const MD(1, 3) Jdphi = Jimg * Jdpi * point_hat;
                const MD(1, 3) Jdp = -Jimg * Jdpi;
                const MD(1, 3) JdR = Jdphi * ctx.Jdphi_dR + Jdp * ctx.Jdp_dR;
                const MD(1, 3) Jdt = Jdp * ctx.Jdp_dt;
                const double residual = current_exposure * current_value -
                                        reference_exposure * reference_patch[patch_size_total * level + patch_index];
                Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(state_dim);
                jacobian.segment<3>(0) = JdR.transpose();
                jacobian.segment<3>(3) = Jdt.transpose();
                if (exposure_estimate_en) jacobian[state->exposureIndex(ctx.camera_id)] = current_value;
                if (estimate_extrinsic)
                {
                  const MD(1, 3) Jimg_Jpi = Jimg * Jdpi;
                  if (allow_extrinsic_rotation)
                    jacobian.segment<3>(state->extrinsicRotIndex(ctx.camera_id)) =
                        (Jimg_Jpi * Jpc_dRcl).transpose();
                  if (allow_extrinsic_translation)
                    jacobian.segment<3>(state->extrinsicTransIndex(ctx.camera_id)) = Jimg_Jpi.transpose();
                }
                hessian.noalias() += jacobian * jacobian.transpose();
                gradient.noalias() += jacobian * residual;
                patch_error += residual * residual;
                ++measurement_count;
              }
            }
          }
          ctx.visual_submap->errors[point_index] = patch_error;
          error += patch_error;
        }
      }

      if (online_extrinsic_active && online_extrinsic_prior_factor_en)
        applyOnlineExtrinsicPriors(hessian, gradient, allow_extrinsic_rotation, allow_extrinsic_translation);
      compute_jacobian_time += omp_get_wtime() - linearize_start;
      if (measurement_count == 0) break;
      error /= measurement_count;
      if (error > last_error)
      {
        *state = old_state;
        syncCameraExtrinsicsFromState(*state);
        break;
      }

      old_state = *state;
      last_error = error;
      const double update_start = omp_get_wtime();
      const Eigen::MatrixXd K1 = (hessian + (state->cov / img_point_cov).inverse()).inverse();
      const Eigen::VectorXd prior_delta = *state_propagat - *state;
      G = K1 * hessian;
      Eigen::VectorXd solution = -K1 * gradient + prior_delta - G * prior_delta;
      limitOnlineExtrinsicUpdate(solution, allow_extrinsic_rotation, allow_extrinsic_translation);
      *state += solution;
      syncCameraExtrinsicsFromState(*state);
      update_ekf_time += omp_get_wtime() - update_start;
      if (solution.segment<3>(0).norm() * 57.3 < 0.001 && solution.segment<3>(3).norm() * 100.0 < 0.001) break;
    }
  }
  state->cov -= G * state->cov;
  if (online_extrinsic_active && frame_count % 20 == 0)
  {
    for (const PerCameraData &ctx : cameras_)
    {
      if (!isOnlineExtrinsicEnabledForCamera(ctx.camera_id)) continue;
      const M3D dR_cl = state->Rcl_prior[ctx.camera_id].transpose() * state->Rcl[ctx.camera_id];
      const V3D dR_deg = Log(dR_cl) * kRadiansToDegrees;
      const V3D dP = state->Pcl[ctx.camera_id] - state->Pcl_prior[ctx.camera_id];
      printf("[ VIO Extrinsic ] frame=%d camera_id=%d dR_deg=(%.6f, %.6f, %.6f) dP_m=(%.6f, %.6f, %.6f) Pcl=(%.6f, %.6f, %.6f)\n",
             frame_count, ctx.camera_id, dR_deg[0], dR_deg[1], dR_deg[2], dP[0], dP[1], dP[2],
             state->Pcl[ctx.camera_id][0], state->Pcl[ctx.camera_id][1], state->Pcl[ctx.camera_id][2]);
    }
  }
  for (PerCameraData &ctx : cameras_) updateFrameState(ctx, *state);
}

void VIOManager::updateStateVirtualS2(cv::Mat img, int level)
{
  if (!virtual_fisheye_patch_en || !virtual_s2_optimize_en) return;
  if (level < 0 || level >= patch_pyrimid_level) return;

  int total_observations = 0;
  for (const PerCameraData &ctx : cameras_) total_observations += ctx.total_points;
  if (total_observations == 0) return;

  G = Eigen::MatrixXd::Zero(state->stateDim(), state->stateDim());
  StatesGroup old_state = *state;
  double last_error = std::numeric_limits<double>::max();
  const bool online_extrinsic_active = online_extrinsic_en &&
      frame_count >= online_extrinsic_start_frame &&
      total_observations >= online_extrinsic_min_tracks;
  const bool allow_extrinsic_rotation = online_extrinsic_active && online_extrinsic_rot_en;
  const bool allow_extrinsic_translation = online_extrinsic_active && online_extrinsic_trans_en;

  for (int iteration = 0; iteration < max_iterations; ++iteration)
  {
    const double linearize_start = omp_get_wtime();
    const int state_dim = state->stateDim();
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(state_dim);
    double error = 0.0;
    int measurement_count = 0;

    for (PerCameraData &ctx : cameras_)
    {
      if (ctx.total_points == 0 || ctx.visual_submap == nullptr) continue;
      const cv::Mat &raw_img = (ctx.new_frame != nullptr) ? ctx.new_frame->img_ : img;
      if (raw_img.empty()) continue;
      const M3D Rwi = state->rot_end;
      const V3D Pwi = state->pos_end;
      const bool estimate_extrinsic = isOnlineExtrinsicEnabledForCamera(ctx.camera_id) &&
                                      (allow_extrinsic_rotation || allow_extrinsic_translation);
      ctx.Rcw = ctx.Rci * Rwi.transpose();
      ctx.Pcw = -ctx.Rci * Rwi.transpose() * Pwi + ctx.Pci;
      ctx.Jdp_dt = ctx.Rci * Rwi.transpose();
      const double current_exposure = state->inv_expo_time[ctx.camera_id];

      for (int point_index = 0; point_index < ctx.total_points; ++point_index)
      {
        VisualPoint *point = ctx.visual_submap->voxel_points[point_index];
        if (point == nullptr || point_index >= static_cast<int>(ctx.visual_submap->virtual_track_patches.size())) continue;
        const int search_level = ctx.visual_submap->search_levels[point_index];
        const int pyramid_level = level + search_level;
        const int scale = 1 << pyramid_level;
        const std::vector<float> &reference_patch = ctx.visual_submap->warp_patch[point_index];
        const double reference_exposure = ctx.visual_submap->inv_expo_list[point_index];
        const VirtualTrackPatch &track = ctx.visual_submap->virtual_track_patches[point_index];
        const V3D point_c = ctx.Rcw * point->pos_ + ctx.Pcw;
        if (!point_c.array().isFinite().all()) continue;
        const double point_c_norm = point_c.norm();
        if (!std::isfinite(point_c_norm) || point_c_norm <= kS2Eps) continue;

        M3D Jpc_dRcl = M3D::Zero();
        if (estimate_extrinsic)
        {
          const V3D point_i = Rwi.transpose() * (point->pos_ - Pwi);
          const V3D point_l = Rli * point_i + Pli;
          M3D point_l_hat;
          point_l_hat << SKEW_SYM_MATRX(point_l);
          Jpc_dRcl = -ctx.Rcl * point_l_hat;
        }

        M3D point_c_hat;
        point_c_hat << SKEW_SYM_MATRX(point_c);
        double patch_error = 0.0;
        for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
        {
          const V2D offset = (core_patch_offsets_[patch_index] * static_cast<float>(scale)).cast<double>();
          float current_value = 0.0f;
          MD(1, 3) J_photo_center;
          if (!linearizeVirtualS2Sample(ctx, raw_img, point_c, track, offset, scale,
                                        current_exposure, current_value, J_photo_center))
            continue;

          const MD(1, 3) Jdphi = J_photo_center * point_c_hat;
          const MD(1, 3) Jdp = -J_photo_center;
          const MD(1, 3) JdR = Jdphi * ctx.Jdphi_dR + Jdp * ctx.Jdp_dR;
          const MD(1, 3) Jdt = Jdp * ctx.Jdp_dt;
          const double residual = current_exposure * current_value -
                                  reference_exposure * reference_patch[patch_size_total * level + patch_index];

          Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(state_dim);
          jacobian.segment<3>(0) = JdR.transpose();
          jacobian.segment<3>(3) = Jdt.transpose();
          if (exposure_estimate_en) jacobian[state->exposureIndex(ctx.camera_id)] = current_value;
          if (estimate_extrinsic)
          {
            if (allow_extrinsic_rotation)
              jacobian.segment<3>(state->extrinsicRotIndex(ctx.camera_id)) =
                  (J_photo_center * Jpc_dRcl).transpose();
            if (allow_extrinsic_translation)
              jacobian.segment<3>(state->extrinsicTransIndex(ctx.camera_id)) = J_photo_center.transpose();
          }
          hessian.noalias() += jacobian * jacobian.transpose();
          gradient.noalias() += jacobian * residual;
          patch_error += residual * residual;
          ++measurement_count;
        }
        ctx.visual_submap->errors[point_index] = patch_error;
        error += patch_error;
      }
    }

    if (online_extrinsic_active && online_extrinsic_prior_factor_en)
      applyOnlineExtrinsicPriors(hessian, gradient, allow_extrinsic_rotation, allow_extrinsic_translation);
    compute_jacobian_time += omp_get_wtime() - linearize_start;
    if (measurement_count == 0) return;
    error /= measurement_count;
    if (error > last_error)
    {
      *state = old_state;
      syncCameraExtrinsicsFromState(*state);
      break;
    }

    old_state = *state;
    last_error = error;
    const double update_start = omp_get_wtime();
    const Eigen::MatrixXd K1 = (hessian + (state->cov / img_point_cov).inverse()).inverse();
    const Eigen::VectorXd prior_delta = *state_propagat - *state;
    G = K1 * hessian;
    Eigen::VectorXd solution = -K1 * gradient + prior_delta - G * prior_delta;
    limitOnlineExtrinsicUpdate(solution, allow_extrinsic_rotation, allow_extrinsic_translation);
    *state += solution;
    syncCameraExtrinsicsFromState(*state);
    update_ekf_time += omp_get_wtime() - update_start;
    if (solution.segment<3>(0).norm() * 57.3 < 0.001 && solution.segment<3>(3).norm() * 100.0 < 0.001) break;
  }
  state->cov -= G * state->cov;
  for (PerCameraData &ctx : cameras_) updateFrameState(ctx, *state);
}

void VIOManager::generateVisualMapPointsVirtual(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg)
{
  if (pg.size() <= 10) return;

  auto consider_candidate = [&](const pointWithVar &candidate, int source_type, int source_index) {
    if (candidate.normal == V3D::Zero()) return;
    V2D raw_px;
    if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(candidate.point_w), border, raw_px)) return;
    const int grid_col = static_cast<int>(raw_px[0] / ctx.grid_size);
    const int grid_row = static_cast<int>(raw_px[1] / ctx.grid_size);
    if (grid_col < 0 || grid_col >= ctx.grid_n_width || grid_row < 0 || grid_row >= ctx.grid_n_height) return;
    const int index = grid_row * ctx.grid_n_width + grid_col;
    if (ctx.grid_num[index] == TYPE_MAP) return;

    const float score = vk::shiTomasiScore(img, raw_px[0], raw_px[1]);
    if (!std::isfinite(score)) return;
    if (score > ctx.scan_value[index])
    {
      ctx.scan_value[index] = score;
      ctx.append_voxel_points[index] = candidate;
      ctx.append_voxel_source_type[index] = source_type;
      ctx.append_voxel_source_index[index] = source_index;
      ctx.grid_num[index] = TYPE_POINTCLOUD;
    }
  };

  for (int i = 0; i < static_cast<int>(pg.size()); ++i) consider_candidate(pg[i], SOURCE_PG, i);
  for (const auto &candidate : ctx.visual_submap->add_from_voxel_map)
    consider_candidate(candidate, SOURCE_RAYCAST_PLANE, -1);

  for (int i = 0; i < ctx.length; ++i)
  {
    if (ctx.grid_num[i] != TYPE_POINTCLOUD) continue;
    const pointWithVar &pt_var = ctx.append_voxel_points[i];
    V2D raw_px;
    if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt_var.point_w), 1, raw_px)) continue;

    std::vector<float> patch(patch_size_total);
    cv::Mat virtual_support_img;
    cv::Point virtual_source_origin;
    SE3<double> T_v_w;
    M3D R_v_from_c, R_c_from_v;
    if (!createVirtualFeaturePatch(ctx, img, ctx.new_frame->T_f_w_, pt_var.point_w, patch.data(),
                                   virtual_support_img, virtual_source_origin, T_v_w, R_v_from_c, R_c_from_v))
    {
      ++rejected_virtual_support_oob_;
      continue;
    }

    PendingNewPointObservation pending;
    pending.camera_id = ctx.camera_id;
    pending.source_type = ctx.append_voxel_source_type[i];
    pending.source_index = ctx.append_voxel_source_index[i];
    pending.pt_var = pt_var;
    pending.px = raw_px;
    pending.bearing = ctx.cam->cam2world(raw_px);
    pending.patch = std::move(patch);
    if (ref_patch_dump_en)
      maybeInitializeRefPatchDumpProbe(ctx, pt_var.point_w, pt_var.normal, raw_px, pending.bearing, pending.patch.data());
    pending.img = virtual_support_img;
    pending.virtual_source_origin = virtual_source_origin;
    pending.T_f_w = ctx.new_frame->T_f_w_;
    pending.T_v_w = T_v_w;
    pending.R_v_from_c = R_v_from_c;
    pending.R_c_from_v = R_c_from_v;
    pending.virtual_patch_valid = true;
    pending.inv_expo_time = state->inv_expo_time[ctx.camera_id];
    ctx.pending_new_points.push_back(std::move(pending));
  }
  printf("[ VIO Virtual ] camera_id=%d selected %zu pending observations\n", ctx.camera_id, ctx.pending_new_points.size());
}

void VIOManager::generateVisualMapPoints(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg)
{
  if (virtual_fisheye_patch_en)
  {
    generateVisualMapPointsVirtual(ctx, img, pg);
    return;
  }
  if (pg.size() <= 10) return;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    V3D pt = pg[i].point_w;
    V2D pc(ctx.new_frame->w2c(pt));

    if (ctx.cam->isInFrame(pc.cast<int>(), border))
    {
      int index = static_cast<int>(pc[1] / ctx.grid_size) * ctx.grid_n_width + static_cast<int>(pc[0] / ctx.grid_size);

      if (ctx.grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > ctx.scan_value[index])
        {
          ctx.scan_value[index] = cur_value;
          ctx.append_voxel_points[index] = pg[i];
          ctx.append_voxel_source_type[index] = SOURCE_PG;
          ctx.append_voxel_source_index[index] = i;
          ctx.grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < ctx.visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = ctx.visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(ctx.new_frame->w2c(pt));

    if (ctx.cam->isInFrame(pc.cast<int>(), border))
    {
      int index = static_cast<int>(pc[1] / ctx.grid_size) * ctx.grid_n_width + static_cast<int>(pc[0] / ctx.grid_size);

      if (ctx.grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > ctx.scan_value[index])
        {
          ctx.scan_value[index] = cur_value;
          ctx.append_voxel_points[index] = ctx.visual_submap->add_from_voxel_map[j];
          ctx.append_voxel_source_type[index] = SOURCE_RAYCAST_PLANE;
          ctx.append_voxel_source_index[index] = -1;
          ctx.grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  for (int i = 0; i < ctx.length; i++)
  {
    if (ctx.grid_num[i] == TYPE_POINTCLOUD)
    {
      pointWithVar pt_var = ctx.append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(ctx.new_frame->T_f_w_.rotationMatrix() * pt_var.normal);
      V3D dir(ctx.new_frame->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(ctx.new_frame->w2c(pt));

      PendingNewPointObservation pending;
      pending.camera_id = ctx.camera_id;
      pending.source_type = ctx.append_voxel_source_type[i];
      pending.source_index = ctx.append_voxel_source_index[i];
      pending.pt_var = pt_var;
      pending.px = pc;
      pending.bearing = ctx.cam->cam2world(pc);
      pending.patch.resize(patch_size_total);
      getImagePatch(ctx, img, pc, pending.patch.data(), 0);
      pending.img = img;
      pending.T_f_w = ctx.new_frame->T_f_w_;
      pending.inv_expo_time = state->inv_expo_time[ctx.camera_id];
      ctx.pending_new_points.push_back(std::move(pending));
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] camera_id=%d selected %zu pending observations\n", ctx.camera_id, ctx.pending_new_points.size());
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::commitPendingNewPoints()
{
  std::unordered_map<int, VisualPoint *> created_from_pg;
  std::vector<VisualPoint *> created_points;
  for (PerCameraData &ctx : cameras_)
  {
    for (PendingNewPointObservation &pending : ctx.pending_new_points)
    {
      VisualPoint *point = nullptr;
      if (pending.source_type == SOURCE_PG && pending.source_index >= 0)
      {
        const auto found = created_from_pg.find(pending.source_index);
        if (found != created_from_pg.end()) point = found->second;
      }
      if (point == nullptr)
      {
        point = new VisualPoint(pending.pt_var.point_w);
        point->ensureCameraCount(numCameras());
        point->covariance_ = pending.pt_var.var;
        point->is_normal_initialized_ = true;
        const V3D dir = pending.T_f_w * pending.pt_var.point_w;
        const V3D normal_c = pending.T_f_w.rotationMatrix() * pending.pt_var.normal;
        point->normal_ = dir.normalized().dot(normal_c) < 0.0 ? -pending.pt_var.normal : pending.pt_var.normal;
        point->previous_normal_ = point->normal_;
        created_points.push_back(point);
        if (pending.source_type == SOURCE_PG && pending.source_index >= 0)
          created_from_pg[pending.source_index] = point;
      }

      float *patch = new float[pending.patch.size()];
      std::copy(pending.patch.begin(), pending.patch.end(), patch);
      Feature *feature = new Feature(point, patch, pending.px, pending.bearing, pending.T_f_w, pending.level,
                                     pending.camera_id, ctx.new_frame->timestamp_);
      if (pending.virtual_patch_valid && virtual_sparse_patch_en)
      {
        feature->virtual_source_roi_ = pending.img;
        feature->virtual_source_origin_ = pending.virtual_source_origin;
        feature->virtual_support_materialized_ = false;
      }
      else
      {
        feature->img_ = pending.img;
        feature->virtual_support_materialized_ = pending.virtual_patch_valid;
      }
      feature->id_ = ctx.new_frame->id_;
      feature->inv_expo_time_ = pending.inv_expo_time;
      feature->T_v_w_ = pending.T_v_w;
      feature->R_v_from_c_ = pending.R_v_from_c;
      feature->R_c_from_v_ = pending.R_c_from_v;
      feature->virtual_patch_valid_ = pending.virtual_patch_valid;
      point->addFrameRef(feature);
      if (cross_camera_reference_en)
      {
        if (!point->has_ref_patch_)
        {
          point->ref_patch = feature;
          point->has_ref_patch_ = true;
        }
      }
      else
      {
        point->ref_patch_by_camera_[pending.camera_id] = feature;
        point->has_ref_patch_by_camera_[pending.camera_id] = 1;
      }
    }
  }
  for (VisualPoint *point : created_points) insertPointIntoVoxelMap(point);
  printf("[ VIO ] committed %zu new shared visual map points\n", created_points.size());
}

void VIOManager::updateVisualMapPointsVirtual(PerCameraData &ctx, const cv::Mat &img)
{
  if (ctx.total_points == 0) return;

  int update_num = 0;
  const SE3<double> pose_cur = ctx.new_frame->T_f_w_;
  for (int i = 0; i < ctx.total_points; ++i)
  {
    VisualPoint *pt = ctx.visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    {
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D raw_px;
    if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, raw_px)) continue;
    Feature *last_feature = nullptr;
    for (Feature *feature : pt->obs_)
      if (feature != nullptr && feature->camera_id_ == ctx.camera_id) { last_feature = feature; break; }
    bool add_flag = last_feature == nullptr;
    if (last_feature != nullptr)
    {
      const SE3<double> delta_pose = last_feature->T_f_w_ * pose_cur.inverse();
      const double delta_p = delta_pose.translation().norm();
      const double trace = delta_pose.rotationMatrix().trace();
      const double delta_theta = trace > 3.0 - 1e-6 ? 0.0 : std::acos(std::clamp(0.5 * (trace - 1.0), -1.0, 1.0));
      if (delta_p > 0.5 || delta_theta > 0.3 || (raw_px - last_feature->px_).norm() > 40.0) add_flag = true;
    }

    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(ctx.new_frame->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
    }
    if (!add_flag) continue;

    std::unique_ptr<float[]> patch(new float[patch_size_total]);
    cv::Mat virtual_support_img;
    cv::Point virtual_source_origin;
    SE3<double> T_v_w;
    M3D R_v_from_c, R_c_from_v;
    if (!createVirtualFeaturePatch(ctx, img, ctx.new_frame->T_f_w_, pt->pos_, patch.get(),
                                   virtual_support_img, virtual_source_origin, T_v_w, R_v_from_c, R_c_from_v))
    {
      ++rejected_virtual_support_oob_;
      continue;
    }

    const V3D bearing = ctx.cam->cam2world(raw_px);
    Feature *ftr_new = new Feature(pt, patch.release(), raw_px, bearing, ctx.new_frame->T_f_w_,
                                   ctx.visual_submap->search_levels[i], ctx.camera_id, ctx.new_frame->timestamp_);
    if (virtual_sparse_patch_en)
    {
      ftr_new->virtual_source_roi_ = virtual_support_img;
      ftr_new->virtual_source_origin_ = virtual_source_origin;
      ftr_new->virtual_support_materialized_ = false;
    }
    else
    {
      ftr_new->img_ = virtual_support_img;
      ftr_new->virtual_support_materialized_ = true;
    }
    ftr_new->id_ = ctx.new_frame->id_;
    ftr_new->inv_expo_time_ = state->inv_expo_time[ctx.camera_id];
    ftr_new->T_v_w_ = T_v_w;
    ftr_new->R_v_from_c_ = R_v_from_c;
    ftr_new->R_c_from_v_ = R_c_from_v;
    ftr_new->virtual_patch_valid_ = true;
    pt->addFrameRef(ftr_new);
    ctx.update_flag[i] = 1;
    ++update_num;
  }
  printf("[ VIO Virtual ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateVisualMapPoints(PerCameraData &ctx, const cv::Mat &img)
{
  if (virtual_fisheye_patch_en)
  {
    updateVisualMapPointsVirtual(ctx, img);
    return;
  }
  if (ctx.total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = ctx.new_frame->T_f_w_;
  for (int i = 0; i < ctx.total_points; i++)
  {
    VisualPoint *pt = ctx.visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(ctx.new_frame->w2c(pt->pos_));
    bool add_flag = false;
    
    float *patch_temp = new float[patch_size_total];
    getImagePatch(ctx, img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = nullptr;
    for (Feature *feature : pt->obs_)
      if (feature != nullptr && feature->camera_id_ == ctx.camera_id) { last_feature = feature; break; }
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    if (last_feature == nullptr) add_flag = true;
    else
    {
      SE3 pose_ref = last_feature->T_f_w_;
      SE3 delta_pose = pose_ref * pose_cur.inverse();
      double delta_p = delta_pose.translation().norm();
      double delta_theta = (delta_pose.rotationMatrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotationMatrix().trace() - 1));
      if (delta_p > 0.5 || delta_theta > 0.3 || (pc - last_feature->px_).norm() > 40) add_flag = true;
    }

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(ctx.new_frame->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      update_num += 1;
      ctx.update_flag[i] = 1;
      Vector3d f = ctx.cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, ctx.new_frame->T_f_w_,
                                     ctx.visual_submap->search_levels[i], ctx.camera_id, ctx.new_frame->timestamp_);
      ftr_new->img_ = img;
      ftr_new->id_ = ctx.new_frame->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time[ctx.camera_id];
      pt->addFrameRef(ftr_new);
    }
    else
    {
      delete[] patch_temp;
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(PerCameraData &ctx, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (ctx.total_points == 0) return;

  for (int i = 0; i < static_cast<int>(ctx.visual_submap->voxel_points.size()); i++)
  {
    VisualPoint *pt = ctx.visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (ctx.update_flag[i] == 0) continue;

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
    Feature *best_reference = nullptr;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      if (ref_patch_temp == nullptr) continue;
      if (!cross_camera_reference_en && ref_patch_temp->camera_id_ != ctx.camera_id) continue;
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
        if (*itm == nullptr || *itm == ref_patch_temp) continue;
        if (!cross_camera_reference_en && (*itm)->camera_id_ != ctx.camera_id) continue;
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

      if (count > 0) NCC /= count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        best_reference = ref_patch_temp;
      }
    }

    if (best_reference != nullptr)
    {
      pt->ensureCameraCount(numCameras());
      if (cross_camera_reference_en)
      {
        pt->ref_patch = best_reference;
        pt->has_ref_patch_ = true;
      }
      else
      {
        pt->ref_patch_by_camera_[ctx.camera_id] = best_reference;
        pt->has_ref_patch_by_camera_[ctx.camera_id] = 1;
      }
    }

  }
}

#if 0  // Legacy single-camera diagnostic renderer.
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

#endif

void VIOManager::projectPatchFromRefToCur(PerCameraData &ctx,
                                          const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  (void)ctx;
  (void)plane_map;
  static bool warned = false;
  if (!warned)
  {
    printf("[ VIO ] reference/current patch dump is disabled in the multi-camera target.\n");
    warned = true;
  }
}

#if 0  // Superseded by the dynamic multi-camera joint linearization above.
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
        // [MODIFY] 使用第一次生成的参�?patch，不再每帧重�?
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
    if (virtual_s2_optimize_en)
      updateStateVirtualS2(img, level);
    else
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

#endif

void VIOManager::updateFrameState(PerCameraData &ctx, const StatesGroup &state_value)
{
  if (ctx.camera_id >= 0 && ctx.camera_id < state_value.num_cameras)
  {
    ctx.Rcl = state_value.Rcl[ctx.camera_id];
    ctx.Pcl = state_value.Pcl[ctx.camera_id];
    updateCameraExtrinsicDerived(ctx);
  }
  M3D Rwi(state_value.rot_end);
  V3D Pwi(state_value.pos_end);
  ctx.Rcw = ctx.Rci * Rwi.transpose();
  ctx.Pcw = -ctx.Rci * Rwi.transpose() * Pwi + ctx.Pci;
  if (ctx.new_frame != nullptr)
    ctx.new_frame->T_f_w_ = SE3(Eigen::Quaterniond(ctx.Rcw).normalized().toRotationMatrix(), ctx.Pcw);
}

void VIOManager::plotTrackedPoints(PerCameraData &ctx)
{
  int total_points = ctx.visual_submap->voxel_points.size();
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
    VisualPoint *pt = ctx.visual_submap->voxel_points[i];
    V2D pc;
    if (virtual_fisheye_patch_en)
    {
      if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, pc)) continue;
    }
    else
    {
      pc = ctx.new_frame->w2c(pt->pos_);
    }

    if (ctx.visual_submap->errors[i] <= ctx.visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(ctx.img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8);
    }
    else
    {
      cv::circle(ctx.img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8);
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
    for (const auto &rejected_pt : ctx.rejected_visual_points_for_draw)
    {
      cv::circle(ctx.img_cp, rejected_pt.first, 5, rejectedPointColor(rejected_pt.second), -1, 8);
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(const cv::Mat &img, V2D pc) const
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
  const uint8_t *img_ptr = img.data + ((v_ref_i)*img.cols + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[3] + w_ref_bl * img_ptr[img.cols * 3] + w_ref_br * img_ptr[img.cols * 3 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[4] + w_ref_bl * img_ptr[1 + img.cols * 3] + w_ref_br * img_ptr[img.cols * 3 + 4];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[5] + w_ref_bl * img_ptr[2 + img.cols * 3] + w_ref_br * img_ptr[img.cols * 3 + 5];
  V3F pixel(B, G, R);
  return pixel;
}

bool VIOManager::getColorFromCamera(int camera_id, const V3D &p_w, V3F &bgr, double *cam_range) const
{
  if (camera_id < 0 || camera_id >= numCameras()) return false;
  const PerCameraData &ctx = cameras_[camera_id];
  if (ctx.cam == nullptr || ctx.new_frame == nullptr || ctx.img_rgb.empty() || ctx.img_rgb.channels() != 3) return false;
  const V3D point_c = ctx.new_frame->w2f(p_w);
  if (!point_c.array().isFinite().all()) return false;
  if (cam_range != nullptr) *cam_range = point_c.norm();
  const V2D pixel = ctx.cam->world2cam(point_c);
  if (!pixel.array().isFinite().all() || !ctx.cam->isInFrame(pixel.cast<int>(), 1)) return false;
  bgr = getInterpolatedPixel(ctx.img_rgb, pixel);
  return bgr.array().isFinite().all();
}

void VIOManager::dumpDataForColmap()
{
  if (cameras_.empty()) return;
  PerCameraData &ctx = cameras_.front();
  if (ctx.pinhole_cam == nullptr)
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
  ctx.pinhole_cam->undistortImage(ctx.img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(ctx.new_frame->T_f_w_.rotationMatrix());
  Eigen::Vector3d t = ctx.new_frame->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6�?
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID�?)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

bool VIOManager::inOpticalFlowBorder(const PerCameraData &ctx, const cv::Point2f &pt) const
{
  const int img_x = cvRound(pt.x);
  const int img_y = cvRound(pt.y);
  const int border_size = 1;
  return border_size <= img_x && img_x < ctx.width - border_size && border_size <= img_y && img_y < ctx.height - border_size;
}

V3D VIOManager::getOpticalFlowBearing(const PerCameraData &ctx, const cv::Point2f &px) const
{
  V3D bearing = ctx.cam->cam2world(px.x, px.y);
  if (bearing.norm() > 1e-12) bearing.normalize();
  return bearing;
}

void VIOManager::setOpticalFlowMask(const PerCameraData &ctx, cv::Mat &mask)
{
  mask = cv::Mat(ctx.height, ctx.width, CV_8UC1, cv::Scalar(255));

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
    if (x < 0 || x >= ctx.width || y < 0 || y >= ctx.height) continue;
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
  const PerCameraData &ctx = cameras_.front();
  OpticalFlowObservation obs;
  obs.frame_id = optical_flow_frame_id;
  obs.timestamp = img_time;
  obs.px = px;
  obs.bearing = getOpticalFlowBearing(ctx, px);
  obs.T_f_w = ctx.new_frame->T_f_w_;
  track.observations.push_back(obs);

  track.history.push_back(px);
  while (static_cast<int>(track.history.size()) > optical_flow_track_history_size) track.history.pop_front();
}

bool VIOManager::triangulateOpticalFlowTrack(OpticalFlowTrack &track)
{
  const PerCameraData &ctx = cameras_.front();
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
      if (!projectRawFisheyeIfValid(ctx, point_c, 1, raw_px)) return false;
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
  const PerCameraData &ctx = cameras_.front();
  if (ctx.img_rgb.empty()) return;
  if (ctx.img_rgb.channels() == 3)
  {
    optical_flow_debug_img = ctx.img_rgb.clone();
  }
  else
  {
    cv::cvtColor(ctx.img_rgb, optical_flow_debug_img, CV_GRAY2BGR);
  }

  for (const auto &pt : rejected_pts)
  {
    if (!inOpticalFlowBorder(ctx, pt)) continue;
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
  if (cameras_.empty()) return;
  PerCameraData &ctx = cameras_.front();
  if (ctx.width != img.cols || ctx.height != img.rows)
  {
    if (img.empty()) printf("[ OpticalFlow ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * ctx.image_resize_factor, img.rows * ctx.image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }

  ctx.img_rgb = img.clone();
  ctx.img_cp = img.clone();

  cv::Mat cur_img;
  if (img.channels() == 3)
  {
    cv::cvtColor(img, cur_img, CV_BGR2GRAY);
  }
  else
  {
    cur_img = img.clone();
  }

  ctx.new_frame.reset(new Frame(ctx.cam, cur_img));
  ctx.new_frame->camera_id_ = 0;
  ctx.new_frame->timestamp_ = img_time;
  updateFrameState(ctx, *state);

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
      if (status[i] && !inOpticalFlowBorder(ctx, optical_flow_cur_pts[i]))
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
  setOpticalFlowMask(ctx, mask);
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
  (void)pg;
  (void)feat_map;
  if (cameras_.empty()) return;
  PerCameraData &ctx = cameras_.front();
  if (ctx.width != img.cols || ctx.height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * ctx.image_resize_factor, img.rows * ctx.image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  ctx.img_rgb = img.clone();
  ctx.img_cp = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  ctx.new_frame.reset(new Frame(ctx.cam, img, -1, 0, img_time));
  updateFrameState(ctx, *state);
}

void VIOManager::processMultiCameraFrameFake(const MeasureGroup &meas)
{
  if (!meas.has_multi_cam_frame)
    throw std::runtime_error("color-only multi-camera mode requires MeasureGroup::multi_cam_frame");

  const MultiCameraFrame &mf = meas.multi_cam_frame;
  if (static_cast<int>(mf.images.size()) != numCameras())
    throw std::runtime_error("MultiCameraFrame image count does not match VIOManager camera count");

  for (int camera_id = 0; camera_id < numCameras(); ++camera_id)
  {
    PerCameraData &ctx = cameras_[camera_id];
    cv::Mat image = mf.images[camera_id].clone();
    if (image.empty())
      throw std::runtime_error("empty image for camera_id=" + std::to_string(camera_id));

    if (ctx.width != image.cols || ctx.height != image.rows)
      cv::resize(image, image,
                 cv::Size(image.cols * ctx.image_resize_factor, image.rows * ctx.image_resize_factor),
                 0, 0, CV_INTER_LINEAR);

    ctx.img_rgb = image.clone();
    ctx.img_cp = image.clone();
    if (image.channels() == 3) cv::cvtColor(image, image, CV_BGR2GRAY);

    ctx.new_frame.reset(new Frame(ctx.cam, image, mf.frame_id, camera_id, mf.timestamp));
    updateFrameState(ctx, *state);
  }
}

void VIOManager::processMultiCameraFrame(const MeasureGroup &meas, vector<pointWithVar> &pg,
                                         const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (!meas.has_multi_cam_frame) throw std::runtime_error("direct VIO requires MeasureGroup::multi_cam_frame");
  const MultiCameraFrame &mf = meas.multi_cam_frame;
  if (static_cast<int>(mf.images.size()) != numCameras())
    throw std::runtime_error("MultiCameraFrame image count does not match VIOManager camera count");

  const double frame_start = omp_get_wtime();
  printf("[ VIO Debug ] processMultiCameraFrame begin frame=%d cameras=%d pg=%zu feat_map=%zu plane_map=%zu virtual=%d cross_ref=%d normal=%d inverse=%d raycast=%d\n",
         mf.frame_id, numCameras(), pg.size(), feat_map.size(), plane_map.size(), virtual_fisheye_patch_en ? 1 : 0,
         cross_camera_reference_en ? 1 : 0, normal_en ? 1 : 0, inverse_composition_en ? 1 : 0, raycast_en ? 1 : 0);
  fflush(stdout);
  for (int camera_id = 0; camera_id < numCameras(); ++camera_id)
  {
    PerCameraData &ctx = cameras_[camera_id];
    cv::Mat image = mf.images[camera_id].clone();
    if (image.empty()) throw std::runtime_error("empty image for camera_id=" + std::to_string(camera_id));
    if (ctx.width != image.cols || ctx.height != image.rows)
      cv::resize(image, image, cv::Size(image.cols * ctx.image_resize_factor, image.rows * ctx.image_resize_factor), 0, 0, CV_INTER_LINEAR);
    ctx.img_rgb = image.clone();
    ctx.img_cp = image.clone();
    if (image.channels() == 3) cv::cvtColor(image, image, CV_BGR2GRAY);
    ctx.new_frame.reset(new Frame(ctx.cam, image, mf.frame_id, camera_id, mf.timestamp));
    updateFrameState(ctx, *state);
    resetGrid(ctx);
    printf("[ VIO Debug ] frame setup camera_id=%d frame=%d image=%dx%d type=%d ns=%s\n",
           ctx.camera_id, mf.frame_id, image.cols, image.rows, image.type(), ctx.camera_namespace.c_str());
    fflush(stdout);
  }
  const double frame_setup_end = omp_get_wtime();

  for (PerCameraData &ctx : cameras_)
  {
    printf("[ VIO Debug ] retrieve begin camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    fflush(stdout);
    retrieveFromVisualSparseMap(ctx, ctx.new_frame->img_, pg, plane_map);
    printf("[ VIO Debug ] retrieve end camera_id=%d frame=%d total_points=%d\n",
           ctx.camera_id, mf.frame_id, ctx.total_points);
    fflush(stdout);
    printf("[ VIO Debug ] generate begin camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    fflush(stdout);
    generateVisualMapPoints(ctx, ctx.new_frame->img_, pg);
    printf("[ VIO Debug ] generate end camera_id=%d frame=%d pending=%zu\n",
           ctx.camera_id, mf.frame_id, ctx.pending_new_points.size());
    fflush(stdout);
  }
  const double retrieve_end = omp_get_wtime();

  printf("[ VIO Debug ] ekf begin frame=%d\n", mf.frame_id);
  fflush(stdout);
  computeJacobianAndUpdateEKF();
  printf("[ VIO Debug ] ekf end frame=%d\n", mf.frame_id);
  fflush(stdout);
  const double ekf_end = omp_get_wtime();

  if (ref_patch_dump_en && !cameras_.empty())
    processRefPatchDumpProbe(cameras_[kRefPatchDumpCameraId], cameras_[kRefPatchDumpCameraId].new_frame->img_);

  for (PerCameraData &ctx : cameras_)
  {
    for (PendingNewPointObservation &pending : ctx.pending_new_points)
    {
      pending.T_f_w = ctx.new_frame->T_f_w_;
      if (pending.virtual_patch_valid) pending.T_v_w = composeVirtualPose(pending.R_v_from_c, pending.T_f_w);
    }
    printf("[ VIO Debug ] updateVisualMap begin camera_id=%d frame=%d pending=%zu\n",
           ctx.camera_id, mf.frame_id, ctx.pending_new_points.size());
    fflush(stdout);
    updateVisualMapPoints(ctx, ctx.new_frame->img_);
    printf("[ VIO Debug ] updateVisualMap end camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    fflush(stdout);
  }
  const double map_update_end = omp_get_wtime();
  printf("[ VIO Debug ] commitPendingNewPoints begin frame=%d\n", mf.frame_id);
  fflush(stdout);
  commitPendingNewPoints();
  printf("[ VIO Debug ] commitPendingNewPoints end frame=%d feat_map=%zu\n", mf.frame_id, feat_map.size());
  fflush(stdout);
  const double commit_end = omp_get_wtime();
  for (PerCameraData &ctx : cameras_)
  {
    printf("[ VIO Debug ] updateReference begin camera_id=%d frame=%d total_points=%d\n",
           ctx.camera_id, mf.frame_id, ctx.total_points);
    fflush(stdout);
    updateReferencePatch(ctx, plane_map);
    printf("[ VIO Debug ] updateReference end camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    fflush(stdout);
    plotTrackedPoints(ctx);
    if (plot_flag) projectPatchFromRefToCur(ctx, plane_map);
  }
  if (colmap_output_en && numCameras() == 1) dumpDataForColmap();
  const double reference_end = omp_get_wtime();
  ++frame_count;
  const double elapsed = reference_end - frame_start;
  ave_total = ave_total * (frame_count - 1) / frame_count + elapsed / frame_count;
  printf("[ VIO ] multi-camera frame=%d cameras=%d observations=%zu elapsed=%.6f s\n",
         mf.frame_id, numCameras(), feat_map.size(), elapsed);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Camera Count", numCameras());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Frame Setup", frame_setup_end - frame_start);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Retrieve + Generate", retrieve_end - frame_setup_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Joint EKF Update", ekf_end - retrieve_end);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Update Visual Map", map_update_end - ekf_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Commit New Points", commit_end - map_update_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Reference + Debug", reference_end - commit_end);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", elapsed);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
}
