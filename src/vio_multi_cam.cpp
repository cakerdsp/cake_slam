/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio_multi_cam.h"
#include "directional_update.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
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
constexpr double kInterpPi = 3.14159265358979323846;
constexpr int kRuntimeRawSupportDumpSize = 45;
constexpr int kLegacyFixedStateDim = 19;

bool normalizePatchValues(const float *values, int count, double min_std,
                          Eigen::VectorXd &normalized, double *stddev = nullptr)
{
  if (values == nullptr || count <= 0 || !std::isfinite(min_std) || min_std <= 0.0) return false;
  normalized.resize(count);
  double mean = 0.0;
  for (int i = 0; i < count; ++i)
  {
    if (!std::isfinite(values[i])) return false;
    mean += values[i];
  }
  mean /= static_cast<double>(count);
  double squared_norm = 0.0;
  for (int i = 0; i < count; ++i)
  {
    normalized[i] = static_cast<double>(values[i]) - mean;
    squared_norm += normalized[i] * normalized[i];
  }
  const double sigma = std::sqrt(squared_norm / static_cast<double>(count));
  if (!std::isfinite(sigma) || sigma < min_std) return false;
  normalized /= sigma;
  if (stddev != nullptr) *stddev = sigma;
  return normalized.array().isFinite().all();
}

bool normalizePatchValues(const std::vector<double> &values, double min_std,
                          Eigen::VectorXd &normalized, double *stddev = nullptr)
{
  if (values.empty() || !std::isfinite(min_std) || min_std <= 0.0) return false;
  normalized = Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
  if (!normalized.array().isFinite().all()) return false;
  const double mean = normalized.mean();
  normalized.array() -= mean;
  const double sigma = std::sqrt(normalized.squaredNorm() / static_cast<double>(values.size()));
  if (!std::isfinite(sigma) || sigma < min_std) return false;
  normalized /= sigma;
  if (stddev != nullptr) *stddev = sigma;
  return normalized.array().isFinite().all();
}

bool normalizePatchWithJacobian(const std::vector<double> &values,
                                const Eigen::MatrixXd &raw_jacobian,
                                double min_std,
                                Eigen::VectorXd &normalized,
                                Eigen::MatrixXd &normalized_jacobian)
{
  if (values.empty() || raw_jacobian.rows() != static_cast<Eigen::Index>(values.size())) return false;
  double sigma = 0.0;
  if (!normalizePatchValues(values, min_std, normalized, &sigma)) return false;

  const Eigen::Map<const Eigen::VectorXd> values_map(
      values.data(), static_cast<Eigen::Index>(values.size()));
  const Eigen::VectorXd centered =
      values_map - Eigen::VectorXd::Constant(values_map.size(), values_map.mean());
  const Eigen::RowVectorXd row_mean = raw_jacobian.colwise().mean();
  normalized_jacobian = raw_jacobian.rowwise() - row_mean;
  const Eigen::RowVectorXd projected = centered.transpose() * raw_jacobian;
  const double count = static_cast<double>(values.size());
  // q = C p / sigma, sigma^2 = (p^T C p) / N:
  // dq/dx = C J / sigma - (C p) ((C p)^T J) / (N sigma^3).
  // Evaluate the two rank-one operations directly; never materialize the N x N centering matrix.
  normalized_jacobian =
      normalized_jacobian / sigma -
      centered * projected / (count * sigma * sigma * sigma);
  return normalized_jacobian.array().isFinite().all();
}

bool normalizedPatchMetrics(const float *reference, const float *current, int count,
                            double min_std, double &sse, double &ncc)
{
  Eigen::VectorXd normalized_reference;
  Eigen::VectorXd normalized_current;
  if (!normalizePatchValues(reference, count, min_std, normalized_reference) ||
      !normalizePatchValues(current, count, min_std, normalized_current))
    return false;
  const Eigen::VectorXd residual = normalized_current - normalized_reference;
  sse = residual.squaredNorm();
  ncc = normalized_reference.dot(normalized_current) / static_cast<double>(count);
  return std::isfinite(sse) && std::isfinite(ncc);
}

double normalizedPatchRobustSqrtWeight(const Eigen::VectorXd &residual,
                                       bool robust_enabled, double huber_delta)
{
  if (!robust_enabled || residual.size() == 0) return 1.0;
  const double patch_rmse =
      std::sqrt(residual.squaredNorm() / static_cast<double>(residual.size()));
  if (!std::isfinite(patch_rmse) || patch_rmse <= huber_delta || patch_rmse <= 1.0e-12) return 1.0;
  return std::sqrt(huber_delta / patch_rmse);
}

double tukeySqrtWeight(double residual, double cutoff)
{
  if (!std::isfinite(residual) || !std::isfinite(cutoff) || cutoff <= 0.0) return 0.0;
  const double normalized_abs_residual = std::abs(residual) / cutoff;
  if (normalized_abs_residual >= 1.0) return 0.0;
  return 1.0 - normalized_abs_residual * normalized_abs_residual;
}

std::string normalizeVirtualInterpMode(std::string mode)
{
  for (char &ch : mode) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return mode;
}

int interpolationKernelRadius(VirtualInterpMode mode)
{
  switch (mode)
  {
  case VirtualInterpMode::BICUBIC: return 2;
  case VirtualInterpMode::LANCZOS: return 4;
  case VirtualInterpMode::BILINEAR:
  default: return 1;
  }
}

int interpolationBorderMargin(VirtualInterpMode mode)
{
  return interpolationKernelRadius(mode) + 1;
}

float clampInterpValue(double value)
{
  return static_cast<float>(std::clamp(value, 0.0, 255.0));
}

double cubicInterpWeight(double x)
{
  x = std::fabs(x);
  constexpr double a = -0.5;
  if (x <= 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
  if (x < 2.0) return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
  return 0.0;
}

double sinc(double x)
{
  if (std::fabs(x) < 1.0e-12) return 1.0;
  const double pix = kInterpPi * x;
  return std::sin(pix) / pix;
}

double lanczosInterpWeight(double x)
{
  x = std::fabs(x);
  constexpr double a = 4.0;
  if (x >= a) return 0.0;
  return sinc(x) * sinc(x / a);
}

std::string sanitizeRuntimeSupportDumpFolder(const std::string &folder)
{
  std::string sanitized;
  sanitized.reserve(folder.size());
  for (unsigned char ch : folder)
  {
    if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.')
      sanitized.push_back(static_cast<char>(ch));
    else
      sanitized.push_back('_');
  }
  if (sanitized.empty() || sanitized == "." || sanitized == "..") sanitized = "runtime_support";
  return sanitized;
}

cv::Mat makeFloatMatDisplay(const cv::Mat &values, const cv::Mat &valid_mask = cv::Mat())
{
  if (values.empty() || values.type() != CV_32FC1) return cv::Mat();
  cv::Mat display(values.rows, values.cols, CV_8UC1, cv::Scalar(0));
  const bool use_mask = !valid_mask.empty() && valid_mask.type() == CV_8UC1 &&
                        valid_mask.rows == values.rows && valid_mask.cols == values.cols;
  for (int y = 0; y < values.rows; ++y)
  {
    const float *src = values.ptr<float>(y);
    const uint8_t *mask = use_mask ? valid_mask.ptr<uint8_t>(y) : nullptr;
    uint8_t *dst = display.ptr<uint8_t>(y);
    for (int x = 0; x < values.cols; ++x)
    {
      if (mask != nullptr && mask[x] == 0) continue;
      const float value = src[x];
      if (std::isfinite(value)) dst[x] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
    }
  }
  return display;
}

cv::Mat makeMarkedFloatMatDisplay(const cv::Mat &values, const cv::Mat &valid_mask, const V2D &center_px)
{
  const cv::Mat gray = makeFloatMatDisplay(values, valid_mask);
  if (gray.empty()) return gray;

  cv::Mat display;
  cv::cvtColor(gray, display, cv::COLOR_GRAY2BGR);
  if (center_px.array().isFinite().all())
  {
    const int x = static_cast<int>(std::lround(center_px[0]));
    const int y = static_cast<int>(std::lround(center_px[1]));
    if (x >= 0 && x < display.cols && y >= 0 && y < display.rows)
      display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
  }
  return display;
}


cv::Mat makeMarkedRawSupportDisplay(const cv::Mat &image, const V2D &center_px, int support_size)
{
  if (image.empty() || image.type() != CV_8UC1 || support_size <= 0 || !center_px.array().isFinite().all())
    return cv::Mat();

  const int half_size = support_size / 2;
  const int cx = static_cast<int>(std::lround(center_px[0]));
  const int cy = static_cast<int>(std::lround(center_px[1]));
  const int src_x0 = std::max(0, cx - half_size);
  const int src_y0 = std::max(0, cy - half_size);
  const int src_x1 = std::min(image.cols, cx - half_size + support_size);
  const int src_y1 = std::min(image.rows, cy - half_size + support_size);
  if (src_x0 >= src_x1 || src_y0 >= src_y1) return cv::Mat();

  cv::Mat display(support_size, support_size, CV_8UC3, cv::Scalar(0, 0, 0));
  const int dst_x0 = src_x0 - (cx - half_size);
  const int dst_y0 = src_y0 - (cy - half_size);
  cv::Mat roi_gray = image(cv::Rect(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0));
  cv::Mat roi_bgr;
  cv::cvtColor(roi_gray, roi_bgr, cv::COLOR_GRAY2BGR);
  roi_bgr.copyTo(display(cv::Rect(dst_x0, dst_y0, roi_bgr.cols, roi_bgr.rows)));

  if (half_size >= 0 && half_size < display.cols && half_size < display.rows)
    display.at<cv::Vec3b>(half_size, half_size) = cv::Vec3b(0, 0, 255);
  return display;
}

cv::Mat makeFloatPatchDisplay(const std::vector<float> &values, int patch_size, int offset = 0)
{
  if (patch_size <= 0 || offset < 0 || values.size() < static_cast<size_t>(offset + patch_size * patch_size))
    return cv::Mat();
  cv::Mat display(patch_size, patch_size, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < patch_size; ++y)
  {
    uint8_t *dst = display.ptr<uint8_t>(y);
    for (int x = 0; x < patch_size; ++x)
    {
      const float value = values[offset + y * patch_size + x];
      if (std::isfinite(value)) dst[x] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
    }
  }
  return display;
}

bool writeImageIfAvailable(const std::filesystem::path &path, const cv::Mat &image)
{
  return !image.empty() && cv::imwrite(path.string(), image);
}

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
  if (!ctx.Rcl.allFinite() || ctx.Rcl.determinant() <= 0.0)
    throw std::invalid_argument("camera Rcl must be finite with positive determinant");
  Eigen::Quaterniond q_cl(ctx.Rcl);
  if (!q_cl.coeffs().allFinite() || q_cl.norm() <= 1.0e-12)
    throw std::invalid_argument("camera Rcl cannot be projected to SO(3)");
  ctx.Rcl = q_cl.normalized().toRotationMatrix();
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

void VIOManager::setCameraTimeOffsetGroups(const std::vector<int> &groups)
{
  if (static_cast<int>(groups.size()) != numCameras())
    throw std::invalid_argument("camera time-offset group count must match camera count");
  camera_time_offset_groups = groups;
  for (PerCameraData &ctx : cameras_)
  {
    if (ctx.camera_id >= 0 && ctx.camera_id < static_cast<int>(groups.size()))
      ctx.time_offset_group = groups[ctx.camera_id];
  }
}

void VIOManager::setCurrentUnbiasedGyr(const V3D &gyr)
{
  current_unbiased_gyr_ = gyr;
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
  bool changed = false;
  for (PerCameraData &ctx : cameras_)
  {
    if (ctx.camera_id < 0 || ctx.camera_id >= state_value.num_cameras) continue;
    const M3D dR_cl = ctx.Rcl.transpose() * state_value.Rcl[ctx.camera_id];
    const double rot_delta = Log(dR_cl).norm();
    const double trans_delta = (ctx.Pcl - state_value.Pcl[ctx.camera_id]).norm();
    if (rot_delta > 1.0e-12 || trans_delta > 1.0e-12) changed = true;
    ctx.Rcl = state_value.Rcl[ctx.camera_id];
    ctx.Pcl = state_value.Pcl[ctx.camera_id];
    updateCameraExtrinsicDerived(ctx);
  }
  if (changed)
  {
    ++extrinsic_version_;
    for (PerCameraData &ctx : cameras_)
    {
      for (auto &entry : ctx.warp_map) delete entry.second;
      ctx.warp_map.clear();
    }
  }
}

bool VIOManager::isOnlineExtrinsicEnabledForCamera(int camera_id) const
{
  if (!online_extrinsic_en || camera_id < 0 || camera_id >= numCameras()) return false;
  if (online_extrinsic_camera_mask.empty()) return true;
  if (camera_id >= static_cast<int>(online_extrinsic_camera_mask.size())) return false;
  return online_extrinsic_camera_mask[camera_id] != 0;
}

bool VIOManager::isOnlineTimeOffsetEnabledForGroup(int group_id) const
{
  if (!online_time_offset_en || state == nullptr || group_id < 0 ||
      group_id >= state->num_time_offset_groups)
    return false;
  if (online_time_offset_group_mask.empty()) return true;
  if (group_id >= static_cast<int>(online_time_offset_group_mask.size())) return false;
  return online_time_offset_group_mask[group_id] != 0;
}

void VIOManager::applyOnlineExtrinsicPriors(Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient,
                                            bool allow_rotation, bool allow_translation) const
{
  if (state == nullptr || !online_extrinsic_en) return;
  constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
  const double rot_std = online_extrinsic_prior_rot_std_deg * kDegToRad;
  const double trans_std = online_extrinsic_prior_trans_std_m;
  if (rot_std <= 0.0 || trans_std <= 0.0) return;
  const double measurement_cov = photometricNoiseCovariance();
  const double rot_info = measurement_cov / (rot_std * rot_std);
  const double trans_info = measurement_cov / (trans_std * trans_std);

  for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
  {
    if (!isOnlineExtrinsicEnabledForCamera(camera_id)) continue;
    if (camera_id >= static_cast<int>(cameras_.size()) ||
        cameras_[camera_id].total_points < online_extrinsic_min_tracks)
      continue;
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

void VIOManager::applyOnlineTimeOffsetPriors(Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient,
                                             const std::vector<uint8_t> &active_groups) const
{
  if (state == nullptr || !online_time_offset_en) return;
  const double std_s = online_time_offset_prior_std_ms * 1.0e-3;
  if (!std::isfinite(std_s) || std_s <= 0.0) return;
  const double info = photometricNoiseCovariance() / (std_s * std_s);
  for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
  {
    if (group_id >= static_cast<int>(active_groups.size()) || active_groups[group_id] == 0) continue;
    const int idx = state->timeOffsetIndex(group_id);
    const double prior = group_id < state->time_offset_prior.size() ? state->time_offset_prior[group_id] : 0.0;
    hessian(idx, idx) += info;
    gradient[idx] += info * (state->time_offset[group_id] - prior);
  }
}

bool VIOManager::calibrationUpdateWithinTrustRegion(
    const Eigen::VectorXd &solution, bool allow_rotation, bool allow_translation,
    const std::vector<uint8_t> &active_time_groups) const
{
  if (state == nullptr) return true;
  constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
  const double max_rot = std::max(0.0, online_extrinsic_max_rot_update_deg) * kDegToRad;
  const double max_trans = std::max(0.0, online_extrinsic_max_trans_update_m);
  for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
  {
    if (!isOnlineExtrinsicEnabledForCamera(camera_id)) continue;
    if (allow_rotation && max_rot > 0.0 &&
        solution.segment<3>(state->extrinsicRotIndex(camera_id)).norm() > max_rot)
      return false;
    if (allow_translation && max_trans > 0.0 &&
        solution.segment<3>(state->extrinsicTransIndex(camera_id)).norm() > max_trans)
      return false;
  }
  const double max_time_update = std::max(0.0, online_time_offset_max_update_ms) * 1.0e-3;
  const double max_abs_time = std::max(0.0, online_time_offset_max_abs_ms) * 1.0e-3;
  for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
  {
    if (group_id >= static_cast<int>(active_time_groups.size()) || active_time_groups[group_id] == 0) continue;
    const int idx = state->timeOffsetIndex(group_id);
    if (max_time_update > 0.0 && std::fabs(solution[idx]) > max_time_update) return false;
    const double prior = group_id < state->time_offset_prior.size() ? state->time_offset_prior[group_id] : 0.0;
    if (max_abs_time > 0.0 &&
        std::fabs(state->time_offset[group_id] + solution[idx] - prior) > max_abs_time)
      return false;
  }
  return true;
}

void VIOManager::deactivateCalibrationBlocks(Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient,
                                             bool deactivate_extrinsic,
                                             const std::vector<uint8_t> &deactivate_time_groups) const
{
  if (state == nullptr) return;
  auto zeroIndex = [&](int index) {
    hessian.row(index).setZero();
    hessian.col(index).setZero();
    gradient[index] = 0.0;
  };
  if (deactivate_extrinsic)
  {
    for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
    {
      for (int k = 0; k < 6; ++k) zeroIndex(state->extrinsicIndex(camera_id) + k);
    }
  }
  for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
  {
    if (group_id < static_cast<int>(deactivate_time_groups.size()) && deactivate_time_groups[group_id])
      zeroIndex(state->timeOffsetIndex(group_id));
  }
}

void VIOManager::deactivateInactiveCalibrationBlocks(
    Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient,
    const std::vector<uint8_t> &active_extrinsic_rot,
    const std::vector<uint8_t> &active_extrinsic_trans,
    const std::vector<uint8_t> &active_time_groups) const
{
  if (state == nullptr) return;
  auto zeroIndex = [&](int index) {
    hessian.row(index).setZero();
    hessian.col(index).setZero();
    gradient[index] = 0.0;
  };
  for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
  {
    const bool rot_active = camera_id < static_cast<int>(active_extrinsic_rot.size()) &&
                            active_extrinsic_rot[camera_id] != 0;
    const bool trans_active = camera_id < static_cast<int>(active_extrinsic_trans.size()) &&
                              active_extrinsic_trans[camera_id] != 0;
    if (!rot_active)
    {
      const int ridx = state->extrinsicRotIndex(camera_id);
      for (int k = 0; k < 3; ++k) zeroIndex(ridx + k);
    }
    if (!trans_active)
    {
      const int tidx = state->extrinsicTransIndex(camera_id);
      for (int k = 0; k < 3; ++k) zeroIndex(tidx + k);
    }
  }
  for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
  {
    const bool active = group_id < static_cast<int>(active_time_groups.size()) &&
                        active_time_groups[group_id] != 0;
    if (!active) zeroIndex(state->timeOffsetIndex(group_id));
  }
}

void VIOManager::restoreInactiveCalibrationCovariance(
    const Eigen::MatrixXd &prior_cov,
    const std::vector<uint8_t> &active_extrinsic_rot,
    const std::vector<uint8_t> &active_extrinsic_trans,
    const std::vector<uint8_t> &active_time_groups) const
{
  if (state == nullptr || prior_cov.rows() != state->cov.rows() || prior_cov.cols() != state->cov.cols()) return;
  auto restoreIndex = [&](int index) {
    state->cov.row(index) = prior_cov.row(index);
    state->cov.col(index) = prior_cov.col(index);
  };
  for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
  {
    const bool rot_active = camera_id < static_cast<int>(active_extrinsic_rot.size()) &&
                            active_extrinsic_rot[camera_id] != 0;
    const bool trans_active = camera_id < static_cast<int>(active_extrinsic_trans.size()) &&
                              active_extrinsic_trans[camera_id] != 0;
    if (!rot_active)
    {
      const int ridx = state->extrinsicRotIndex(camera_id);
      for (int k = 0; k < 3; ++k) restoreIndex(ridx + k);
    }
    if (!trans_active)
    {
      const int tidx = state->extrinsicTransIndex(camera_id);
      for (int k = 0; k < 3; ++k) restoreIndex(tidx + k);
    }
  }
  for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
  {
    const bool active = group_id < static_cast<int>(active_time_groups.size()) &&
                        active_time_groups[group_id] != 0;
    if (!active) restoreIndex(state->timeOffsetIndex(group_id));
  }
}

bool VIOManager::refreshReferenceCalibration(Feature &feature)
{
  if (feature.camera_id_ < 0 || feature.camera_id_ >= numCameras()) return false;
  if (feature.extrinsic_version_ == extrinsic_version_) return true;
  PerCameraData &ctx = cameras_[feature.camera_id_];
  const M3D Rcw = ctx.Rci * feature.Rwi_ref_.transpose();
  const V3D Pcw = -ctx.Rci * feature.Rwi_ref_.transpose() * feature.Pwi_ref_ + ctx.Pci;
  feature.T_f_w_ = SE3d(Eigen::Quaterniond(Rcw).normalized().toRotationMatrix(), Pcw);
  if (feature.virtual_patch_valid_)
    feature.T_v_w_ = composeVirtualPose(feature.R_v_from_c_, feature.T_f_w_);
  feature.extrinsic_version_ = extrinsic_version_;
  return true;
}

void VIOManager::fillPendingObservationTiming(PendingNewPointObservation &pending,
                                              const PerCameraData &ctx) const
{
  if (ctx.new_frame == nullptr) return;
  pending.raw_timestamp = ctx.new_frame->raw_timestamp_;
  pending.corrected_timestamp = ctx.new_frame->corrected_timestamp_;
  pending.capture_timestamp = ctx.new_frame->capture_timestamp_;
  pending.td_used = ctx.new_frame->td_used_;
  pending.exposure_time_offset = ctx.new_frame->exposure_time_offset_;
  pending.time_offset_group = ctx.new_frame->time_offset_group_;
  pending.Rwi_ref = ctx.Rwi;
  pending.Pwi_ref = ctx.Pwi;
  pending.extrinsic_version = extrinsic_version_;
}

void VIOManager::initializeVIO()
{
  if (cameras_.empty()) throw std::runtime_error("VIOManager cameras were not configured");
  validateVisualMapManageConfig();
  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);
  if (zncc_residual_en)
  {
    if (inverse_composition_en)
      throw std::runtime_error("zncc_residual_en does not support inverse_composition_en");
    if (virtual_s2_optimize_en)
      throw std::runtime_error("zncc_residual_en does not support virtual_s2_optimize_en");
    if (!std::isfinite(zncc_min_std) || zncc_min_std <= 0.0)
      throw std::runtime_error("zncc_min_std must be finite and positive");
    if (!std::isfinite(zncc_residual_cov) || zncc_residual_cov <= 0.0)
      throw std::runtime_error("zncc_residual_cov must be finite and positive");
    if (!std::isfinite(zncc_huber_delta) || zncc_huber_delta <= 0.0)
      throw std::runtime_error("zncc_huber_delta must be finite and positive");
    if (exposure_estimate_en)
      throw std::runtime_error("zncc_residual_en requires exposure_estimate_en=false");
    printf("[ VIO ZNCC Residual ] enabled for all same-camera and cross-camera patches: min_std=%.3f cov=%.6f robust=%d huber_delta=%.3f\n",
           zncc_min_std, zncc_residual_cov, (zncc_robust_en && !tukey_robust_en) ? 1 : 0, zncc_huber_delta);
  }
  if (tukey_robust_en)
  {
    if (!std::isfinite(outlier_threshold) || outlier_threshold <= 0.0)
      throw std::runtime_error("tukey_robust_en requires outlier_threshold to be finite and positive");
    printf("[ VIO Tukey Robust ] enabled for every active visual residual: cutoff=outlier_threshold=%.3f\n",
           outlier_threshold);
    if (zncc_residual_en && zncc_robust_en)
      printf("[ VIO Tukey Robust ] overriding ZNCC patch Huber weighting.\n");
  }
  runtime_support_dump_initialized_ = false;
  runtime_support_dump_next_point_id_ = 0;
  runtime_support_dump_best_track_count_ = 0;
  runtime_support_dump_best_point_ = nullptr;
  runtime_support_dump_effective_folder_.clear();
  if (runtime_support_dump_en && !virtual_fisheye_patch_en)
    printf("[ VIO Runtime Support Dump ] Enabled for raw patch path.\n");

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

    virtual_interp_mode = normalizeVirtualInterpMode(virtual_interp_mode);
    if (virtual_interp_mode == "bilinear")
      virtual_interp_mode_enum = VirtualInterpMode::BILINEAR;
    else if (virtual_interp_mode == "bicubic")
      virtual_interp_mode_enum = VirtualInterpMode::BICUBIC;
    else if (virtual_interp_mode == "lanczos")
      virtual_interp_mode_enum = VirtualInterpMode::LANCZOS;
    else
      throw std::runtime_error("virtual_interp_mode must be bilinear, bicubic, or lanczos");

    if (virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::FORWARD_SPLAT &&
        virtual_interp_mode_enum != VirtualInterpMode::BILINEAR)
      printf("[ VIO Virtual Interp ] virtual_interp_mode=%s affects support reads; forward_splat deposition remains bilinear. Use pull_exact for raw pull interpolation.\n",
             virtual_interp_mode.c_str());

    const int max_scale = 1 << ((patch_pyrimid_level - 1) + virtual_max_search_level);
    virtual_support_radius = patch_size_half * max_scale + virtual_patch_margin + interpolationBorderMargin(virtual_interp_mode_enum);
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

  if (visual_ref_post_ekf_build_en)
    printf("[ VIO Ref Build ] New VisualPoint references are materialized from the post-EKF camera pose.\n");

  if (visual_map_manage_en)
  {
    printf("[ VIO Map Config ] manage=1 shadow=%d ref_select=%d fallback=%d lifecycle=%d coverage=%d nis=%d seed=%d footprint=%d information=%d replacement=%d retirement=%d post_ekf=%d mode=%s\n",
           visual_map_manage_shadow_en ? 1 : 0, visual_ref_current_select_en ? 1 : 0,
           visual_ref_fallback_en ? 1 : 0, visual_ref_lifecycle_en ? 1 : 0,
           visual_ref_view_coverage_en ? 1 : 0, visual_ref_nis_en ? 1 : 0,
           visual_point_seed_validation_en ? 1 : 0, visual_point_footprint_redundancy_en ? 1 : 0,
           visual_point_information_prune_en ? 1 : 0, visual_point_replacement_en ? 1 : 0,
           visual_map_retirement_apply_en ? 1 : 0, visual_ref_post_ekf_build_en ? 1 : 0,
           virtual_fisheye_patch_en ? "virtual" : "raw");
  }

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

  if (usage_stats_window <= 0) throw std::runtime_error("usage_stats_window must be positive");
  resetUsageStatsWindow();
  resetUsageStatsTotals();
  if (usage_stats_en)
  {
    printf("\033[1;35m[ VIO Usage Stats ] enabled: window=%d frames; tables show candidate/accepted usage ratios.\033[0m\n",
           usage_stats_window);
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
  const int fallback_time_groups = camera_time_offset_groups.empty()
                                     ? 1
                                     : (*std::max_element(camera_time_offset_groups.begin(),
                                                          camera_time_offset_groups.end()) + 1);
  const int state_dim = state != nullptr ? state->stateDim()
                                         : BASE_STATE_DIM + numCameras() + fallback_time_groups + 6 * numCameras();
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
  ctx.retrieve_voxel_point_candidates.assign(ctx.length, std::vector<VisualPoint *>());
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

bool VIOManager::passVisualGeometryFilter(const pointWithVar &candidate,
                                          const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
                                          VisualGeomRejectReason &reject_reason) const
{
  reject_reason = VISUAL_GEOM_REJECT_NONE;
  if (!visual_geom_filter_en) return true;

  const V3D &p_w = candidate.point_w;
  const double min_sigma = (std::isfinite(visual_geom_filter_min_sigma) && visual_geom_filter_min_sigma > 0.0)
                               ? visual_geom_filter_min_sigma
                               : 1.0e-12;
  if (!p_w.array().isFinite().all())
  {
    reject_reason = VISUAL_GEOM_REJECT_BAD_POINT;
    return false;
  }

  const double voxel_size = (std::isfinite(visual_geom_filter_voxel_size) && visual_geom_filter_voxel_size > 1.0e-6)
                              ? visual_geom_filter_voxel_size
                              : 0.5;
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; ++j) loc_xyz[j] = static_cast<int64_t>(std::floor(p_w[j] / voxel_size));
  const auto iter = plane_map.find(VOXEL_LOCATION(loc_xyz[0], loc_xyz[1], loc_xyz[2]));
  if (iter == plane_map.end() || iter->second == nullptr)
  {
    reject_reason = VISUAL_GEOM_REJECT_NO_PLANE;
    return false;
  }

  VoxelOctoTree *octo = iter->second->find_correspond(p_w);
  if (octo == nullptr || octo->plane_ptr_ == nullptr)
  {
    reject_reason = VISUAL_GEOM_REJECT_NO_PLANE;
    return false;
  }

  const VoxelPlane &plane = *octo->plane_ptr_;
  if (!plane.is_plane_)
  {
    reject_reason = VISUAL_GEOM_REJECT_NOT_PLANE;
    return false;
  }
  if (visual_geom_filter_min_plane_points > 0 && plane.points_size_ < visual_geom_filter_min_plane_points)
  {
    reject_reason = VISUAL_GEOM_REJECT_PLANE_SIZE;
    return false;
  }
  if (!plane.center_.array().isFinite().all() || !plane.normal_.array().isFinite().all() ||
      !std::isfinite(static_cast<double>(plane.d_)))
  {
    reject_reason = VISUAL_GEOM_REJECT_NOT_PLANE;
    return false;
  }

  V3D normal = plane.normal_;
  const double normal_norm = normal.norm();
  if (!std::isfinite(normal_norm) || normal_norm <= 1.0e-12)
  {
    reject_reason = VISUAL_GEOM_REJECT_NOT_PLANE;
    return false;
  }
  normal /= normal_norm;

  const double candidate_normal_norm = candidate.normal.norm();
  if (std::isfinite(candidate_normal_norm) && candidate_normal_norm > 1.0e-12 &&
      std::isfinite(visual_geom_filter_min_normal_cos) && visual_geom_filter_min_normal_cos > 0.0)
  {
    const double normal_cos = std::fabs((candidate.normal / candidate_normal_norm).dot(normal));
    if (!std::isfinite(normal_cos) || normal_cos < visual_geom_filter_min_normal_cos)
    {
      reject_reason = VISUAL_GEOM_REJECT_NORMAL;
      return false;
    }
  }

  if (!candidate.var.array().isFinite().all())
  {
    reject_reason = VISUAL_GEOM_REJECT_COV;
    return false;
  }
  const double point_cov_trace = candidate.var.trace();
  if (visual_geom_filter_require_point_cov && (!std::isfinite(point_cov_trace) || point_cov_trace <= min_sigma))
  {
    reject_reason = VISUAL_GEOM_REJECT_COV;
    return false;
  }
  if (std::isfinite(visual_geom_filter_max_point_cov_trace) && visual_geom_filter_max_point_cov_trace > 0.0 &&
      point_cov_trace > visual_geom_filter_max_point_cov_trace)
  {
    reject_reason = VISUAL_GEOM_REJECT_COV;
    return false;
  }

  const double normal_cov = normal.dot(candidate.var * normal);
  if (visual_geom_filter_require_point_cov && (!std::isfinite(normal_cov) || normal_cov <= min_sigma))
  {
    reject_reason = VISUAL_GEOM_REJECT_COV;
    return false;
  }
  if (std::isfinite(visual_geom_filter_max_normal_cov) && visual_geom_filter_max_normal_cov > 0.0 &&
      normal_cov > visual_geom_filter_max_normal_cov)
  {
    reject_reason = VISUAL_GEOM_REJECT_COV;
    return false;
  }

  const double signed_dis = normal.dot(p_w) + static_cast<double>(plane.d_);
  if (!std::isfinite(signed_dis))
  {
    reject_reason = VISUAL_GEOM_REJECT_BAD_POINT;
    return false;
  }
  const double dis_to_plane = std::fabs(signed_dis);
  const V3D center_delta = p_w - plane.center_;
  double tangent_sq = center_delta.squaredNorm() - signed_dis * signed_dis;
  if (!std::isfinite(tangent_sq) || tangent_sq < -1.0e-8)
  {
    reject_reason = VISUAL_GEOM_REJECT_RANGE;
    return false;
  }
  tangent_sq = std::max(0.0, tangent_sq);
  const double tangent_dist = std::sqrt(tangent_sq);
  if (std::isfinite(static_cast<double>(plane.radius_)) && plane.radius_ > 0.0f &&
      std::isfinite(visual_geom_filter_radius_multiplier) && visual_geom_filter_radius_multiplier > 0.0 &&
      tangent_dist > visual_geom_filter_radius_multiplier * static_cast<double>(plane.radius_))
  {
    reject_reason = VISUAL_GEOM_REJECT_RANGE;
    return false;
  }

  if (!plane.plane_var_.array().isFinite().all())
  {
    reject_reason = VISUAL_GEOM_REJECT_SIGMA;
    return false;
  }
  Eigen::Matrix<double, 1, 6> J_nq;
  J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
  J_nq.block<1, 3>(0, 3) = -normal;
  const double plane_sigma = (J_nq * plane.plane_var_ * J_nq.transpose())(0, 0);
  const double sigma_l = plane_sigma + normal_cov;
  if (!std::isfinite(sigma_l) || sigma_l <= min_sigma)
  {
    reject_reason = VISUAL_GEOM_REJECT_SIGMA;
    return false;
  }

  const double chi2 = dis_to_plane * dis_to_plane / sigma_l;
  if (!std::isfinite(chi2) || (std::isfinite(visual_geom_filter_max_chi2) && visual_geom_filter_max_chi2 > 0.0 &&
                               chi2 > visual_geom_filter_max_chi2))
  {
    reject_reason = VISUAL_GEOM_REJECT_CHI2;
    return false;
  }

  return true;
}

void VIOManager::validateVisualMapManageConfig() const
{
  if (!visual_map_manage_en) return;
  auto require = [](bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
  };
  require(!visual_ref_fallback_en || visual_ref_current_select_en,
          "visual_ref_fallback_en requires visual_ref_current_select_en");
  require(!visual_point_seed_validation_en || visual_ref_lifecycle_en,
          "visual_point_seed_validation_en requires visual_ref_lifecycle_en");
  require(!visual_point_information_prune_en || visual_point_footprint_redundancy_en,
          "visual_point_information_prune_en requires visual_point_footprint_redundancy_en");
  require(!visual_point_replacement_en ||
              (visual_ref_lifecycle_en && visual_point_seed_validation_en &&
               visual_point_footprint_redundancy_en),
          "visual_point_replacement_en requires ref lifecycle, seed validation, and footprint redundancy");
  require(visual_ref_max_candidates >= 1 && visual_ref_max_candidates <= 4,
          "visual_ref_max_candidates must be in [1, 4]");
  require(visual_ref_validate_min_tests >= 1 && visual_point_seed_min_tests >= 1 &&
              visual_ref_retire_reject_count >= 1 && visual_point_suspect_reject_count >= 1,
          "visual map validation and rejection counts must be positive");
  require(visual_ref_max_count >= 1,
          "visual_ref_max_count must be positive");
  require(visual_ref_validate_min_ratio >= 0.0 && visual_ref_validate_min_ratio <= 1.0 &&
              visual_point_seed_min_ratio >= 0.0 && visual_point_seed_min_ratio <= 1.0,
          "visual map validation ratios must be in [0, 1]");
  require(std::isfinite(visual_ref_coverage_angle_deg) && visual_ref_coverage_angle_deg > 0.0 &&
              visual_ref_coverage_angle_deg <= 180.0,
          "visual_ref_coverage_angle_deg must be in (0, 180]");
  require(std::isfinite(visual_ref_max_anisotropy) && visual_ref_max_anisotropy >= 1.0,
          "visual_ref_max_anisotropy must be at least 1");
  require(std::isfinite(visual_ref_nis_max_per_dof) && visual_ref_nis_max_per_dof > 0.0,
          "visual_ref_nis_max_per_dof must be positive");
  require(visual_point_footprint_iou >= 0.0 && visual_point_footprint_iou <= 1.0,
          "visual_point_footprint_iou must be in [0, 1]");
  require(visual_point_information_retain > 0.0 && visual_point_information_retain <= 1.0,
          "visual_point_information_retain must be in (0, 1]");
  require(visual_point_stale_frames >= 1,
          "visual_point_stale_frames must be positive");
}

bool VIOManager::associateVisualPointSurface(
    const V3D &point_w, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
    int64_t &voxel_x, int64_t &voxel_y, int64_t &voxel_z, const VoxelPlane *&plane) const
{
  plane = nullptr;
  if (!point_w.array().isFinite().all()) return false;
  const double voxel_size = std::isfinite(visual_geom_filter_voxel_size) &&
                                    visual_geom_filter_voxel_size > 1.0e-6
                                ? visual_geom_filter_voxel_size
                                : 0.5;
  voxel_x = static_cast<int64_t>(std::floor(point_w[0] / voxel_size));
  voxel_y = static_cast<int64_t>(std::floor(point_w[1] / voxel_size));
  voxel_z = static_cast<int64_t>(std::floor(point_w[2] / voxel_size));
  const auto found = plane_map.find(VOXEL_LOCATION(voxel_x, voxel_y, voxel_z));
  if (found == plane_map.end() || found->second == nullptr) return false;
  VoxelOctoTree *octo = found->second->find_correspond(point_w);
  if (octo == nullptr || octo->plane_ptr_ == nullptr || !octo->plane_ptr_->is_plane_) return false;
  const VoxelPlane &candidate = *octo->plane_ptr_;
  if (!candidate.normal_.array().isFinite().all() || candidate.normal_.norm() <= 1.0e-9) return false;
  plane = &candidate;
  return true;
}

bool VIOManager::computeManagedFootprint(const PerCameraData &ctx, const SE3d &T_c_w,
                                         const V3D &point_w, const V3D &normal_w,
                                         const Feature *feature, const VoxelPlane &plane,
                                         std::array<V3D, 4> &corners_w) const
{
  (void)normal_w;
  if (ctx.cam == nullptr || !point_w.array().isFinite().all()) return false;
  const bool use_virtual = virtual_fisheye_patch_en;
  V2D raw_center;
  const V3D point_c = T_c_w * point_w;
  if (!projectRawFisheyeIfValid(ctx, point_c, use_virtual ? 1 : border, raw_center)) return false;

  M3D R_c_from_v = M3D::Identity();
  if (use_virtual)
  {
    if (feature != nullptr && feature->virtual_patch_valid_)
      R_c_from_v = feature->R_c_from_v_;
    else
    {
      M3D R_v_from_c;
      if (!buildVirtualFrameRotation(ctx, point_c, raw_center, R_v_from_c, R_c_from_v)) return false;
    }
  }

  const V3D camera_w = T_c_w.inverse().translation();
  const M3D R_w_from_c = T_c_w.rotationMatrix().transpose();
  V3D n = plane.normal_.normalized();
  const double d = -n.dot(plane.center_);
  const double half = static_cast<double>(patch_size_half);
  const std::array<V2D, 4> offsets = {V2D(-half, -half), V2D(half, -half),
                                      V2D(half, half), V2D(-half, half)};
  for (size_t i = 0; i < offsets.size(); ++i)
  {
    V3D ray_c;
    if (use_virtual)
    {
      const V2D virtual_px(virtual_support_radius + offsets[i][0],
                           virtual_support_radius + offsets[i][1]);
      ray_c = R_c_from_v * virtualCam2World(virtual_px);
    }
    else
    {
      ray_c = ctx.cam->cam2world(raw_center + offsets[i]);
    }
    if (!ray_c.array().isFinite().all() || ray_c.norm() <= 1.0e-9) return false;
    const V3D ray_w = (R_w_from_c * ray_c.normalized()).normalized();
    const double denominator = n.dot(ray_w);
    if (!std::isfinite(denominator) || std::fabs(denominator) <= 1.0e-8) return false;
    const double distance = -(n.dot(camera_w) + d) / denominator;
    if (!std::isfinite(distance) || distance <= 0.0) return false;
    corners_w[i] = camera_w + distance * ray_w;
  }
  return true;
}

double VIOManager::managedFootprintIoU(const std::array<V3D, 4> &a,
                                       const std::array<V3D, 4> &b,
                                       const VoxelPlane &plane) const
{
  V3D x = plane.x_normal_.normalized();
  V3D y = plane.y_normal_.normalized();
  if (!x.array().isFinite().all() || !y.array().isFinite().all() ||
      x.norm() <= 1.0e-9 || y.norm() <= 1.0e-9)
    return 0.0;
  auto project = [&](const std::array<V3D, 4> &corners) {
    std::vector<V2D> polygon;
    polygon.reserve(4);
    for (const V3D &corner : corners)
    {
      const V3D delta = corner - plane.center_;
      polygon.emplace_back(delta.dot(x), delta.dot(y));
    }
    double signed_area = 0.0;
    for (int i = 0; i < 4; ++i)
      signed_area += polygon[i][0] * polygon[(i + 1) % 4][1] -
                     polygon[(i + 1) % 4][0] * polygon[i][1];
    if (signed_area < 0.0) std::reverse(polygon.begin(), polygon.end());
    return polygon;
  };
  auto area = [](const std::vector<V2D> &polygon) {
    if (polygon.size() < 3) return 0.0;
    double value = 0.0;
    for (size_t i = 0; i < polygon.size(); ++i)
      value += polygon[i][0] * polygon[(i + 1) % polygon.size()][1] -
               polygon[(i + 1) % polygon.size()][0] * polygon[i][1];
    return 0.5 * std::fabs(value);
  };
  std::vector<V2D> subject = project(a);
  const std::vector<V2D> clip = project(b);
  for (size_t edge = 0; edge < clip.size() && !subject.empty(); ++edge)
  {
    const V2D c0 = clip[edge];
    const V2D c1 = clip[(edge + 1) % clip.size()];
    auto inside = [&](const V2D &p) {
      return (c1[0] - c0[0]) * (p[1] - c0[1]) -
                 (c1[1] - c0[1]) * (p[0] - c0[0]) >= -1.0e-10;
    };
    auto intersection = [&](const V2D &p0, const V2D &p1) -> V2D {
      const V2D r = p1 - p0;
      const V2D s = c1 - c0;
      const double denominator = r[0] * s[1] - r[1] * s[0];
      if (std::fabs(denominator) <= 1.0e-12) return p1;
      const V2D delta = c0 - p0;
      const double t = (delta[0] * s[1] - delta[1] * s[0]) / denominator;
      return (p0 + std::clamp(t, 0.0, 1.0) * r).eval();
    };
    std::vector<V2D> output;
    V2D previous = subject.back();
    bool previous_inside = inside(previous);
    for (const V2D &current : subject)
    {
      const bool current_inside = inside(current);
      if (current_inside != previous_inside) output.push_back(intersection(previous, current));
      if (current_inside) output.push_back(current);
      previous = current;
      previous_inside = current_inside;
    }
    subject.swap(output);
  }
  const double area_a = area(project(a));
  const double area_b = area(project(b));
  const double intersection_area = area(subject);
  const double union_area = area_a + area_b - intersection_area;
  return union_area > 1.0e-12 ? std::clamp(intersection_area / union_area, 0.0, 1.0) : 0.0;
}

VisualPoint *VIOManager::findRedundantVisualPoint(
    const PendingNewPointObservation &pending,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
    double &best_iou)
{
  best_iou = 0.0;
  if (!visual_map_manage_en || !visual_point_footprint_redundancy_en ||
      !pending.surface_valid || !pending.footprint_valid)
    return nullptr;
  const auto surface = plane_map.find(VOXEL_LOCATION(pending.surface_voxel_x,
                                                     pending.surface_voxel_y,
                                                     pending.surface_voxel_z));
  if (surface == plane_map.end() || surface->second == nullptr) return nullptr;
  VoxelOctoTree *octo = surface->second->find_correspond(pending.pt_var.point_w);
  if (octo == nullptr || octo->plane_ptr_ == nullptr || !octo->plane_ptr_->is_plane_ ||
      octo->plane_ptr_->id_ != pending.surface_plane_id)
    return nullptr;
  const VoxelPlane &plane = *octo->plane_ptr_;

  const double map_voxel_size = 0.5;
  const int64_t center_x = static_cast<int64_t>(std::floor(pending.pt_var.point_w[0] / map_voxel_size));
  const int64_t center_y = static_cast<int64_t>(std::floor(pending.pt_var.point_w[1] / map_voxel_size));
  const int64_t center_z = static_cast<int64_t>(std::floor(pending.pt_var.point_w[2] / map_voxel_size));
  VisualPoint *best = nullptr;
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dz = -1; dz <= 1; ++dz)
      {
        const auto found = feat_map.find(VOXEL_LOCATION(center_x + dx, center_y + dy, center_z + dz));
        if (found == feat_map.end() || found->second == nullptr) continue;
        for (VisualPoint *point : found->second->voxel_points)
        {
          if (point == nullptr || point->pending_delete_ || point->surface_plane_id_ != pending.surface_plane_id) continue;
          for (Feature *feature : point->obs_)
          {
            if (feature == nullptr || feature->pending_delete_ || !feature->footprint_valid_ ||
                feature->surface_plane_id_ != pending.surface_plane_id)
              continue;
            ++visual_map_manage_stats_.footprint_compared;
            const double iou = managedFootprintIoU(pending.footprint_corners_w,
                                                   feature->footprint_corners_w_, plane);
            if (iou > best_iou)
            {
              best_iou = iou;
              best = point;
            }
          }
        }
      }
  return best_iou >= visual_point_footprint_iou ? best : nullptr;
}

bool VIOManager::shouldReplaceRedundantPoint(const PendingNewPointObservation &pending,
                                             const VisualPoint &existing) const
{
  if (existing.pending_delete_) return false;
  const bool candidate_geometry_valid = std::isfinite(pending.geometry_chi2) &&
      (!(std::isfinite(visual_geom_filter_max_chi2) && visual_geom_filter_max_chi2 > 0.0) ||
       pending.geometry_chi2 <= visual_geom_filter_max_chi2);
  const bool candidate_geometry_better = candidate_geometry_valid &&
      (!std::isfinite(existing.geometry_chi2_) || pending.geometry_chi2 < existing.geometry_chi2_);
  if (visual_point_replacement_en && existing.state_ == VisualPoint::State::SUSPECT)
    return candidate_geometry_better;
  if (!visual_point_information_prune_en) return false;
  Eigen::Matrix<double, 6, 6> information =
      existing.accumulated_pose_information_ / std::max(1, existing.accepted_test_count_);
  information.diagonal().array() += 1.0e-9;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(information);
  if (solver.info() != Eigen::Success) return true;
  const double max_eigenvalue = solver.eigenvalues().maxCoeff();
  const double min_eigenvalue = solver.eigenvalues().minCoeff();
  const double isotropic_retention = max_eigenvalue > 1.0e-12 ? min_eigenvalue / max_eigenvalue : 0.0;
  const double candidate_geometry = std::isfinite(pending.geometry_chi2)
                                        ? 1.0 / (1.0 + std::max(0.0, pending.geometry_chi2))
                                        : 0.0;
  return isotropic_retention < 1.0 - visual_point_information_retain &&
         candidate_geometry > 0.0 && candidate_geometry_better;
}
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

bool VIOManager::sampleRawCorePatchForDump(const PerCameraData &ctx, const cv::Mat &img, const V2D &pc,
                                           int scale, std::vector<float> &patch) const
{
  patch.clear();
  if (scale <= 0 || patch_size <= 0 || patch_size_total <= 0 || img.empty() || img.type() != CV_8UC1 ||
      !pc.array().isFinite().all())
    return false;

  const int required_border = (patch_size_half + 1) * scale + 1;
  if (ctx.cam != nullptr && !ctx.cam->isInFrame(pc.cast<int>(), required_border)) return false;

  const int u_i = static_cast<int>(std::floor(pc[0] / scale)) * scale;
  const int v_i = static_cast<int>(std::floor(pc[1] / scale)) * scale;
  const double subpix_u = (pc[0] - u_i) / scale;
  const double subpix_v = (pc[1] - v_i) / scale;
  const double w_tl = (1.0 - subpix_u) * (1.0 - subpix_v);
  const double w_tr = subpix_u * (1.0 - subpix_v);
  const double w_bl = (1.0 - subpix_u) * subpix_v;
  const double w_br = subpix_u * subpix_v;

  patch.assign(patch_size_total, 0.0f);
  for (int row_index = 0; row_index < patch_size; ++row_index)
  {
    const int row = v_i - patch_size_half * scale + row_index * scale;
    const int row_next = row + scale;
    if (row < 0 || row_next < 0 || row >= img.rows || row_next >= img.rows) return false;
    const uint8_t *row0 = img.ptr<uint8_t>(row);
    const uint8_t *row1 = img.ptr<uint8_t>(row_next);
    for (int col_index = 0; col_index < patch_size; ++col_index)
    {
      const int col = u_i - patch_size_half * scale + col_index * scale;
      const int col_next = col + scale;
      if (col < 0 || col_next < 0 || col >= img.cols || col_next >= img.cols) return false;
      const double value = w_tl * row0[col] + w_tr * row0[col_next] +
                           w_bl * row1[col] + w_br * row1[col_next];
      patch[row_index * patch_size + col_index] = static_cast<float>(value);
    }
  }
  return true;
}
SE3d VIOManager::composeVirtualPose(const M3D &R_v_from_c, const SE3d &T_c_w) const
{
  // {}^V T_W = {}^V T_C * {}^C T_W. The virtual and raw cameras share the optical center.
  return SE3d(R_v_from_c * T_c_w.rotationMatrix(), R_v_from_c * T_c_w.translation());
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
  if (ctx.cam == nullptr || ctx.width <= 0 || ctx.height <= 0 ||
      !ray_or_point_in_raw_camera.array().isFinite().all() ||
      ray_or_point_in_raw_camera.norm() <= virtual_min_z)
    return false;
  const int border_req = std::max(0, required_border);
  raw_px = ctx.cam->world2cam(ray_or_point_in_raw_camera);
  if (!raw_px.array().isFinite().all()) return false;
  if (raw_px[0] < border_req || raw_px[1] < border_req || raw_px[0] >= ctx.width - border_req - 1 ||
      raw_px[1] >= ctx.height - border_req - 1)
    return false;
  return ctx.cam->isInFrame(raw_px.cast<int>(), border_req);
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
    return interpolateRawVirtualImage(raw_img, u, v, sampled);
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
  const int required_border = scale + interpolationBorderMargin(virtual_interp_mode_enum);
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
      if (!projectRawFisheyeIfValid(ctx, ray_c, interpolationBorderMargin(virtual_interp_mode_enum), raw_px)) continue;
      const double local_u = raw_px[0] - raw_origin.x;
      const double local_v = raw_px[1] - raw_origin.y;
      float sampled = 0.0f;
      if (!interpolateRawVirtualImage(raw_img, local_u, local_v, sampled)) continue;
      values[x] = sampled;
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

  const int interp_margin = interpolationBorderMargin(virtual_interp_mode_enum);
  const int x0 = std::max(0, static_cast<int>(std::floor(min_u)) - interp_margin);
  const int y0 = std::max(0, static_cast<int>(std::floor(min_v)) - interp_margin);
  const int x1 = std::min(raw_img.cols, static_cast<int>(std::ceil(max_u)) + interp_margin + 1);
  const int y1 = std::min(raw_img.rows, static_cast<int>(std::ceil(max_v)) + interp_margin + 1);
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

bool VIOManager::interpolateRawVirtualImage(const cv::Mat &img, double u, double v, float &value) const
{
  value = 0.0f;
  if (!std::isfinite(u) || !std::isfinite(v) || img.empty() || img.type() != CV_8UC1) return false;

  const int x = static_cast<int>(std::floor(u));
  const int y = static_cast<int>(std::floor(v));
  if (virtual_interp_mode_enum == VirtualInterpMode::BILINEAR)
  {
    if (x < 0 || y < 0 || x + 1 >= img.cols || y + 1 >= img.rows) return false;
    value = static_cast<float>(vk::interpolateMat_8u(img, u, v));
    return std::isfinite(value);
  }

  const int radius = interpolationKernelRadius(virtual_interp_mode_enum);
  const int start_x = x - (radius - 1);
  const int end_x = x + radius;
  const int start_y = y - (radius - 1);
  const int end_y = y + radius;
  if (start_x < 0 || start_y < 0 || end_x >= img.cols || end_y >= img.rows) return false;

  const bool use_bicubic = virtual_interp_mode_enum == VirtualInterpMode::BICUBIC;
  double weighted_sum = 0.0;
  double weight_sum = 0.0;
  for (int yy = start_y; yy <= end_y; ++yy)
  {
    const double wy = use_bicubic ? cubicInterpWeight(v - yy) : lanczosInterpWeight(v - yy);
    const uint8_t *row = img.ptr<uint8_t>(yy);
    for (int xx = start_x; xx <= end_x; ++xx)
    {
      const double wx = use_bicubic ? cubicInterpWeight(u - xx) : lanczosInterpWeight(u - xx);
      const double w = wx * wy;
      weighted_sum += w * static_cast<double>(row[xx]);
      weight_sum += w;
    }
  }

  if (std::fabs(weight_sum) < 1.0e-12) return false;
  const double sampled = weighted_sum / weight_sum;
  if (!std::isfinite(sampled)) return false;
  value = clampInterpValue(sampled);
  return true;
}

bool VIOManager::interpolateFloatVirtualImage(const cv::Mat &img, const cv::Mat *valid_mask, double u, double v, float &value) const
{
  value = 0.0f;
  if (!std::isfinite(u) || !std::isfinite(v) || img.empty() || img.type() != CV_32FC1) return false;
  if (valid_mask != nullptr && (valid_mask->empty() || valid_mask->type() != CV_8UC1 ||
                                valid_mask->rows != img.rows || valid_mask->cols != img.cols))
    return false;

  auto read_sample = [&](int yy, int xx, double &sample) -> bool {
    if (valid_mask != nullptr && valid_mask->ptr<uint8_t>(yy)[xx] == 0) return false;
    const float value_at_pixel = img.ptr<float>(yy)[xx];
    if (!std::isfinite(value_at_pixel)) return false;
    sample = static_cast<double>(value_at_pixel);
    return true;
  };

  const int x = static_cast<int>(std::floor(u));
  const int y = static_cast<int>(std::floor(v));
  if (virtual_interp_mode_enum == VirtualInterpMode::BILINEAR)
  {
    if (x < 0 || y < 0 || x + 1 >= img.cols || y + 1 >= img.rows) return false;
    double tl = 0.0, tr = 0.0, bl = 0.0, br = 0.0;
    if (!read_sample(y, x, tl) || !read_sample(y, x + 1, tr) ||
        !read_sample(y + 1, x, bl) || !read_sample(y + 1, x + 1, br))
      return false;
    const double dx = u - x;
    const double dy = v - y;
    value = clampInterpValue((1.0 - dy) * ((1.0 - dx) * tl + dx * tr) +
                             dy * ((1.0 - dx) * bl + dx * br));
    return true;
  }

  const int radius = interpolationKernelRadius(virtual_interp_mode_enum);
  const int start_x = x - (radius - 1);
  const int end_x = x + radius;
  const int start_y = y - (radius - 1);
  const int end_y = y + radius;
  if (start_x < 0 || start_y < 0 || end_x >= img.cols || end_y >= img.rows) return false;

  const bool use_bicubic = virtual_interp_mode_enum == VirtualInterpMode::BICUBIC;
  double weighted_sum = 0.0;
  double weight_sum = 0.0;
  for (int yy = start_y; yy <= end_y; ++yy)
  {
    const double wy = use_bicubic ? cubicInterpWeight(v - yy) : lanczosInterpWeight(v - yy);
    for (int xx = start_x; xx <= end_x; ++xx)
    {
      double sample = 0.0;
      if (!read_sample(yy, xx, sample)) return false;
      const double wx = use_bicubic ? cubicInterpWeight(u - xx) : lanczosInterpWeight(u - xx);
      const double w = wx * wy;
      weighted_sum += w * sample;
      weight_sum += w;
    }
  }

  if (std::fabs(weight_sum) < 1.0e-12) return false;
  const double sampled = weighted_sum / weight_sum;
  if (!std::isfinite(sampled)) return false;
  value = clampInterpValue(sampled);
  return true;
}

bool VIOManager::interpolateVirtualFloat(const cv::Mat &img, const cv::Mat &valid_mask, float u, float v, float &value) const
{
  return interpolateFloatVirtualImage(img, &valid_mask, u, v, value);
}

bool VIOManager::interpolateStoredVirtualImage(const cv::Mat &img, float u, float v, float &value) const
{
  return interpolateFloatVirtualImage(img, nullptr, u, v, value);
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
  if (!projectRawFisheyeIfValid(ctx, ray_c, interpolationBorderMargin(virtual_interp_mode_enum), raw_px)) return false;
  return interpolateRawVirtualImage(raw_img, raw_px[0], raw_px[1], value);
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
                                            const SE3d &T_c_w, const V3D &point_w,
                                            float *core_patch, cv::Mat &virtual_support_img,
                                            cv::Point &virtual_source_origin, SE3d &T_v_w,
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

  const SE3d T_v_w = composeVirtualPose(R_v_from_c, ctx.new_frame->T_f_w_);
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
                                               const V3D &point_c, const SE3d &T_v_w,
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
    const SE3d T_canchor_csource =
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
  const SE3d T_vanchor_vsource = ref_patch_dump_probe_.anchor_T_v_w * T_v_w.inverse();
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

void VIOManager::initializeRuntimeSupportDump()
{
  if (runtime_support_dump_initialized_) return;
  runtime_support_dump_effective_folder_ = sanitizeRuntimeSupportDumpFolder(runtime_support_dump_folder);
  const std::filesystem::path root = std::filesystem::path(ROOT_DIR) / "Log" / "result" / "patchs" / runtime_support_dump_effective_folder_;
  std::filesystem::create_directories(root / "updates");

  const std::filesystem::path index_path = root / "index.csv";
  if (!std::filesystem::exists(index_path) || std::filesystem::file_size(index_path) == 0)
  {
    std::ofstream index(index_path);
    index << "event,frame_id,timestamp,camera_id,point_dump_id,track_count,submap_index,ref_frame_id,ref_camera_id,ref_timestamp,search_level,error,ncc,raw_u,raw_v,ref_support,cur_support,cur_mask,warped_ref,current_core\n";
  }

  std::ofstream selection(root / "selection.txt");
  selection << "This folder is written by debug.runtime_support_dump_en.\n"
            << "It stores accepted runtime support evidence from the raw or virtual optimization path, not the ref_patch_dump probe path.\n"
            << "folder=" << runtime_support_dump_effective_folder_ << "\n";
  runtime_support_dump_initialized_ = true;
}

void VIOManager::dumpRuntimeSupportObservation(const PerCameraData &ctx, const VisualPoint &point, const Feature &ref_ftr,
                                               const V2D &current_raw_center_px, const VirtualTrackPatch &track,
                                               const std::vector<float> &warped_reference,
                                               const std::vector<float> &current_core, int track_count,
                                               int submap_index, double error, double ncc)
{
  if (!runtime_support_dump_en || ctx.new_frame == nullptr) return;
  initializeRuntimeSupportDump();

  const std::filesystem::path root = std::filesystem::path(ROOT_DIR) / "Log" / "result" / "patchs" / runtime_support_dump_effective_folder_;
  const std::filesystem::path updates_dir = root / "updates";
  std::ostringstream base_name;
  base_name << "frame_" << std::setw(6) << std::setfill('0') << ctx.new_frame->id_
            << "_point_" << std::setw(4) << std::setfill('0') << point.runtime_support_dump_id_
            << "_count_" << std::setw(4) << std::setfill('0') << track_count;
  const std::string base = base_name.str();

  const std::filesystem::path ref_support_path = updates_dir / (base + "_ref_support.png");
  const std::filesystem::path cur_support_path = updates_dir / (base + "_cur_support.png");
  const std::filesystem::path warped_ref_path = updates_dir / (base + "_warped_ref_l0.png");
  const std::filesystem::path current_core_path = updates_dir / (base + "_current_core_l0.png");

  const V2D ref_support_center(virtual_support_radius, virtual_support_radius);
  const V3D point_vcur = track.T_vcur_w_seed * point.pos_;
  const V2D cur_support_center = virtualProject(point_vcur);
  const cv::Mat ref_support_display = makeMarkedFloatMatDisplay(ref_ftr.img_, cv::Mat(), ref_support_center);
  const cv::Mat cur_support_display =
      makeMarkedFloatMatDisplay(track.cur_support.values, track.cur_support.valid_mask, cur_support_center);
  const cv::Mat warped_ref_display = makeFloatPatchDisplay(warped_reference, patch_size, 0);
  const cv::Mat current_core_display = makeFloatPatchDisplay(current_core, patch_size, 0);

  const bool ref_saved = writeImageIfAvailable(ref_support_path, ref_support_display);
  const bool cur_saved = writeImageIfAvailable(cur_support_path, cur_support_display);
  const bool warped_saved = writeImageIfAvailable(warped_ref_path, warped_ref_display);
  const bool core_saved = writeImageIfAvailable(current_core_path, current_core_display);

  if (ref_saved) writeImageIfAvailable(root / "best_ref_support.png", ref_support_display);
  if (cur_saved) writeImageIfAvailable(root / "best_cur_support.png", cur_support_display);
  if (warped_saved) writeImageIfAvailable(root / "best_warped_ref_l0.png", warped_ref_display);
  if (core_saved) writeImageIfAvailable(root / "best_current_core_l0.png", current_core_display);

  std::ofstream index(root / "index.csv", std::ios::app);
  index << "best_update," << ctx.new_frame->id_ << ','
        << std::fixed << std::setprecision(9) << ctx.new_frame->timestamp_ << ','
        << ctx.camera_id << ',' << point.runtime_support_dump_id_ << ',' << track_count << ','
        << submap_index << ',' << ref_ftr.id_ << ',' << ref_ftr.camera_id_ << ','
        << std::setprecision(9) << ref_ftr.timestamp_ << ',' << track.search_level << ','
        << std::setprecision(6) << error << ',' << ncc << ','
        << current_raw_center_px[0] << ',' << current_raw_center_px[1] << ','
        << (ref_saved ? std::filesystem::relative(ref_support_path, root).generic_string() : "") << ','
        << (cur_saved ? std::filesystem::relative(cur_support_path, root).generic_string() : "") << ','
        << "" << ','
        << (warped_saved ? std::filesystem::relative(warped_ref_path, root).generic_string() : "") << ','
        << (core_saved ? std::filesystem::relative(current_core_path, root).generic_string() : "") << '\n';

  printf("[ VIO Runtime Support Dump ] best point=%d tracks=%d frame=%d ref_saved=%d cur_saved=%d core_saved=%d folder=%s\n",
         point.runtime_support_dump_id_, track_count, ctx.new_frame->id_, ref_saved ? 1 : 0,
         cur_saved ? 1 : 0, core_saved ? 1 : 0, runtime_support_dump_effective_folder_.c_str());
}

void VIOManager::dumpRuntimeSupportRawObservation(const PerCameraData &ctx, const VisualPoint &point, const Feature &ref_ftr,
                                                   const V2D &current_raw_center_px, int search_level,
                                                   const std::vector<float> &warped_reference,
                                                   const std::vector<float> &current_core, int track_count,
                                                   int submap_index, double error, double ncc)
{
  if (!runtime_support_dump_en || ctx.new_frame == nullptr) return;
  initializeRuntimeSupportDump();

  const std::filesystem::path root = std::filesystem::path(ROOT_DIR) / "Log" / "result" / "patchs" / runtime_support_dump_effective_folder_;
  const std::filesystem::path updates_dir = root / "updates";
  std::ostringstream base_name;
  base_name << "frame_" << std::setw(6) << std::setfill('0') << ctx.new_frame->id_
            << "_point_" << std::setw(4) << std::setfill('0') << point.runtime_support_dump_id_
            << "_count_" << std::setw(4) << std::setfill('0') << track_count;
  const std::string base = base_name.str();

  const std::filesystem::path ref_support_path = updates_dir / (base + "_ref_support.png");
  const std::filesystem::path cur_support_path = updates_dir / (base + "_cur_support.png");
  const std::filesystem::path warped_ref_path = updates_dir / (base + "_warped_ref_l0.png");
  const std::filesystem::path current_core_path = updates_dir / (base + "_current_core_l0.png");

  const cv::Mat ref_support_display = makeMarkedRawSupportDisplay(ref_ftr.img_, ref_ftr.px_, kRuntimeRawSupportDumpSize);
  const cv::Mat cur_support_display = makeMarkedRawSupportDisplay(ctx.new_frame->img_, current_raw_center_px,
                                                                   kRuntimeRawSupportDumpSize);
  const cv::Mat warped_ref_display = makeFloatPatchDisplay(warped_reference, patch_size, 0);
  const cv::Mat current_core_display = makeFloatPatchDisplay(current_core, patch_size, 0);

  const bool ref_saved = writeImageIfAvailable(ref_support_path, ref_support_display);
  const bool cur_saved = writeImageIfAvailable(cur_support_path, cur_support_display);
  const bool warped_saved = writeImageIfAvailable(warped_ref_path, warped_ref_display);
  const bool core_saved = writeImageIfAvailable(current_core_path, current_core_display);

  if (ref_saved) writeImageIfAvailable(root / "best_ref_support.png", ref_support_display);
  if (cur_saved) writeImageIfAvailable(root / "best_cur_support.png", cur_support_display);
  if (warped_saved) writeImageIfAvailable(root / "best_warped_ref_l0.png", warped_ref_display);
  if (core_saved) writeImageIfAvailable(root / "best_current_core_l0.png", current_core_display);

  std::ofstream index(root / "index.csv", std::ios::app);
  index << "best_update_raw," << ctx.new_frame->id_ << ','
        << std::fixed << std::setprecision(9) << ctx.new_frame->timestamp_ << ','
        << ctx.camera_id << ',' << point.runtime_support_dump_id_ << ',' << track_count << ','
        << submap_index << ',' << ref_ftr.id_ << ',' << ref_ftr.camera_id_ << ','
        << std::setprecision(9) << ref_ftr.timestamp_ << ',' << search_level << ','
        << std::setprecision(6) << error << ',' << ncc << ','
        << current_raw_center_px[0] << ',' << current_raw_center_px[1] << ','
        << (ref_saved ? std::filesystem::relative(ref_support_path, root).generic_string() : "") << ','
        << (cur_saved ? std::filesystem::relative(cur_support_path, root).generic_string() : "") << ','
        << ','
        << (warped_saved ? std::filesystem::relative(warped_ref_path, root).generic_string() : "") << ','
        << (core_saved ? std::filesystem::relative(current_core_path, root).generic_string() : "") << '\n';

  printf("[ VIO Runtime Support Dump ] raw best point=%d tracks=%d frame=%d ref_saved=%d cur_saved=%d core_saved=%d folder=%s\n",
         point.runtime_support_dump_id_, track_count, ctx.new_frame->id_, ref_saved ? 1 : 0,
         cur_saved ? 1 : 0, core_saved ? 1 : 0, runtime_support_dump_effective_folder_.c_str());
}

bool VIOManager::getWarpMatrixAffineVirtual(const V3D &xyz_ref, const SE3d &T_vcur_vref, int level_ref, int pyramid_level,
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

bool VIOManager::getWarpMatrixAffineHomographyVirtual(const V3D &xyz_ref, const V3D &normal_ref, const SE3d &T_vcur_vref,
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
  const bool apply_map_management = visual_map_manage_en && !visual_map_manage_shadow_en;
  int64_t location[3];
  for (int j = 0; j < 3; ++j)
  {
    if (apply_map_management)
      location[j] = static_cast<int64_t>(std::floor(pt_w[j] / voxel_size));
    else
    {
      float legacy_location = static_cast<float>(pt_w[j] / voxel_size);
      if (legacy_location < 0.0f) legacy_location -= 1.0f;
      location[j] = static_cast<int64_t>(legacy_location);
    }
  }
  VOXEL_LOCATION position(location[0], location[1], location[2]);
  if (visual_map_manage_en)
  {
    pt_new->map_voxel_x_ = position.x;
    pt_new->map_voxel_y_ = position.y;
    pt_new->map_voxel_z_ = position.z;
  }
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    VOXEL_POINTS *ot = new VOXEL_POINTS(apply_map_management ? 1 : 0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(const PerCameraData &ref_ctx, const PerCameraData &cur_ctx, const V2D &px_ref,
                                               const V3D &xyz_ref, const V3D &normal_ref, const SE3d &T_cur_ref,
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
                                     const Vector3d &f_ref, const double depth_ref, const SE3d &T_cur_ref,
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
    // printf("[ VIO Debug ] warpAffine reject det=%.6e finite=%d px=(%.2f,%.2f) img=%dx%d type=%d level_ref=%d search=%d pyramid=%d half=%d patch_null=%d\n",
    //        debug_affine_det, A_cur_ref.array().isFinite().all() ? 1 : 0, px_ref[0], px_ref[1], img_ref.cols, img_ref.rows, img_ref.type(),
    //        level_ref, search_level, pyramid_level, halfpatch_size, patch == nullptr ? 1 : 0);
    // fflush(stdout);
    return false;
  }

  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (!A_ref_cur.array().isFinite().all())
  {
    // printf("[ VIO Debug ] warpAffine reject inverse_nonfinite det=%.6e\n", debug_affine_det);
    // fflush(stdout);
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

int VIOManager::usageRegionBin(const PerCameraData &ctx, const V2D &px) const
{
  if (ctx.width <= 1 || ctx.height <= 1 || !px.array().isFinite().all()) return -1;
  const double cx = 0.5 * static_cast<double>(ctx.width - 1);
  const double cy = 0.5 * static_cast<double>(ctx.height - 1);
  const double max_radius = std::sqrt(cx * cx + cy * cy);
  if (max_radius <= 1.0e-9) return -1;
  const double radius = std::sqrt((px[0] - cx) * (px[0] - cx) + (px[1] - cy) * (px[1] - cy)) / max_radius;
  if (radius < 0.30) return 0;
  if (radius < 0.60) return 1;
  if (radius < 0.80) return 2;
  return 3;
}

int VIOManager::usageViewAngleBin(const PerCameraData &ctx, const Feature &ref_ftr, const VisualPoint &pt) const
{
  if (ctx.new_frame == nullptr) return -1;
  const V3D ref_view = ref_ftr.pos() - pt.pos_;
  const V3D cur_view = ctx.new_frame->pos() - pt.pos_;
  const double ref_norm = ref_view.norm();
  const double cur_norm = cur_view.norm();
  if (!std::isfinite(ref_norm) || !std::isfinite(cur_norm) || ref_norm <= 1.0e-9 || cur_norm <= 1.0e-9) return -1;
  double cos_angle = ref_view.dot(cur_view) / (ref_norm * cur_norm);
  cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
  const double angle_deg = std::acos(cos_angle) * kRadiansToDegrees;
  if (angle_deg < 5.0) return 0;
  if (angle_deg < 15.0) return 1;
  if (angle_deg < 30.0) return 2;
  if (angle_deg < 60.0) return 3;
  return 4;
}

int VIOManager::usageFootprintBin(const Matrix2d &A_cur_ref) const
{
  const double det = std::fabs(A_cur_ref.determinant());
  if (!A_cur_ref.array().isFinite().all() || !std::isfinite(det) || det <= 1.0e-12) return 0;
  const double ratio = std::max(det, 1.0 / det);
  if (ratio < 2.0) return 1;
  if (ratio < 4.0) return 2;
  if (ratio < 8.0) return 3;
  return 4;
}

int VIOManager::usageAnisotropyBin(const Matrix2d &A_cur_ref) const
{
  if (!A_cur_ref.array().isFinite().all()) return -1;
  Eigen::JacobiSVD<Matrix2d> svd(A_cur_ref);
  const V2D singular = svd.singularValues();
  const double min_sv = std::min(singular[0], singular[1]);
  const double max_sv = std::max(singular[0], singular[1]);
  if (!std::isfinite(min_sv) || !std::isfinite(max_sv) || min_sv <= 1.0e-12) return -1;
  const double ratio = max_sv / min_sv;
  if (ratio < 1.5) return 0;
  if (ratio < 2.0) return 1;
  if (ratio < 4.0) return 2;
  return 3;
}

int VIOManager::usageNccBin(double ncc) const
{
  if (!std::isfinite(ncc)) return -1;
  const double clamped = std::max(-1.0, std::min(1.0, ncc));
  if (clamped >= 1.0) return kUsageNccBinCount - 1;
  return std::max(0, std::min(kUsageNccBinCount - 1,
                              static_cast<int>(std::floor((clamped + 1.0) * 10.0))));
}

int VIOManager::usageRmsBin(double rms) const
{
  if (!std::isfinite(rms) || rms < 0.0) return -1;
  static const double upper_bounds[kUsageRmsBinCount - 1] = {2.0, 4.0, 6.0, 8.0, 10.0,
                                                             15.0, 20.0, 30.0, 50.0};
  for (int i = 0; i < kUsageRmsBinCount - 1; ++i)
    if (rms < upper_bounds[i]) return i;
  return kUsageRmsBinCount - 1;
}

void VIOManager::resetUsageStatsWindow()
{
  usage_stats_frames_ = 0;
  usage_camera_pairs_.assign(std::max(0, numCameras() * numCameras()), UsageStatsCell());
  if (cross_camera_current_residual_en)
  {
    usage_current_camera_pairs_.assign(std::max(0, numCameras() * numCameras()), UsageStatsCell());
    for (UsageStatsCell &cell : usage_current_view_angle_bins_) cell.reset();
    for (UsageStatsCell &cell : usage_current_footprint_bins_) cell.reset();
    for (UsageStatsCell &cell : usage_current_anisotropy_bins_) cell.reset();
  }
  for (UsageStatsCell &cell : usage_cross_region_pairs_) cell.reset();
  for (UsageStatsCell &cell : usage_view_angle_bins_) cell.reset();
  for (UsageStatsCell &cell : usage_footprint_bins_) cell.reset();
  for (UsageStatsCell &cell : usage_anisotropy_bins_) cell.reset();
  usage_pose_all_.reset();
  usage_pose_same_.reset();
  usage_pose_cross_.reset();
  if (cross_camera_current_residual_en) usage_pose_current_cross_.reset();
}

void VIOManager::resetUsageStatsTotals()
{
  usage_stats_total_frames_ = 0;
  usage_total_camera_pairs_.assign(std::max(0, numCameras() * numCameras()), UsageStatsCell());
  if (cross_camera_current_residual_en)
  {
    usage_total_current_camera_pairs_.assign(std::max(0, numCameras() * numCameras()), UsageStatsCell());
    for (UsageStatsCell &cell : usage_total_current_view_angle_bins_) cell.reset();
    for (UsageStatsCell &cell : usage_total_current_footprint_bins_) cell.reset();
    for (UsageStatsCell &cell : usage_total_current_anisotropy_bins_) cell.reset();
  }
  for (UsageStatsCell &cell : usage_total_cross_region_pairs_) cell.reset();
  for (UsageStatsCell &cell : usage_total_view_angle_bins_) cell.reset();
  for (UsageStatsCell &cell : usage_total_footprint_bins_) cell.reset();
  for (UsageStatsCell &cell : usage_total_anisotropy_bins_) cell.reset();
  usage_total_pose_all_.reset();
  usage_total_pose_same_.reset();
  usage_total_pose_cross_.reset();
  if (cross_camera_current_residual_en) usage_total_pose_current_cross_.reset();
}

void VIOManager::recordCurrentCrossCameraUsage(const CurrentCrossCameraPair &pair, int stage,
                                               int pyramid_level, int residuals)
{
  if (!usage_stats_en) return;
  const int camera_count = numCameras();
  if (pair.source_camera_id < 0 || pair.target_camera_id < 0 ||
      pair.source_camera_id >= camera_count || pair.target_camera_id >= camera_count ||
      pair.source_camera_id == pair.target_camera_id)
    return;
  const size_t expected = static_cast<size_t>(camera_count * camera_count);
  if (usage_current_camera_pairs_.size() != expected || usage_total_current_camera_pairs_.size() != expected)
  {
    resetUsageStatsWindow();
    resetUsageStatsTotals();
  }
  const int level_count = std::max(1, patch_pyrimid_level);
  const int pair_index = pair.source_camera_id * camera_count + pair.target_camera_id;
  auto updateCell = [&](UsageStatsCell &cell) {
    cell.ensureLevels(level_count);
    if (stage == kUsageEkfStage)
    {
      if (pyramid_level < 0 || pyramid_level >= level_count || residuals <= 0) return;
      ++cell.ekf_patches;
      cell.ekf_residuals += residuals;
      UsageMetricStats &metric = cell.levels[pyramid_level].stages[kUsageEkfStage];
      const bool valid = pyramid_level < static_cast<int>(pair.level_active.size()) && pair.level_active[pyramid_level] != 0;
      const double ncc = pyramid_level < static_cast<int>(pair.ncc_levels.size())
                             ? pair.ncc_levels[pyramid_level] : std::numeric_limits<double>::quiet_NaN();
      const double sse = pyramid_level < static_cast<int>(pair.sse_levels.size())
                             ? pair.sse_levels[pyramid_level] : std::numeric_limits<double>::quiet_NaN();
      if (valid && std::isfinite(ncc))
      {
        ++metric.ncc_count;
        metric.ncc_sum += ncc;
        if (ncc >= nccThresholdForLevel(pyramid_level)) ++metric.ncc_gate_pass_count;
        const int bin = usageNccBin(ncc);
        if (bin >= 0) ++metric.ncc_hist[bin];
      }
      if (valid && std::isfinite(sse))
      {
        metric.sse_sum += sse;
        metric.sse_samples += patch_size_total;
        ++metric.rms_count;
        const int bin = usageRmsBin(std::sqrt(std::max(0.0, sse) / std::max(1, patch_size_total)));
        if (bin >= 0) ++metric.rms_hist[bin];
      }
      return;
    }

    if (stage == kUsagePreGateStage) ++cell.candidates;
    if (stage == kUsageAcceptedStage)
    {
      ++cell.accepted;
      const int active_levels = static_cast<int>(std::count_if(
          pair.level_active.begin(), pair.level_active.end(), [](uint8_t active) { return active != 0; }));
      cell.theoretical_residuals += static_cast<long long>(patch_size_total) * active_levels;
    }
    bool all_valid = true;
    bool all_gate_pass = true;
    double ncc_min = std::numeric_limits<double>::infinity();
    double ncc_sum = 0.0;
    double threshold_sum = 0.0;
    for (int level = 0; level < level_count; ++level)
    {
      const std::vector<uint8_t> &stage_mask =
          stage == kUsagePreGateStage ? pair.level_valid : pair.level_active;
      const bool valid = level < static_cast<int>(stage_mask.size()) && stage_mask[level] != 0;
      const double ncc = level < static_cast<int>(pair.ncc_levels.size())
                             ? pair.ncc_levels[level] : std::numeric_limits<double>::quiet_NaN();
      const double sse = level < static_cast<int>(pair.sse_levels.size())
                             ? pair.sse_levels[level] : std::numeric_limits<double>::quiet_NaN();
      UsageMetricStats &metric = cell.levels[level].stages[stage];
      if (!valid || !std::isfinite(ncc))
        all_valid = false;
      else
      {
        ++metric.ncc_count;
        metric.ncc_sum += ncc;
        const double threshold = nccThresholdForLevel(level);
        if (ncc >= threshold) ++metric.ncc_gate_pass_count;
        else all_gate_pass = false;
        threshold_sum += threshold;
        const int bin = usageNccBin(ncc);
        if (bin >= 0) ++metric.ncc_hist[bin];
        ncc_min = std::min(ncc_min, ncc);
        ncc_sum += ncc;
      }
      if (valid && std::isfinite(sse))
      {
        metric.sse_sum += sse;
        metric.sse_samples += patch_size_total;
        ++metric.rms_count;
        const int bin = usageRmsBin(std::sqrt(std::max(0.0, sse) / std::max(1, patch_size_total)));
        if (bin >= 0) ++metric.rms_hist[bin];
      }
    }
    if (all_valid)
    {
      UsageJointStats &joint = cell.joint_stages[stage];
      const double macro = ncc_sum / level_count;
      ++joint.count;
      joint.ncc_min_sum += ncc_min;
      joint.ncc_macro_sum += macro;
      if (all_gate_pass) ++joint.min_gate_pass_count;
      if (macro >= threshold_sum / level_count) ++joint.macro_gate_pass_count;
      const int min_bin = usageNccBin(ncc_min);
      const int macro_bin = usageNccBin(macro);
      if (min_bin >= 0) ++joint.ncc_min_hist[min_bin];
      if (macro_bin >= 0) ++joint.ncc_macro_hist[macro_bin];
    }
  };
  updateCell(usage_current_camera_pairs_[pair_index]);
  updateCell(usage_total_current_camera_pairs_[pair_index]);
  if (pair.view_angle_bin >= 0 && pair.view_angle_bin < 5)
  {
    updateCell(usage_current_view_angle_bins_[pair.view_angle_bin]);
    updateCell(usage_total_current_view_angle_bins_[pair.view_angle_bin]);
  }
  if (pair.footprint_bin >= 0 && pair.footprint_bin < 5)
  {
    updateCell(usage_current_footprint_bins_[pair.footprint_bin]);
    updateCell(usage_total_current_footprint_bins_[pair.footprint_bin]);
  }
  if (pair.anisotropy_bin >= 0 && pair.anisotropy_bin < 4)
  {
    updateCell(usage_current_anisotropy_bins_[pair.anisotropy_bin]);
    updateCell(usage_total_current_anisotropy_bins_[pair.anisotropy_bin]);
  }
}

void VIOManager::recordUsageObservation(const PerCameraData &ctx, const Feature &ref_ftr, const VisualPoint &pt,
                                        const V2D &cur_px, const Matrix2d &A_cur_ref, bool accepted,
                                        const std::vector<double> &sse_levels,
                                        const std::vector<double> &ncc_levels,
                                        const std::vector<uint8_t> &level_valid)
{
  if (!usage_stats_en) return;
  const int camera_count = numCameras();
  if (camera_count <= 0 || ctx.camera_id < 0 || ctx.camera_id >= camera_count ||
      ref_ftr.camera_id_ < 0 || ref_ftr.camera_id_ >= camera_count)
    return;
  const size_t expected_pairs = static_cast<size_t>(camera_count * camera_count);
  if (usage_camera_pairs_.size() != expected_pairs || usage_total_camera_pairs_.size() != expected_pairs)
  {
    resetUsageStatsWindow();
    resetUsageStatsTotals();
  }

  const PerCameraData &ref_ctx = cameras_[ref_ftr.camera_id_];
  const int ref_region = usageRegionBin(ref_ctx, ref_ftr.px_);
  const int cur_region = usageRegionBin(ctx, cur_px);
  const int view_bin = usageViewAngleBin(ctx, ref_ftr, pt);
  const int footprint_bin = usageFootprintBin(A_cur_ref);
  const int anisotropy_bin = usageAnisotropyBin(A_cur_ref);
  const bool cross_camera = ref_ftr.camera_id_ != ctx.camera_id;
  const int level_count = std::max(1, patch_pyrimid_level);

  auto updateCell = [&](UsageStatsCell &cell) {
    cell.ensureLevels(level_count);
    if (!accepted)
    {
      ++cell.candidates;
      return;
    }

    ++cell.accepted;
    const int active_levels = static_cast<int>(std::count_if(
        level_valid.begin(), level_valid.end(), [](uint8_t valid) { return valid != 0; }));
    cell.theoretical_residuals += static_cast<long long>(patch_size_total) * active_levels;
    bool all_level_valid = true;
    bool all_gate_pass = true;
    double ncc_min = std::numeric_limits<double>::infinity();
    double ncc_sum = 0.0;
    double threshold_sum = 0.0;
    for (int level = 0; level < level_count; ++level)
    {
      const bool valid = level < static_cast<int>(level_valid.size()) && level_valid[level] != 0;
      const double ncc = level < static_cast<int>(ncc_levels.size())
                             ? ncc_levels[level]
                             : std::numeric_limits<double>::quiet_NaN();
      const double sse = level < static_cast<int>(sse_levels.size())
                             ? sse_levels[level]
                             : std::numeric_limits<double>::quiet_NaN();
      UsageMetricStats &metric = cell.levels[level].stages[kUsageAcceptedStage];
      if (!valid || !std::isfinite(ncc))
      {
        all_level_valid = false;
      }
      else
      {
        ++metric.ncc_count;
        metric.ncc_sum += ncc;
        const double threshold = nccThresholdForLevel(level);
        if (ncc >= threshold) ++metric.ncc_gate_pass_count;
        else all_gate_pass = false;
        threshold_sum += threshold;
        const int ncc_bin = usageNccBin(ncc);
        if (ncc_bin >= 0) ++metric.ncc_hist[ncc_bin];
        ncc_min = std::min(ncc_min, ncc);
        ncc_sum += ncc;
      }
      if (valid && std::isfinite(sse))
      {
        const double rms = std::sqrt(std::max(0.0, sse) /
                                     static_cast<double>(std::max(1, patch_size_total)));
        metric.sse_sum += sse;
        metric.sse_samples += patch_size_total;
        ++metric.rms_count;
        const int rms_bin = usageRmsBin(rms);
        if (rms_bin >= 0) ++metric.rms_hist[rms_bin];
      }
    }
    if (all_level_valid)
    {
      UsageJointStats &joint = cell.joint_stages[kUsageAcceptedStage];
      const double ncc_macro = ncc_sum / static_cast<double>(level_count);
      ++joint.count;
      joint.ncc_min_sum += ncc_min;
      joint.ncc_macro_sum += ncc_macro;
      if (all_gate_pass) ++joint.min_gate_pass_count;
      if (ncc_macro >= threshold_sum / level_count) ++joint.macro_gate_pass_count;
      const int min_bin = usageNccBin(ncc_min);
      const int macro_bin = usageNccBin(ncc_macro);
      if (min_bin >= 0) ++joint.ncc_min_hist[min_bin];
      if (macro_bin >= 0) ++joint.ncc_macro_hist[macro_bin];
    }
  };

  const int camera_pair_idx = ref_ftr.camera_id_ * camera_count + ctx.camera_id;
  updateCell(usage_camera_pairs_[camera_pair_idx]);
  updateCell(usage_total_camera_pairs_[camera_pair_idx]);
  if (ref_region >= 0 && cur_region >= 0)
  {
    const int region_pair_idx = ref_region * 4 + cur_region;
    const int cross_region_pair_idx = (cross_camera ? 16 : 0) + region_pair_idx;
    updateCell(usage_cross_region_pairs_[cross_region_pair_idx]);
    updateCell(usage_total_cross_region_pairs_[cross_region_pair_idx]);
  }
  if (view_bin >= 0)
  {
    updateCell(usage_view_angle_bins_[view_bin]);
    updateCell(usage_total_view_angle_bins_[view_bin]);
  }
  if (footprint_bin >= 0)
  {
    updateCell(usage_footprint_bins_[footprint_bin]);
    updateCell(usage_total_footprint_bins_[footprint_bin]);
  }
  if (anisotropy_bin >= 0)
  {
    updateCell(usage_anisotropy_bins_[anisotropy_bin]);
    updateCell(usage_total_anisotropy_bins_[anisotropy_bin]);
  }
}

void VIOManager::recordUsagePreGateMetrics(const PerCameraData &ctx, const Feature &ref_ftr, const VisualPoint &pt,
                                           const V2D &cur_px, const Matrix2d &A_cur_ref,
                                           const std::vector<double> &sse_levels,
                                           const std::vector<double> &ncc_levels,
                                           const std::vector<uint8_t> &level_valid)
{
  if (!usage_stats_en) return;
  const int camera_count = numCameras();
  if (camera_count <= 0 || ctx.camera_id < 0 || ctx.camera_id >= camera_count ||
      ref_ftr.camera_id_ < 0 || ref_ftr.camera_id_ >= camera_count)
    return;
  const size_t expected_pairs = static_cast<size_t>(camera_count * camera_count);
  if (usage_camera_pairs_.size() != expected_pairs || usage_total_camera_pairs_.size() != expected_pairs)
  {
    resetUsageStatsWindow();
    resetUsageStatsTotals();
  }

  const PerCameraData &ref_ctx = cameras_[ref_ftr.camera_id_];
  const int ref_region = usageRegionBin(ref_ctx, ref_ftr.px_);
  const int cur_region = usageRegionBin(ctx, cur_px);
  const int view_bin = usageViewAngleBin(ctx, ref_ftr, pt);
  const int footprint_bin = usageFootprintBin(A_cur_ref);
  const int anisotropy_bin = usageAnisotropyBin(A_cur_ref);
  const bool cross_camera = ref_ftr.camera_id_ != ctx.camera_id;
  const int level_count = std::max(1, patch_pyrimid_level);

  auto updateCell = [&](UsageStatsCell &cell) {
    cell.ensureLevels(level_count);
    bool all_level_valid = true;
    bool all_gate_pass = true;
    double ncc_min = std::numeric_limits<double>::infinity();
    double ncc_sum = 0.0;
    double threshold_sum = 0.0;
    for (int level = 0; level < level_count; ++level)
    {
      const bool valid = level < static_cast<int>(level_valid.size()) && level_valid[level] != 0;
      const double ncc = level < static_cast<int>(ncc_levels.size())
                             ? ncc_levels[level]
                             : std::numeric_limits<double>::quiet_NaN();
      const double sse = level < static_cast<int>(sse_levels.size())
                             ? sse_levels[level]
                             : std::numeric_limits<double>::quiet_NaN();
      UsageMetricStats &metric = cell.levels[level].stages[kUsagePreGateStage];
      if (!valid || !std::isfinite(ncc))
      {
        all_level_valid = false;
      }
      else
      {
        ++metric.ncc_count;
        metric.ncc_sum += ncc;
        const double threshold = nccThresholdForLevel(level);
        if (ncc >= threshold) ++metric.ncc_gate_pass_count;
        else all_gate_pass = false;
        threshold_sum += threshold;
        const int ncc_bin = usageNccBin(ncc);
        if (ncc_bin >= 0) ++metric.ncc_hist[ncc_bin];
        ncc_min = std::min(ncc_min, ncc);
        ncc_sum += ncc;
      }
      if (valid && std::isfinite(sse))
      {
        const double rms = std::sqrt(std::max(0.0, sse) /
                                     static_cast<double>(std::max(1, patch_size_total)));
        metric.sse_sum += sse;
        metric.sse_samples += patch_size_total;
        ++metric.rms_count;
        const int rms_bin = usageRmsBin(rms);
        if (rms_bin >= 0) ++metric.rms_hist[rms_bin];
      }
    }
    if (all_level_valid)
    {
      UsageJointStats &joint = cell.joint_stages[kUsagePreGateStage];
      const double ncc_macro = ncc_sum / static_cast<double>(level_count);
      ++joint.count;
      joint.ncc_min_sum += ncc_min;
      joint.ncc_macro_sum += ncc_macro;
      if (all_gate_pass) ++joint.min_gate_pass_count;
      if (ncc_macro >= threshold_sum / level_count) ++joint.macro_gate_pass_count;
      const int min_bin = usageNccBin(ncc_min);
      const int macro_bin = usageNccBin(ncc_macro);
      if (min_bin >= 0) ++joint.ncc_min_hist[min_bin];
      if (macro_bin >= 0) ++joint.ncc_macro_hist[macro_bin];
    }
  };

  const int camera_pair_idx = ref_ftr.camera_id_ * camera_count + ctx.camera_id;
  updateCell(usage_camera_pairs_[camera_pair_idx]);
  updateCell(usage_total_camera_pairs_[camera_pair_idx]);
  if (ref_region >= 0 && cur_region >= 0)
  {
    const int region_pair_idx = ref_region * 4 + cur_region;
    const int cross_region_pair_idx = (cross_camera ? 16 : 0) + region_pair_idx;
    updateCell(usage_cross_region_pairs_[cross_region_pair_idx]);
    updateCell(usage_total_cross_region_pairs_[cross_region_pair_idx]);
  }
  if (view_bin >= 0)
  {
    updateCell(usage_view_angle_bins_[view_bin]);
    updateCell(usage_total_view_angle_bins_[view_bin]);
  }
  if (footprint_bin >= 0)
  {
    updateCell(usage_footprint_bins_[footprint_bin]);
    updateCell(usage_total_footprint_bins_[footprint_bin]);
  }
  if (anisotropy_bin >= 0)
  {
    updateCell(usage_anisotropy_bins_[anisotropy_bin]);
    updateCell(usage_total_anisotropy_bins_[anisotropy_bin]);
  }
}

void VIOManager::recordUsageEkfContribution(const PerCameraData &ctx, const Feature &ref_ftr, const VisualPoint &pt,
                                             const V2D &cur_px, const Matrix2d &A_cur_ref, int pyramid_level,
                                             int residuals, double sse, double ncc, bool level_valid)
{
  if (!usage_stats_en || residuals <= 0) return;
  const int camera_count = numCameras();
  if (camera_count <= 0 || ctx.camera_id < 0 || ctx.camera_id >= camera_count ||
      ref_ftr.camera_id_ < 0 || ref_ftr.camera_id_ >= camera_count)
    return;
  const size_t expected_pairs = static_cast<size_t>(camera_count * camera_count);
  if (usage_camera_pairs_.size() != expected_pairs || usage_total_camera_pairs_.size() != expected_pairs)
  {
    resetUsageStatsWindow();
    resetUsageStatsTotals();
  }

  const PerCameraData &ref_ctx = cameras_[ref_ftr.camera_id_];
  const int ref_region = usageRegionBin(ref_ctx, ref_ftr.px_);
  const int cur_region = usageRegionBin(ctx, cur_px);
  const int view_bin = usageViewAngleBin(ctx, ref_ftr, pt);
  const int footprint_bin = usageFootprintBin(A_cur_ref);
  const int anisotropy_bin = usageAnisotropyBin(A_cur_ref);
  const bool cross_camera = ref_ftr.camera_id_ != ctx.camera_id;
  const int level_count = std::max(1, patch_pyrimid_level);
  if (pyramid_level < 0 || pyramid_level >= level_count) return;

  auto updateCell = [&](UsageStatsCell &cell) {
    cell.ensureLevels(level_count);
    ++cell.ekf_patches;
    cell.ekf_residuals += residuals;
    UsageMetricStats &metric = cell.levels[pyramid_level].stages[kUsageEkfStage];
    if (level_valid && std::isfinite(ncc))
    {
      ++metric.ncc_count;
      metric.ncc_sum += ncc;
      if (ncc >= nccThresholdForLevel(pyramid_level)) ++metric.ncc_gate_pass_count;
      const int ncc_bin = usageNccBin(ncc);
      if (ncc_bin >= 0) ++metric.ncc_hist[ncc_bin];
    }
    if (level_valid && std::isfinite(sse))
    {
      const double rms = std::sqrt(std::max(0.0, sse) /
                                   static_cast<double>(std::max(1, patch_size_total)));
      metric.sse_sum += sse;
      metric.sse_samples += patch_size_total;
      ++metric.rms_count;
      const int rms_bin = usageRmsBin(rms);
      if (rms_bin >= 0) ++metric.rms_hist[rms_bin];
    }
  };

  const int camera_pair_idx = ref_ftr.camera_id_ * camera_count + ctx.camera_id;
  updateCell(usage_camera_pairs_[camera_pair_idx]);
  updateCell(usage_total_camera_pairs_[camera_pair_idx]);
  if (ref_region >= 0 && cur_region >= 0)
  {
    const int region_pair_idx = ref_region * 4 + cur_region;
    const int cross_region_pair_idx = (cross_camera ? 16 : 0) + region_pair_idx;
    updateCell(usage_cross_region_pairs_[cross_region_pair_idx]);
    updateCell(usage_total_cross_region_pairs_[cross_region_pair_idx]);
  }
  if (view_bin >= 0)
  {
    updateCell(usage_view_angle_bins_[view_bin]);
    updateCell(usage_total_view_angle_bins_[view_bin]);
  }
  if (footprint_bin >= 0)
  {
    updateCell(usage_footprint_bins_[footprint_bin]);
    updateCell(usage_total_footprint_bins_[footprint_bin]);
  }
  if (anisotropy_bin >= 0)
  {
    updateCell(usage_anisotropy_bins_[anisotropy_bin]);
    updateCell(usage_total_anisotropy_bins_[anisotropy_bin]);
  }
}

void VIOManager::recordUsagePoseFrameInfo(const Eigen::MatrixXd &prior_cov, const Eigen::MatrixXd &posterior_cov,
                                          const Eigen::MatrixXd &h_base, const Eigen::MatrixXd &h_same,
                                          const Eigen::MatrixXd &h_cross, const Eigen::MatrixXd &h_current_cross,
                                          long long patches_all, long long residuals_all,
                                          long long patches_same, long long residuals_same,
                                          long long patches_cross, long long residuals_cross,
                                          long long patches_current_cross, long long residuals_current_cross)
{
  if (!usage_stats_en || prior_cov.rows() == 0 || prior_cov.rows() != prior_cov.cols() ||
      posterior_cov.rows() != prior_cov.rows() || posterior_cov.cols() != prior_cov.cols())
    return;

  auto symmetrize = [](const Eigen::MatrixXd &m) {
    return 0.5 * (m + m.transpose());
  };

  auto poseBlock = [](const Eigen::MatrixXd &m) {
    Eigen::Matrix<double, kUsagePoseDim, kUsagePoseDim> out =
        Eigen::Matrix<double, kUsagePoseDim, kUsagePoseDim>::Identity() * 1.0e-12;
    const int dim = std::min<int>(kUsagePoseDim, std::min(m.rows(), m.cols()));
    if (dim > 0) out.topLeftCorner(dim, dim) = m.topLeftCorner(dim, dim);
    out = 0.5 * (out + out.transpose());
    out.diagonal().array() += 1.0e-12;
    return out;
  };

  auto logDetSpd = [](const Eigen::Matrix<double, kUsagePoseDim, kUsagePoseDim> &m) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kUsagePoseDim, kUsagePoseDim>> solver(m);
    if (solver.info() != Eigen::Success) return std::numeric_limits<double>::quiet_NaN();
    double value = 0.0;
    for (int i = 0; i < kUsagePoseDim; ++i)
      value += std::log(std::max(1.0e-18, solver.eigenvalues()[i]));
    return value;
  };

  struct Gain
  {
    bool valid = false;
    double ig = 0.0;
    double worst_ig = 0.0;
  };

  auto computeCovarianceGain = [&](const Eigen::MatrixXd &prior, const Eigen::MatrixXd &posterior) {
    Gain gain;
    if (prior.rows() != posterior.rows() || prior.cols() != posterior.cols() || prior.rows() == 0) return gain;
    Eigen::MatrixXd prior_spd = symmetrize(prior);
    Eigen::MatrixXd posterior_spd = symmetrize(posterior);
    prior_spd.diagonal().array() += 1.0e-12;
    posterior_spd.diagonal().array() += 1.0e-12;
    const auto prior_pose = poseBlock(prior_spd);
    const auto posterior_pose = poseBlock(posterior_spd);
    const double logdet_prior = logDetSpd(prior_pose);
    const double logdet_posterior = logDetSpd(posterior_pose);
    if (!std::isfinite(logdet_prior) || !std::isfinite(logdet_posterior)) return gain;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, kUsagePoseDim, kUsagePoseDim>>
        solver(prior_pose, posterior_pose);
    if (solver.info() != Eigen::Success) return gain;
    const double min_ratio = std::max(1.0e-18, solver.eigenvalues().minCoeff());
    gain.ig = 0.5 * (logdet_prior - logdet_posterior);
    gain.worst_ig = 0.5 * std::log(min_ratio);
    gain.valid = std::isfinite(gain.ig) && std::isfinite(gain.worst_ig);
    return gain;
  };

  auto posteriorFromHessian = [&](const Eigen::MatrixXd &hessian) -> Eigen::MatrixXd {
    Eigen::MatrixXd empty;
    if (hessian.rows() != prior_cov.rows() || hessian.cols() != prior_cov.cols()) return empty;
    Eigen::MatrixXd prior_spd = symmetrize(prior_cov);
    prior_spd.diagonal().array() += 1.0e-12;
    const Eigen::LDLT<Eigen::MatrixXd> prior_ldlt(prior_spd);
    if (prior_ldlt.info() != Eigen::Success) return empty;
    const double measurement_cov = photometricNoiseCovariance();
    const double inv_noise = (std::isfinite(measurement_cov) && measurement_cov > 1.0e-12)
                                 ? 1.0 / measurement_cov
                                 : 1.0;
    Eigen::MatrixXd information =
        prior_ldlt.solve(Eigen::MatrixXd::Identity(prior_spd.rows(), prior_spd.cols())) + inv_noise * symmetrize(hessian);
    information = symmetrize(information);
    information.diagonal().array() += 1.0e-12;
    const Eigen::LDLT<Eigen::MatrixXd> info_ldlt(information);
    if (info_ldlt.info() != Eigen::Success) return empty;
    Eigen::MatrixXd posterior = info_ldlt.solve(Eigen::MatrixXd::Identity(information.rows(), information.cols()));
    return symmetrize(posterior).eval();
  };

  const Gain realized_gain = computeCovarianceGain(prior_cov, posterior_cov);
  const Eigen::MatrixXd base_posterior = posteriorFromHessian(h_base);
  std::array<double, 3> shapley = {};
  if (!cross_camera_current_residual_en)
  {
    const Eigen::MatrixXd same_posterior = posteriorFromHessian(h_base + h_same);
    const Eigen::MatrixXd cross_posterior = posteriorFromHessian(h_base + h_cross);
    const Eigen::MatrixXd visual_posterior = posteriorFromHessian(h_base + h_same + h_cross);
    const Gain same_gain = base_posterior.size() > 0 && same_posterior.size() > 0
                               ? computeCovarianceGain(base_posterior, same_posterior) : Gain();
    const Gain cross_gain = base_posterior.size() > 0 && cross_posterior.size() > 0
                                ? computeCovarianceGain(base_posterior, cross_posterior) : Gain();
    const Gain visual_gain = base_posterior.size() > 0 && visual_posterior.size() > 0
                                 ? computeCovarianceGain(base_posterior, visual_posterior) : Gain();
    const double ig_same = same_gain.valid ? same_gain.ig : 0.0;
    const double ig_cross = cross_gain.valid ? cross_gain.ig : 0.0;
    const double ig_visual = visual_gain.valid ? visual_gain.ig : 0.0;
    shapley[0] = 0.5 * (ig_same + ig_visual - ig_cross);
    shapley[1] = 0.5 * (ig_cross + ig_visual - ig_same);
  }
  else
  {
    const Eigen::MatrixXd source_hessians[3] = {h_same, h_cross, h_current_cross};
    std::array<double, 8> subset_gain = {};
    for (int mask = 1; mask < 8; ++mask)
    {
      Eigen::MatrixXd subset_hessian = h_base;
      for (int player = 0; player < 3; ++player)
        if ((mask & (1 << player)) != 0) subset_hessian += source_hessians[player];
      const Eigen::MatrixXd subset_posterior = posteriorFromHessian(subset_hessian);
      const Gain gain = base_posterior.size() > 0 && subset_posterior.size() > 0
                            ? computeCovarianceGain(base_posterior, subset_posterior) : Gain();
      subset_gain[mask] = gain.valid ? gain.ig : 0.0;
    }
    for (int player = 0; player < 3; ++player)
    {
      for (int subset = 0; subset < 8; ++subset)
      {
        if ((subset & (1 << player)) != 0) continue;
        const int subset_size = ((subset & 1) != 0) + ((subset & 2) != 0) + ((subset & 4) != 0);
        const double weight = subset_size == 0 ? 1.0 / 3.0 : (subset_size == 1 ? 1.0 / 6.0 : 1.0 / 3.0);
        shapley[player] += weight * (subset_gain[subset | (1 << player)] - subset_gain[subset]);
      }
    }
  }

  auto updateCell = [](PoseInfoStatsCell &cell, double information_gain, double worst_direction_ig,
                       long long patches, long long residuals) {
    if (residuals > 0)
    {
      ++cell.active_frames;
      cell.active_information_gains.push_back(information_gain);
    }
    cell.patches += patches;
    cell.residuals += residuals;
    cell.information_gain_sum += information_gain;
    if (residuals > 0 && std::isfinite(worst_direction_ig))
      cell.worst_direction_ig_sum += worst_direction_ig;
  };

  const double realized_ig = realized_gain.valid ? realized_gain.ig : 0.0;
  const double realized_worst_ig = realized_gain.valid ? realized_gain.worst_ig : 0.0;
  const double not_applicable = std::numeric_limits<double>::quiet_NaN();
  updateCell(usage_pose_all_, realized_ig, realized_worst_ig, patches_all, residuals_all);
  updateCell(usage_pose_same_, shapley[0], not_applicable, patches_same, residuals_same);
  updateCell(usage_pose_cross_, shapley[1], not_applicable, patches_cross, residuals_cross);
  if (cross_camera_current_residual_en)
    updateCell(usage_pose_current_cross_, shapley[2], not_applicable, patches_current_cross, residuals_current_cross);
  updateCell(usage_total_pose_all_, realized_ig, realized_worst_ig, patches_all, residuals_all);
  updateCell(usage_total_pose_same_, shapley[0], not_applicable, patches_same, residuals_same);
  updateCell(usage_total_pose_cross_, shapley[1], not_applicable, patches_cross, residuals_cross);
  if (cross_camera_current_residual_en)
    updateCell(usage_total_pose_current_cross_, shapley[2], not_applicable,
               patches_current_cross, residuals_current_cross);
}
void VIOManager::printUsageStatsTable(int frame_id)
{
  if (!usage_stats_en) return;

  const char *color = "\033[1;35m";
  const char *reset = "\033[0m";

  auto mergeCell = [&](UsageStatsCell &dst, const UsageStatsCell &src) {
    const int level_count = std::max(static_cast<int>(dst.levels.size()), static_cast<int>(src.levels.size()));
    dst.ensureLevels(level_count);
    dst.candidates += src.candidates;
    dst.accepted += src.accepted;
    dst.theoretical_residuals += src.theoretical_residuals;
    dst.ekf_patches += src.ekf_patches;
    dst.ekf_residuals += src.ekf_residuals;
    for (int stage = 0; stage < 2; ++stage)
    {
      UsageJointStats &dst_joint = dst.joint_stages[stage];
      const UsageJointStats &src_joint = src.joint_stages[stage];
      dst_joint.count += src_joint.count;
      dst_joint.min_gate_pass_count += src_joint.min_gate_pass_count;
      dst_joint.macro_gate_pass_count += src_joint.macro_gate_pass_count;
      dst_joint.ncc_min_sum += src_joint.ncc_min_sum;
      dst_joint.ncc_macro_sum += src_joint.ncc_macro_sum;
      for (int i = 0; i < kUsageNccBinCount; ++i)
      {
        dst_joint.ncc_min_hist[i] += src_joint.ncc_min_hist[i];
        dst_joint.ncc_macro_hist[i] += src_joint.ncc_macro_hist[i];
      }
    }
    for (int level = 0; level < static_cast<int>(src.levels.size()); ++level)
    {
      for (int stage = 0; stage < kUsageStageCount; ++stage)
      {
        UsageMetricStats &dst_metric = dst.levels[level].stages[stage];
        const UsageMetricStats &src_metric = src.levels[level].stages[stage];
        dst_metric.ncc_count += src_metric.ncc_count;
        dst_metric.ncc_gate_pass_count += src_metric.ncc_gate_pass_count;
        dst_metric.sse_samples += src_metric.sse_samples;
        dst_metric.rms_count += src_metric.rms_count;
        dst_metric.ncc_sum += src_metric.ncc_sum;
        dst_metric.sse_sum += src_metric.sse_sum;
        for (int i = 0; i < kUsageNccBinCount; ++i)
          dst_metric.ncc_hist[i] += src_metric.ncc_hist[i];
        for (int i = 0; i < kUsageRmsBinCount; ++i)
          dst_metric.rms_hist[i] += src_metric.rms_hist[i];
      }
    }
  };

  auto formatDouble = [](double value, int precision = 3) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
  };

  auto formatPercent = [&](long long count, long long total) {
    return total > 0 ? formatDouble(100.0 * static_cast<double>(count) / static_cast<double>(total), 2) : std::string("-");
  };

  auto printLine = [&](const std::string &line) {
    printf("%s%s%s\n", color, line.c_str(), reset);
  };

  auto printDivider = [&]() {
    printLine(std::string(180, '-'));
  };

  auto meanText = [&](double sum, long long count, int precision) {
    return count > 0 ? formatDouble(sum / static_cast<double>(count), precision) : std::string("-");
  };

  auto nccBinName = [](int idx) {
    const double lower = -1.0 + 0.1 * idx;
    const double upper = lower + 0.1;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << lower << ".." << upper;
    return stream.str();
  };

  auto nccQuantile = [&](const std::array<long long, kUsageNccBinCount> &hist, long long count,
                         double quantile) {
    if (count <= 0) return std::string("-");
    const long long target = std::max(1LL, static_cast<long long>(std::ceil(quantile * count)));
    long long cumulative = 0;
    for (int i = 0; i < kUsageNccBinCount; ++i)
    {
      cumulative += hist[i];
      if (cumulative >= target) return formatDouble(-0.95 + 0.1 * i, 2);
    }
    return formatDouble(0.95, 2);
  };

  auto nccBelow = [](const std::array<long long, kUsageNccBinCount> &hist, double threshold) {
    long long count = 0;
    for (int i = 0; i < kUsageNccBinCount; ++i)
    {
      const double upper = -0.9 + 0.1 * i;
      if (upper <= threshold + 1.0e-12) count += hist[i];
    }
    return count;
  };

  auto rmsQuantile = [&](const std::array<long long, kUsageRmsBinCount> &hist, long long count,
                         double quantile) {
    static const double representatives[kUsageRmsBinCount] = {1.0, 3.0, 5.0, 7.0, 9.0,
                                                               12.5, 17.5, 25.0, 40.0, 50.0};
    if (count <= 0) return std::string("-");
    const long long target = std::max(1LL, static_cast<long long>(std::ceil(quantile * count)));
    long long cumulative = 0;
    for (int i = 0; i < kUsageRmsBinCount; ++i)
    {
      cumulative += hist[i];
      if (cumulative >= target) return formatDouble(representatives[i], 1);
    }
    return formatDouble(representatives[kUsageRmsBinCount - 1], 1);
  };

  auto vectorQuantile = [&](const std::vector<double> &values, double quantile) {
    if (values.empty()) return std::string("-");
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const size_t idx = std::min(sorted.size() - 1,
                                static_cast<size_t>(std::floor(quantile * static_cast<double>(sorted.size() - 1))));
    return formatDouble(sorted[idx], 5);
  };

  auto printStatsSet = [&](const char *title, const char *frame_label, long long frame_count,
                           const std::vector<UsageStatsCell> &camera_pairs,
                           const std::vector<UsageStatsCell> &current_camera_pairs,
                           const std::array<UsageStatsCell, 5> &current_view_angle_bins,
                           const std::array<UsageStatsCell, 5> &current_footprint_bins,
                           const std::array<UsageStatsCell, 4> &current_anisotropy_bins,
                           const std::array<UsageStatsCell, 32> &cross_region_pairs,
                           const std::array<UsageStatsCell, 5> &view_angle_bins,
                           const std::array<UsageStatsCell, 5> &footprint_bins,
                           const std::array<UsageStatsCell, 4> &anisotropy_bins,
                           const PoseInfoStatsCell &pose_all,
                           const PoseInfoStatsCell &pose_same,
                           const PoseInfoStatsCell &pose_cross,
                           const PoseInfoStatsCell &pose_current_cross,
                           bool detailed) {
    UsageStatsCell total;
    UsageStatsCell same_camera;
    UsageStatsCell cross_camera;
    UsageStatsCell current_cross_camera;
    total.ensureLevels(std::max(1, patch_pyrimid_level));
    same_camera.ensureLevels(std::max(1, patch_pyrimid_level));
    cross_camera.ensureLevels(std::max(1, patch_pyrimid_level));
    if (cross_camera_current_residual_en)
      current_cross_camera.ensureLevels(std::max(1, patch_pyrimid_level));
    const int camera_count = numCameras();
    for (int ref_cam = 0; ref_cam < camera_count; ++ref_cam)
    {
      for (int cur_cam = 0; cur_cam < camera_count; ++cur_cam)
      {
        const size_t idx = static_cast<size_t>(ref_cam * camera_count + cur_cam);
        if (idx >= camera_pairs.size()) continue;
        const UsageStatsCell &cell = camera_pairs[idx];
        mergeCell(total, cell);
        if (ref_cam == cur_cam)
          mergeCell(same_camera, cell);
        else
          mergeCell(cross_camera, cell);
      }
    }
    if (cross_camera_current_residual_en)
      for (const UsageStatsCell &cell : current_camera_pairs)
      {
        mergeCell(current_cross_camera, cell);
        mergeCell(total, cell);
      }

    auto printFlowHeader = [&](const char *section_title) {
      printDivider();
      printLine(section_title);
      printDivider();
      printLine("Category               CandidateAttempts AllLvlRdy  Ready%  AcceptedTracks  Acc%  EKFLvlPatch   EKFRes  EKFRes/F  EKF/Theory%");
      printDivider();
    };

    auto printFlowRow = [&](const std::string &name, const UsageStatsCell &cell) {
      const long long ready = cell.joint_stages[kUsagePreGateStage].count;
      std::ostringstream row;
      row << std::left << std::setw(28) << name
          << std::right << std::setw(18) << cell.candidates
          << std::setw(10) << ready
          << std::setw(8) << formatPercent(ready, cell.candidates)
          << std::setw(16) << cell.accepted
          << std::setw(7) << formatPercent(cell.accepted, cell.candidates)
          << std::setw(13) << cell.ekf_patches
          << std::setw(9) << cell.ekf_residuals
          << std::setw(10) << (frame_count > 0 ? formatDouble(static_cast<double>(cell.ekf_residuals) / frame_count, 1) : "-")
          << std::setw(12) << formatPercent(cell.ekf_residuals, cell.theoretical_residuals);
      printLine(row.str());
    };

    auto printMetricSummaryHeader = [&](const char *section_title) {
      printDivider();
      printLine(section_title);
      printDivider();
      printLine("Stage       Group               Metric    Samples    Mean    P10~    P50~    P90~    <0.3%    <0.5%   >=gate%  RMSmicro  RMSP50~  RMSP90~");
      printDivider();
    };

    auto printLevelMetricRow = [&](const char *stage_name, const std::string &group, const std::string &metric_name,
                                   const UsageMetricStats &metric) {
      const std::string rms_micro = metric.sse_samples > 0
          ? formatDouble(std::sqrt(std::max(0.0, metric.sse_sum) / static_cast<double>(metric.sse_samples)), 2)
          : std::string("-");
      std::ostringstream row;
      row << std::left << std::setw(12) << stage_name
          << std::setw(20) << group
          << std::setw(10) << metric_name
          << std::right << std::setw(9) << metric.ncc_count
          << std::setw(8) << meanText(metric.ncc_sum, metric.ncc_count, 3)
          << std::setw(8) << nccQuantile(metric.ncc_hist, metric.ncc_count, 0.10)
          << std::setw(8) << nccQuantile(metric.ncc_hist, metric.ncc_count, 0.50)
          << std::setw(8) << nccQuantile(metric.ncc_hist, metric.ncc_count, 0.90)
          << std::setw(10) << formatPercent(nccBelow(metric.ncc_hist, 0.3), metric.ncc_count)
          << std::setw(10) << formatPercent(nccBelow(metric.ncc_hist, 0.5), metric.ncc_count)
          << std::setw(10) << formatPercent(metric.ncc_gate_pass_count, metric.ncc_count)
          << std::setw(10) << rms_micro
          << std::setw(10) << rmsQuantile(metric.rms_hist, metric.rms_count, 0.50)
          << std::setw(10) << rmsQuantile(metric.rms_hist, metric.rms_count, 0.90);
      printLine(row.str());
    };

    auto printJointMetricRow = [&](const char *stage_name, const std::string &group, const char *metric_name,
                                   const UsageJointStats &joint, bool use_min) {
      const auto &hist = use_min ? joint.ncc_min_hist : joint.ncc_macro_hist;
      const double sum = use_min ? joint.ncc_min_sum : joint.ncc_macro_sum;
      const long long gate_pass = use_min ? joint.min_gate_pass_count : joint.macro_gate_pass_count;
      std::ostringstream row;
      row << std::left << std::setw(12) << stage_name
          << std::setw(20) << group
          << std::setw(10) << metric_name
          << std::right << std::setw(9) << joint.count
          << std::setw(8) << meanText(sum, joint.count, 3)
          << std::setw(8) << nccQuantile(hist, joint.count, 0.10)
          << std::setw(8) << nccQuantile(hist, joint.count, 0.50)
          << std::setw(8) << nccQuantile(hist, joint.count, 0.90)
          << std::setw(10) << formatPercent(nccBelow(hist, 0.3), joint.count)
          << std::setw(10) << formatPercent(nccBelow(hist, 0.5), joint.count)
          << std::setw(10) << formatPercent(gate_pass, joint.count)
          << std::setw(10) << "-" << std::setw(10) << "-" << std::setw(10) << "-";
      printLine(row.str());
    };

    auto printGroupMetrics = [&](const std::string &group, const UsageStatsCell &cell) {
      static const char *stage_names[kUsageStageCount] = {"PRE_GATE", "POST_GATE", "EKF_USED"};
      for (int stage = 0; stage < kUsageStageCount; ++stage)
      {
        for (int level = 0; level < std::max(1, patch_pyrimid_level); ++level)
          if (level < static_cast<int>(cell.levels.size()))
            printLevelMetricRow(stage_names[stage], group, "L" + std::to_string(level),
                                cell.levels[level].stages[stage]);
        if (stage < 2)
        {
          printJointMetricRow(stage_names[stage], group, "MIN", cell.joint_stages[stage], true);
          printJointMetricRow(stage_names[stage], group, "MACRO", cell.joint_stages[stage], false);
        }
      }
    };

    auto printPoseSections = [&]() {
      printDivider();
      printLine("POSE_REALIZED_COVARIANCE_GAIN");
      printDivider();
      printLine("Category                 RunFrames  Active  Active%  FinalLvlPatch  FinalRes  IG/frame  IG/active  MedianAct  P10Act  P90Act  WorstIG/act");
      printDivider();
      auto printRealizedRow = [&](const PoseInfoStatsCell &cell) {
        std::ostringstream row;
        row << std::left << std::setw(24) << "all_actual_posterior"
            << std::right << std::setw(9) << frame_count
            << std::setw(8) << cell.active_frames
            << std::setw(9) << formatPercent(cell.active_frames, frame_count)
            << std::setw(13) << cell.patches
            << std::setw(9) << cell.residuals
            << std::setw(10) << (frame_count > 0 ? formatDouble(cell.information_gain_sum / frame_count, 5) : "-")
            << std::setw(11) << (cell.active_frames > 0 ? formatDouble(cell.information_gain_sum / cell.active_frames, 5) : "-")
            << std::setw(11) << vectorQuantile(cell.active_information_gains, 0.50)
            << std::setw(9) << vectorQuantile(cell.active_information_gains, 0.10)
            << std::setw(9) << vectorQuantile(cell.active_information_gains, 0.90)
            << std::setw(13) << (cell.active_frames > 0
                                     ? formatDouble(cell.worst_direction_ig_sum / cell.active_frames, 5)
                                     : "-");
        printLine(row.str());
      };
      printRealizedRow(pose_all);

      printDivider();
      printLine("POSE_SOURCE_ATTRIBUTION_SHAPLEY");
      printLine(cross_camera_current_residual_en
                    ? "pose_scope: realized gain uses actual P_before/P_after; exact 3-source Shapley uses same-camera, historical cross-camera, and current-current cross-camera Hessians from the final covariance linearization, conditioned on base factors"
                    : "pose_scope: realized gain uses actual P_before/P_after; exact 2-source Shapley uses same-camera and historical cross-camera Hessians from the final covariance linearization, conditioned on base factors");
      printDivider();
      printLine("Source                    RunFrames  Active  Active%  FinalLvlPatch  FinalRes  IG/frame  IG/active  MedianAct  P10Act  P90Act  Share%");
      printDivider();
      const double attribution_total = pose_same.information_gain_sum + pose_cross.information_gain_sum +
          (cross_camera_current_residual_en ? pose_current_cross.information_gain_sum : 0.0);
      auto printAttributionRow = [&](const std::string &name, const PoseInfoStatsCell &cell) {
        std::ostringstream row;
        row << std::left << std::setw(26) << name
            << std::right << std::setw(7) << frame_count
            << std::setw(8) << cell.active_frames
            << std::setw(9) << formatPercent(cell.active_frames, frame_count)
            << std::setw(13) << cell.patches
            << std::setw(9) << cell.residuals
            << std::setw(10) << (frame_count > 0 ? formatDouble(cell.information_gain_sum / frame_count, 5) : "-")
            << std::setw(11) << (cell.active_frames > 0 ? formatDouble(cell.information_gain_sum / cell.active_frames, 5) : "-")
            << std::setw(11) << vectorQuantile(cell.active_information_gains, 0.50)
            << std::setw(9) << vectorQuantile(cell.active_information_gains, 0.10)
            << std::setw(9) << vectorQuantile(cell.active_information_gains, 0.90)
            << std::setw(9) << (std::fabs(attribution_total) > 1.0e-12
                                    ? formatDouble(100.0 * cell.information_gain_sum / attribution_total, 2)
                                    : "-");
        printLine(row.str());
      };
      printAttributionRow("same_cam_shapley", pose_same);
      printAttributionRow("cross_cam_shapley", pose_cross);
      if (cross_camera_current_residual_en)
        printAttributionRow("current_cross_cam_shapley", pose_current_cross);
    };

    auto printNccDistribution = [&](const std::string &stage_name, const std::string &metric_name,
                                    const std::array<long long, kUsageNccBinCount> &total_hist,
                                    const std::array<long long, kUsageNccBinCount> &same_hist,
                                    const std::array<long long, kUsageNccBinCount> &cross_hist,
                                    const std::array<long long, kUsageNccBinCount> &current_hist) {
      long long total_count = 0;
      long long same_count = 0;
      long long cross_count = 0;
      long long current_count = 0;
      for (int i = 0; i < kUsageNccBinCount; ++i)
      {
        total_count += total_hist[i];
        same_count += same_hist[i];
        cross_count += cross_hist[i];
        if (cross_camera_current_residual_en) current_count += current_hist[i];
      }
      printDivider();
      printLine("NCC_DISTRIBUTION stage=" + stage_name + " metric=" + metric_name);
      printDivider();
      printLine(cross_camera_current_residual_en
                    ? "NCC_bin          all_count    all%   same_count   same%  hist_cross  cross%  current_cc   curr%"
                    : "NCC_bin          all_count    all%   same_count   same%  hist_cross  cross%");
      printDivider();
      for (int i = 0; i < kUsageNccBinCount; ++i)
      {
        std::ostringstream row;
        row << std::left << std::setw(16) << nccBinName(i)
            << std::right << std::setw(10) << total_hist[i]
            << std::setw(8) << formatPercent(total_hist[i], total_count)
            << std::setw(13) << same_hist[i]
            << std::setw(8) << formatPercent(same_hist[i], same_count)
            << std::setw(13) << cross_hist[i]
            << std::setw(8) << formatPercent(cross_hist[i], cross_count);
        if (cross_camera_current_residual_en)
          row << std::setw(13) << current_hist[i]
              << std::setw(8) << formatPercent(current_hist[i], current_count);
        printLine(row.str());
      }
    };

    auto printDiagnosticHeader = [&](const char *section_title) {
      printDivider();
      printLine(section_title);
      printDivider();
      std::ostringstream header;
      header << std::left << std::setw(30) << "Category"
             << std::right << std::setw(10) << "Attempts"
             << std::setw(10) << "Ready"
             << std::setw(10) << "Accept"
             << std::setw(10) << "EKFRes"
             << std::setw(10) << "PreMIN"
             << std::setw(10) << "PostMIN";
      for (int level = 0; level < std::max(1, patch_pyrimid_level); ++level)
        header << std::setw(10) << ("PreL" + std::to_string(level))
               << std::setw(10) << ("PostL" + std::to_string(level));
      printLine(header.str());
      printDivider();
    };

    auto printDiagnosticRow = [&](const std::string &name, const UsageStatsCell &cell) {
      const UsageJointStats &pre_joint = cell.joint_stages[kUsagePreGateStage];
      const UsageJointStats &accepted_joint = cell.joint_stages[kUsageAcceptedStage];
      std::ostringstream row;
      row << std::left << std::setw(30) << name
          << std::right << std::setw(10) << cell.candidates
          << std::setw(10) << pre_joint.count
          << std::setw(10) << cell.accepted
          << std::setw(10) << cell.ekf_residuals
          << std::setw(10) << meanText(pre_joint.ncc_min_sum, pre_joint.count, 3)
          << std::setw(10) << meanText(accepted_joint.ncc_min_sum, accepted_joint.count, 3);
      for (int level = 0; level < std::max(1, patch_pyrimid_level); ++level)
      {
        if (level < static_cast<int>(cell.levels.size()))
        {
          const UsageMetricStats &pre = cell.levels[level].stages[kUsagePreGateStage];
          const UsageMetricStats &post = cell.levels[level].stages[kUsageAcceptedStage];
          row << std::setw(10) << meanText(pre.ncc_sum, pre.ncc_count, 3)
              << std::setw(10) << meanText(post.ncc_sum, post.ncc_count, 3);
        }
        else
        {
          row << std::setw(10) << "-" << std::setw(10) << "-";
        }
      }
      printLine(row.str());
    };

    printDivider();
    printLine(title);
    std::ostringstream ncc_threshold_text;
    ncc_threshold_text << "[";
    for (int level = 0; level < std::max(1, patch_pyrimid_level); ++level)
    {
      if (level > 0) ncc_threshold_text << ",";
      ncc_threshold_text << formatDouble(nccThresholdForLevel(level), 3);
    }
    ncc_threshold_text << "]";
    std::ostringstream summary;
    summary << "frame=" << frame_id << " " << frame_label << "=" << frame_count
            << " candidate_attempts=" << total.candidates
            << " accepted_tracks=" << total.accepted
            << " theoretical_residuals=" << total.theoretical_residuals
            << " actual_ekf_residuals=" << total.ekf_residuals
            << " pyramid_levels=" << std::max(1, patch_pyrimid_level)
            << " ncc_gate=" << (ncc_en ? "on" : "off")
            << " ncc_thresholds_L0_first=" << ncc_threshold_text.str()
            << " current_cross_residual=" << (cross_camera_current_residual_en ? "on" : "off");
    printLine(summary.str());
    printLine("metric_scope: AllLvlReady requires every configured level to have valid PRE_GATE NCC; each level is gated independently; POST_GATE and EKF_USED contain only that level's admitted patches");

    printFlowHeader("TRACK_FLOW_BY_GROUP");
    printFlowRow("same_cam", same_camera);
    printFlowRow("cross_cam", cross_camera);
    if (cross_camera_current_residual_en)
      printFlowRow("current_cross_cam", current_cross_camera);

    printMetricSummaryHeader("MULTILEVEL_TEXTURE_HEALTH_BY_GROUP");
    printGroupMetrics("same_cam", same_camera);
    printGroupMetrics("cross_cam", cross_camera);
    if (cross_camera_current_residual_en)
      printGroupMetrics("current_cross_cam", current_cross_camera);

    printPoseSections();

    printFlowHeader("TRACK_FLOW_BY_CAMERA_PAIR");
    for (int ref_cam = 0; ref_cam < camera_count; ++ref_cam)
    {
      for (int cur_cam = 0; cur_cam < camera_count; ++cur_cam)
      {
        const size_t idx = static_cast<size_t>(ref_cam * camera_count + cur_cam);
        if (idx >= camera_pairs.size()) continue;
        const UsageStatsCell &cell = camera_pairs[idx];
        if (cell.candidates == 0 && cell.accepted == 0) continue;
        printFlowRow("cam" + std::to_string(ref_cam) + "->cam" + std::to_string(cur_cam), cell);
      }
    }
    if (cross_camera_current_residual_en)
    for (int source_cam = 0; source_cam < camera_count; ++source_cam)
    {
      for (int target_cam = source_cam + 1; target_cam < camera_count; ++target_cam)
      {
        const size_t idx = static_cast<size_t>(source_cam * camera_count + target_cam);
        if (idx >= current_camera_pairs.size()) continue;
        const UsageStatsCell &cell = current_camera_pairs[idx];
        if (cell.candidates == 0 && cell.accepted == 0) continue;
        printFlowRow("cam" + std::to_string(source_cam) + "<->cam" + std::to_string(target_cam) + " current", cell);
      }
    }

    if (detailed)
    {
      for (int stage = 0; stage < 2; ++stage)
      {
        const std::string stage_name = stage == kUsagePreGateStage ? "PRE_GATE" : "POST_GATE";
        for (int level = 0; level < std::max(1, patch_pyrimid_level); ++level)
        {
          if (level >= static_cast<int>(total.levels.size()) ||
              level >= static_cast<int>(same_camera.levels.size()) ||
              level >= static_cast<int>(cross_camera.levels.size()) ||
              (cross_camera_current_residual_en &&
               level >= static_cast<int>(current_cross_camera.levels.size())))
            continue;
          static const std::array<long long, kUsageNccBinCount> empty_ncc_hist = {};
          printNccDistribution(stage_name, "L" + std::to_string(level),
                               total.levels[level].stages[stage].ncc_hist,
                               same_camera.levels[level].stages[stage].ncc_hist,
                               cross_camera.levels[level].stages[stage].ncc_hist,
                               cross_camera_current_residual_en
                                   ? current_cross_camera.levels[level].stages[stage].ncc_hist
                                   : empty_ncc_hist);
        }
        printNccDistribution(stage_name, "MIN",
                             total.joint_stages[stage].ncc_min_hist,
                             same_camera.joint_stages[stage].ncc_min_hist,
                             cross_camera.joint_stages[stage].ncc_min_hist,
                             current_cross_camera.joint_stages[stage].ncc_min_hist);
      }

      printDiagnosticHeader("TEXTURE_DIAGNOSTICS_BY_CAMERA_PAIR");
      for (int ref_cam = 0; ref_cam < camera_count; ++ref_cam)
      {
        for (int cur_cam = 0; cur_cam < camera_count; ++cur_cam)
        {
          const size_t idx = static_cast<size_t>(ref_cam * camera_count + cur_cam);
          if (idx >= camera_pairs.size()) continue;
          const UsageStatsCell &cell = camera_pairs[idx];
          if (cell.candidates == 0 && cell.accepted == 0) continue;
          printDiagnosticRow("cam" + std::to_string(ref_cam) + "->cam" + std::to_string(cur_cam), cell);
        }
      }
      if (cross_camera_current_residual_en)
      for (int source_cam = 0; source_cam < camera_count; ++source_cam)
      {
        for (int target_cam = source_cam + 1; target_cam < camera_count; ++target_cam)
        {
          const size_t idx = static_cast<size_t>(source_cam * camera_count + target_cam);
          if (idx >= current_camera_pairs.size()) continue;
          const UsageStatsCell &cell = current_camera_pairs[idx];
          if (cell.candidates == 0 && cell.accepted == 0) continue;
          printDiagnosticRow("cam" + std::to_string(source_cam) + "<->cam" +
                             std::to_string(target_cam) + " current", cell);
        }
      }

      static const char *region_names[] = {"center", "mid", "edge", "outer"};
      printDiagnosticHeader("TEXTURE_DIAGNOSTICS_BY_SAME_CROSS_REGION");
      for (int cross = 0; cross < 2; ++cross)
      {
        for (int ref_region = 0; ref_region < 4; ++ref_region)
        {
          for (int cur_region = 0; cur_region < 4; ++cur_region)
          {
            const UsageStatsCell &cell = cross_region_pairs[cross * 16 + ref_region * 4 + cur_region];
            if (cell.candidates == 0 && cell.accepted == 0) continue;
            printDiagnosticRow(std::string(cross ? "cross " : "same ") + region_names[ref_region] + "->" +
                                   region_names[cur_region],
                               cell);
          }
        }
      }

      static const char *angle_names[] = {"0-5deg", "5-15deg", "15-30deg", "30-60deg", ">60deg"};
      printDiagnosticHeader("TEXTURE_DIAGNOSTICS_BY_VIEW_ANGLE");
      for (int i = 0; i < 5; ++i)
        if (view_angle_bins[i].candidates > 0 || view_angle_bins[i].accepted > 0)
          printDiagnosticRow(angle_names[i], view_angle_bins[i]);

      if (cross_camera_current_residual_en)
      {
        printDiagnosticHeader("CURRENT_CROSS_TEXTURE_BY_VIEW_ANGLE");
        for (int i = 0; i < 5; ++i)
          if (current_view_angle_bins[i].candidates > 0 || current_view_angle_bins[i].accepted > 0)
            printDiagnosticRow(angle_names[i], current_view_angle_bins[i]);
      }

      static const char *footprint_names[] = {"invalid", "1-2x", "2-4x", "4-8x", ">8x"};
      printDiagnosticHeader("TEXTURE_DIAGNOSTICS_BY_FOOTPRINT");
      for (int i = 0; i < 5; ++i)
        if (footprint_bins[i].candidates > 0 || footprint_bins[i].accepted > 0)
          printDiagnosticRow(footprint_names[i], footprint_bins[i]);

      if (cross_camera_current_residual_en)
      {
        printDiagnosticHeader("CURRENT_CROSS_TEXTURE_BY_FOOTPRINT");
        for (int i = 0; i < 5; ++i)
          if (current_footprint_bins[i].candidates > 0 || current_footprint_bins[i].accepted > 0)
            printDiagnosticRow(footprint_names[i], current_footprint_bins[i]);
      }

      static const char *anisotropy_names[] = {"1-1.5x", "1.5-2x", "2-4x", ">4x"};
      printDiagnosticHeader("TEXTURE_DIAGNOSTICS_BY_ANISOTROPY");
      for (int i = 0; i < 4; ++i)
        if (anisotropy_bins[i].candidates > 0 || anisotropy_bins[i].accepted > 0)
          printDiagnosticRow(anisotropy_names[i], anisotropy_bins[i]);

      if (cross_camera_current_residual_en)
      {
        printDiagnosticHeader("CURRENT_CROSS_TEXTURE_BY_ANISOTROPY");
        for (int i = 0; i < 4; ++i)
          if (current_anisotropy_bins[i].candidates > 0 || current_anisotropy_bins[i].accepted > 0)
            printDiagnosticRow(anisotropy_names[i], current_anisotropy_bins[i]);
      }
    }
    printDivider();
  };

  printStatsSet("VIO Usage Stats (Window)", "window_frames", usage_stats_frames_, usage_camera_pairs_,
                usage_current_camera_pairs_,
                usage_current_view_angle_bins_, usage_current_footprint_bins_, usage_current_anisotropy_bins_,
                usage_cross_region_pairs_, usage_view_angle_bins_, usage_footprint_bins_, usage_anisotropy_bins_,
                usage_pose_all_, usage_pose_same_, usage_pose_cross_, usage_pose_current_cross_, false);
  printStatsSet("VIO Usage Stats (Run Total)", "total_frames", usage_stats_total_frames_, usage_total_camera_pairs_,
                usage_total_current_camera_pairs_,
                usage_total_current_view_angle_bins_, usage_total_current_footprint_bins_,
                usage_total_current_anisotropy_bins_,
                usage_total_cross_region_pairs_, usage_total_view_angle_bins_, usage_total_footprint_bins_,
                usage_total_anisotropy_bins_, usage_total_pose_all_,
                usage_total_pose_same_, usage_total_pose_cross_, usage_total_pose_current_cross_, true);
}
void VIOManager::maybePrintUsageStatsTable(int frame_id)
{
  if (!usage_stats_en) return;
  ++usage_stats_frames_;
  ++usage_stats_total_frames_;
  if (usage_stats_frames_ < std::max(1, usage_stats_window)) return;
  printUsageStatsTable(frame_id);
  resetUsageStatsWindow();
}

void VIOManager::printOnlineCalibrationStatsTable(int frame_id) const
{
  if (state == nullptr || (!online_extrinsic_en && !online_time_offset_en)) return;
  const int state_dim = state->stateDim();
  const bool cov_valid = state->cov.rows() == state_dim && state->cov.cols() == state_dim;
  const auto safeSigma = [](double var) {
    return std::sqrt(std::max(0.0, var));
  };

  printf("\033[1;35m+--------------------------------------------------------------------------------------------------------------+\033[0m\n");
  printf("\033[1;35m|                                      VIO Online Calibration Stats                                             |\033[0m\n");
  printf("\033[1;35m+--------------------------------------------------------------------------------------------------------------+\033[0m\n");
  printf("\033[1;35m| frame=%d visual_updates=%lld obs=%d | ext try/accept=%lld/%lld | td try/accept=%lld/%lld |\033[0m\n",
         frame_id, online_calib_visual_update_count_, online_calib_last_total_observations_,
         online_extrinsic_attempt_count_, online_extrinsic_accept_count_,
         online_time_offset_attempt_count_, online_time_offset_accept_count_);
  printf("\033[1;35m| last max update: dR=%.6f deg | dP=%.6f cm | dtd=%.6f ms |\033[0m\n",
         online_calib_last_max_rot_update_deg_,
         online_calib_last_max_trans_update_cm_,
         online_calib_last_max_time_update_ms_);

  if (online_extrinsic_en)
  {
    printf("\033[1;35m+--------------------------------------------------------------------------------------------------------------+\033[0m\n");
    printf("\033[1;35m| Extrinsic per camera: cam en actR actT accepts dR_prior_deg(x y z) dP_prior_cm(x y z) sigma(rot_deg trans_cm) |\033[0m\n");
    printf("\033[1;35m+--------------------------------------------------------------------------------------------------------------+\033[0m\n");
    const int camera_count = std::min(numCameras(), state->num_cameras);
    for (int camera_id = 0; camera_id < camera_count; ++camera_id)
    {
      const bool enabled = isOnlineExtrinsicEnabledForCamera(camera_id);
      const int active_rot = camera_id < static_cast<int>(online_calib_last_active_extrinsic_rot_.size())
                                 ? online_calib_last_active_extrinsic_rot_[camera_id]
                                 : 0;
      const int active_trans = camera_id < static_cast<int>(online_calib_last_active_extrinsic_trans_.size())
                                   ? online_calib_last_active_extrinsic_trans_[camera_id]
                                   : 0;
      const long long accepts = camera_id < static_cast<int>(online_extrinsic_camera_accept_count_.size())
                                    ? online_extrinsic_camera_accept_count_[camera_id]
                                    : 0;
      V3D dR_prior_deg = V3D::Zero();
      V3D dP_prior_cm = V3D::Zero();
      if (camera_id < static_cast<int>(state->Rcl.size()) &&
          camera_id < static_cast<int>(state->Rcl_prior.size()))
      {
        const M3D dR_cl = state->Rcl_prior[camera_id].transpose() * state->Rcl[camera_id];
        dR_prior_deg = Log(dR_cl) * kRadiansToDegrees;
      }
      if (camera_id < static_cast<int>(state->Pcl.size()) &&
          camera_id < static_cast<int>(state->Pcl_prior.size()))
        dP_prior_cm = (state->Pcl[camera_id] - state->Pcl_prior[camera_id]) * 100.0;

      double rot_sigma_deg = 0.0;
      double trans_sigma_cm = 0.0;
      if (cov_valid)
      {
        const int ridx = state->extrinsicRotIndex(camera_id);
        const int tidx = state->extrinsicTransIndex(camera_id);
        if (ridx + 2 < state->cov.rows())
        {
          const double rot_var =
              (state->cov(ridx, ridx) + state->cov(ridx + 1, ridx + 1) + state->cov(ridx + 2, ridx + 2)) / 3.0;
          rot_sigma_deg = safeSigma(rot_var) * kRadiansToDegrees;
        }
        if (tidx + 2 < state->cov.rows())
        {
          const double trans_var =
              (state->cov(tidx, tidx) + state->cov(tidx + 1, tidx + 1) + state->cov(tidx + 2, tidx + 2)) / 3.0;
          trans_sigma_cm = safeSigma(trans_var) * 100.0;
        }
      }

      printf("\033[1;35m| cam=%d en=%d act=%d/%d accepts=%lld dR=(%.6f %.6f %.6f) dP=(%.6f %.6f %.6f) sig=(%.6f %.6f) |\033[0m\n",
             camera_id, enabled ? 1 : 0, active_rot, active_trans, accepts,
             dR_prior_deg[0], dR_prior_deg[1], dR_prior_deg[2],
             dP_prior_cm[0], dP_prior_cm[1], dP_prior_cm[2],
             rot_sigma_deg, trans_sigma_cm);
      if (camera_id < static_cast<int>(state->Rcl.size()) &&
          camera_id < static_cast<int>(state->Pcl.size()))
      {
        const M3D &Rcl = state->Rcl[camera_id];
        const V3D &Pcl = state->Pcl[camera_id];
        printf("\033[1;35m|   Rcl: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f] Pcl: [%.6f, %.6f, %.6f] last=(%.6fdeg %.6fcm) |\033[0m\n",
               Rcl(0, 0), Rcl(0, 1), Rcl(0, 2),
               Rcl(1, 0), Rcl(1, 1), Rcl(1, 2),
               Rcl(2, 0), Rcl(2, 1), Rcl(2, 2),
               Pcl[0], Pcl[1], Pcl[2],
               camera_id < static_cast<int>(online_calib_last_extrinsic_rot_update_deg_.size())
                   ? online_calib_last_extrinsic_rot_update_deg_[camera_id]
                   : 0.0,
               camera_id < static_cast<int>(online_calib_last_extrinsic_trans_update_cm_.size())
                   ? online_calib_last_extrinsic_trans_update_cm_[camera_id]
                   : 0.0);
      }
    }
  }

  if (online_time_offset_en)
  {
    printf("\033[1;35m+--------------------------------------------------------------------------------------------------------------+\033[0m\n");
    printf("\033[1;35m| Time offset per group: group en active accepts tracks avg_px_vel td_ms delta_prior_ms last_update_ms sigma_ms   |\033[0m\n");
    printf("\033[1;35m+--------------------------------------------------------------------------------------------------------------+\033[0m\n");
    for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
    {
      const bool enabled = isOnlineTimeOffsetEnabledForGroup(group_id);
      const int active = group_id < static_cast<int>(online_calib_last_active_time_groups_.size())
                             ? online_calib_last_active_time_groups_[group_id]
                             : 0;
      const long long accepts = group_id < static_cast<int>(online_time_offset_group_accept_count_.size())
                                    ? online_time_offset_group_accept_count_[group_id]
                                    : 0;
      const int tracks = group_id < static_cast<int>(online_calib_last_time_tracks_.size())
                             ? online_calib_last_time_tracks_[group_id]
                             : 0;
      const double avg_pixel_velocity = group_id < static_cast<int>(online_calib_last_time_avg_pixel_velocity_.size())
                                            ? online_calib_last_time_avg_pixel_velocity_[group_id]
                                            : 0.0;
      const double last_update_ms = group_id < static_cast<int>(online_calib_last_time_update_ms_.size())
                                        ? online_calib_last_time_update_ms_[group_id]
                                        : 0.0;
      const double td_ms = group_id < state->time_offset.size() ? state->time_offset[group_id] * 1.0e3 : 0.0;
      const double prior_ms =
          group_id < state->time_offset_prior.size() ? state->time_offset_prior[group_id] * 1.0e3 : 0.0;
      double sigma_ms = 0.0;
      if (cov_valid)
      {
        const int idx = state->timeOffsetIndex(group_id);
        if (idx < state->cov.rows()) sigma_ms = safeSigma(state->cov(idx, idx)) * 1.0e3;
      }
      printf("\033[1;35m| group=%d en=%d active=%d accepts=%lld tracks=%d avg_px_vel=%.6f td=%.6f delta=%.6f last=%.6f sigma=%.6f |\033[0m\n",
             group_id, enabled ? 1 : 0, active, accepts, tracks, avg_pixel_velocity,
             td_ms, td_ms - prior_ms, last_update_ms, sigma_ms);
    }
  }
  printf("\033[1;35m+--------------------------------------------------------------------------------------------------------------+\033[0m\n");
}

void VIOManager::retrieveFromVisualSparseMapVirtual(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                                    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
                                                    int raw_score_point_quota)
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

  auto addRetrieveCandidate = [&](int index, VisualPoint *pt, float cur_dist) {
    if (index < 0 || index >= ctx.length || pt == nullptr) return;
    ctx.grid_num[index] = TYPE_MAP;
    if (virtual_raw_score_select_en)
    {
      std::vector<VisualPoint *> &grid_candidates = ctx.retrieve_voxel_point_candidates[index];
      if (std::find(grid_candidates.begin(), grid_candidates.end(), pt) == grid_candidates.end())
        grid_candidates.push_back(pt);
    }
    if (cur_dist <= ctx.map_dist[index])
    {
      ctx.map_dist[index] = cur_dist;
      ctx.retrieve_voxel_points[index] = pt;
    }
  };

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
      const float cur_dist = static_cast<float>((ctx.new_frame->pos() - pt->pos_).norm());
      addRetrieveCandidate(index, pt, cur_dist);
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
            const float cur_dist = static_cast<float>((ctx.new_frame->pos() - pt->pos_).norm());
            addRetrieveCandidate(index, pt, cur_dist);
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
    double shi_tomasi_score = 0.0;
    int reference_rank = 0;
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
    vector<float> current_core;
    std::vector<double> level_sse;
    std::vector<double> level_ncc;
    std::vector<uint8_t> level_valid;
    std::vector<uint8_t> level_active;
    float error = 0.0f;
    double inverse_reference_exposure = 0.0;
    double ncc = std::numeric_limits<double>::quiet_NaN();
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
  auto computeShiTomasiCandidateScore = [&](const V2D &raw_px) {
    const float score = vk::shiTomasiScore(img, static_cast<int>(raw_px[0]), static_cast<int>(raw_px[1]));
    return std::isfinite(score) ? static_cast<double>(score) : 0.0;
  };

  vector<VirtualCandidate> candidates;
  candidates.reserve(ctx.length);
  for (int i = 0; i < ctx.length; ++i)
  {
    if (ctx.grid_num[i] != TYPE_MAP) continue;
    ++virtual_map_grid_count_;
    const std::vector<VisualPoint *> &grid_candidates = ctx.retrieve_voxel_point_candidates[i];
    const int grid_candidate_count = virtual_raw_score_select_en && !grid_candidates.empty()
                                         ? static_cast<int>(grid_candidates.size())
                                         : 1;
    for (int grid_candidate_index = 0; grid_candidate_index < grid_candidate_count; ++grid_candidate_index)
    {
    VisualPoint *pt = virtual_raw_score_select_en && !grid_candidates.empty()
                          ? grid_candidates[grid_candidate_index]
                          : ctx.retrieve_voxel_points[i];
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

    if (visual_map_manage_en && visual_ref_current_select_en && visual_map_manage_shadow_en)
    {
      const int candidate_limit = visual_ref_fallback_en ? visual_ref_max_candidates : 1;
      if (!selectManagedReferenceCandidates(ctx, *pt, candidate_limit).empty())
        ++visual_map_manage_stats_.dynamic_selected;
    }
    if (visual_map_manage_en && visual_ref_current_select_en && !visual_map_manage_shadow_en)
    {
      const int candidate_limit = visual_ref_fallback_en ? visual_ref_max_candidates : 1;
      std::vector<Feature *> managed_refs = selectManagedReferenceCandidates(ctx, *pt, candidate_limit);
      if (managed_refs.empty())
      {
        ++virtual_candidate_ref_missing_count_;
        recordRejectedPoint(raw_px, REJECT_DRAW_REF_MISSING);
        continue;
      }
      for (int rank = 0; rank < static_cast<int>(managed_refs.size()); ++rank)
      {
        Feature *managed_ref = managed_refs[rank];
        if (managed_ref == nullptr || !managed_ref->virtual_patch_valid_) continue;
        VirtualCandidate candidate;
        candidate.point = pt;
        candidate.reference = managed_ref;
        candidate.point_c_seed = pt_c_seed;
        candidate.current_raw_center_px = raw_px;
        candidate.shi_tomasi_score = computeShiTomasiCandidateScore(raw_px);
        candidate.reference_rank = rank;
        candidates.push_back(candidate);
      }
      ++visual_map_manage_stats_.dynamic_selected;
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
    candidate.shi_tomasi_score = computeShiTomasiCandidateScore(raw_px);
    candidates.push_back(candidate);
    }
  }

  if (virtual_raw_score_select_en && raw_score_point_quota >= 0)
  {
    std::vector<int> primary_indices;
    primary_indices.reserve(candidates.size());
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
      if (candidates[i].reference_rank == 0) primary_indices.push_back(i);

    const int keep_points = std::min(raw_score_point_quota, static_cast<int>(primary_indices.size()));
    if (keep_points < static_cast<int>(primary_indices.size()))
      std::partial_sort(primary_indices.begin(), primary_indices.begin() + keep_points, primary_indices.end(),
                        [&](int lhs, int rhs) {
                          return candidates[lhs].shi_tomasi_score > candidates[rhs].shi_tomasi_score;
                        });

    std::set<VisualPoint *> selected_points;
    for (int i = 0; i < keep_points; ++i) selected_points.insert(candidates[primary_indices[i]].point);

    vector<VirtualCandidate> selected_candidates;
    selected_candidates.reserve(candidates.size());
    for (const VirtualCandidate &candidate : candidates)
      if (selected_points.count(candidate.point) != 0) selected_candidates.push_back(candidate);
    candidates.swap(selected_candidates);
  }
  virtual_candidate_select_time_ = omp_get_wtime() - candidate_select_start;
  virtual_candidate_count_ = static_cast<int>(candidates.size());

  vector<VirtualCandidateResult> results(candidates.size());
  const double parallel_track_start = omp_get_wtime();
  for (int fallback_pass = 0; fallback_pass < 2; ++fallback_pass)
  {
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index)
    {
    const VirtualCandidate &candidate = candidates[candidate_index];
    const bool is_primary = candidate.reference_rank == 0;
    if (is_primary != (fallback_pass == 0)) continue;
    if (!is_primary)
    {
      int primary_index = candidate_index - 1;
      while (primary_index >= 0 && candidates[primary_index].point == candidate.point &&
             candidates[primary_index].reference_rank != 0)
        --primary_index;
      if (primary_index < 0 || candidates[primary_index].point != candidate.point) continue;
      const int primary_rejection = results[primary_index].rejection;
      if (primary_rejection != VIRTUAL_REJECT_NCC && primary_rejection != VIRTUAL_REJECT_PHOTOMETRIC)
        continue;
    }
    VirtualCandidateResult &result = results[candidate_index];
    VisualPoint *pt = candidate.point;
    Feature *ref_ftr = candidate.reference;
    if (ref_ftr == nullptr || !refreshReferenceCalibration(*ref_ftr))
    {
      result.rejection = VIRTUAL_REJECT_REFERENCE_SUPPORT;
      continue;
    }

    if (!buildVirtualFrameRotation(ctx, candidate.point_c_seed, candidate.current_raw_center_px,
                                   result.track.R_vcur_from_ccur_seed, result.track.R_ccur_from_vcur_seed))
    {
      result.rejection = VIRTUAL_REJECT_ROTATION;
      continue;
    }
    result.track.T_vcur_w_seed = composeVirtualPose(result.track.R_vcur_from_ccur_seed, ctx.new_frame->T_f_w_);
    result.track.cur_support.T_v_w_seed = result.track.T_vcur_w_seed;

    const SE3d T_vcur_vref = result.track.T_vcur_w_seed * ref_ftr->T_v_w_.inverse();
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
    if (visual_map_manage_en && visual_ref_current_select_en)
    {
      const Eigen::JacobiSVD<Matrix2d> affine_svd(result.track.A_cur_ref);
      const V2D singular = affine_svd.singularValues();
      if (!singular.array().isFinite().all() || singular[1] <= 1.0e-9 ||
          singular[0] / singular[1] > visual_ref_max_anisotropy)
      {
        result.rejection = VIRTUAL_REJECT_AFFINE_MATRIX;
        continue;
      }
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
      // Reuse the stored reference support instead of regenerating it every frame.
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
    const int usage_levels = std::max(1, patch_pyrimid_level);
    const bool evaluate_all_levels = zncc_residual_en || ncc_en || usage_stats_en;
    vector<float> current_core_all(evaluate_all_levels ? warp_len : patch_size_total, 0.0f);
    result.level_sse.assign(usage_levels, std::numeric_limits<double>::quiet_NaN());
    result.level_ncc.assign(usage_levels, std::numeric_limits<double>::quiet_NaN());
    result.level_valid.assign(usage_levels, 0);
    result.level_active.assign(usage_levels, 0);
    const V3D point_vcur = result.track.T_vcur_w_seed * pt->pos_;
    if (point_vcur[2] <= virtual_min_z)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_Z;
      continue;
    }
    const V2D virtual_center = virtualProject(point_vcur);
    const bool current_core_ok = virtual_sparse_patch_en
        ? sampleSparseVirtualCorePatch(ctx, img, result.track.R_ccur_from_vcur_seed,
                                       virtual_center, 1, current_core_all.data())
        : sampleVirtualCorePatch(result.track.cur_support, virtual_center, 1, current_core_all.data());
    result.level_valid[0] = current_core_ok ? 1 : 0;
    if (!current_core_ok)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_CORE;
      continue;
    }
    if (evaluate_all_levels)
    {
      for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
      {
        const int scale = 1 << pyramid_level;
        const bool level_ok = virtual_sparse_patch_en
            ? sampleSparseVirtualCorePatch(ctx, img, result.track.R_ccur_from_vcur_seed,
                                           virtual_center, scale,
                                           current_core_all.data() + patch_size_total * pyramid_level)
            : sampleVirtualCorePatch(result.track.cur_support, virtual_center, scale,
                                     current_core_all.data() + patch_size_total * pyramid_level);
        result.level_valid[pyramid_level] = level_ok ? 1 : 0;
      }
    }

    if (zncc_residual_en)
    {
      for (int pyramid_level = 0; pyramid_level < patch_pyrimid_level; ++pyramid_level)
      {
        if (result.level_valid[pyramid_level] == 0) continue;
        const int offset = patch_size_total * pyramid_level;
        if (!normalizedPatchMetrics(result.warped_reference.data() + offset,
                                    current_core_all.data() + offset,
                                    patch_size_total, zncc_min_std,
                                    result.level_sse[pyramid_level],
                                    result.level_ncc[pyramid_level]))
        {
          result.level_valid[pyramid_level] = 0;
        }
      }
      if (result.level_valid[0] == 0)
      {
        result.current_core_time = omp_get_wtime() - current_core_start;
        result.rejection = VIRTUAL_REJECT_PHOTOMETRIC;
        continue;
      }
      result.error = static_cast<float>(result.level_sse[0]);
    }
    else
    {
      double level0_error = 0.0;
      for (int k = 0; k < patch_size_total; ++k)
      {
        const double residual = ref_ftr->inv_expo_time_ * result.warped_reference[k] -
                                state->inv_expo_time[ctx.camera_id] * current_core_all[k];
        level0_error += residual * residual;
      }
      result.error = static_cast<float>(level0_error);
      result.level_sse[0] = level0_error;
      if (evaluate_all_levels)
      {
        for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
        {
          if (result.level_valid[pyramid_level] == 0) continue;
          const int offset = patch_size_total * pyramid_level;
          double level_error = 0.0;
          for (int k = 0; k < patch_size_total; ++k)
          {
            const double residual = ref_ftr->inv_expo_time_ * result.warped_reference[offset + k] -
                                    state->inv_expo_time[ctx.camera_id] * current_core_all[offset + k];
            level_error += residual * residual;
          }
          result.level_sse[pyramid_level] = level_error;
        }
      }
      if (ncc_en || usage_stats_en)
      {
        result.level_ncc[0] = calculateNCC(result.warped_reference.data(), current_core_all.data(), patch_size_total);
        if (evaluate_all_levels)
        {
          for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
          {
            if (result.level_valid[pyramid_level] == 0) continue;
            const int offset = patch_size_total * pyramid_level;
            result.level_ncc[pyramid_level] =
                calculateNCC(result.warped_reference.data() + offset, current_core_all.data() + offset, patch_size_total);
          }
        }
      }
    }
    result.ncc = result.level_ncc[0];
    bool any_level_active = false;
    bool any_photometric_reject = false;
    if (zncc_residual_en)
    {
      for (int level = 0; level < usage_levels; ++level)
      {
        if (result.level_valid[level] == 0 || !std::isfinite(result.level_ncc[level]) ||
            !std::isfinite(result.level_sse[level]))
          continue;
        if (ncc_en && result.level_ncc[level] < nccThresholdForLevel(level)) continue;
        result.level_active[level] = 1;
        any_level_active = true;
      }
    }
    else if (ncc_en)
    {
      for (int level = 0; level < usage_levels; ++level)
      {
        if (result.level_valid[level] == 0 || !std::isfinite(result.level_ncc[level]) ||
            !std::isfinite(result.level_sse[level]))
          continue;
        if (result.level_ncc[level] < nccThresholdForLevel(level)) continue;
        if (result.level_sse[level] > outlier_threshold * patch_size_total)
        {
          any_photometric_reject = true;
          continue;
        }
        result.level_active[level] = 1;
        any_level_active = true;
      }
    }
    else if (result.error <= outlier_threshold * patch_size_total)
    {
      std::fill(result.level_active.begin(), result.level_active.end(), 1);
      any_level_active = true;
    }
    else
    {
      any_photometric_reject = true;
    }
    if (!any_level_active)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = any_photometric_reject ? VIRTUAL_REJECT_PHOTOMETRIC : VIRTUAL_REJECT_NCC;
      continue;
    }
    if (result.level_active[0] == 0)
      for (int level = 1; level < usage_levels; ++level)
        if (result.level_active[level] != 0)
        {
          result.error = static_cast<float>(result.level_sse[level]);
          break;
        }
    result.current_core_time = omp_get_wtime() - current_core_start;

    if (runtime_support_dump_en)
      result.current_core.assign(current_core_all.begin(), current_core_all.begin() + patch_size_total);
    result.inverse_reference_exposure = ref_ftr->inv_expo_time_;
    result.track.valid = true;
    result.valid = true;
    }
  }
  virtual_parallel_track_time_ = omp_get_wtime() - parallel_track_start;

  const double result_collect_start = omp_get_wtime();
  vector<int> virtual_search_level_count(std::max(virtual_max_search_level + 1, 1), 0);
  vector<int> virtual_warp_fail_search_level_count(std::max(virtual_max_search_level + 1, 1), 0);
  vector<int> virtual_warp_fail_pyramid_level_count(std::max(patch_pyrimid_level, 1), 0);
  std::set<VisualPoint *> accepted_managed_points;
  std::set<VisualPoint *> fallback_allowed_points;
  for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index)
  {
    const VirtualCandidate &collect_candidate = candidates[candidate_index];
    if (collect_candidate.reference_rank > 0)
    {
      if (accepted_managed_points.count(collect_candidate.point) != 0) continue;
      if (fallback_allowed_points.count(collect_candidate.point) == 0) continue;
      ++visual_map_manage_stats_.fallback_attempted;
    }
    VirtualCandidateResult &result = results[candidate_index];
    if (usage_stats_en)
    {
      const VirtualCandidate &candidate = candidates[candidate_index];
      const Matrix2d usage_affine = result.search_level >= 0 ? result.track.A_cur_ref : Matrix2d::Zero();
      recordUsageObservation(ctx, *candidate.reference, *candidate.point, candidate.current_raw_center_px, usage_affine, false);
      recordUsagePreGateMetrics(ctx, *candidate.reference, *candidate.point, candidate.current_raw_center_px,
                                usage_affine, result.level_sse, result.level_ncc, result.level_valid);
    }
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
      if (collect_candidate.reference_rank == 0 &&
          (result.rejection == VIRTUAL_REJECT_NCC || result.rejection == VIRTUAL_REJECT_PHOTOMETRIC))
        fallback_allowed_points.insert(collect_candidate.point);
      if (result.rejection == VIRTUAL_REJECT_NCC || result.rejection == VIRTUAL_REJECT_PHOTOMETRIC)
        recordManagedReferenceRejection(*collect_candidate.point, *collect_candidate.reference,
                                        ctx.new_frame->id_, ctx.camera_id);
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
    if (collect_candidate.reference_rank > 0) ++visual_map_manage_stats_.fallback_accepted;
    accepted_managed_points.insert(collect_candidate.point);
    if (usage_stats_en)
    {
      const VirtualCandidate &candidate = candidates[candidate_index];
      recordUsageObservation(ctx, *candidate.reference, *candidate.point, candidate.current_raw_center_px,
                             result.track.A_cur_ref, true, result.level_sse, result.level_ncc, result.level_active);
    }
    const VirtualCandidate &accepted_candidate = candidates[candidate_index];
    if (runtime_support_dump_en && accepted_candidate.point != nullptr)
    {
      if (accepted_candidate.point->runtime_support_dump_id_ < 0)
        accepted_candidate.point->runtime_support_dump_id_ = runtime_support_dump_next_point_id_++;
      const int track_count = ++accepted_candidate.point->runtime_support_track_count_;
      if (track_count > runtime_support_dump_best_track_count_)
      {
        runtime_support_dump_best_track_count_ = track_count;
        runtime_support_dump_best_point_ = accepted_candidate.point;
        const int submap_index = static_cast<int>(ctx.visual_submap->voxel_points.size());
        dumpRuntimeSupportObservation(ctx, *accepted_candidate.point, *accepted_candidate.reference,
                                      accepted_candidate.current_raw_center_px, result.track,
                                      result.warped_reference, result.current_core,
                                      track_count, submap_index, result.error, result.ncc);
      }
    }
    ctx.visual_submap->voxel_points.push_back(candidates[candidate_index].point);
    ctx.visual_submap->reference_features.push_back(candidates[candidate_index].reference);
    ctx.visual_submap->propa_errors.push_back(result.error);
    ctx.visual_submap->search_levels.push_back(result.track.search_level);
    ctx.visual_submap->warp_affines.push_back(result.track.A_cur_ref);
    ctx.visual_submap->errors.push_back(result.error);
    ctx.visual_submap->warp_patch.push_back(std::move(result.warped_reference));
    ctx.visual_submap->inv_expo_list.push_back(result.inverse_reference_exposure);
    ctx.visual_submap->level_active.push_back(result.level_active);
    if (usage_stats_en)
    {
      ctx.visual_submap->usage_ncc_levels.push_back(result.level_ncc);
      ctx.visual_submap->usage_sse_levels.push_back(result.level_sse);
      ctx.visual_submap->usage_level_valid.push_back(result.level_active);
    }
    ctx.visual_submap->virtual_track_patches.push_back(std::move(result.track));
    appendManagedSubmapMetadata(ctx, *candidates[candidate_index].point,
                                *candidates[candidate_index].reference);
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
                                             const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
                                             int raw_score_point_quota)
{
  if (virtual_fisheye_patch_en)
  {
    retrieveFromVisualSparseMapVirtual(ctx, img, pg, plane_map, raw_score_point_quota);
    return;
  }
  if (feat_map.size() <= 0)
  {
    // printf("[ VIO Debug ] retrieve skip camera_id=%d reason=empty_feat_map pg=%zu img=%dx%d\n",
    //        ctx.camera_id, pg.size(), img.cols, img.rows);
    // fflush(stdout);
    return;
  }
  // printf("[ VIO Debug ] retrieve start camera_id=%d img=%dx%d pg=%zu feat_map=%zu cross_ref=%d normal=%d raycast=%d border=%d grid=%d\n",
  //        ctx.camera_id, img.cols, img.rows, pg.size(), feat_map.size(), cross_camera_reference_en ? 1 : 0,
  //        normal_en ? 1 : 0, raycast_en ? 1 : 0, border, ctx.length);
  // fflush(stdout);
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
      loc_xyz[j] = static_cast<int>(std::floor(pt_w[j] / voxel_size));
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

    V2D px = ctx.cam->world2cam(pt_c);
    if (px.array().isFinite().all() && ctx.cam->isInFrame(px.cast<int>(), border))
    {
      // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
      float range = static_cast<float>(pt_c.norm());
      int col = int(px[0]);
      int row = int(px[1]);
      it[ctx.width * row + col] = range;
      ++debug_depth_samples;
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // printf("[ VIO Debug ] retrieve depth camera_id=%d depth_samples=%d sub_voxels=%zu elapsed=%.6f\n",
  //        ctx.camera_id, debug_depth_samples, ctx.sub_feat_map.size(), omp_get_wtime() - ts0);
  // fflush(stdout);

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;
  auto addRetrieveCandidate = [&](int index, VisualPoint *pt, float cur_dist) {
    if (index < 0 || index >= ctx.length || pt == nullptr) return;
    ctx.grid_num[index] = TYPE_MAP;
    if (virtual_raw_score_select_en)
    {
      std::vector<VisualPoint *> &grid_candidates = ctx.retrieve_voxel_point_candidates[index];
      if (std::find(grid_candidates.begin(), grid_candidates.end(), pt) == grid_candidates.end())
        grid_candidates.push_back(pt);
    }
    if (cur_dist <= ctx.map_dist[index])
    {
      ctx.map_dist[index] = cur_dist;
      ctx.retrieve_voxel_points[index] = pt;
    }
  };

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

        // V3D norm_vec(ctx.new_frame->T_f_w_.rotationMatrix() * pt->normal_);
        // V3D dir(ctx.new_frame->T_f_w_ * pt->pos_);
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(ctx.new_frame->w2c(pt->pos_));
        if (ctx.cam->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / ctx.grid_size) * ctx.grid_n_width + static_cast<int>(pc[0] / ctx.grid_size);
          Vector3d obs_vec(ctx.new_frame->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          addRetrieveCandidate(index, pt, cur_dist);
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
          loc_xyz[j] = static_cast<int>(std::floor(sample_point_w[j] / voxel_size));
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

            // V3D norm_vec(ctx.new_frame->T_f_w_.rotationMatrix() * pt->normal_);
            // V3D dir(ctx.new_frame->T_f_w_ * pt->pos_);
            // dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(ctx.new_frame->w2c(pt->pos_));

            if (ctx.cam->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / ctx.grid_size) * ctx.grid_n_width + static_cast<int>(pc[0] / ctx.grid_size);
              Vector3d obs_vec(ctx.new_frame->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              addRetrieveCandidate(index, pt, cur_dist);
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

  // printf("[ VIO Debug ] retrieve map camera_id=%d sub_voxels=%zu deleted_voxels=%zu raycast=%d\n",
  //        ctx.camera_id, ctx.sub_feat_map.size(), DeleteKeyList.size(), raycast_en ? 1 : 0);
  // fflush(stdout);
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
      const std::vector<VisualPoint *> &grid_candidates = ctx.retrieve_voxel_point_candidates[i];
      const int grid_candidate_count = virtual_raw_score_select_en && !grid_candidates.empty()
                                           ? static_cast<int>(grid_candidates.size())
                                           : 1;
      for (int grid_candidate_index = 0; grid_candidate_index < grid_candidate_count; ++grid_candidate_index)
      {
      ++debug_grid_candidates;
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = virtual_raw_score_select_en && !grid_candidates.empty()
                            ? grid_candidates[grid_candidate_index]
                            : ctx.retrieve_voxel_points[i];
      if (pt == nullptr) continue;
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

          double delta_dist = abs(pt_cam.norm() - depth);

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
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      std::vector<Feature *> reference_candidates;
      if (visual_map_manage_en && visual_ref_current_select_en && visual_map_manage_shadow_en)
      {
        const int candidate_limit = visual_ref_fallback_en ? visual_ref_max_candidates : 1;
        if (!selectManagedReferenceCandidates(ctx, *pt, candidate_limit).empty())
          ++visual_map_manage_stats_.dynamic_selected;
      }
      if (visual_map_manage_en && visual_ref_current_select_en && !visual_map_manage_shadow_en)
      {
        const int candidate_limit = visual_ref_fallback_en ? visual_ref_max_candidates : 1;
        reference_candidates = selectManagedReferenceCandidates(ctx, *pt, candidate_limit);
        ++visual_map_manage_stats_.dynamic_selected;
      }
      else
      {
        Feature *ref_ftr = nullptr;
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
              if (ref_patch_temp == nullptr || ref_patch_temp->pending_delete_ ||
                  ref_patch_temp->ref_state_ == Feature::RefState::RETIRED)
                continue;
              if (!cross_camera_reference_en && ref_patch_temp->camera_id_ != ctx.camera_id) continue;
              float *patch_temp = ref_patch_temp->patch_;
              float phtometric_errors = 0.0;
              int count = 0;
              for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
              {
                if (*itm == nullptr || *itm == ref_patch_temp || (*itm)->pending_delete_) continue;
                if (!cross_camera_reference_en && (*itm)->camera_id_ != ctx.camera_id) continue;
                float *patch_cache = (*itm)->patch_;
                for (int ind = 0; ind < patch_size_total; ind++)
                  phtometric_errors += (patch_temp[ind] - patch_cache[ind]) *
                                       (patch_temp[ind] - patch_cache[ind]);
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
        else if (!pt->getCloseViewObs(ctx.new_frame->pos(), ref_ftr, pc,
                                      cross_camera_reference_en ? -1 : ctx.camera_id))
          ref_ftr = nullptr;
        if (ref_ftr != nullptr) reference_candidates.push_back(ref_ftr);
      }
      if (reference_candidates.empty()) continue;

      for (int reference_rank = 0; reference_rank < static_cast<int>(reference_candidates.size()); ++reference_rank)
      {
        Feature *ref_ftr = reference_candidates[reference_rank];
        if (reference_rank > 0) ++visual_map_manage_stats_.fallback_attempted;
        if (ref_ftr == nullptr || ref_ftr->camera_id_ < 0 || ref_ftr->camera_id_ >= numCameras())
        {
          ++debug_ref_invalid;
          break;
        }
        if (!refreshReferenceCalibration(*ref_ftr))
        {
          ++debug_ref_invalid;
          break;
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
      if (usage_stats_en) recordUsageObservation(ctx, *ref_ftr, *pt, pc, A_cur_ref_zero, false);
      if (!debug_affine_ok)
      {
        ++debug_affine_bad;
        if (debug_affine_bad <= 5)
        {
          // printf("[ VIO Debug ] retrieve affine suspicious camera_id=%d ref_camera_id=%d det=%.6e search_level=%d finite=%d\n",
          //        ctx.camera_id, ref_ftr->camera_id_, debug_affine_det, search_level, debug_affine_ok ? 1 : 0);
          // fflush(stdout);
        }
      }
      // Shadow mode must remain observational: it may evaluate managed
      // references above, but it must not gate the legacy raw tracking path.
      if (visual_map_manage_en && visual_ref_current_select_en && !visual_map_manage_shadow_en)
      {
        if (!debug_affine_ok) break;
        const Eigen::JacobiSVD<Matrix2d> affine_svd(A_cur_ref_zero);
        const V2D singular = affine_svd.singularValues();
        if (!singular.array().isFinite().all() || singular[1] <= 1.0e-9 ||
            singular[0] / singular[1] > visual_ref_max_anisotropy)
          break;
      }
      if (debug_warp_logs < 8)
      {
        // printf("[ VIO Debug ] retrieve warp camera_id=%d ref_camera_id=%d search_level=%d det=%.6e finite=%d ref_px=(%.2f,%.2f) ref_img=%dx%d ref_level=%d\n",
        //        ctx.camera_id, ref_ftr->camera_id_, search_level, debug_affine_det, debug_affine_ok ? 1 : 0,
        //        ref_ftr->px_[0], ref_ftr->px_[1], ref_ftr->img_.cols, ref_ftr->img_.rows, ref_ftr->level_);
        // fflush(stdout);
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
      if (!warp_ok) break;

      std::vector<double> usage_sse_levels(std::max(1, patch_pyrimid_level), std::numeric_limits<double>::quiet_NaN());
      std::vector<double> usage_ncc_levels(std::max(1, patch_pyrimid_level), std::numeric_limits<double>::quiet_NaN());
      std::vector<uint8_t> usage_level_valid(std::max(1, patch_pyrimid_level), 0);
      std::vector<uint8_t> level_active(std::max(1, patch_pyrimid_level), 0);
      getImagePatch(ctx, img, pc, patch_buffer.data(), 0);
      usage_level_valid[0] = 1;

      float error = 0.0;
      const bool evaluate_all_levels = zncc_residual_en || ncc_en || usage_stats_en;
      std::vector<float> current_patch_all;
      if (evaluate_all_levels)
      {
        current_patch_all.assign(warp_len, 0.0f);
        std::copy(patch_buffer.begin(), patch_buffer.begin() + patch_size_total, current_patch_all.begin());
        for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
        {
          getImagePatch(ctx, img, pc, current_patch_all.data(), pyramid_level);
          usage_level_valid[pyramid_level] = 1;
        }
      }

      double usage_ncc = std::numeric_limits<double>::quiet_NaN();
      if (zncc_residual_en)
      {
        for (int pyramid_level = 0; pyramid_level < std::max(1, patch_pyrimid_level); ++pyramid_level)
        {
          const int offset = patch_size_total * pyramid_level;
          const float *current_level = pyramid_level == 0
                                           ? patch_buffer.data()
                                           : current_patch_all.data() + offset;
          if (!normalizedPatchMetrics(patch_wrap.data() + offset, current_level,
                                      patch_size_total, zncc_min_std,
                                      usage_sse_levels[pyramid_level],
                                      usage_ncc_levels[pyramid_level]))
            usage_level_valid[pyramid_level] = 0;
        }
        if (usage_level_valid[0] == 0)
        {
          recordManagedReferenceRejection(*pt, *ref_ftr, ctx.new_frame->id_, ctx.camera_id);
          continue;
        }
        error = static_cast<float>(usage_sse_levels[0]);
        usage_ncc = usage_ncc_levels[0];
      }
      else
      {
        for (int pyramid_level = 0; pyramid_level < std::max(1, patch_pyrimid_level); ++pyramid_level)
        {
          if (pyramid_level > 0 && !evaluate_all_levels) break;
          const int offset = patch_size_total * pyramid_level;
          const float *current_level = pyramid_level == 0
                                           ? patch_buffer.data()
                                           : current_patch_all.data() + offset;
          double level_error = 0.0;
          for (int ind = 0; ind < patch_size_total; ++ind)
          {
            const double residual = ref_ftr->inv_expo_time_ * patch_wrap[offset + ind] -
                                    state->inv_expo_time[ctx.camera_id] * current_level[ind];
            level_error += residual * residual;
          }
          usage_sse_levels[pyramid_level] = level_error;
        }
        error = static_cast<float>(usage_sse_levels[0]);
      }
      if (!zncc_residual_en && (ncc_en || usage_stats_en))
      {
        usage_ncc_levels[0] = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        usage_ncc = usage_ncc_levels[0];
        if (evaluate_all_levels)
        {
          for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
          {
            const int offset = patch_size_total * pyramid_level;
            usage_ncc_levels[pyramid_level] =
                calculateNCC(patch_wrap.data() + offset, current_patch_all.data() + offset, patch_size_total);
          }
        }
      }
      if (usage_stats_en) recordUsagePreGateMetrics(ctx, *ref_ftr, *pt, pc, A_cur_ref_zero,
                                                     usage_sse_levels, usage_ncc_levels, usage_level_valid);
      bool any_level_active = false;
      if (zncc_residual_en)
      {
        for (int level = 0; level < std::max(1, patch_pyrimid_level); ++level)
        {
          if (usage_level_valid[level] == 0 || !std::isfinite(usage_ncc_levels[level]) ||
              !std::isfinite(usage_sse_levels[level]))
            continue;
          if (ncc_en && usage_ncc_levels[level] < nccThresholdForLevel(level)) continue;
          level_active[level] = 1;
          any_level_active = true;
        }
      }
      else if (ncc_en)
      {
        for (int level = 0; level < std::max(1, patch_pyrimid_level); ++level)
        {
          if (usage_level_valid[level] == 0 || !std::isfinite(usage_ncc_levels[level]) ||
              !std::isfinite(usage_sse_levels[level]))
            continue;
          if (usage_ncc_levels[level] < nccThresholdForLevel(level)) continue;
          if (usage_sse_levels[level] > outlier_threshold * patch_size_total) continue;
          level_active[level] = 1;
          any_level_active = true;
        }
      }
      else if (error <= outlier_threshold * patch_size_total)
      {
        std::fill(level_active.begin(), level_active.end(), 1);
        any_level_active = true;
      }
      if (!any_level_active)
      {
        recordManagedReferenceRejection(*pt, *ref_ftr, ctx.new_frame->id_, ctx.camera_id);
        continue;
      }
      if (level_active[0] == 0)
        for (int level = 1; level < std::max(1, patch_pyrimid_level); ++level)
          if (level_active[level] != 0)
          {
            error = static_cast<float>(usage_sse_levels[level]);
            break;
          }

      if (runtime_support_dump_en && pt != nullptr)
      {
        if (pt->runtime_support_dump_id_ < 0)
          pt->runtime_support_dump_id_ = runtime_support_dump_next_point_id_++;
        const int track_count = ++pt->runtime_support_track_count_;
        if (track_count > runtime_support_dump_best_track_count_)
        {
          runtime_support_dump_best_track_count_ = track_count;
          runtime_support_dump_best_point_ = pt;
          const int submap_index = static_cast<int>(ctx.visual_submap->voxel_points.size());
          std::vector<float> dump_current_core;
          const int safe_search_level = (search_level >= 0 && search_level < 20) ? search_level : 0;
          const int dump_scale = 1 << safe_search_level;
          if (!sampleRawCorePatchForDump(ctx, img, pc, dump_scale, dump_current_core))
            dump_current_core.assign(patch_buffer.begin(), patch_buffer.begin() + patch_size_total);
          dumpRuntimeSupportRawObservation(ctx, *pt, *ref_ftr, pc, search_level, patch_wrap,
                                           dump_current_core, track_count, submap_index,
                                           error, usage_ncc);
        }
      }
      if (usage_stats_en) recordUsageObservation(ctx, *ref_ftr, *pt, pc, A_cur_ref_zero, true,
                                                  usage_sse_levels, usage_ncc_levels, level_active);
      ctx.visual_submap->voxel_points.push_back(pt);
      ctx.visual_submap->reference_features.push_back(ref_ftr);
      ctx.visual_submap->propa_errors.push_back(error);
      ctx.visual_submap->search_levels.push_back(search_level);
      ctx.visual_submap->warp_affines.push_back(A_cur_ref_zero);
      ctx.visual_submap->errors.push_back(error);
      ctx.visual_submap->warp_patch.push_back(patch_wrap);
      ctx.visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);
      ctx.visual_submap->level_active.push_back(level_active);
      if (usage_stats_en)
      {
        ctx.visual_submap->usage_ncc_levels.push_back(usage_ncc_levels);
        ctx.visual_submap->usage_sse_levels.push_back(usage_sse_levels);
        ctx.visual_submap->usage_level_valid.push_back(level_active);
      }
      appendManagedSubmapMetadata(ctx, *pt, *ref_ftr);
      if (reference_rank > 0) ++visual_map_manage_stats_.fallback_accepted;

      ++debug_accepted;
      break;

      // t_5 += omp_get_wtime() - t_1;
      }
      }
    }
  }
  ctx.total_points = ctx.visual_submap->voxel_points.size();
  if (virtual_raw_score_select_en && raw_score_point_quota >= 0 && ctx.total_points > raw_score_point_quota)
  {
    std::vector<int> selected_indices(ctx.total_points);
    std::iota(selected_indices.begin(), selected_indices.end(), 0);
    std::partial_sort(selected_indices.begin(), selected_indices.begin() + raw_score_point_quota, selected_indices.end(),
                      [&](int lhs, int rhs) {
                        const float lhs_error = lhs < static_cast<int>(ctx.visual_submap->errors.size())
                                                    ? ctx.visual_submap->errors[lhs]
                                                    : std::numeric_limits<float>::max();
                        const float rhs_error = rhs < static_cast<int>(ctx.visual_submap->errors.size())
                                                    ? ctx.visual_submap->errors[rhs]
                                                    : std::numeric_limits<float>::max();
                        return lhs_error < rhs_error;
                      });
    selected_indices.resize(raw_score_point_quota);
    std::sort(selected_indices.begin(), selected_indices.end());

    auto pruneVector = [&](auto &values) {
      using VectorType = std::decay_t<decltype(values)>;
      VectorType pruned;
      pruned.reserve(selected_indices.size());
      for (int index : selected_indices)
        if (index >= 0 && index < static_cast<int>(values.size())) pruned.push_back(std::move(values[index]));
      values.swap(pruned);
    };

    pruneVector(ctx.visual_submap->propa_errors);
    pruneVector(ctx.visual_submap->errors);
    pruneVector(ctx.visual_submap->warp_patch);
    pruneVector(ctx.visual_submap->search_levels);
    pruneVector(ctx.visual_submap->warp_affines);
    pruneVector(ctx.visual_submap->voxel_points);
    pruneVector(ctx.visual_submap->reference_features);
    pruneVector(ctx.visual_submap->inv_expo_list);
    pruneVector(ctx.visual_submap->level_active);
    if (usage_stats_en)
    {
      pruneVector(ctx.visual_submap->usage_ncc_levels);
      pruneVector(ctx.visual_submap->usage_sse_levels);
      pruneVector(ctx.visual_submap->usage_level_valid);
    }
    ctx.total_points = static_cast<int>(ctx.visual_submap->voxel_points.size());
  }

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  // printf("[ VIO ] camera_id=%d retrieve %d points from visual sparse map\n", ctx.camera_id, ctx.total_points);
  // printf("[ VIO Debug ] retrieve summary camera_id=%d grid_candidates=%d ref_invalid=%d cross_refs=%d warp_attempts=%d affine_bad=%d accepted=%d elapsed=%.6f\n",
  //        ctx.camera_id, debug_grid_candidates, debug_ref_invalid, debug_cross_refs, debug_warp_attempts, debug_affine_bad, debug_accepted,
  //        omp_get_wtime() - ts0);
  // fflush(stdout);
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

void VIOManager::buildCurrentCrossCameraPairs()
{
  current_cross_camera_pairs_.clear();
  if (!cross_camera_current_residual_en || numCameras() < 2) return;

  struct AcceptedObservation
  {
    int camera_id = -1;
    int submap_index = -1;
  };
  std::unordered_map<VisualPoint *, std::vector<AcceptedObservation>> observations_by_point;
  for (const PerCameraData &ctx : cameras_)
  {
    if (ctx.visual_submap == nullptr || ctx.new_frame == nullptr) continue;
    const int count = std::min<int>(ctx.total_points, ctx.visual_submap->voxel_points.size());
    for (int index = 0; index < count; ++index)
    {
      VisualPoint *point = ctx.visual_submap->voxel_points[index];
      if (point == nullptr || point->pending_delete_ || !point->is_normal_initialized_) continue;
      if (index < static_cast<int>(ctx.visual_submap->contributes_to_ekf.size()) &&
          ctx.visual_submap->contributes_to_ekf[index] == 0)
        continue;
      observations_by_point[point].push_back({ctx.camera_id, index});
    }
  }

  const int metric_levels = std::max(1, patch_pyrimid_level);
  for (auto &entry : observations_by_point)
  {
    VisualPoint *point = entry.first;
    std::vector<AcceptedObservation> &observations = entry.second;
    if (observations.size() < 2) continue;
    std::sort(observations.begin(), observations.end(), [](const AcceptedObservation &a, const AcceptedObservation &b) {
      return a.camera_id < b.camera_id;
    });
    for (size_t source_i = 0; source_i + 1 < observations.size(); ++source_i)
    {
      for (size_t target_i = source_i + 1; target_i < observations.size(); ++target_i)
      {
        const AcceptedObservation source_observation = observations[source_i];
        const AcceptedObservation target_observation = observations[target_i];
        if (source_observation.camera_id == target_observation.camera_id) continue;
        if (source_observation.camera_id < 0 || source_observation.camera_id >= numCameras() ||
            target_observation.camera_id < 0 || target_observation.camera_id >= numCameras() ||
            source_observation.camera_id >= state->num_cameras ||
            target_observation.camera_id >= state->num_cameras ||
            source_observation.submap_index < 0 || target_observation.submap_index < 0)
          continue;
        PerCameraData &source = cameras_[source_observation.camera_id];
        PerCameraData &target = cameras_[target_observation.camera_id];
        if (source.cam == nullptr || target.cam == nullptr || source.visual_submap == nullptr ||
            target.visual_submap == nullptr || source.new_frame == nullptr || target.new_frame == nullptr ||
            source.new_frame->img_.empty() || target.new_frame->img_.empty() ||
            source.new_frame->img_.type() != CV_8UC1 || target.new_frame->img_.type() != CV_8UC1 ||
            source_observation.submap_index >= static_cast<int>(source.visual_submap->voxel_points.size()) ||
            target_observation.submap_index >= static_cast<int>(target.visual_submap->voxel_points.size()) ||
            source_observation.submap_index >= static_cast<int>(source.visual_submap->search_levels.size()) ||
            target_observation.submap_index >= static_cast<int>(target.visual_submap->search_levels.size()) ||
            source.visual_submap->voxel_points[source_observation.submap_index] != point ||
            target.visual_submap->voxel_points[target_observation.submap_index] != point)
          continue;
        CurrentCrossCameraPair pair;
        pair.point = point;
        pair.source_camera_id = source.camera_id;
        pair.target_camera_id = target.camera_id;
        pair.source_submap_index = source_observation.submap_index;
        pair.target_submap_index = target_observation.submap_index;
        const int source_search = source.visual_submap->search_levels[source_observation.submap_index];
        const int target_search = target.visual_submap->search_levels[target_observation.submap_index];
        pair.search_level = std::max(0, std::max(source_search, target_search));
        const int max_scale_exponent = pair.search_level + metric_levels - 1;
        if (max_scale_exponent < 0 || max_scale_exponent >= 30) continue;
        pair.sse_levels.assign(metric_levels, std::numeric_limits<double>::quiet_NaN());
        pair.ncc_levels.assign(metric_levels, std::numeric_limits<double>::quiet_NaN());
        pair.level_valid.assign(metric_levels, 0);
        pair.level_active.assign(metric_levels, 0);

        updateFrameState(source, *state);
        updateFrameState(target, *state);
        const SE3d &T_source_w = source.new_frame->T_f_w_;
        const SE3d &T_target_w = target.new_frame->T_f_w_;
        const V3D source_view = T_source_w.inverse().translation() - point->pos_;
        const V3D target_view = T_target_w.inverse().translation() - point->pos_;
        if (source_view.norm() > 1.0e-9 && target_view.norm() > 1.0e-9)
        {
          const double cosine = std::max(-1.0, std::min(1.0,
              source_view.dot(target_view) / (source_view.norm() * target_view.norm())));
          const double angle_deg = std::acos(cosine) * kRadiansToDegrees;
          pair.view_angle_bin = angle_deg < 5.0 ? 0 : (angle_deg < 15.0 ? 1 :
                                (angle_deg < 30.0 ? 2 : (angle_deg < 60.0 ? 3 : 4)));
        }
        bool affine_ok = false;
        V2D source_center = V2D::Zero();
        V2D target_center = V2D::Zero();
        const VirtualTrackPatch *source_track = nullptr;
        const VirtualTrackPatch *target_track = nullptr;
        if (virtual_fisheye_patch_en)
        {
          if (source_observation.submap_index < static_cast<int>(source.visual_submap->virtual_track_patches.size()) &&
              target_observation.submap_index < static_cast<int>(target.visual_submap->virtual_track_patches.size()))
          {
            source_track = &source.visual_submap->virtual_track_patches[source_observation.submap_index];
            target_track = &target.visual_submap->virtual_track_patches[target_observation.submap_index];
            const V3D point_vsource = source_track->T_vcur_w_seed * point->pos_;
            const V3D point_vtarget = target_track->T_vcur_w_seed * point->pos_;
            const V3D normal_vsource = source_track->T_vcur_w_seed.rotationMatrix() * point->normal_;
            const SE3d T_vtarget_vsource = target_track->T_vcur_w_seed * source_track->T_vcur_w_seed.inverse();
            if (source_track->valid && target_track->valid && point_vsource[2] > virtual_min_z &&
                point_vtarget[2] > virtual_min_z && normal_vsource.norm() > virtual_min_z)
            {
              source_center = virtualProject(point_vsource);
              target_center = virtualProject(point_vtarget);
              affine_ok = getWarpMatrixAffineHomographyVirtual(point_vsource, normal_vsource.normalized(),
                                                                T_vtarget_vsource, 0, pair.A_target_source);
            }
          }
        }
        else
        {
          const V3D point_source = T_source_w * point->pos_;
          const V3D point_target = T_target_w * point->pos_;
          const V3D normal_source = T_source_w.rotationMatrix() * point->normal_;
          const int interpolation_border = interpolationBorderMargin(virtual_interp_mode_enum);
          if (point_source.array().isFinite().all() && point_target.array().isFinite().all() &&
              normal_source.array().isFinite().all() && normal_source.norm() > 1.0e-9 &&
              projectRawFisheyeIfValid(source, point_source, interpolation_border, source_center) &&
              projectRawFisheyeIfValid(target, point_target, interpolation_border, target_center))
          {
            getWarpMatrixAffineHomography(source, target, source_center, point_source, normal_source.normalized(),
                                           T_target_w * T_source_w.inverse(), 0, pair.A_target_source);
            const double affine_det = pair.A_target_source.determinant();
            affine_ok = pair.A_target_source.array().isFinite().all() && std::isfinite(affine_det) &&
                        std::fabs(affine_det) > 1.0e-9 &&
                        pair.A_target_source.inverse().array().isFinite().all();
          }
        }

        if (affine_ok)
        {
          pair.footprint_bin = usageFootprintBin(pair.A_target_source);
          pair.anisotropy_bin = usageAnisotropyBin(pair.A_target_source);
          const Matrix2d A_source_target = pair.A_target_source.inverse();
          std::vector<float> source_patch(patch_size_total);
          std::vector<float> target_patch(patch_size_total);
          for (int level = 0; level < metric_levels; ++level)
          {
            const int scale = 1 << (level + pair.search_level);
            bool level_ok = true;
            double sse = 0.0;
            for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
            {
              const V2D target_offset = (core_patch_offsets_[patch_index] * static_cast<float>(scale)).cast<double>();
              const V2D source_offset = A_source_target * target_offset;
              float source_value = 0.0f;
              float target_value = 0.0f;
              if (virtual_fisheye_patch_en)
              {
                const cv::Mat &source_img = source.new_frame->img_;
                const cv::Mat &target_img = target.new_frame->img_;
                const bool source_ok = virtual_sparse_patch_en
                    ? sampleSparseVirtualValue(source, source_img, source_track->R_ccur_from_vcur_seed,
                                               source_center + source_offset, source_value)
                    : interpolateVirtualFloat(source_track->cur_support.values, source_track->cur_support.valid_mask,
                                              source_center[0] + source_offset[0], source_center[1] + source_offset[1], source_value);
                const bool target_ok = virtual_sparse_patch_en
                    ? sampleSparseVirtualValue(target, target_img, target_track->R_ccur_from_vcur_seed,
                                               target_center + target_offset, target_value)
                    : interpolateVirtualFloat(target_track->cur_support.values, target_track->cur_support.valid_mask,
                                              target_center[0] + target_offset[0], target_center[1] + target_offset[1], target_value);
                level_ok = source_ok && target_ok;
              }
              else
              {
                const V2D source_sample = source_center + source_offset;
                const V2D target_sample = target_center + target_offset;
                const int sample_border = interpolationBorderMargin(virtual_interp_mode_enum);
                level_ok = source_sample.array().isFinite().all() && target_sample.array().isFinite().all() &&
                           source_sample[0] >= sample_border && source_sample[1] >= sample_border &&
                           source_sample[0] < source.new_frame->img_.cols - sample_border - 1 &&
                           source_sample[1] < source.new_frame->img_.rows - sample_border - 1 &&
                           target_sample[0] >= sample_border && target_sample[1] >= sample_border &&
                           target_sample[0] < target.new_frame->img_.cols - sample_border - 1 &&
                           target_sample[1] < target.new_frame->img_.rows - sample_border - 1 &&
                           source.cam->isInFrame(source_sample.cast<int>(), sample_border) &&
                           target.cam->isInFrame(target_sample.cast<int>(), sample_border) &&
                           interpolateRawVirtualImage(source.new_frame->img_, source_sample[0], source_sample[1], source_value) &&
                           interpolateRawVirtualImage(target.new_frame->img_, target_sample[0], target_sample[1], target_value);
              }
              if (!level_ok) break;
              source_patch[patch_index] = source_value;
              target_patch[patch_index] = target_value;
              if (!zncc_residual_en)
              {
                const double residual = state->inv_expo_time[source.camera_id] * source_value -
                                        state->inv_expo_time[target.camera_id] * target_value;
                sse += residual * residual;
              }
            }
            if (!level_ok) continue;
            double ncc = std::numeric_limits<double>::quiet_NaN();
            if (zncc_residual_en &&
                !normalizedPatchMetrics(source_patch.data(), target_patch.data(), patch_size_total,
                                        zncc_min_std, sse, ncc))
              continue;
            pair.level_valid[level] = 1;
            pair.sse_levels[level] = sse;
            pair.ncc_levels[level] = zncc_residual_en
                                         ? ncc
                                         : calculateNCC(source_patch.data(), target_patch.data(), patch_size_total);
          }
        }

        recordCurrentCrossCameraUsage(pair, kUsagePreGateStage);
        bool any_level_active = false;
        if (zncc_residual_en)
        {
          for (int level = 0; level < metric_levels; ++level)
          {
            const bool valid = pair.level_valid[level] != 0 &&
                               std::isfinite(pair.ncc_levels[level]) &&
                               std::isfinite(pair.sse_levels[level]);
            if (valid && (!ncc_en || pair.ncc_levels[level] >= nccThresholdForLevel(level)))
            {
              pair.level_active[level] = 1;
              any_level_active = true;
            }
          }
        }
        else if (ncc_en)
        {
          for (int level = 0; level < metric_levels; ++level)
          {
            const bool valid = pair.level_valid[level] != 0 &&
                               std::isfinite(pair.ncc_levels[level]) &&
                               std::isfinite(pair.sse_levels[level]);
            if (valid && pair.ncc_levels[level] >= nccThresholdForLevel(level) &&
                pair.sse_levels[level] <= outlier_threshold * patch_size_total)
            {
              pair.level_active[level] = 1;
              any_level_active = true;
            }
          }
        }
        else if (!pair.level_valid.empty() && pair.level_valid[0] != 0 &&
                 std::isfinite(pair.sse_levels[0]) &&
                 pair.sse_levels[0] <= outlier_threshold * patch_size_total)
        {
          pair.level_active = pair.level_valid;
          any_level_active = std::any_of(pair.level_active.begin(), pair.level_active.end(),
                                         [](uint8_t active) { return active != 0; });
        }
        pair.accepted = affine_ok && any_level_active;
        if (pair.accepted) recordCurrentCrossCameraUsage(pair, kUsageAcceptedStage);
        current_cross_camera_pairs_.push_back(std::move(pair));
      }
    }
  }

  std::unordered_map<VisualPoint *, int> accepted_pair_count;
  for (const CurrentCrossCameraPair &pair : current_cross_camera_pairs_)
    if (pair.accepted) ++accepted_pair_count[pair.point];
  for (CurrentCrossCameraPair &pair : current_cross_camera_pairs_)
    if (pair.accepted) pair.hessian_weight = 1.0 / std::max(1, accepted_pair_count[pair.point]);
}

void VIOManager::computeJacobianAndUpdateEKF()
{
  compute_jacobian_time = update_ekf_time = 0.0;
  vio_linearized_residual_count_ = 0;
  int total_observations = 0;
  for (const PerCameraData &ctx : cameras_) total_observations += ctx.total_points;
  if (total_observations == 0) return;
  const double measurement_cov = photometricNoiseCovariance();
  if (!std::isfinite(measurement_cov) || measurement_cov <= 0.0)
    throw std::runtime_error("photometric residual covariance must be finite and positive");
  ++online_calib_visual_update_count_;
  online_calib_last_total_observations_ = total_observations;
  const int calib_camera_count = std::max(0, state->num_cameras);
  const int calib_group_count = std::max(0, state->num_time_offset_groups);
  if (static_cast<int>(online_extrinsic_camera_accept_count_.size()) != calib_camera_count)
    online_extrinsic_camera_accept_count_.assign(calib_camera_count, 0);
  if (static_cast<int>(online_time_offset_group_accept_count_.size()) != calib_group_count)
    online_time_offset_group_accept_count_.assign(calib_group_count, 0);
  std::vector<int> last_time_tracks(calib_group_count, 0);
  std::vector<double> last_time_avg_pixel_velocity(calib_group_count, 0.0);
  std::vector<double> last_time_update_ms(calib_group_count, 0.0);
  std::vector<double> last_extrinsic_rot_update_deg(calib_camera_count, 0.0);
  std::vector<double> last_extrinsic_trans_update_cm(calib_camera_count, 0.0);
  std::vector<int> rollback_last_time_tracks = last_time_tracks;
  std::vector<double> rollback_last_time_avg_pixel_velocity = last_time_avg_pixel_velocity;
  std::vector<double> rollback_last_time_update_ms = last_time_update_ms;
  std::vector<double> rollback_last_extrinsic_rot_update_deg = last_extrinsic_rot_update_deg;
  std::vector<double> rollback_last_extrinsic_trans_update_cm = last_extrinsic_trans_update_cm;
  double last_max_rot_update_deg = 0.0;
  double last_max_trans_update_cm = 0.0;
  double last_max_time_update_ms = 0.0;
  double rollback_last_max_rot_update_deg = last_max_rot_update_deg;
  double rollback_last_max_trans_update_cm = last_max_trans_update_cm;
  double rollback_last_max_time_update_ms = last_max_time_update_ms;
  bool attempted_extrinsic_this_frame = false;
  bool attempted_time_this_frame = false;
  if (cross_camera_current_residual_en) buildCurrentCrossCameraPairs();
  G = Eigen::MatrixXd::Zero(state->stateDim(), state->stateDim());
  const bool online_extrinsic_active = online_extrinsic_en &&
      frame_count >= online_extrinsic_start_frame &&
      total_observations >= online_extrinsic_min_tracks;
  const bool allow_extrinsic_rotation = online_extrinsic_active && online_extrinsic_rot_en;
  const bool allow_extrinsic_translation = online_extrinsic_active && online_extrinsic_trans_en;
  const StatesGroup state_before_visual_update = *state;
  const Eigen::MatrixXd cov_before_visual_update = state->cov;
  std::vector<uint8_t> final_active_extrinsic_rot(state->num_cameras, 0);
  std::vector<uint8_t> final_active_extrinsic_trans(state->num_cameras, 0);
  std::vector<uint8_t> final_active_time_groups(state->num_time_offset_groups, 0);
  Eigen::MatrixXd rollback_G = G;
  directional_update::Result final_directional_result;
  directional_update::Result rollback_directional_result;
  Eigen::MatrixXd final_directional_posterior_covariance;
  Eigen::MatrixXd rollback_directional_posterior_covariance;
  std::vector<uint8_t> rollback_active_extrinsic_rot = final_active_extrinsic_rot;
  std::vector<uint8_t> rollback_active_extrinsic_trans = final_active_extrinsic_trans;
  std::vector<uint8_t> rollback_active_time_groups = final_active_time_groups;
  const Eigen::MatrixXd usage_prior_cov = usage_stats_en ? state->cov : Eigen::MatrixXd();
  const int usage_state_dim = state->stateDim();
  // These snapshots are overwritten only when G is updated, so they always correspond to the exact
  // linearization used by the final covariance update rather than a sum over pyramid levels/iterations.
  Eigen::MatrixXd usage_final_h_base;
  Eigen::MatrixXd usage_final_h_same;
  Eigen::MatrixXd usage_final_h_cross;
  Eigen::MatrixXd usage_final_h_current_cross;
  if (usage_stats_en)
  {
    usage_final_h_base = Eigen::MatrixXd::Zero(usage_state_dim, usage_state_dim);
    usage_final_h_same = Eigen::MatrixXd::Zero(usage_state_dim, usage_state_dim);
    usage_final_h_cross = Eigen::MatrixXd::Zero(usage_state_dim, usage_state_dim);
    if (cross_camera_current_residual_en)
      usage_final_h_current_cross = Eigen::MatrixXd::Zero(usage_state_dim, usage_state_dim);
  }
  long long usage_final_patches_all = 0;
  long long usage_final_patches_same = 0;
  long long usage_final_patches_cross = 0;
  long long usage_final_patches_current_cross = 0;
  long long usage_final_residuals_all = 0;
  long long usage_final_residuals_same = 0;
  long long usage_final_residuals_cross = 0;
  long long usage_final_residuals_current_cross = 0;
  Eigen::MatrixXd rollback_usage_final_h_base = usage_final_h_base;
  Eigen::MatrixXd rollback_usage_final_h_same = usage_final_h_same;
  Eigen::MatrixXd rollback_usage_final_h_cross = usage_final_h_cross;
  Eigen::MatrixXd rollback_usage_final_h_current_cross = usage_final_h_current_cross;
  long long rollback_usage_final_patches_all = usage_final_patches_all;
  long long rollback_usage_final_patches_same = usage_final_patches_same;
  long long rollback_usage_final_patches_cross = usage_final_patches_cross;
  long long rollback_usage_final_patches_current_cross = usage_final_patches_current_cross;
  long long rollback_usage_final_residuals_all = usage_final_residuals_all;
  long long rollback_usage_final_residuals_same = usage_final_residuals_same;
  long long rollback_usage_final_residuals_cross = usage_final_residuals_cross;
  long long rollback_usage_final_residuals_current_cross = usage_final_residuals_current_cross;

  for (int level = patch_pyrimid_level - 1; level >= 0; --level)
  {
    StatesGroup old_state = *state;
    if (inverse_composition_en) buildFixedTemplateGradientCache(level);
    double last_error = std::numeric_limits<double>::max();
    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
      const double linearize_start = omp_get_wtime();
      const int full_state_dim = state->stateDim();
      Eigen::MatrixXd usage_iter_h_all;
      Eigen::MatrixXd usage_iter_h_same;
      Eigen::MatrixXd usage_iter_h_cross;
      Eigen::MatrixXd usage_iter_h_current_cross;
      long long usage_iter_patches_all = 0;
      long long usage_iter_patches_same = 0;
      long long usage_iter_patches_cross = 0;
      long long usage_iter_patches_current_cross = 0;
      long long usage_iter_residuals_all = 0;
      long long usage_iter_residuals_same = 0;
      long long usage_iter_residuals_cross = 0;
      long long usage_iter_residuals_current_cross = 0;
      Eigen::MatrixXd usage_effective_h_all;
      Eigen::MatrixXd usage_effective_h_same;
      Eigen::MatrixXd usage_effective_h_cross;
      Eigen::MatrixXd usage_effective_h_current_cross;
      directional_update::Result iteration_directional_result;
      double error = 0.0;
      int measurement_count = 0;
      for (PerCameraData &ctx : cameras_) updateFrameState(ctx, *state);
      std::vector<uint8_t> active_time_groups(state->num_time_offset_groups, 0);
      if (online_time_offset_en && frame_count >= online_time_offset_start_frame &&
          (online_time_offset_min_update_interval <= 0 ||
           frame_count % std::max(1, online_time_offset_min_update_interval) == 0))
      {
        std::vector<int> group_tracks(state->num_time_offset_groups, 0);
        std::vector<double> group_pixel_velocity(state->num_time_offset_groups, 0.0);
        for (PerCameraData &ctx : cameras_)
        {
          if (ctx.total_points <= 0 || ctx.visual_submap == nullptr || ctx.new_frame == nullptr) continue;
          const int group_id = ctx.time_offset_group;
          if (!isOnlineTimeOffsetEnabledForGroup(group_id)) continue;
          const int count = std::min<int>(ctx.total_points, ctx.visual_submap->voxel_points.size());
          for (int i = 0; i < count; ++i)
          {
            VisualPoint *point = ctx.visual_submap->voxel_points[i];
            if (point == nullptr) continue;
            const V3D point_c = ctx.Rcw * point->pos_ + ctx.Pcw;
            if (!point_c.array().isFinite().all()) continue;
            MD(2, 3) Jdpi;
            computeProjectionJacobian(ctx, point_c, Jdpi);
            const V3D point_i = ctx.Rwi.transpose() * (point->pos_ - ctx.Pwi);
            M3D point_i_hat;
            point_i_hat << SKEW_SYM_MATRX(point_i);
            const V3D dpc_dtd = ctx.Rci * (point_i_hat * ctx.gyro_i - ctx.Rwi.transpose() * ctx.Vwi);
            const double pixel_speed = (Jdpi * dpc_dtd).norm();
            if (!std::isfinite(pixel_speed)) continue;
            ++group_tracks[group_id];
            group_pixel_velocity[group_id] += pixel_speed;
          }
        }
        for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
        {
          const double avg_pixel_velocity =
              group_tracks[group_id] > 0 ? group_pixel_velocity[group_id] / group_tracks[group_id] : 0.0;
          if (group_id < static_cast<int>(last_time_tracks.size()))
          {
            last_time_tracks[group_id] = group_tracks[group_id];
            last_time_avg_pixel_velocity[group_id] = avg_pixel_velocity;
          }
          if (isOnlineTimeOffsetEnabledForGroup(group_id) &&
              group_tracks[group_id] >= online_time_offset_min_tracks &&
              avg_pixel_velocity >= online_time_offset_min_pixel_velocity)
          {
            active_time_groups[group_id] = 1;
            attempted_time_this_frame = true;
          }
        }
      }
      std::vector<uint8_t> active_extrinsic_rot(state->num_cameras, 0);
      std::vector<uint8_t> active_extrinsic_trans(state->num_cameras, 0);
      for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
      {
        const bool camera_active = camera_id < static_cast<int>(cameras_.size()) &&
                                   isOnlineExtrinsicEnabledForCamera(camera_id) &&
                                   cameras_[camera_id].total_points >= online_extrinsic_min_tracks;
        if (camera_active && allow_extrinsic_rotation) active_extrinsic_rot[camera_id] = 1;
        if (camera_active && allow_extrinsic_translation) active_extrinsic_trans[camera_id] = 1;
        if (active_extrinsic_rot[camera_id] != 0 || active_extrinsic_trans[camera_id] != 0)
          attempted_extrinsic_this_frame = true;
      }

      std::vector<int> solve_to_full;
      std::vector<int> full_to_solve(full_state_dim, -1);
      auto addSolveIndex = [&](int full_index) {
        if (full_index < 0 || full_index >= full_state_dim || full_to_solve[full_index] >= 0) return;
        full_to_solve[full_index] = static_cast<int>(solve_to_full.size());
        solve_to_full.push_back(full_index);
      };
      auto addSolveBlock = [&](int full_index, int size) {
        for (int k = 0; k < size; ++k) addSolveIndex(full_index + k);
      };
      addSolveBlock(0, 6);
      addSolveBlock(state->velocityIndex(), 3);
      addSolveBlock(state->gyroBiasIndex(), 3);
      addSolveBlock(state->accelBiasIndex(), 3);
      addSolveBlock(state->gravityIndex(), 3);
      if (exposure_estimate_en)
        for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
          addSolveIndex(state->exposureIndex(camera_id));
      for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
        if (group_id < static_cast<int>(active_time_groups.size()) && active_time_groups[group_id] != 0)
          addSolveIndex(state->timeOffsetIndex(group_id));
      for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
      {
        if (camera_id < static_cast<int>(active_extrinsic_rot.size()) && active_extrinsic_rot[camera_id] != 0)
          addSolveBlock(state->extrinsicRotIndex(camera_id), 3);
        if (camera_id < static_cast<int>(active_extrinsic_trans.size()) && active_extrinsic_trans[camera_id] != 0)
          addSolveBlock(state->extrinsicTransIndex(camera_id), 3);
      }
      const int solve_dim = static_cast<int>(solve_to_full.size());
      Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(solve_dim);
      auto addJacobianValue = [&](Eigen::VectorXd &jacobian, int full_index, double value) {
        if (full_index < 0 || full_index >= full_state_dim) return;
        const int solve_index = full_to_solve[full_index];
        if (solve_index >= 0) jacobian[solve_index] = value;
      };
      auto addJacobianSegment = [&](Eigen::VectorXd &jacobian, int full_index, const auto &value) {
        for (int k = 0; k < value.size(); ++k)
          addJacobianValue(jacobian, full_index + k, value[k]);
      };
      struct SparseResidualJacobian
      {
        enum : int { kMaxEntries = 24 };
        std::array<int, kMaxEntries> index = {};
        std::array<double, kMaxEntries> value = {};
        int size = 0;

        void clear() { size = 0; }

        void add(int solve_index, double v)
        {
          if (solve_index < 0) return;
          for (int k = 0; k < size; ++k)
          {
            if (index[k] == solve_index)
            {
              value[k] += v;
              return;
            }
          }
          if (size >= kMaxEntries)
            throw std::runtime_error("SparseResidualJacobian capacity exceeded");
          index[size] = solve_index;
          value[size] = v;
          ++size;
        }
      };
      auto addSparseJacobianValue = [&](SparseResidualJacobian &jacobian, int full_index, double value) {
        if (full_index < 0 || full_index >= full_state_dim) return;
        jacobian.add(full_to_solve[full_index], value);
      };
      auto addSparseJacobianSegment = [&](SparseResidualJacobian &jacobian, int full_index, const auto &value) {
        for (int k = 0; k < value.size(); ++k)
          addSparseJacobianValue(jacobian, full_index + k, value[k]);
      };
      auto sparseJacobianToDense = [&](const SparseResidualJacobian &sparse_jacobian,
                                       Eigen::VectorXd &dense_jacobian) {
        if (dense_jacobian.size() != solve_dim) dense_jacobian.resize(solve_dim);
        dense_jacobian.setZero();
        for (int k = 0; k < sparse_jacobian.size; ++k)
          dense_jacobian[sparse_jacobian.index[k]] = sparse_jacobian.value[k];
      };
      auto fullVectorToSolve = [&](const Eigen::VectorXd &full_vector) {
        Eigen::VectorXd reduced(solve_dim);
        for (int k = 0; k < solve_dim; ++k) reduced[k] = full_vector[solve_to_full[k]];
        return reduced;
      };
      auto fullCovToSolve = [&](const Eigen::MatrixXd &full_cov) {
        Eigen::MatrixXd reduced(solve_dim, solve_dim);
        for (int r = 0; r < solve_dim; ++r)
          for (int c = 0; c < solve_dim; ++c)
            reduced(r, c) = full_cov(solve_to_full[r], solve_to_full[c]);
        return reduced;
      };
      auto solveMatrixToFull = [&](const Eigen::MatrixXd &reduced) {
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(full_state_dim, full_state_dim);
        for (int r = 0; r < reduced.rows(); ++r)
          for (int c = 0; c < reduced.cols(); ++c)
            full(solve_to_full[r], solve_to_full[c]) = reduced(r, c);
        return full;
      };
      auto solveVectorToFull = [&](const Eigen::VectorXd &reduced) {
        Eigen::VectorXd full = Eigen::VectorXd::Zero(full_state_dim);
        for (int k = 0; k < reduced.size(); ++k) full[solve_to_full[k]] = reduced[k];
        return full;
      };
      if (usage_stats_en)
      {
        usage_iter_h_all = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
        usage_iter_h_same = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
        usage_iter_h_cross = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
        if (cross_camera_current_residual_en)
          usage_iter_h_current_cross = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
      }

      for (PerCameraData &ctx : cameras_)
      {
        if (ctx.total_points == 0 || ctx.visual_submap == nullptr || ctx.new_frame == nullptr) continue;
        const cv::Mat &img = ctx.new_frame->img_;
        const M3D Rwi = ctx.Rwi;
        const V3D Pwi = ctx.Pwi;
        const bool estimate_extrinsic =
            ctx.camera_id >= 0 && ctx.camera_id < state->num_cameras &&
            (active_extrinsic_rot[ctx.camera_id] != 0 || active_extrinsic_trans[ctx.camera_id] != 0);
        const double current_exposure = state->inv_expo_time[ctx.camera_id];

        for (int point_index = 0; point_index < ctx.total_points; ++point_index)
        {
          VisualPoint *point = ctx.visual_submap->voxel_points[point_index];
          if (point == nullptr) continue;
          if (point_index < static_cast<int>(ctx.visual_submap->level_active.size()) &&
              (level >= static_cast<int>(ctx.visual_submap->level_active[point_index].size()) ||
               ctx.visual_submap->level_active[point_index][level] == 0))
            continue;
          const int group_id = ctx.time_offset_group;
          const bool estimate_time_offset =
              group_id >= 0 && group_id < state->num_time_offset_groups &&
              group_id < static_cast<int>(active_time_groups.size()) &&
              active_time_groups[group_id] != 0;
          V3D point_i_for_time = V3D::Zero();
          V3D dpc_dtd = V3D::Zero();
          if (estimate_time_offset || estimate_extrinsic)
          {
            point_i_for_time = Rwi.transpose() * (point->pos_ - Pwi);
            if (estimate_time_offset)
            {
              M3D point_i_for_time_hat;
              point_i_for_time_hat << SKEW_SYM_MATRX(point_i_for_time);
              dpc_dtd = ctx.Rci * (point_i_for_time_hat * ctx.gyro_i - Rwi.transpose() * ctx.Vwi);
            }
          }
          const int search_level = ctx.visual_submap->search_levels[point_index];
          const int pyramid_level = level + search_level;
          const int scale = 1 << pyramid_level;
          const double inv_scale = 1.0 / scale;
          const std::vector<float> &reference_patch = ctx.visual_submap->warp_patch[point_index];
          const double reference_exposure = ctx.visual_submap->inv_expo_list[point_index];
          double patch_error = 0.0;
          const bool contributes_to_ekf = point_index >= static_cast<int>(ctx.visual_submap->contributes_to_ekf.size()) ||
                                          ctx.visual_submap->contributes_to_ekf[point_index] != 0;
          Eigen::Matrix<double, 6, 6> local_pose_information = Eigen::Matrix<double, 6, 6>::Zero();
          Eigen::MatrixXd local_hessian;
          Eigen::VectorXd local_gradient;
          Eigen::MatrixXd usage_local_hessian;
          int usage_local_dof = 0;
          if (visual_map_manage_en)
          {
            local_hessian = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
            local_gradient = Eigen::VectorXd::Zero(solve_dim);
          }
          else if (usage_stats_en)
          {
            usage_local_hessian = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
          }
          double local_squared_error = 0.0;
          int local_dof = 0;
          auto accumulateObservation = [&](const Eigen::VectorXd &raw_jacobian, double raw_residual) {
            ++vio_linearized_residual_count_;
            const double sqrt_robust_weight =
                tukey_robust_en ? tukeySqrtWeight(raw_residual, outlier_threshold) : 1.0;
            const Eigen::VectorXd jacobian = sqrt_robust_weight * raw_jacobian;
            const double residual = sqrt_robust_weight * raw_residual;
            if (visual_map_manage_en)
            {
              if (iteration == 0 && jacobian.size() >= 6)
              {
                const Eigen::Matrix<double, 6, 1> pose_jacobian = jacobian.head<6>();
                local_pose_information.noalias() += pose_jacobian * pose_jacobian.transpose();
              }
              local_squared_error += residual * residual;
              ++local_dof;
            }
            if (visual_map_manage_en)
            {
              local_hessian.noalias() += jacobian * jacobian.transpose();
              local_gradient.noalias() += jacobian * residual;
            }
            else
            {
              hessian.noalias() += jacobian * jacobian.transpose();
              gradient.noalias() += jacobian * residual;
              ++measurement_count;
              if (usage_stats_en)
              {
                usage_local_hessian.noalias() += jacobian * jacobian.transpose();
                ++usage_local_dof;
              }
            }
            patch_error += residual * residual;
          };
          auto accumulateSparseObservation =
              [&](const SparseResidualJacobian &raw_jacobian, double raw_residual) {
            ++vio_linearized_residual_count_;
            const double sqrt_robust_weight =
                tukey_robust_en ? tukeySqrtWeight(raw_residual, outlier_threshold) : 1.0;
            const double residual = sqrt_robust_weight * raw_residual;
            if (visual_map_manage_en)
            {
              if (iteration == 0)
              {
                for (int r = 0; r < raw_jacobian.size; ++r)
                {
                  const int row = raw_jacobian.index[r];
                  if (row < 0 || row >= 6) continue;
                  const double row_value = sqrt_robust_weight * raw_jacobian.value[r];
                  for (int c = 0; c < raw_jacobian.size; ++c)
                  {
                    const int col = raw_jacobian.index[c];
                    if (col < 0 || col >= 6) continue;
                    local_pose_information(row, col) +=
                        row_value * sqrt_robust_weight * raw_jacobian.value[c];
                  }
                }
              }
              local_squared_error += residual * residual;
              ++local_dof;
            }
            Eigen::MatrixXd &target_hessian = visual_map_manage_en ? local_hessian : hessian;
            Eigen::VectorXd &target_gradient = visual_map_manage_en ? local_gradient : gradient;
            for (int r = 0; r < raw_jacobian.size; ++r)
            {
              const int row = raw_jacobian.index[r];
              const double row_value = sqrt_robust_weight * raw_jacobian.value[r];
              target_gradient[row] += row_value * residual;
              for (int c = 0; c < raw_jacobian.size; ++c)
              {
                const int col = raw_jacobian.index[c];
                target_hessian(row, col) += row_value * sqrt_robust_weight * raw_jacobian.value[c];
              }
            }
            if (!visual_map_manage_en)
            {
              ++measurement_count;
              if (usage_stats_en)
              {
                for (int r = 0; r < raw_jacobian.size; ++r)
                {
                  const int row = raw_jacobian.index[r];
                  const double row_value = sqrt_robust_weight * raw_jacobian.value[r];
                  for (int c = 0; c < raw_jacobian.size; ++c)
                  {
                    const int col = raw_jacobian.index[c];
                    usage_local_hessian(row, col) +=
                        row_value * sqrt_robust_weight * raw_jacobian.value[c];
                  }
                }
                ++usage_local_dof;
              }
            }
            patch_error += residual * residual;
          };
          auto accumulateNormalizedReferencePatch =
              [&](const std::vector<double> &current_values,
                  const Eigen::MatrixXd &raw_current_jacobian) -> bool {
            Eigen::VectorXd normalized_current;
            Eigen::MatrixXd normalized_current_jacobian;
            Eigen::VectorXd normalized_reference;
            if (!normalizePatchWithJacobian(current_values, raw_current_jacobian, zncc_min_std,
                                            normalized_current, normalized_current_jacobian) ||
                !normalizePatchValues(reference_patch.data() + patch_size_total * level,
                                      patch_size_total, zncc_min_std, normalized_reference))
              return false;
            Eigen::VectorXd residual = normalized_current - normalized_reference;
            const double sqrt_robust_weight =
                normalizedPatchRobustSqrtWeight(residual, zncc_robust_en && !tukey_robust_en,
                                                zncc_huber_delta);
            residual *= sqrt_robust_weight;
            normalized_current_jacobian *= sqrt_robust_weight;
            for (int row = 0; row < patch_size_total; ++row)
              accumulateObservation(normalized_current_jacobian.row(row).transpose(), residual[row]);
            return true;
          };
          auto addMotionTimeJacobian = [&](Eigen::VectorXd &jacobian, const MD(1, 3) &J_photo_center) {
            addJacobianSegment(jacobian, state->velocityIndex(), (J_photo_center * ctx.dpc_dvel).transpose());
            if (estimate_time_offset)
            {
              addJacobianValue(jacobian, state->timeOffsetIndex(group_id), (J_photo_center * dpc_dtd)(0, 0));
            }
          };
          auto addSparseMotionTimeJacobian =
              [&](SparseResidualJacobian &jacobian, const MD(1, 3) &J_photo_center) {
            addSparseJacobianSegment(jacobian, state->velocityIndex(),
                                     (J_photo_center * ctx.dpc_dvel).transpose());
            if (estimate_time_offset)
            {
              addSparseJacobianValue(jacobian, state->timeOffsetIndex(group_id),
                                     (J_photo_center * dpc_dtd)(0, 0));
            }
          };
          Feature *usage_reference = point_index < static_cast<int>(ctx.visual_submap->reference_features.size())
                                         ? ctx.visual_submap->reference_features[point_index]
                                         : nullptr;
          V2D usage_current_px_for_stats = usage_reference != nullptr ? usage_reference->px_ : V2D::Zero();
          Matrix2d usage_affine_for_stats = Matrix2d::Zero();
          bool usage_level_valid = false;
          double usage_level_ncc = std::numeric_limits<double>::quiet_NaN();
          double usage_level_sse = std::numeric_limits<double>::quiet_NaN();
          if (usage_stats_en)
          {
            if (point_index < static_cast<int>(ctx.visual_submap->warp_affines.size()))
              usage_affine_for_stats = ctx.visual_submap->warp_affines[point_index];
            usage_level_valid = point_index < static_cast<int>(ctx.visual_submap->usage_level_valid.size()) &&
                                level < static_cast<int>(ctx.visual_submap->usage_level_valid[point_index].size()) &&
                                ctx.visual_submap->usage_level_valid[point_index][level] != 0;
            if (point_index < static_cast<int>(ctx.visual_submap->usage_ncc_levels.size()) &&
                level < static_cast<int>(ctx.visual_submap->usage_ncc_levels[point_index].size()))
              usage_level_ncc = ctx.visual_submap->usage_ncc_levels[point_index][level];
            if (point_index < static_cast<int>(ctx.visual_submap->usage_sse_levels.size()) &&
                level < static_cast<int>(ctx.visual_submap->usage_sse_levels[point_index].size()))
              usage_level_sse = ctx.visual_submap->usage_sse_levels[point_index][level];
          }
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
            V2D usage_cur_px = V2D::Zero();
            if (point_c.array().isFinite().all()) usage_cur_px = ctx.cam->world2cam(point_c);
            usage_current_px_for_stats = usage_cur_px;
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
                SparseResidualJacobian jacobian;
                jacobian.clear();
                addSparseJacobianSegment(jacobian, 0, JdR.transpose());
                addSparseJacobianSegment(jacobian, 3, Jdt.transpose());
                addSparseMotionTimeJacobian(jacobian, J_photo_center);
                if (exposure_estimate_en)
                  addSparseJacobianValue(jacobian, state->exposureIndex(ctx.camera_id), current_value);
                if (estimate_extrinsic)
                {
                  if (active_extrinsic_rot[ctx.camera_id] != 0)
                    addSparseJacobianSegment(jacobian, state->extrinsicRotIndex(ctx.camera_id),
                                             (J_photo_center * Jpc_dRcl).transpose());
                  if (active_extrinsic_trans[ctx.camera_id] != 0)
                    addSparseJacobianSegment(jacobian, state->extrinsicTransIndex(ctx.camera_id),
                                             J_photo_center.transpose());
                }
                accumulateSparseObservation(jacobian, residual);
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
              std::vector<double> normalized_current_values;
              Eigen::MatrixXd normalized_current_jacobian;
              bool normalized_patch_complete = true;
              if (zncc_residual_en)
              {
                normalized_current_values.resize(patch_size_total);
                normalized_current_jacobian = Eigen::MatrixXd::Zero(patch_size_total, solve_dim);
              }
              Eigen::VectorXd jacobian_buf(solve_dim);
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
                  if (!current_gradient_ok)
                  {
                    if (zncc_residual_en)
                    {
                      normalized_patch_complete = false;
                      break;
                    }
                    continue;
                  }
                  Jimg << image_gradient[0], image_gradient[1];
                  Jimg *= (zncc_residual_en ? 1.0 : current_exposure) * inv_scale;
                }
                const MD(1, 3) Jimg_Jpi_R = Jimg * Jdpi * track.R_vcur_from_ccur_seed;
                const MD(1, 3) Jdphi = Jimg_Jpi_R * point_c_hat;
                const MD(1, 3) Jdp = -Jimg_Jpi_R;
                const MD(1, 3) JdR = Jdphi * ctx.Jdphi_dR + Jdp * ctx.Jdp_dR;
                const MD(1, 3) Jdt = Jdp * ctx.Jdp_dt;
                SparseResidualJacobian jacobian;
                jacobian.clear();
                addSparseJacobianSegment(jacobian, 0, JdR.transpose());
                addSparseJacobianSegment(jacobian, 3, Jdt.transpose());
                addSparseMotionTimeJacobian(jacobian, Jimg_Jpi_R);
                if (!zncc_residual_en && exposure_estimate_en)
                  addSparseJacobianValue(jacobian, state->exposureIndex(ctx.camera_id), current_value);
                if (estimate_extrinsic)
                {
                  if (active_extrinsic_rot[ctx.camera_id] != 0)
                    addSparseJacobianSegment(jacobian, state->extrinsicRotIndex(ctx.camera_id),
                                             (Jimg_Jpi_R * Jpc_dRcl).transpose());
                  if (active_extrinsic_trans[ctx.camera_id] != 0)
                    addSparseJacobianSegment(jacobian, state->extrinsicTransIndex(ctx.camera_id),
                                             Jimg_Jpi_R.transpose());
                }
                if (zncc_residual_en)
                {
                  normalized_current_values[patch_index] = current_value;
                  sparseJacobianToDense(jacobian, jacobian_buf);
                  normalized_current_jacobian.row(patch_index) = jacobian_buf.transpose();
                }
                else
                {
                  const double residual = current_exposure * current_value -
                                          reference_exposure * reference_patch[patch_size_total * level + patch_index];
                  accumulateSparseObservation(jacobian, residual);
                }
              }
              if (zncc_residual_en &&
                  (!normalized_patch_complete ||
                   !accumulateNormalizedReferencePatch(normalized_current_values, normalized_current_jacobian)))
                continue;
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
            usage_current_px_for_stats = pixel;
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
            std::vector<double> normalized_current_values;
            Eigen::MatrixXd normalized_current_jacobian;
            if (zncc_residual_en)
            {
              normalized_current_values.resize(patch_size_total);
              normalized_current_jacobian = Eigen::MatrixXd::Zero(patch_size_total, solve_dim);
            }
            Eigen::VectorXd jacobian_buf(solve_dim);
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
                  Jimg *= (zncc_residual_en ? 1.0 : current_exposure) * inv_scale;
                }
                const MD(1, 3) Jimg_Jpi = Jimg * Jdpi;
                const MD(1, 3) Jdphi = Jimg_Jpi * point_hat;
                const MD(1, 3) Jdp = -Jimg_Jpi;
                const MD(1, 3) JdR = Jdphi * ctx.Jdphi_dR + Jdp * ctx.Jdp_dR;
                const MD(1, 3) Jdt = Jdp * ctx.Jdp_dt;
                SparseResidualJacobian jacobian;
                jacobian.clear();
                addSparseJacobianSegment(jacobian, 0, JdR.transpose());
                addSparseJacobianSegment(jacobian, 3, Jdt.transpose());
                addSparseMotionTimeJacobian(jacobian, Jimg_Jpi);
                if (!zncc_residual_en && exposure_estimate_en)
                  addSparseJacobianValue(jacobian, state->exposureIndex(ctx.camera_id), current_value);
                if (estimate_extrinsic)
                {
                  if (active_extrinsic_rot[ctx.camera_id] != 0)
                    addSparseJacobianSegment(jacobian, state->extrinsicRotIndex(ctx.camera_id),
                                             (Jimg_Jpi * Jpc_dRcl).transpose());
                  if (active_extrinsic_trans[ctx.camera_id] != 0)
                    addSparseJacobianSegment(jacobian, state->extrinsicTransIndex(ctx.camera_id),
                                             Jimg_Jpi.transpose());
                }
                if (zncc_residual_en)
                {
                  normalized_current_values[patch_index] = current_value;
                  sparseJacobianToDense(jacobian, jacobian_buf);
                  normalized_current_jacobian.row(patch_index) = jacobian_buf.transpose();
                }
                else
                {
                  const double residual = current_exposure * current_value -
                                          reference_exposure * reference_patch[patch_size_total * level + patch_index];
                  accumulateSparseObservation(jacobian, residual);
                }
              }
            }
            if (zncc_residual_en &&
                !accumulateNormalizedReferencePatch(normalized_current_values, normalized_current_jacobian))
              continue;
          }
          ctx.visual_submap->errors[point_index] = patch_error;
          if (!visual_map_manage_en)
          {
            if (usage_stats_en && usage_reference != nullptr && usage_local_dof > 0)
            {
              const bool cross_camera = usage_reference->camera_id_ != ctx.camera_id;
              usage_iter_h_all.noalias() += usage_local_hessian;
              ++usage_iter_patches_all;
              usage_iter_residuals_all += usage_local_dof;
              if (cross_camera)
              {
                usage_iter_h_cross.noalias() += usage_local_hessian;
                ++usage_iter_patches_cross;
                usage_iter_residuals_cross += usage_local_dof;
              }
              else
              {
                usage_iter_h_same.noalias() += usage_local_hessian;
                ++usage_iter_patches_same;
                usage_iter_residuals_same += usage_local_dof;
              }
              if (iteration == 0)
                recordUsageEkfContribution(ctx, *usage_reference, *point, usage_current_px_for_stats,
                                           usage_affine_for_stats, level, usage_local_dof, usage_level_sse,
                                           usage_level_ncc, usage_level_valid);
            }
            error += patch_error;
            continue;
          }
          double nis = std::numeric_limits<double>::quiet_NaN();
          if (!visual_ref_nis_en)
          {
            // Photometric acceptance is used below; avoid the full-state NIS solve.
          }
          else if (iteration > 0 && point_index < static_cast<int>(ctx.visual_submap->observation_nis.size()))
            nis = ctx.visual_submap->observation_nis[point_index];
          else if (local_dof > 0)
          {
            const double inv_r = 1.0 / measurement_cov;
            const Eigen::MatrixXd A = inv_r * local_hessian;
            const Eigen::VectorXd b = inv_r * local_gradient;
            const double c = inv_r * local_squared_error;
            Eigen::MatrixXd P = fullCovToSolve(state->cov);
            if (usage_reference != nullptr && usage_reference->birth_pose_cov_.array().isFinite().all())
            {
              // updateFrameState() already projects the composed camera rotation to SO(3)
              // when it creates T_f_w_. Reuse that valid pose instead of asking Sophus
              // to construct an SE3 directly from the numerically approximate ctx.Rcw.
              const SE3d &T_cur_w = ctx.new_frame->T_f_w_;
              const SE3d T_cur_ref = T_cur_w * usage_reference->T_f_w_.inverse();
              const M3D R = T_cur_ref.rotationMatrix();
              const V3D t = T_cur_ref.translation();
              Eigen::Matrix<double, 6, 6> adjoint = Eigen::Matrix<double, 6, 6>::Zero();
              M3D t_hat;
              t_hat << SKEW_SYM_MATRX(t);
              adjoint.block<3, 3>(0, 0) = R;
              adjoint.block<3, 3>(3, 0) = t_hat * R;
              adjoint.block<3, 3>(3, 3) = R;
              P.block<6, 6>(0, 0) +=
                  adjoint * usage_reference->birth_pose_cov_ * adjoint.transpose();
            }
            if (point->covariance_.array().isFinite().all())
              P.block<3, 3>(3, 3) += point->covariance_;
            P.diagonal().array() += 1.0e-12;
            const Eigen::LDLT<Eigen::MatrixXd> p_ldlt(P);
            if (p_ldlt.info() == Eigen::Success)
            {
              const Eigen::MatrixXd system =
                  p_ldlt.solve(Eigen::MatrixXd::Identity(solve_dim, solve_dim)) + A;
              const Eigen::LDLT<Eigen::MatrixXd> system_ldlt(system);
              if (system_ldlt.info() == Eigen::Success)
                nis = std::max(0.0, c - b.dot(system_ldlt.solve(b)));
            }
          }
          if (iteration == 0 && point_index < static_cast<int>(ctx.visual_submap->pose_information.size()))
          {
            ctx.visual_submap->pose_information[point_index] = local_pose_information;
            ctx.visual_submap->observation_dof[point_index] = local_dof;
            ctx.visual_submap->observation_nis[point_index] = nis;
          }
          const bool nis_pass = !visual_map_manage_en || visual_map_manage_shadow_en || !visual_ref_nis_en ||
                                (local_dof > 0 && std::isfinite(nis) &&
                                 nis / static_cast<double>(local_dof) <= visual_ref_nis_max_per_dof);
          if (contributes_to_ekf && nis_pass)
          {
            if (usage_stats_en && usage_reference != nullptr && local_dof > 0)
            {
              const bool cross_camera = usage_reference->camera_id_ != ctx.camera_id;
              usage_iter_h_all.noalias() += local_hessian;
              ++usage_iter_patches_all;
              usage_iter_residuals_all += local_dof;
              if (cross_camera)
              {
                usage_iter_h_cross.noalias() += local_hessian;
                ++usage_iter_patches_cross;
                usage_iter_residuals_cross += local_dof;
              }
              else
              {
                usage_iter_h_same.noalias() += local_hessian;
                ++usage_iter_patches_same;
                usage_iter_residuals_same += local_dof;
              }
              if (iteration == 0)
                recordUsageEkfContribution(ctx, *usage_reference, *point, usage_current_px_for_stats,
                                           usage_affine_for_stats, level, local_dof, usage_level_sse,
                                           usage_level_ncc, usage_level_valid);
            }
            hessian.noalias() += local_hessian;
            gradient.noalias() += local_gradient;
            measurement_count += local_dof;
            error += patch_error;
          }
        }
      }

      if (cross_camera_current_residual_en)
      {
        // Keep the optional current-current linearization out of this already very large hot function.
        // Otherwise GCC's whole-function register allocation and code layout can regress even when the
        // runtime option is disabled. The closure is constructed and called only on the enabled path.
        auto linearize_current_cross_camera = [&]() __attribute__((noinline)) {
          for (CurrentCrossCameraPair &pair : current_cross_camera_pairs_)
          {
        if (!pair.accepted || pair.point == nullptr || pair.point->pending_delete_ ||
            !pair.point->pos_.array().isFinite().all())
          continue;
        if (level >= static_cast<int>(pair.level_active.size()) || pair.level_active[level] == 0) continue;
        if (pair.source_camera_id < 0 || pair.source_camera_id >= numCameras() ||
            pair.target_camera_id < 0 || pair.target_camera_id >= numCameras() ||
            pair.source_camera_id == pair.target_camera_id ||
            pair.source_submap_index < 0 || pair.target_submap_index < 0)
          continue;
        PerCameraData &source = cameras_[pair.source_camera_id];
        PerCameraData &target = cameras_[pair.target_camera_id];
        if (source.new_frame == nullptr || target.new_frame == nullptr ||
            source.visual_submap == nullptr || target.visual_submap == nullptr)
          continue;
        if (pair.source_submap_index >= static_cast<int>(source.visual_submap->voxel_points.size()) ||
            pair.target_submap_index >= static_cast<int>(target.visual_submap->voxel_points.size()) ||
            source.visual_submap->voxel_points[pair.source_submap_index] != pair.point ||
            target.visual_submap->voxel_points[pair.target_submap_index] != pair.point)
          continue;
        const int scale_exponent = level + pair.search_level;
        if (scale_exponent < 0 || scale_exponent >= 30) continue;
        const int scale = 1 << scale_exponent;
        if (!std::isfinite(pair.hessian_weight) || pair.hessian_weight < 0.0) continue;
        const double sqrt_weight = std::sqrt(std::max(0.0, pair.hessian_weight));
        const double affine_det = pair.A_target_source.determinant();
        if (!pair.A_target_source.array().isFinite().all() || !std::isfinite(affine_det) ||
            std::fabs(affine_det) <= 1.0e-9)
          continue;
        const Matrix2d A_source_target = pair.A_target_source.inverse();
        if (!A_source_target.array().isFinite().all()) continue;
        const V3D point_w = pair.point->pos_;
        updateFrameState(source, *state);
        updateFrameState(target, *state);
        const VirtualTrackPatch *source_track = nullptr;
        const VirtualTrackPatch *target_track = nullptr;
        if (virtual_fisheye_patch_en)
        {
          if (pair.source_submap_index >= static_cast<int>(source.visual_submap->virtual_track_patches.size()) ||
              pair.target_submap_index >= static_cast<int>(target.visual_submap->virtual_track_patches.size()))
            continue;
          source_track = &source.visual_submap->virtual_track_patches[pair.source_submap_index];
          target_track = &target.visual_submap->virtual_track_patches[pair.target_submap_index];
        }

        auto linearizeCurrentSample = [&](PerCameraData &ctx, const VirtualTrackPatch *track,
                                          const V2D &offset, float &value, Eigen::VectorXd &jacobian) -> bool {
          if (jacobian.size() != solve_dim) jacobian.resize(solve_dim);
          jacobian.setZero();
          if (ctx.camera_id < 0 || ctx.camera_id >= state->num_cameras || ctx.cam == nullptr ||
              ctx.new_frame == nullptr)
            return false;
          const double exposure = zncc_residual_en ? 1.0 : state->inv_expo_time[ctx.camera_id];
          if (!std::isfinite(exposure)) return false;
          const V3D point_c = ctx.Rcw * point_w + ctx.Pcw;
          if (!point_c.array().isFinite().all()) return false;
          const bool estimate_extrinsic =
              ctx.camera_id >= 0 && ctx.camera_id < state->num_cameras &&
              (active_extrinsic_rot[ctx.camera_id] != 0 || active_extrinsic_trans[ctx.camera_id] != 0);
          M3D Jpc_dRcl = M3D::Zero();
          if (estimate_extrinsic)
          {
            const V3D point_i = ctx.Rwi.transpose() * (point_w - ctx.Pwi);
            const V3D point_l = Rli * point_i + Pli;
            M3D point_l_hat;
            point_l_hat << SKEW_SYM_MATRX(point_l);
            Jpc_dRcl = -ctx.Rcl * point_l_hat;
          }

          MD(1, 3) J_photo_center;
          if (virtual_fisheye_patch_en)
          {
            if (track == nullptr || !track->valid) return false;
            if (virtual_s2_optimize_en)
            {
              if (!linearizeVirtualS2Sample(ctx, ctx.new_frame->img_, point_c, *track, offset, scale,
                                            exposure, value, J_photo_center))
                return false;
            }
            else
            {
              const V3D point_v = track->R_vcur_from_ccur_seed * point_c;
              if (point_v[2] <= virtual_min_z) return false;
              const V2D center = virtualProject(point_v);
              V2D gradient;
              const bool sample_ok = virtual_sparse_patch_en
                  ? sampleSparseVirtualValueAndGradient(ctx, ctx.new_frame->img_, track->R_ccur_from_vcur_seed,
                                                        center + offset, scale, value, gradient)
                  : sampleVirtualValueAndGradient(track->cur_support, center + offset, scale, value, gradient);
              if (!sample_ok) return false;
              MD(1, 2) Jimg;
              Jimg << gradient[0], gradient[1];
              Jimg *= exposure / scale;
              MD(2, 3) Jdpi;
              computeVirtualProjectionJacobian(point_v, Jdpi);
              J_photo_center = Jimg * Jdpi * track->R_vcur_from_ccur_seed;
            }
          }
          else
          {
            V2D center;
            MD(2, 3) Jdpi;
            const int center_border = interpolationBorderMargin(virtual_interp_mode_enum);
            if (raw_camera_model_jacobian_en)
            {
              if (!projectRawCameraWithJacobian(ctx, point_c, center_border, center, Jdpi)) return false;
            }
            else
            {
              if (!projectRawFisheyeIfValid(ctx, point_c, center_border, center)) return false;
              computeProjectionJacobian(ctx, point_c, Jdpi);
              if (!Jdpi.array().isFinite().all()) return false;
            }
            const V2D sample_px = center + offset;
            const int sample_border = scale + interpolationBorderMargin(virtual_interp_mode_enum);
            if (!sample_px.array().isFinite().all() ||
                sample_px[0] < sample_border || sample_px[1] < sample_border ||
                sample_px[0] >= ctx.new_frame->img_.cols - sample_border - 1 ||
                sample_px[1] >= ctx.new_frame->img_.rows - sample_border - 1 ||
                !ctx.cam->isInFrame(sample_px.cast<int>(), sample_border))
              return false;
            V2D gradient;
            if (!sampleRawImageValueAndGradient(ctx.new_frame->img_, sample_px, scale, value, gradient))
              return false;
            MD(1, 2) Jimg;
            Jimg << gradient[0], gradient[1];
            Jimg *= exposure / scale;
            J_photo_center = Jimg * Jdpi;
          }
          if (!J_photo_center.array().isFinite().all()) return false;

          M3D point_c_hat;
          point_c_hat << SKEW_SYM_MATRX(point_c);
          const MD(1, 3) Jdphi = J_photo_center * point_c_hat;
          const MD(1, 3) Jdp = -J_photo_center;
          addJacobianSegment(jacobian, 0, (Jdphi * ctx.Jdphi_dR + Jdp * ctx.Jdp_dR).transpose());
          addJacobianSegment(jacobian, 3, (Jdp * ctx.Jdp_dt).transpose());
          addJacobianSegment(jacobian, state->velocityIndex(), (J_photo_center * ctx.dpc_dvel).transpose());
          const int group_id = ctx.time_offset_group;
          if (group_id >= 0 && group_id < state->num_time_offset_groups &&
              group_id < static_cast<int>(active_time_groups.size()) &&
              active_time_groups[group_id] != 0)
          {
            const V3D point_i_time = ctx.Rwi.transpose() * (point_w - ctx.Pwi);
            M3D point_i_time_hat;
            point_i_time_hat << SKEW_SYM_MATRX(point_i_time);
            const V3D dpc_dtd =
                ctx.Rci * (point_i_time_hat * ctx.gyro_i - ctx.Rwi.transpose() * ctx.Vwi);
            addJacobianValue(jacobian, state->timeOffsetIndex(group_id), (J_photo_center * dpc_dtd)(0, 0));
          }
          if (!zncc_residual_en && exposure_estimate_en)
            addJacobianValue(jacobian, state->exposureIndex(ctx.camera_id), value);
          if (estimate_extrinsic)
          {
            if (active_extrinsic_rot[ctx.camera_id] != 0)
              addJacobianSegment(jacobian, state->extrinsicRotIndex(ctx.camera_id),
                                 (J_photo_center * Jpc_dRcl).transpose());
            if (active_extrinsic_trans[ctx.camera_id] != 0)
              addJacobianSegment(jacobian, state->extrinsicTransIndex(ctx.camera_id),
                                 J_photo_center.transpose());
          }
          return jacobian.array().isFinite().all() && std::isfinite(value);
        };

        Eigen::MatrixXd pair_hessian = Eigen::MatrixXd::Zero(solve_dim, solve_dim);
        int pair_residuals = 0;
        double pair_error = 0.0;
        std::vector<double> source_values;
        std::vector<double> target_values;
        Eigen::MatrixXd source_jacobians;
        Eigen::MatrixXd target_jacobians;
        bool complete_pair_patch = true;
        if (zncc_residual_en)
        {
          source_values.resize(patch_size_total);
          target_values.resize(patch_size_total);
          source_jacobians = Eigen::MatrixXd::Zero(patch_size_total, solve_dim);
          target_jacobians = Eigen::MatrixXd::Zero(patch_size_total, solve_dim);
        }
        Eigen::VectorXd source_jacobian(solve_dim);
        Eigen::VectorXd target_jacobian(solve_dim);
        for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
        {
          const V2D target_offset = (core_patch_offsets_[patch_index] * static_cast<float>(scale)).cast<double>();
          const V2D source_offset = A_source_target * target_offset;
          float source_value = 0.0f;
          float target_value = 0.0f;
          if (!linearizeCurrentSample(source, source_track, source_offset, source_value, source_jacobian) ||
              !linearizeCurrentSample(target, target_track, target_offset, target_value, target_jacobian))
          {
            if (zncc_residual_en)
            {
              complete_pair_patch = false;
              break;
            }
            continue;
          }
          if (zncc_residual_en)
          {
            source_values[patch_index] = source_value;
            target_values[patch_index] = target_value;
            source_jacobians.row(patch_index) = source_jacobian.transpose();
            target_jacobians.row(patch_index) = target_jacobian.transpose();
            continue;
          }
          const double raw_residual =
              state->inv_expo_time[source.camera_id] * source_value -
              state->inv_expo_time[target.camera_id] * target_value;
          const double sqrt_robust_weight =
              tukey_robust_en ? tukeySqrtWeight(raw_residual, outlier_threshold) : 1.0;
          const double residual = sqrt_weight * sqrt_robust_weight * raw_residual;
          const Eigen::VectorXd jacobian =
              sqrt_weight * sqrt_robust_weight * (source_jacobian - target_jacobian);
          hessian.noalias() += jacobian * jacobian.transpose();
          gradient.noalias() += jacobian * residual;
          pair_hessian.noalias() += jacobian * jacobian.transpose();
          pair_error += residual * residual;
          ++pair_residuals;
          ++vio_linearized_residual_count_;
          ++measurement_count;
        }
        if (zncc_residual_en)
        {
          if (!complete_pair_patch) continue;
          Eigen::VectorXd normalized_source;
          Eigen::VectorXd normalized_target;
          Eigen::MatrixXd normalized_source_jacobian;
          Eigen::MatrixXd normalized_target_jacobian;
          if (!normalizePatchWithJacobian(source_values, source_jacobians, zncc_min_std,
                                          normalized_source, normalized_source_jacobian) ||
              !normalizePatchWithJacobian(target_values, target_jacobians, zncc_min_std,
                                          normalized_target, normalized_target_jacobian))
            continue;
          Eigen::VectorXd residuals = normalized_source - normalized_target;
          Eigen::MatrixXd jacobians = normalized_source_jacobian - normalized_target_jacobian;
          const double sqrt_robust_weight =
              normalizedPatchRobustSqrtWeight(residuals, zncc_robust_en && !tukey_robust_en,
                                              zncc_huber_delta);
          residuals *= sqrt_robust_weight;
          jacobians *= sqrt_robust_weight;
          for (int row = 0; row < patch_size_total; ++row)
          {
            const double tukey_sqrt_weight =
                tukey_robust_en ? tukeySqrtWeight(residuals[row], outlier_threshold) : 1.0;
            const double combined_sqrt_weight = sqrt_weight * tukey_sqrt_weight;
            const Eigen::VectorXd jacobian =
                combined_sqrt_weight * jacobians.row(row).transpose();
            const double residual = combined_sqrt_weight * residuals[row];
            hessian.noalias() += jacobian * jacobian.transpose();
            gradient.noalias() += jacobian * residual;
            pair_hessian.noalias() += jacobian * jacobian.transpose();
            pair_error += residual * residual;
            ++pair_residuals;
            ++vio_linearized_residual_count_;
            ++measurement_count;
          }
        }
        if (pair_residuals <= 0) continue;
        error += pair_error;
        if (usage_stats_en)
        {
          usage_iter_h_all.noalias() += pair_hessian;
          usage_iter_h_current_cross.noalias() += pair_hessian;
          ++usage_iter_patches_all;
          ++usage_iter_patches_current_cross;
          usage_iter_residuals_all += pair_residuals;
          usage_iter_residuals_current_cross += pair_residuals;
          if (iteration == 0) recordCurrentCrossCameraUsage(pair, kUsageEkfStage, level, pair_residuals);
        }
          }
        };
        linearize_current_cross_camera();
      }

      if (usage_stats_en)
      {
        usage_effective_h_all = usage_iter_h_all;
        usage_effective_h_same = usage_iter_h_same;
        usage_effective_h_cross = usage_iter_h_cross;
        if (cross_camera_current_residual_en)
          usage_effective_h_current_cross = usage_iter_h_current_cross;
      }

      if (measurement_count > 0 && directional_update_en)
      {
        if (!directional_update::filterInformation(
                fullCovToSolve(state->cov), hessian / measurement_cov, gradient / measurement_cov,
                directional_drop_variance_reduction,
                directional_full_variance_reduction,
                iteration_directional_result))
        {
          *state = state_before_visual_update;
          G.setZero();
          syncCameraExtrinsicsFromState(*state);
          const std::string reason = iteration_directional_result.error.empty()
              ? "photometric residual covariance must be finite and positive"
              : iteration_directional_result.error;
          std::cerr << "[ Directional VIO ] update rejected: " << reason << std::endl;
          return;
        }
        hessian = iteration_directional_result.information * measurement_cov;
        gradient = iteration_directional_result.information_vector * measurement_cov;

        if (usage_stats_en)
        {
          std::string component_error;
          const bool components_valid =
              directional_update::filterInformationComponent(
                  usage_iter_h_all, iteration_directional_result,
                  usage_effective_h_all, component_error) &&
              directional_update::filterInformationComponent(
                  usage_iter_h_same, iteration_directional_result,
                  usage_effective_h_same, component_error) &&
              directional_update::filterInformationComponent(
                  usage_iter_h_cross, iteration_directional_result,
                  usage_effective_h_cross, component_error) &&
              (!cross_camera_current_residual_en ||
               directional_update::filterInformationComponent(
                   usage_iter_h_current_cross, iteration_directional_result,
                   usage_effective_h_current_cross, component_error));
          if (!components_valid)
          {
            *state = state_before_visual_update;
            G.setZero();
            syncCameraExtrinsicsFromState(*state);
            std::cerr << "[ Directional VIO ] usage-information filtering rejected: "
                      << component_error << std::endl;
            return;
          }
        }
      }

      // Calibration regularizers are intentionally excluded from observability
      // classification and added only after filtering the visual measurements.
      if (online_extrinsic_active && online_extrinsic_prior_factor_en)
      {
        constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
        const double rot_std = online_extrinsic_prior_rot_std_deg * kDegToRad;
        const double trans_std = online_extrinsic_prior_trans_std_m;
        if (rot_std > 0.0 && trans_std > 0.0)
        {
          const double rot_info = measurement_cov / (rot_std * rot_std);
          const double trans_info = measurement_cov / (trans_std * trans_std);
          for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
          {
            if (!isOnlineExtrinsicEnabledForCamera(camera_id)) continue;
            if (camera_id >= static_cast<int>(cameras_.size()) ||
                cameras_[camera_id].total_points < online_extrinsic_min_tracks)
              continue;
            if (allow_extrinsic_rotation && camera_id < static_cast<int>(active_extrinsic_rot.size()) &&
                active_extrinsic_rot[camera_id] != 0)
            {
              const int ridx = state->extrinsicRotIndex(camera_id);
              const int solve_index = full_to_solve[ridx];
              if (solve_index >= 0)
              {
                const M3D dR_cl = state->Rcl_prior[camera_id].transpose() * state->Rcl[camera_id];
                const V3D rot_error = Log(dR_cl);
                hessian.block<3, 3>(solve_index, solve_index).diagonal().array() += rot_info;
                gradient.segment<3>(solve_index) += rot_info * rot_error;
              }
            }
            if (allow_extrinsic_translation && camera_id < static_cast<int>(active_extrinsic_trans.size()) &&
                active_extrinsic_trans[camera_id] != 0)
            {
              const int tidx = state->extrinsicTransIndex(camera_id);
              const int solve_index = full_to_solve[tidx];
              if (solve_index >= 0)
              {
                const V3D trans_error = state->Pcl[camera_id] - state->Pcl_prior[camera_id];
                hessian.block<3, 3>(solve_index, solve_index).diagonal().array() += trans_info;
                gradient.segment<3>(solve_index) += trans_info * trans_error;
              }
            }
          }
        }
      }
      if (online_time_offset_en)
      {
        const double std_s = online_time_offset_prior_std_ms * 1.0e-3;
        if (std::isfinite(std_s) && std_s > 0.0)
        {
          const double info = measurement_cov / (std_s * std_s);
          for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
          {
            if (group_id >= static_cast<int>(active_time_groups.size()) || active_time_groups[group_id] == 0) continue;
            const int full_index = state->timeOffsetIndex(group_id);
            const int solve_index = full_to_solve[full_index];
            if (solve_index < 0) continue;
            const double prior = group_id < state->time_offset_prior.size() ? state->time_offset_prior[group_id] : 0.0;
            hessian(solve_index, solve_index) += info;
            gradient[solve_index] += info * (state->time_offset[group_id] - prior);
          }
        }
      }
      compute_jacobian_time += omp_get_wtime() - linearize_start;
      if (measurement_count == 0) break;
      error /= measurement_count;
      if (error > last_error)
      {
        *state = old_state;
        G = rollback_G;
        final_active_extrinsic_rot = rollback_active_extrinsic_rot;
        final_active_extrinsic_trans = rollback_active_extrinsic_trans;
        final_active_time_groups = rollback_active_time_groups;
        last_time_tracks = rollback_last_time_tracks;
        last_time_avg_pixel_velocity = rollback_last_time_avg_pixel_velocity;
        last_time_update_ms = rollback_last_time_update_ms;
        last_extrinsic_rot_update_deg = rollback_last_extrinsic_rot_update_deg;
        last_extrinsic_trans_update_cm = rollback_last_extrinsic_trans_update_cm;
        last_max_rot_update_deg = rollback_last_max_rot_update_deg;
        last_max_trans_update_cm = rollback_last_max_trans_update_cm;
        last_max_time_update_ms = rollback_last_max_time_update_ms;
        usage_final_h_base = rollback_usage_final_h_base;
        usage_final_h_same = rollback_usage_final_h_same;
        usage_final_h_cross = rollback_usage_final_h_cross;
        usage_final_h_current_cross = rollback_usage_final_h_current_cross;
        usage_final_patches_all = rollback_usage_final_patches_all;
        usage_final_patches_same = rollback_usage_final_patches_same;
        usage_final_patches_cross = rollback_usage_final_patches_cross;
        usage_final_patches_current_cross = rollback_usage_final_patches_current_cross;
        usage_final_residuals_all = rollback_usage_final_residuals_all;
        usage_final_residuals_same = rollback_usage_final_residuals_same;
        usage_final_residuals_cross = rollback_usage_final_residuals_cross;
        usage_final_residuals_current_cross = rollback_usage_final_residuals_current_cross;
        final_directional_result = rollback_directional_result;
        final_directional_posterior_covariance = rollback_directional_posterior_covariance;
        syncCameraExtrinsicsFromState(*state);
        break;
      }

      old_state = *state;
      last_error = error;
      const double update_start = omp_get_wtime();
      rollback_usage_final_h_base = usage_final_h_base;
      rollback_usage_final_h_same = usage_final_h_same;
      rollback_usage_final_h_cross = usage_final_h_cross;
      rollback_usage_final_h_current_cross = usage_final_h_current_cross;
      rollback_usage_final_patches_all = usage_final_patches_all;
      rollback_usage_final_patches_same = usage_final_patches_same;
      rollback_usage_final_patches_cross = usage_final_patches_cross;
      rollback_usage_final_patches_current_cross = usage_final_patches_current_cross;
      rollback_usage_final_residuals_all = usage_final_residuals_all;
      rollback_usage_final_residuals_same = usage_final_residuals_same;
      rollback_usage_final_residuals_cross = usage_final_residuals_cross;
      rollback_usage_final_residuals_current_cross = usage_final_residuals_current_cross;
      if (usage_stats_en)
      {
        usage_final_h_base = solveMatrixToFull(hessian - usage_effective_h_all);
        usage_final_h_same = solveMatrixToFull(usage_effective_h_same);
        usage_final_h_cross = solveMatrixToFull(usage_effective_h_cross);
        if (cross_camera_current_residual_en)
          usage_final_h_current_cross = solveMatrixToFull(usage_effective_h_current_cross);
        usage_final_patches_all = usage_iter_patches_all;
        usage_final_patches_same = usage_iter_patches_same;
        usage_final_patches_cross = usage_iter_patches_cross;
        usage_final_patches_current_cross = usage_iter_patches_current_cross;
        usage_final_residuals_all = usage_iter_residuals_all;
        usage_final_residuals_same = usage_iter_residuals_same;
        usage_final_residuals_cross = usage_iter_residuals_cross;
        usage_final_residuals_current_cross = usage_iter_residuals_current_cross;
      }
      const Eigen::VectorXd prior_delta_full = *state_propagat - *state;
      const Eigen::VectorXd prior_delta = fullVectorToSolve(prior_delta_full);
      Eigen::MatrixXd solve_hessian = hessian;
      Eigen::VectorXd solve_gradient = gradient;
      rollback_G = G;
      rollback_active_extrinsic_rot = final_active_extrinsic_rot;
      rollback_active_extrinsic_trans = final_active_extrinsic_trans;
      rollback_active_time_groups = final_active_time_groups;
      rollback_last_time_tracks = last_time_tracks;
      rollback_last_time_avg_pixel_velocity = last_time_avg_pixel_velocity;
      rollback_last_time_update_ms = last_time_update_ms;
      rollback_last_extrinsic_rot_update_deg = last_extrinsic_rot_update_deg;
      rollback_last_extrinsic_trans_update_cm = last_extrinsic_trans_update_cm;
      rollback_last_max_rot_update_deg = last_max_rot_update_deg;
      rollback_last_max_trans_update_cm = last_max_trans_update_cm;
      rollback_last_max_time_update_ms = last_max_time_update_ms;
      rollback_directional_result = final_directional_result;
      rollback_directional_posterior_covariance = final_directional_posterior_covariance;
      Eigen::MatrixXd K1;
      Eigen::MatrixXd solve_posterior_covariance;
      std::string solve_error;
      const Eigen::MatrixXd solve_cov = fullCovToSolve(state->cov);
      auto factorSolveSystem = [&]() {
        if (directional_update_en)
        {
          if (!directional_update::posteriorCovariance(
                  solve_cov, solve_hessian / measurement_cov,
                  solve_posterior_covariance, solve_error))
            return false;
          K1 = solve_posterior_covariance / measurement_cov;
        }
        else
        {
          K1 = (solve_hessian + (solve_cov / measurement_cov).inverse()).inverse();
        }
        return true;
      };
      if (!factorSolveSystem())
      {
        *state = state_before_visual_update;
        G.setZero();
        syncCameraExtrinsicsFromState(*state);
        std::cerr << "[ Directional VIO ] system factorization rejected: "
                  << solve_error << std::endl;
        return;
      }
      Eigen::MatrixXd G_reduced = K1 * solve_hessian;
      G = solveMatrixToFull(G_reduced);
      Eigen::VectorXd solution = solveVectorToFull(-K1 * solve_gradient + prior_delta - G_reduced * prior_delta);
      if (!calibrationUpdateWithinTrustRegion(solution, allow_extrinsic_rotation,
                                              allow_extrinsic_translation, active_time_groups))
      {
        std::vector<uint8_t> deactivate_time_groups = active_time_groups;
        auto zeroReducedIndex = [&](int full_index) {
          const int solve_index = (full_index >= 0 && full_index < full_state_dim) ? full_to_solve[full_index] : -1;
          if (solve_index < 0) return;
          solve_hessian.row(solve_index).setZero();
          solve_hessian.col(solve_index).setZero();
          solve_gradient[solve_index] = 0.0;
        };
        for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
          for (int k = 0; k < 6; ++k)
            zeroReducedIndex(state->extrinsicIndex(camera_id) + k);
        for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
          if (group_id < static_cast<int>(deactivate_time_groups.size()) && deactivate_time_groups[group_id])
            zeroReducedIndex(state->timeOffsetIndex(group_id));
        if (!factorSolveSystem())
        {
          *state = state_before_visual_update;
          G.setZero();
          syncCameraExtrinsicsFromState(*state);
          std::cerr << "[ Directional VIO ] trust-region system factorization rejected: "
                    << solve_error << std::endl;
          return;
        }
        G_reduced = K1 * solve_hessian;
        G = solveMatrixToFull(G_reduced);
        solution = solveVectorToFull(-K1 * solve_gradient + prior_delta - G_reduced * prior_delta);
        for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
        {
          solution.segment<3>(state->extrinsicRotIndex(camera_id)).setZero();
          solution.segment<3>(state->extrinsicTransIndex(camera_id)).setZero();
        }
        for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
        {
          if (group_id < static_cast<int>(active_time_groups.size()) && active_time_groups[group_id])
            solution[state->timeOffsetIndex(group_id)] = 0.0;
        }
        std::fill(final_active_extrinsic_rot.begin(), final_active_extrinsic_rot.end(), 0);
        std::fill(final_active_extrinsic_trans.begin(), final_active_extrinsic_trans.end(), 0);
        std::fill(final_active_time_groups.begin(), final_active_time_groups.end(), 0);
      }
      else
      {
        final_active_extrinsic_rot = active_extrinsic_rot;
        final_active_extrinsic_trans = active_extrinsic_trans;
        final_active_time_groups = active_time_groups;
        for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
        {
          if (camera_id < static_cast<int>(active_extrinsic_rot.size()) &&
              active_extrinsic_rot[camera_id] != 0)
          {
            const double rot_update_deg =
                solution.segment<3>(state->extrinsicRotIndex(camera_id)).norm() * kRadiansToDegrees;
            if (camera_id < static_cast<int>(last_extrinsic_rot_update_deg.size()))
              last_extrinsic_rot_update_deg[camera_id] =
                  std::max(last_extrinsic_rot_update_deg[camera_id], rot_update_deg);
            last_max_rot_update_deg = std::max(last_max_rot_update_deg, rot_update_deg);
          }
          if (camera_id < static_cast<int>(active_extrinsic_trans.size()) &&
              active_extrinsic_trans[camera_id] != 0)
          {
            const double trans_update_cm =
                solution.segment<3>(state->extrinsicTransIndex(camera_id)).norm() * 100.0;
            if (camera_id < static_cast<int>(last_extrinsic_trans_update_cm.size()))
              last_extrinsic_trans_update_cm[camera_id] =
                  std::max(last_extrinsic_trans_update_cm[camera_id], trans_update_cm);
            last_max_trans_update_cm = std::max(last_max_trans_update_cm, trans_update_cm);
          }
        }
        for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
        {
          if (group_id < static_cast<int>(active_time_groups.size()) &&
              active_time_groups[group_id] != 0)
          {
            const double time_update_ms = std::fabs(solution[state->timeOffsetIndex(group_id)]) * 1.0e3;
            if (group_id < static_cast<int>(last_time_update_ms.size()))
              last_time_update_ms[group_id] = std::max(last_time_update_ms[group_id], time_update_ms);
            last_max_time_update_ms = std::max(last_max_time_update_ms, time_update_ms);
          }
        }
      }
      if (directional_update_en)
      {
        final_directional_result = iteration_directional_result;
        final_directional_posterior_covariance = cov_before_visual_update;
        for (int r = 0; r < solve_posterior_covariance.rows(); ++r)
          for (int c = 0; c < solve_posterior_covariance.cols(); ++c)
            final_directional_posterior_covariance(solve_to_full[r], solve_to_full[c]) =
                solve_posterior_covariance(r, c);
      }
      *state += solution;
      syncCameraExtrinsicsFromState(*state);
      update_ekf_time += omp_get_wtime() - update_start;
      double max_extrinsic_update = 0.0;
      if (allow_extrinsic_rotation || allow_extrinsic_translation)
      {
        for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
        {
          if (camera_id < static_cast<int>(active_extrinsic_rot.size()) &&
              active_extrinsic_rot[camera_id] != 0)
            max_extrinsic_update = std::max(max_extrinsic_update,
                                            solution.segment<3>(state->extrinsicRotIndex(camera_id)).norm());
          if (camera_id < static_cast<int>(active_extrinsic_trans.size()) &&
              active_extrinsic_trans[camera_id] != 0)
            max_extrinsic_update = std::max(max_extrinsic_update,
                                            solution.segment<3>(state->extrinsicTransIndex(camera_id)).norm());
        }
      }
      double max_time_update = 0.0;
      for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
      {
        if (group_id < static_cast<int>(active_time_groups.size()) && active_time_groups[group_id])
          max_time_update = std::max(max_time_update, std::fabs(solution[state->timeOffsetIndex(group_id)]));
      }
      if (solution.segment<3>(0).norm() * 57.3 < 0.001 &&
          solution.segment<3>(3).norm() * 100.0 < 0.001 &&
          max_extrinsic_update < 1.0e-6 &&
          max_time_update < 1.0e-6)
        break;
    }
  }
  if (directional_update_en)
  {
    if (final_directional_posterior_covariance.rows() == state->stateDim() &&
        final_directional_posterior_covariance.cols() == state->stateDim())
    {
      state->cov = final_directional_posterior_covariance;
    }
    else
    {
      state->cov = cov_before_visual_update;
    }
  }
  else
  {
    state->cov -= G * state->cov;
  }
  restoreInactiveCalibrationCovariance(cov_before_visual_update,
                                       final_active_extrinsic_rot,
                                       final_active_extrinsic_trans,
                                       final_active_time_groups);
  if (directional_update_en)
  {
    state->cov = directional_update::symmetrize(state->cov);
    if (final_directional_result.valid)
    {
      std::cout << "[ Directional VIO ] rho_max="
                << final_directional_result.variance_reductions.maxCoeff()
                << " rho_mean=" << final_directional_result.variance_reductions.mean()
                << " active=" << final_directional_result.active_rank
                << " full=" << final_directional_result.full_rank
                << " dim=" << state->stateDim() << std::endl;
    }
  }
  recordUsagePoseFrameInfo(usage_prior_cov, state->cov, usage_final_h_base,
                           usage_final_h_same, usage_final_h_cross, usage_final_h_current_cross,
                           usage_final_patches_all, usage_final_residuals_all,
                           usage_final_patches_same, usage_final_residuals_same,
                           usage_final_patches_cross, usage_final_residuals_cross,
                           usage_final_patches_current_cross, usage_final_residuals_current_cross);
  online_calib_last_active_extrinsic_rot_ = final_active_extrinsic_rot;
  online_calib_last_active_extrinsic_trans_ = final_active_extrinsic_trans;
  online_calib_last_active_time_groups_ = final_active_time_groups;
  online_calib_last_time_tracks_ = last_time_tracks;
  online_calib_last_time_avg_pixel_velocity_ = last_time_avg_pixel_velocity;
  online_calib_last_time_update_ms_ = last_time_update_ms;
  online_calib_last_extrinsic_rot_update_deg_ = last_extrinsic_rot_update_deg;
  online_calib_last_extrinsic_trans_update_cm_ = last_extrinsic_trans_update_cm;
  online_calib_last_max_rot_update_deg_ = last_max_rot_update_deg;
  online_calib_last_max_trans_update_cm_ = last_max_trans_update_cm;
  online_calib_last_max_time_update_ms_ = last_max_time_update_ms;

  constexpr double kAcceptedUpdateEps = 1.0e-12;
  bool accepted_extrinsic_this_frame = false;
  for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
  {
    const double rot_update = camera_id < static_cast<int>(last_extrinsic_rot_update_deg.size())
                                  ? last_extrinsic_rot_update_deg[camera_id]
                                  : 0.0;
    const double trans_update = camera_id < static_cast<int>(last_extrinsic_trans_update_cm.size())
                                    ? last_extrinsic_trans_update_cm[camera_id]
                                    : 0.0;
    if (rot_update > kAcceptedUpdateEps || trans_update > kAcceptedUpdateEps)
    {
      accepted_extrinsic_this_frame = true;
      if (camera_id < static_cast<int>(online_extrinsic_camera_accept_count_.size()))
        ++online_extrinsic_camera_accept_count_[camera_id];
    }
  }
  bool accepted_time_this_frame = false;
  for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
  {
    const double time_update = group_id < static_cast<int>(last_time_update_ms.size())
                                   ? last_time_update_ms[group_id]
                                   : 0.0;
    if (time_update > kAcceptedUpdateEps)
    {
      accepted_time_this_frame = true;
      if (group_id < static_cast<int>(online_time_offset_group_accept_count_.size()))
        ++online_time_offset_group_accept_count_[group_id];
    }
  }
  if (attempted_extrinsic_this_frame) ++online_extrinsic_attempt_count_;
  if (accepted_extrinsic_this_frame) ++online_extrinsic_accept_count_;
  if (attempted_time_this_frame) ++online_time_offset_attempt_count_;
  if (accepted_time_this_frame) ++online_time_offset_accept_count_;
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
  const Eigen::MatrixXd cov_before_visual_update = state->cov;
  std::vector<uint8_t> final_active_extrinsic_rot(state->num_cameras, 0);
  std::vector<uint8_t> final_active_extrinsic_trans(state->num_cameras, 0);
  std::vector<uint8_t> final_active_time_groups(state->num_time_offset_groups, 0);
  Eigen::MatrixXd rollback_G = G;
  std::vector<uint8_t> rollback_active_extrinsic_rot = final_active_extrinsic_rot;
  std::vector<uint8_t> rollback_active_extrinsic_trans = final_active_extrinsic_trans;
  std::vector<uint8_t> rollback_active_time_groups = final_active_time_groups;

  for (int iteration = 0; iteration < max_iterations; ++iteration)
  {
    const double linearize_start = omp_get_wtime();
    const int state_dim = state->stateDim();
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(state_dim);
    double error = 0.0;
    int measurement_count = 0;
    for (PerCameraData &ctx : cameras_) updateFrameState(ctx, *state);
    std::vector<uint8_t> active_time_groups(state->num_time_offset_groups, 0);
    if (online_time_offset_en && frame_count >= online_time_offset_start_frame &&
        (online_time_offset_min_update_interval <= 0 ||
         frame_count % std::max(1, online_time_offset_min_update_interval) == 0))
    {
      std::vector<int> group_tracks(state->num_time_offset_groups, 0);
      std::vector<double> group_pixel_velocity(state->num_time_offset_groups, 0.0);
      for (PerCameraData &ctx : cameras_)
      {
        if (ctx.total_points <= 0 || ctx.visual_submap == nullptr) continue;
        const int group_id = ctx.time_offset_group;
        if (!isOnlineTimeOffsetEnabledForGroup(group_id)) continue;
        const int count = std::min<int>(ctx.total_points, ctx.visual_submap->voxel_points.size());
        for (int i = 0; i < count; ++i)
        {
          VisualPoint *point = ctx.visual_submap->voxel_points[i];
          if (point == nullptr || i >= static_cast<int>(ctx.visual_submap->virtual_track_patches.size())) continue;
          const VirtualTrackPatch &track = ctx.visual_submap->virtual_track_patches[i];
          if (!track.valid) continue;
          const V3D point_c = ctx.Rcw * point->pos_ + ctx.Pcw;
          const V3D point_v = track.R_vcur_from_ccur_seed * point_c;
          if (!point_v.array().isFinite().all() || point_v.norm() <= kS2Eps) continue;
          MD(2, 3) Jdpi;
          computeVirtualProjectionJacobian(point_v, Jdpi);
          const V3D point_i = ctx.Rwi.transpose() * (point->pos_ - ctx.Pwi);
          M3D point_i_hat;
          point_i_hat << SKEW_SYM_MATRX(point_i);
          const V3D dpc_dtd = ctx.Rci * (point_i_hat * ctx.gyro_i - ctx.Rwi.transpose() * ctx.Vwi);
          const double pixel_speed = (Jdpi * track.R_vcur_from_ccur_seed * dpc_dtd).norm();
          if (!std::isfinite(pixel_speed)) continue;
          ++group_tracks[group_id];
          group_pixel_velocity[group_id] += pixel_speed;
        }
      }
      for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
      {
        const double avg_pixel_velocity =
            group_tracks[group_id] > 0 ? group_pixel_velocity[group_id] / group_tracks[group_id] : 0.0;
        if (isOnlineTimeOffsetEnabledForGroup(group_id) &&
            group_tracks[group_id] >= online_time_offset_min_tracks &&
            avg_pixel_velocity >= online_time_offset_min_pixel_velocity)
          active_time_groups[group_id] = 1;
      }
    }
    std::vector<uint8_t> active_extrinsic_rot(state->num_cameras, 0);
    std::vector<uint8_t> active_extrinsic_trans(state->num_cameras, 0);
    for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
    {
      const bool camera_active = camera_id < static_cast<int>(cameras_.size()) &&
                                 isOnlineExtrinsicEnabledForCamera(camera_id) &&
                                 cameras_[camera_id].total_points >= online_extrinsic_min_tracks;
      if (camera_active && allow_extrinsic_rotation) active_extrinsic_rot[camera_id] = 1;
      if (camera_active && allow_extrinsic_translation) active_extrinsic_trans[camera_id] = 1;
    }

    for (PerCameraData &ctx : cameras_)
    {
      if (ctx.total_points == 0 || ctx.visual_submap == nullptr) continue;
      const cv::Mat &raw_img = (ctx.new_frame != nullptr) ? ctx.new_frame->img_ : img;
      if (raw_img.empty()) continue;
      const M3D Rwi = ctx.Rwi;
      const V3D Pwi = ctx.Pwi;
      const bool estimate_extrinsic =
          ctx.camera_id >= 0 && ctx.camera_id < state->num_cameras &&
          (active_extrinsic_rot[ctx.camera_id] != 0 || active_extrinsic_trans[ctx.camera_id] != 0);
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
        if (!track.valid) continue;
        const V3D point_c = ctx.Rcw * point->pos_ + ctx.Pcw;
        if (!point_c.array().isFinite().all()) continue;
        const double point_c_norm = point_c.norm();
        if (!std::isfinite(point_c_norm) || point_c_norm <= kS2Eps) continue;
        const V3D point_i_for_time = Rwi.transpose() * (point->pos_ - Pwi);
        M3D point_i_for_time_hat;
        point_i_for_time_hat << SKEW_SYM_MATRX(point_i_for_time);
        const V3D dpc_dtd =
            ctx.Rci * (point_i_for_time_hat * ctx.gyro_i - Rwi.transpose() * ctx.Vwi);

        M3D Jpc_dRcl = M3D::Zero();
        if (estimate_extrinsic)
        {
          const V3D point_l = Rli * point_i_for_time + Pli;
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
          jacobian.segment<3>(state->velocityIndex()) = (J_photo_center * ctx.dpc_dvel).transpose();
          const int group_id = ctx.time_offset_group;
          if (group_id >= 0 && group_id < state->num_time_offset_groups &&
              group_id < static_cast<int>(active_time_groups.size()) &&
              active_time_groups[group_id] != 0)
            jacobian[state->timeOffsetIndex(group_id)] = (J_photo_center * dpc_dtd)(0, 0);
          if (exposure_estimate_en) jacobian[state->exposureIndex(ctx.camera_id)] = current_value;
          if (estimate_extrinsic)
          {
            if (active_extrinsic_rot[ctx.camera_id] != 0)
              jacobian.segment<3>(state->extrinsicRotIndex(ctx.camera_id)) =
                  (J_photo_center * Jpc_dRcl).transpose();
            if (active_extrinsic_trans[ctx.camera_id] != 0)
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
    if (online_time_offset_en)
      applyOnlineTimeOffsetPriors(hessian, gradient, active_time_groups);
    compute_jacobian_time += omp_get_wtime() - linearize_start;
    if (measurement_count == 0) return;
    error /= measurement_count;
    if (error > last_error)
    {
      *state = old_state;
      G = rollback_G;
      final_active_extrinsic_rot = rollback_active_extrinsic_rot;
      final_active_extrinsic_trans = rollback_active_extrinsic_trans;
      final_active_time_groups = rollback_active_time_groups;
      syncCameraExtrinsicsFromState(*state);
      break;
    }

    old_state = *state;
    last_error = error;
    const double update_start = omp_get_wtime();
    const Eigen::VectorXd prior_delta = *state_propagat - *state;
    Eigen::MatrixXd solve_hessian = hessian;
    Eigen::VectorXd solve_gradient = gradient;
    rollback_G = G;
    rollback_active_extrinsic_rot = final_active_extrinsic_rot;
    rollback_active_extrinsic_trans = final_active_extrinsic_trans;
    rollback_active_time_groups = final_active_time_groups;
    deactivateInactiveCalibrationBlocks(solve_hessian, solve_gradient,
                                        active_extrinsic_rot,
                                        active_extrinsic_trans,
                                        active_time_groups);
    Eigen::MatrixXd K1 = (solve_hessian + (state->cov / img_point_cov).inverse()).inverse();
    G = K1 * solve_hessian;
    Eigen::VectorXd solution = -K1 * solve_gradient + prior_delta - G * prior_delta;
    auto zeroInactiveCalibrationInSolution = [&](Eigen::VectorXd &delta) {
      for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
      {
        if (camera_id >= static_cast<int>(active_extrinsic_rot.size()) ||
            active_extrinsic_rot[camera_id] == 0)
          delta.segment<3>(state->extrinsicRotIndex(camera_id)).setZero();
        if (camera_id >= static_cast<int>(active_extrinsic_trans.size()) ||
            active_extrinsic_trans[camera_id] == 0)
          delta.segment<3>(state->extrinsicTransIndex(camera_id)).setZero();
      }
      for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
      {
        if (group_id >= static_cast<int>(active_time_groups.size()) || active_time_groups[group_id] == 0)
          delta[state->timeOffsetIndex(group_id)] = 0.0;
      }
    };
    zeroInactiveCalibrationInSolution(solution);
    if (!calibrationUpdateWithinTrustRegion(solution, allow_extrinsic_rotation,
                                            allow_extrinsic_translation, active_time_groups))
    {
      std::vector<uint8_t> deactivate_time_groups = active_time_groups;
      deactivateCalibrationBlocks(solve_hessian, solve_gradient, true, deactivate_time_groups);
      K1 = (solve_hessian + (state->cov / img_point_cov).inverse()).inverse();
      G = K1 * solve_hessian;
      solution = -K1 * solve_gradient + prior_delta - G * prior_delta;
      zeroInactiveCalibrationInSolution(solution);
      for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
      {
        solution.segment<3>(state->extrinsicRotIndex(camera_id)).setZero();
        solution.segment<3>(state->extrinsicTransIndex(camera_id)).setZero();
      }
      for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
      {
        if (group_id < static_cast<int>(active_time_groups.size()) && active_time_groups[group_id])
          solution[state->timeOffsetIndex(group_id)] = 0.0;
      }
      std::fill(final_active_extrinsic_rot.begin(), final_active_extrinsic_rot.end(), 0);
      std::fill(final_active_extrinsic_trans.begin(), final_active_extrinsic_trans.end(), 0);
      std::fill(final_active_time_groups.begin(), final_active_time_groups.end(), 0);
    }
    else
    {
      final_active_extrinsic_rot = active_extrinsic_rot;
      final_active_extrinsic_trans = active_extrinsic_trans;
      final_active_time_groups = active_time_groups;
    }
    *state += solution;
    syncCameraExtrinsicsFromState(*state);
    update_ekf_time += omp_get_wtime() - update_start;
    double max_extrinsic_update = 0.0;
    for (int camera_id = 0; camera_id < state->num_cameras; ++camera_id)
    {
      if (camera_id < static_cast<int>(active_extrinsic_rot.size()) &&
          active_extrinsic_rot[camera_id] != 0)
        max_extrinsic_update = std::max(max_extrinsic_update,
                                        solution.segment<3>(state->extrinsicRotIndex(camera_id)).norm());
      if (camera_id < static_cast<int>(active_extrinsic_trans.size()) &&
          active_extrinsic_trans[camera_id] != 0)
        max_extrinsic_update = std::max(max_extrinsic_update,
                                        solution.segment<3>(state->extrinsicTransIndex(camera_id)).norm());
    }
    double max_time_update = 0.0;
    for (int group_id = 0; group_id < state->num_time_offset_groups; ++group_id)
    {
      if (group_id < static_cast<int>(active_time_groups.size()) && active_time_groups[group_id])
        max_time_update = std::max(max_time_update, std::fabs(solution[state->timeOffsetIndex(group_id)]));
    }
    if (solution.segment<3>(0).norm() * 57.3 < 0.001 &&
        solution.segment<3>(3).norm() * 100.0 < 0.001 &&
        max_extrinsic_update < 1.0e-6 &&
        max_time_update < 1.0e-6)
      break;
  }
  state->cov -= G * state->cov;
  restoreInactiveCalibrationCovariance(cov_before_visual_update,
                                       final_active_extrinsic_rot,
                                       final_active_extrinsic_trans,
                                       final_active_time_groups);
  for (PerCameraData &ctx : cameras_) updateFrameState(ctx, *state);
}

void VIOManager::generateVisualMapPointsVirtual(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                                 const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (pg.size() <= 10) return;

  std::array<int, VISUAL_GEOM_REJECT_COUNT> geom_reject_counts = {};
  int geom_checked = 0;
  int geom_passed = 0;
  auto pass_candidate_geometry = [&](const pointWithVar &candidate) {
    if (!visual_geom_filter_en) return true;
    ++geom_checked;
    VisualGeomRejectReason reason = VISUAL_GEOM_REJECT_NONE;
    if (!passVisualGeometryFilter(candidate, plane_map, reason))
    {
      const int reason_index = static_cast<int>(reason);
      if (reason_index > VISUAL_GEOM_REJECT_NONE && reason_index < VISUAL_GEOM_REJECT_COUNT)
        ++geom_reject_counts[reason_index];
      return false;
    }
    ++geom_passed;
    return true;
  };
  auto consider_candidate = [&](const pointWithVar &candidate, int source_type, int source_index) {
    if (candidate.normal == V3D::Zero()) return;
    if (!pass_candidate_geometry(candidate)) return;
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

    if (visual_ref_post_ekf_build_en)
    {
      PendingNewPointObservation pending;
      pending.camera_id = ctx.camera_id;
      pending.source_type = ctx.append_voxel_source_type[i];
      pending.source_index = ctx.append_voxel_source_index[i];
      pending.pt_var = pt_var;
      pending.texture_score = ctx.scan_value[i];
      pending.px = raw_px;
      pending.bearing = ctx.cam->cam2world(raw_px);
      pending.T_f_w = ctx.new_frame->T_f_w_;
      fillPendingObservationTiming(pending, ctx);
      pending.inv_expo_time = state->inv_expo_time[ctx.camera_id];
      ctx.pending_new_points.push_back(std::move(pending));
      continue;
    }

    std::vector<float> patch(patch_size_total);
    cv::Mat virtual_support_img;
    cv::Point virtual_source_origin;
    SE3d T_v_w;
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
    pending.texture_score = ctx.scan_value[i];
    pending.px = raw_px;
    pending.bearing = ctx.cam->cam2world(raw_px);
    pending.patch = std::move(patch);
    if (ref_patch_dump_en)
      maybeInitializeRefPatchDumpProbe(ctx, pt_var.point_w, pt_var.normal, raw_px, pending.bearing, pending.patch.data());
    pending.img = virtual_support_img;
    pending.virtual_source_origin = virtual_source_origin;
    pending.T_f_w = ctx.new_frame->T_f_w_;
    fillPendingObservationTiming(pending, ctx);
    pending.T_v_w = T_v_w;
    pending.R_v_from_c = R_v_from_c;
    pending.R_c_from_v = R_c_from_v;
    pending.virtual_patch_valid = true;
    pending.inv_expo_time = state->inv_expo_time[ctx.camera_id];
    ctx.pending_new_points.push_back(std::move(pending));
  }
  if (visual_geom_filter_en && visual_geom_filter_log_en)
  {
    printf("[ VIO Geom Filter ] camera_id=%d mode=virtual checked=%d pass=%d bad=%d no_plane=%d not_plane=%d plane_size=%d range=%d normal=%d cov=%d sigma=%d chi2=%d pending=%zu\n",
           ctx.camera_id, geom_checked, geom_passed,
           geom_reject_counts[VISUAL_GEOM_REJECT_BAD_POINT], geom_reject_counts[VISUAL_GEOM_REJECT_NO_PLANE],
           geom_reject_counts[VISUAL_GEOM_REJECT_NOT_PLANE], geom_reject_counts[VISUAL_GEOM_REJECT_PLANE_SIZE],
           geom_reject_counts[VISUAL_GEOM_REJECT_RANGE], geom_reject_counts[VISUAL_GEOM_REJECT_NORMAL],
           geom_reject_counts[VISUAL_GEOM_REJECT_COV], geom_reject_counts[VISUAL_GEOM_REJECT_SIGMA],
           geom_reject_counts[VISUAL_GEOM_REJECT_CHI2], ctx.pending_new_points.size());
  }
  printf("[ VIO Virtual ] camera_id=%d selected %zu pending observations\n", ctx.camera_id, ctx.pending_new_points.size());
}

void VIOManager::generateVisualMapPoints(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                     const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (virtual_fisheye_patch_en)
  {
    generateVisualMapPointsVirtual(ctx, img, pg, plane_map);
    return;
  }
  if (pg.size() <= 10) return;

  std::array<int, VISUAL_GEOM_REJECT_COUNT> geom_reject_counts = {};
  int geom_checked = 0;
  int geom_passed = 0;
  auto pass_candidate_geometry = [&](const pointWithVar &candidate) {
    if (!visual_geom_filter_en) return true;
    ++geom_checked;
    VisualGeomRejectReason reason = VISUAL_GEOM_REJECT_NONE;
    if (!passVisualGeometryFilter(candidate, plane_map, reason))
    {
      const int reason_index = static_cast<int>(reason);
      if (reason_index > VISUAL_GEOM_REJECT_NONE && reason_index < VISUAL_GEOM_REJECT_COUNT)
        ++geom_reject_counts[reason_index];
      return false;
    }
    ++geom_passed;
    return true;
  };
  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (pg[i].normal == V3D(0, 0, 0)) continue;
    if (!pass_candidate_geometry(pg[i])) continue;

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
    const pointWithVar &candidate = ctx.visual_submap->add_from_voxel_map[j];
    if (!pass_candidate_geometry(candidate)) continue;
    V3D pt = candidate.point_w;
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
          ctx.append_voxel_points[index] = candidate;
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

      if (visual_ref_post_ekf_build_en)
      {
        PendingNewPointObservation pending;
        pending.camera_id = ctx.camera_id;
        pending.source_type = ctx.append_voxel_source_type[i];
        pending.source_index = ctx.append_voxel_source_index[i];
        pending.pt_var = pt_var;
        pending.texture_score = ctx.scan_value[i];
        pending.px = pc;
        pending.bearing = ctx.cam->cam2world(pc);
        pending.T_f_w = ctx.new_frame->T_f_w_;
        fillPendingObservationTiming(pending, ctx);
        pending.inv_expo_time = state->inv_expo_time[ctx.camera_id];
        ctx.pending_new_points.push_back(std::move(pending));
        continue;
      }

      PendingNewPointObservation pending;
      pending.camera_id = ctx.camera_id;
      pending.source_type = ctx.append_voxel_source_type[i];
      pending.source_index = ctx.append_voxel_source_index[i];
      pending.pt_var = pt_var;
      pending.texture_score = ctx.scan_value[i];
      pending.px = pc;
      pending.bearing = ctx.cam->cam2world(pc);
      pending.patch.resize(patch_size_total);
      getImagePatch(ctx, img, pc, pending.patch.data(), 0);
      pending.img = img;
      pending.T_f_w = ctx.new_frame->T_f_w_;
      fillPendingObservationTiming(pending, ctx);
      pending.inv_expo_time = state->inv_expo_time[ctx.camera_id];
      ctx.pending_new_points.push_back(std::move(pending));
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  if (visual_geom_filter_en && visual_geom_filter_log_en)
  {
    printf("[ VIO Geom Filter ] camera_id=%d mode=raw checked=%d pass=%d bad=%d no_plane=%d not_plane=%d plane_size=%d range=%d normal=%d cov=%d sigma=%d chi2=%d pending=%zu\n",
           ctx.camera_id, geom_checked, geom_passed,
           geom_reject_counts[VISUAL_GEOM_REJECT_BAD_POINT], geom_reject_counts[VISUAL_GEOM_REJECT_NO_PLANE],
           geom_reject_counts[VISUAL_GEOM_REJECT_NOT_PLANE], geom_reject_counts[VISUAL_GEOM_REJECT_PLANE_SIZE],
           geom_reject_counts[VISUAL_GEOM_REJECT_RANGE], geom_reject_counts[VISUAL_GEOM_REJECT_NORMAL],
           geom_reject_counts[VISUAL_GEOM_REJECT_COV], geom_reject_counts[VISUAL_GEOM_REJECT_SIGMA],
           geom_reject_counts[VISUAL_GEOM_REJECT_CHI2], ctx.pending_new_points.size());
  }
  printf("[ VIO ] camera_id=%d selected %zu pending observations\n", ctx.camera_id, ctx.pending_new_points.size());
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

std::vector<Feature *> VIOManager::selectManagedReferenceCandidates(const PerCameraData &ctx,
                                                                    VisualPoint &point,
                                                                    int max_candidates) const
{
  std::vector<Feature *> selected;
  if (point.pending_delete_ || point.state_ == VisualPoint::State::RETIRED) return selected;
  const int requested = std::max(1, max_candidates);
  if (!visual_map_manage_en || !visual_ref_current_select_en)
  {
    Feature *legacy = point.referencePatch(ctx.camera_id, cross_camera_reference_en);
    if (legacy != nullptr) selected.push_back(legacy);
    else
    {
      Feature *closest = nullptr;
      point.getCloseViewObs(ctx.new_frame->pos(), closest, V2D::Zero(),
                            cross_camera_reference_en ? -1 : ctx.camera_id);
      if (closest != nullptr) selected.push_back(closest);
    }
    return selected;
  }

  const bool allow_candidate = visual_point_seed_validation_en && point.state_ == VisualPoint::State::SEED;
  const V3D current_direction = (ctx.new_frame->pos() - point.pos_).normalized();
  std::vector<std::pair<double, Feature *>> ranked_validated;
  std::vector<std::pair<double, Feature *>> ranked_candidate;
  ranked_validated.reserve(point.obs_.size());
  ranked_candidate.reserve(point.obs_.size());
  for (Feature *feature : point.obs_)
  {
    if (feature == nullptr || feature->pending_delete_ || feature->ref_state_ == Feature::RefState::RETIRED) continue;
    if (visual_map_manage_shadow_en && shadow_retired_ref_suggestions_.count(feature->ref_id_) != 0) continue;
    if (!cross_camera_reference_en && feature->camera_id_ != ctx.camera_id) continue;
    if (virtual_fisheye_patch_en && !feature->virtual_patch_valid_) continue;
    V3D reference_direction = feature->view_direction_w_;
    if (!reference_direction.array().isFinite().all() || reference_direction.norm() <= 1.0e-9)
      reference_direction = feature->pos() - point.pos_;
    if (!reference_direction.array().isFinite().all() || reference_direction.norm() <= 1.0e-9) continue;
    reference_direction.normalize();
    const double angle = std::acos(std::clamp(current_direction.dot(reference_direction), -1.0, 1.0));
    if (feature->ref_state_ == Feature::RefState::VALIDATED)
      ranked_validated.emplace_back(angle, feature);
    else if (feature->ref_state_ == Feature::RefState::CANDIDATE)
      ranked_candidate.emplace_back(angle, feature);
  }
  auto rank = [](auto &items) {
    std::stable_sort(items.begin(), items.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first) return lhs.first < rhs.first;
    return lhs.second->accepted_test_count_ > rhs.second->accepted_test_count_;
    });
  };
  rank(ranked_validated);
  rank(ranked_candidate);

  std::vector<std::pair<double, Feature *>> ranked;
  const double coverage_angle = visual_ref_coverage_angle_deg / kRadiansToDegrees;
  const bool candidate_needed = allow_candidate || ranked_validated.empty() ||
                                (!ranked_candidate.empty() &&
                                 ranked_validated.front().first > coverage_angle &&
                                 ranked_candidate.front().first < ranked_validated.front().first);
  if (candidate_needed) ranked.insert(ranked.end(), ranked_candidate.begin(), ranked_candidate.end());
  ranked.insert(ranked.end(), ranked_validated.begin(), ranked_validated.end());
  for (const auto &item : ranked)
  {
    selected.push_back(item.second);
    if (static_cast<int>(selected.size()) >= requested) break;
  }
  return selected;
}

bool VIOManager::shouldReferenceContributeToEkf(const VisualPoint &point,
                                                const Feature &reference) const
{
  if (!visual_map_manage_en || visual_map_manage_shadow_en) return true;
  if (point.pending_delete_ || reference.pending_delete_) return false;
  if (point.state_ == VisualPoint::State::RETIRED || reference.ref_state_ == Feature::RefState::RETIRED) return false;
  if (visual_point_seed_validation_en && point.state_ == VisualPoint::State::SEED) return false;
  if (visual_ref_lifecycle_en && reference.ref_state_ != Feature::RefState::VALIDATED) return false;
  return true;
}

void VIOManager::appendManagedSubmapMetadata(PerCameraData &ctx, VisualPoint &point,
                                             Feature &reference)
{
  if (!visual_map_manage_en || ctx.visual_submap == nullptr) return;
  ctx.visual_submap->contributes_to_ekf.push_back(
      shouldReferenceContributeToEkf(point, reference) ? 1 : 0);
  ctx.visual_submap->pose_information.push_back(Eigen::Matrix<double, 6, 6>::Zero());
  ctx.visual_submap->observation_nis.push_back(std::numeric_limits<double>::quiet_NaN());
  ctx.visual_submap->observation_dof.push_back(0);
  point.last_visible_frame_ = ctx.new_frame != nullptr ? ctx.new_frame->id_ : frame_count;
}

void VIOManager::initializeManagedReference(Feature &feature, VisualPoint &point,
                                             const PerCameraData &ctx,
                                             bool initial_point_reference)
{
  (void)initial_point_reference;
  if (!visual_map_manage_en) return;
  feature.ref_id_ = next_visual_ref_id_++;
  feature.birth_frame_id_ = ctx.new_frame != nullptr ? ctx.new_frame->id_ : frame_count;
  feature.last_test_frame_id_ = -1;
  feature.last_test_camera_id_ = -1;
  feature.last_success_frame_id_ = -1;
  feature.ref_state_ = visual_ref_lifecycle_en
                           ? Feature::RefState::CANDIDATE
                           : Feature::RefState::VALIDATED;
  const V3D camera_w = feature.pos();
  const V3D view = camera_w - point.pos_;
  feature.view_range_ = view.norm();
  if (feature.view_range_ > 1.0e-9)
    feature.view_direction_w_ = view / feature.view_range_;
  else
    feature.view_direction_w_.setZero();
  feature.birth_pose_cov_.setZero();
  if (state != nullptr && state->cov.rows() >= 6 && state->cov.cols() >= 6)
    feature.birth_pose_cov_ = state->cov.block<6, 6>(0, 0);
}

void VIOManager::queueReferenceRetirement(VisualPoint &point, Feature &feature)
{
  if (feature.pending_delete_) return;
  if (visual_map_manage_shadow_en || !visual_map_retirement_apply_en)
  {
    if (shadow_retired_ref_suggestions_.insert(feature.ref_id_).second)
      ++visual_map_manage_stats_.ref_retired;
    return;
  }
  feature.ref_state_ = Feature::RefState::RETIRED;
  feature.pending_delete_ = true;
  ++visual_map_manage_stats_.ref_retired;
  if (!visual_map_manage_shadow_en) retired_visual_refs_.emplace_back(&point, &feature);
}

void VIOManager::queuePointRetirement(VisualPoint &point)
{
  if (point.pending_delete_) return;
  if (visual_map_manage_shadow_en || !visual_map_retirement_apply_en)
  {
    if (shadow_retired_point_suggestions_.insert(point.point_id_).second)
      ++visual_map_manage_stats_.point_retired;
    return;
  }
  point.state_ = VisualPoint::State::RETIRED;
  point.pending_delete_ = true;
  ++visual_map_manage_stats_.point_retired;
  if (!visual_map_manage_shadow_en) retired_visual_points_.push_back(&point);
}

void VIOManager::flushVisualMapRetirements()
{
  if (!visual_map_manage_en || !visual_map_retirement_apply_en || visual_map_manage_shadow_en)
  {
    retired_visual_refs_.clear();
    retired_visual_points_.clear();
    return;
  }

  std::set<VisualPoint *> retiring_points(retired_visual_points_.begin(), retired_visual_points_.end());
  if (!retiring_points.empty())
  {
    for (auto &voxel : feat_map)
    {
      if (voxel.second == nullptr) continue;
      for (VisualPoint *point : voxel.second->voxel_points)
      {
        if (point != nullptr && point->challenger_of_ != nullptr &&
            retiring_points.count(point->challenger_of_) != 0)
          point->challenger_of_ = nullptr;
      }
    }
  }
  for (const auto &entry : retired_visual_refs_)
  {
    VisualPoint *point = entry.first;
    Feature *feature = entry.second;
    if (point == nullptr || feature == nullptr || retiring_points.count(point) != 0) continue;
    point->deleteFeatureRef(feature);
  }
  retired_visual_refs_.clear();

  for (VisualPoint *point : retired_visual_points_)
  {
    if (point == nullptr) continue;
    const VOXEL_LOCATION key(point->map_voxel_x_, point->map_voxel_y_, point->map_voxel_z_);
    auto found = feat_map.find(key);
    if (found == feat_map.end() || found->second == nullptr ||
        std::find(found->second->voxel_points.begin(), found->second->voxel_points.end(), point) ==
            found->second->voxel_points.end())
    {
      found = std::find_if(feat_map.begin(), feat_map.end(), [&](const auto &entry) {
        return entry.second != nullptr &&
               std::find(entry.second->voxel_points.begin(), entry.second->voxel_points.end(), point) !=
                   entry.second->voxel_points.end();
      });
    }
    if (found != feat_map.end() && found->second != nullptr)
    {
      auto &points = found->second->voxel_points;
      points.erase(std::remove(points.begin(), points.end(), point), points.end());
      found->second->count = static_cast<int>(points.size());
      if (points.empty())
      {
        delete found->second;
        feat_map.erase(found);
      }
    }
    if (runtime_support_dump_best_point_ == point) runtime_support_dump_best_point_ = nullptr;
    delete point;
  }
  retired_visual_points_.clear();
}

void VIOManager::printVisualMapManageStats(int frame_id) const
{
  if (!visual_map_manage_en || !visual_map_manage_log_en) return;
  printf("[ VIO Map Manage ] frame=%d mode=%s shadow=%d selected=%lld fallback=%lld/%lld seed=%lld/%lld ref_validated=%lld ref_retired=%lld footprint=%lld redundant=%lld information_admitted=%lld replacement=%lld point_retired=%lld\n",
         frame_id, virtual_fisheye_patch_en ? "virtual" : "raw", visual_map_manage_shadow_en ? 1 : 0,
         visual_map_manage_stats_.dynamic_selected, visual_map_manage_stats_.fallback_accepted,
         visual_map_manage_stats_.fallback_attempted, visual_map_manage_stats_.seed_confirmed,
         visual_map_manage_stats_.seed_tested, visual_map_manage_stats_.ref_validated,
         visual_map_manage_stats_.ref_retired, visual_map_manage_stats_.footprint_compared,
         visual_map_manage_stats_.redundant_rejected, visual_map_manage_stats_.information_admitted,
         visual_map_manage_stats_.replacement_suggested,
         visual_map_manage_stats_.point_retired);
}

void VIOManager::updateManagedObservationEvidence(PerCameraData &ctx)
{
  if (!visual_map_manage_en || ctx.visual_submap == nullptr) return;
  const int count = std::min({static_cast<int>(ctx.visual_submap->voxel_points.size()),
                              static_cast<int>(ctx.visual_submap->reference_features.size()),
                              static_cast<int>(ctx.visual_submap->errors.size())});
  for (int i = 0; i < count; ++i)
  {
    VisualPoint *point = ctx.visual_submap->voxel_points[i];
    Feature *reference = ctx.visual_submap->reference_features[i];
    if (point == nullptr || reference == nullptr || point->pending_delete_ || reference->pending_delete_) continue;
    if (visual_map_manage_shadow_en &&
        (shadow_retired_point_suggestions_.count(point->point_id_) != 0 ||
         shadow_retired_ref_suggestions_.count(reference->ref_id_) != 0))
      continue;
    const int dof = i < static_cast<int>(ctx.visual_submap->observation_dof.size())
                        ? ctx.visual_submap->observation_dof[i]
                        : patch_size_total;
    const double nis = i < static_cast<int>(ctx.visual_submap->observation_nis.size())
                           ? ctx.visual_submap->observation_nis[i]
                           : std::numeric_limits<double>::quiet_NaN();
    const double normalized_nis = dof > 0 && std::isfinite(nis)
                                      ? nis / static_cast<double>(dof)
                                      : std::numeric_limits<double>::quiet_NaN();
    const bool accepted = visual_ref_nis_en
                              ? std::isfinite(normalized_nis) && normalized_nis <= visual_ref_nis_max_per_dof
                              : std::isfinite(ctx.visual_submap->errors[i]) &&
                                    (zncc_residual_en ||
                                     ctx.visual_submap->errors[i] <= outlier_threshold * patch_size_total);

    reference->last_test_frame_id_ = ctx.new_frame->id_;
    reference->last_test_camera_id_ = ctx.camera_id;
    reference->last_nis_ = nis;
    if (std::isfinite(normalized_nis))
      reference->nis_ema_ = reference->independent_test_count_ == 0
                                ? normalized_nis
                                : 0.9 * reference->nis_ema_ + 0.1 * normalized_nis;
    ++reference->independent_test_count_;
    ++point->independent_test_count_;
    if (point->state_ == VisualPoint::State::SEED) ++visual_map_manage_stats_.seed_tested;

    if (accepted)
    {
      reference->consecutive_reject_count_ = 0;
      ++reference->accepted_test_count_;
      ++point->accepted_test_count_;
      reference->last_success_frame_id_ = ctx.new_frame->id_;
      point->last_success_frame_ = ctx.new_frame->id_;
      const V3D view = ctx.new_frame->pos() - point->pos_;
      if (view.array().isFinite().all() && view.norm() > 1.0e-9)
      {
        const V3D direction = view.normalized();
        bool represented = false;
        const double sample_separation = 3.0 / kRadiansToDegrees;
        for (const V3D &sample : point->view_samples_)
        {
          if (std::acos(std::clamp(direction.dot(sample), -1.0, 1.0)) <= sample_separation)
          {
            represented = true;
            break;
          }
        }
        if (!represented)
        {
          point->view_samples_.push_back(direction);
          if (point->view_samples_.size() > 64) point->view_samples_.erase(point->view_samples_.begin());
        }
      }
      if (i < static_cast<int>(ctx.visual_submap->pose_information.size()))
      {
        const Eigen::Matrix<double, 6, 6> &information = ctx.visual_submap->pose_information[i];
        reference->mean_pose_information_ +=
            (information - reference->mean_pose_information_) /
            static_cast<double>(reference->accepted_test_count_);
        point->accumulated_pose_information_ += information;
        Eigen::Matrix<double, 6, 6> regularized = information;
        regularized.diagonal().array() += 1.0e-9;
        const double determinant = regularized.determinant();
        if (std::isfinite(determinant) && determinant > 0.0)
          reference->fisher_log_p_sum_ += std::log(determinant);
      }
    }
    else
    {
      ++reference->consecutive_reject_count_;
      ++reference->rejected_test_count_;
      ++point->rejected_test_count_;
    }

    if (visual_ref_lifecycle_en && reference->ref_state_ == Feature::RefState::CANDIDATE &&
        reference->independent_test_count_ >= visual_ref_validate_min_tests)
    {
      const double ratio = static_cast<double>(reference->accepted_test_count_) /
                           std::max(1, reference->independent_test_count_);
      if (ratio >= visual_ref_validate_min_ratio)
      {
        reference->ref_state_ = Feature::RefState::VALIDATED;
        ++visual_map_manage_stats_.ref_validated;
      }
      else if (reference->rejected_test_count_ >= visual_ref_retire_reject_count)
      {
        queueReferenceRetirement(*point, *reference);
      }
    }
    else if (visual_ref_lifecycle_en && reference->ref_state_ == Feature::RefState::VALIDATED &&
             reference->consecutive_reject_count_ >= visual_ref_retire_reject_count)
    {
      queueReferenceRetirement(*point, *reference);
    }

    if (visual_point_seed_validation_en && point->state_ == VisualPoint::State::SEED &&
        point->independent_test_count_ >= visual_point_seed_min_tests)
    {
      const double ratio = static_cast<double>(point->accepted_test_count_) /
                           std::max(1, point->independent_test_count_);
      if (ratio >= visual_point_seed_min_ratio)
      {
        point->state_ = VisualPoint::State::CONFIRMED;
        ++visual_map_manage_stats_.seed_confirmed;
        if (point->challenger_of_ != nullptr && !point->challenger_of_->pending_delete_)
        {
          VisualPoint *existing = point->challenger_of_;
          if (existing->state_ == VisualPoint::State::SUSPECT)
          {
            queuePointRetirement(*existing);
          }
          else if (visual_point_information_prune_en)
          {
            Eigen::Matrix<double, 6, 6> old_info =
                existing->accumulated_pose_information_ / std::max(1, existing->accepted_test_count_);
            Eigen::Matrix<double, 6, 6> new_info =
                point->accumulated_pose_information_ / std::max(1, point->accepted_test_count_);
            Eigen::Matrix<double, 6, 6> all_info = old_info + new_info;
            old_info.diagonal().array() += 1.0e-9;
            new_info.diagonal().array() += 1.0e-9;
            all_info.diagonal().array() += 1.0e-9;
            auto retention = [&](const Eigen::Matrix<double, 6, 6> &subset) {
              const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(subset, all_info);
              return solver.info() == Eigen::Success ? solver.eigenvalues().minCoeff() : 0.0;
            };
            const double old_retention = retention(old_info);
            const double new_retention = retention(new_info);
            if (new_retention >= visual_point_information_retain &&
                new_retention > old_retention)
              queuePointRetirement(*existing);
            else if (old_retention >= visual_point_information_retain)
              queuePointRetirement(*point);
          }
          point->challenger_of_ = nullptr;
        }
      }
      else if (point->rejected_test_count_ >= visual_point_suspect_reject_count)
      {
        point->state_ = VisualPoint::State::SUSPECT;
      }
    }
    else if (visual_ref_lifecycle_en && point->state_ == VisualPoint::State::CONFIRMED &&
             point->rejected_test_count_ >= visual_point_suspect_reject_count &&
             point->accepted_test_count_ * 2 < point->rejected_test_count_)
    {
      point->state_ = VisualPoint::State::SUSPECT;
    }
    bool has_usable_reference = point->hasUsableReference(0, true, false);
    if (visual_map_manage_shadow_en && has_usable_reference)
    {
      has_usable_reference = false;
      for (Feature *feature : point->obs_)
      {
        if (feature != nullptr && feature->ref_state_ == Feature::RefState::VALIDATED &&
            !feature->pending_delete_ && shadow_retired_ref_suggestions_.count(feature->ref_id_) == 0)
        {
          has_usable_reference = true;
          break;
        }
      }
    }
    if (point->state_ == VisualPoint::State::SUSPECT && !has_usable_reference)
      queuePointRetirement(*point);
  }
}

void VIOManager::recordManagedReferenceRejection(VisualPoint &point, Feature &reference,
                                                 int frame_id, int camera_id)
{
  if (!visual_map_manage_en || reference.pending_delete_ || point.pending_delete_) return;
  if (visual_map_manage_shadow_en &&
      (shadow_retired_point_suggestions_.count(point.point_id_) != 0 ||
       shadow_retired_ref_suggestions_.count(reference.ref_id_) != 0))
    return;
  if (reference.last_test_frame_id_ == frame_id && reference.last_test_camera_id_ == camera_id) return;
  reference.last_test_frame_id_ = frame_id;
  reference.last_test_camera_id_ = camera_id;
  ++reference.independent_test_count_;
  ++reference.rejected_test_count_;
  ++reference.consecutive_reject_count_;
  ++point.independent_test_count_;
  ++point.rejected_test_count_;
  if (point.state_ == VisualPoint::State::SEED) ++visual_map_manage_stats_.seed_tested;
  if (visual_ref_lifecycle_en && reference.ref_state_ == Feature::RefState::CANDIDATE &&
      reference.independent_test_count_ >= visual_ref_validate_min_tests &&
      reference.rejected_test_count_ >= visual_ref_retire_reject_count)
    queueReferenceRetirement(point, reference);
  else if (visual_ref_lifecycle_en && reference.ref_state_ == Feature::RefState::VALIDATED &&
           reference.consecutive_reject_count_ >= visual_ref_retire_reject_count)
    queueReferenceRetirement(point, reference);
  if (visual_ref_lifecycle_en && point.rejected_test_count_ >= visual_point_suspect_reject_count &&
      point.accepted_test_count_ * 2 < point.rejected_test_count_)
    point.state_ = VisualPoint::State::SUSPECT;
}

bool VIOManager::shouldCreateManagedReference(const PerCameraData &ctx,
                                              const VisualPoint &point,
                                              const V2D &current_px,
                                              const Matrix2d &current_affine) const
{
  (void)current_px;
  if (!visual_map_manage_en || !visual_ref_view_coverage_en) return true;
  Eigen::JacobiSVD<Matrix2d> svd(current_affine);
  const auto singular = svd.singularValues();
  if (!singular.array().isFinite().all() || singular[1] <= 1.0e-9) return false;
  if (singular[0] / singular[1] > visual_ref_max_anisotropy) return true;
  const V3D current_direction = (ctx.new_frame->pos() - point.pos_).normalized();
  const double threshold = visual_ref_coverage_angle_deg / kRadiansToDegrees;
  for (Feature *feature : point.obs_)
  {
    if (feature == nullptr || feature->pending_delete_ || feature->ref_state_ == Feature::RefState::RETIRED) continue;
    if (!cross_camera_reference_en && feature->camera_id_ != ctx.camera_id) continue;
    V3D direction = feature->view_direction_w_;
    if (!direction.array().isFinite().all() || direction.norm() <= 1.0e-9)
      direction = feature->pos() - point.pos_;
    if (!direction.array().isFinite().all() || direction.norm() <= 1.0e-9) continue;
    direction.normalize();
    const double angle = std::acos(std::clamp(current_direction.dot(direction), -1.0, 1.0));
    if (angle <= threshold) return false;
  }
  return true;
}

void VIOManager::manageReferenceBank(VisualPoint &point)
{
  if (!visual_map_manage_en || !visual_ref_view_coverage_en ||
      static_cast<int>(point.obs_.size()) <= visual_ref_max_count)
    return;
  std::vector<Feature *> refs;
  for (Feature *feature : point.obs_)
    if (feature != nullptr && !feature->pending_delete_ &&
        feature->ref_state_ == Feature::RefState::VALIDATED)
      refs.push_back(feature);
  if (static_cast<int>(refs.size()) <= visual_ref_max_count) return;

  const double threshold = visual_ref_coverage_angle_deg / kRadiansToDegrees;
  std::set<Feature *> kept;
  std::vector<V3D> samples = point.view_samples_;
  if (samples.empty())
    for (Feature *feature : refs) samples.push_back(feature->view_direction_w_);
  std::vector<uint8_t> covered(samples.size(), 0);
  while (static_cast<int>(kept.size()) < visual_ref_max_count)
  {
    Feature *best = nullptr;
    int best_gain = -1;
    int best_success = -1;
    for (Feature *candidate : refs)
    {
      if (kept.count(candidate) != 0) continue;
      int gain = 0;
      for (size_t i = 0; i < samples.size(); ++i)
      {
        if (covered[i]) continue;
        const double angle = std::acos(std::clamp(candidate->view_direction_w_.dot(
            samples[i]), -1.0, 1.0));
        if (angle <= threshold) ++gain;
      }
      if (gain > best_gain || (gain == best_gain && candidate->accepted_test_count_ > best_success))
      {
        best = candidate;
        best_gain = gain;
        best_success = candidate->accepted_test_count_;
      }
    }
    if (best == nullptr) break;
    kept.insert(best);
    for (size_t i = 0; i < samples.size(); ++i)
    {
      const double angle = std::acos(std::clamp(best->view_direction_w_.dot(
          samples[i]), -1.0, 1.0));
      if (angle <= threshold) covered[i] = 1;
    }
    if (std::all_of(covered.begin(), covered.end(), [](uint8_t value) { return value != 0; })) break;
  }
  for (Feature *feature : refs)
    if (kept.count(feature) == 0) queueReferenceRetirement(point, *feature);
}

void VIOManager::materializePendingNewPointObservations(
    PerCameraData &ctx, const cv::Mat &img,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (!visual_ref_post_ekf_build_en || ctx.pending_new_points.empty()) return;

  std::vector<PendingNewPointObservation> materialized;
  materialized.reserve(ctx.pending_new_points.size());
  int projection_reject = 0;
  int support_reject = 0;

  for (PendingNewPointObservation &pending : ctx.pending_new_points)
  {
    pending.patch.assign(patch_size_total, 0.0f);
    pending.img.release();
    pending.virtual_source_origin = cv::Point();
    pending.T_f_w = ctx.new_frame->T_f_w_;
    fillPendingObservationTiming(pending, ctx);
    pending.inv_expo_time = state->inv_expo_time[ctx.camera_id];
    pending.virtual_patch_valid = false;
    pending.T_v_w = pending.T_f_w;
    pending.R_v_from_c.setIdentity();
    pending.R_c_from_v.setIdentity();

    if (virtual_fisheye_patch_en)
    {
      V2D raw_px;
      if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pending.pt_var.point_w), 1, raw_px))
      {
        ++projection_reject;
        continue;
      }

      cv::Mat virtual_support_img;
      cv::Point virtual_source_origin;
      SE3d T_v_w;
      M3D R_v_from_c, R_c_from_v;
      if (!createVirtualFeaturePatch(ctx, img, pending.T_f_w, pending.pt_var.point_w, pending.patch.data(),
                                     virtual_support_img, virtual_source_origin, T_v_w,
                                     R_v_from_c, R_c_from_v))
      {
        ++support_reject;
        ++rejected_virtual_support_oob_;
        continue;
      }

      pending.px = raw_px;
      pending.bearing = ctx.cam->cam2world(raw_px);
      pending.img = virtual_support_img;
      pending.virtual_source_origin = virtual_source_origin;
      pending.T_v_w = T_v_w;
      pending.R_v_from_c = R_v_from_c;
      pending.R_c_from_v = R_c_from_v;
      pending.virtual_patch_valid = true;
      if (ref_patch_dump_en)
        maybeInitializeRefPatchDumpProbe(ctx, pending.pt_var.point_w, pending.pt_var.normal,
                                        pending.px, pending.bearing, pending.patch.data());
    }
    else
    {
      const V2D pc = ctx.new_frame->w2c(pending.pt_var.point_w);
      if (!pc.array().isFinite().all() || !ctx.cam->isInFrame(pc.cast<int>(), border))
      {
        ++projection_reject;
        continue;
      }

      pending.px = pc;
      pending.bearing = ctx.cam->cam2world(pc);
      getImagePatch(ctx, img, pc, pending.patch.data(), 0);
      pending.img = img;
    }

    if (visual_map_manage_en && visual_point_footprint_redundancy_en)
    {
      const VoxelPlane *plane = nullptr;
      pending.surface_valid = associateVisualPointSurface(
          pending.pt_var.point_w, plane_map, pending.surface_voxel_x,
          pending.surface_voxel_y, pending.surface_voxel_z, plane);
      if (pending.surface_valid && plane != nullptr)
      {
        pending.surface_plane_id = plane->id_;
        pending.surface_revision = plane->revision_;
        V3D surface_normal = plane->normal_.normalized();
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = pending.pt_var.point_w - plane->center_;
        J_nq.block<1, 3>(0, 3) = -surface_normal;
        const double sigma = (J_nq * plane->plane_var_ * J_nq.transpose())(0, 0) +
                             surface_normal.dot(pending.pt_var.var * surface_normal);
        const double distance = std::fabs(surface_normal.dot(pending.pt_var.point_w - plane->center_));
        pending.geometry_chi2 = sigma > visual_geom_filter_min_sigma
                                    ? distance * distance / sigma
                                    : std::numeric_limits<double>::infinity();
        pending.footprint_valid = computeManagedFootprint(
            ctx, pending.T_f_w, pending.pt_var.point_w, pending.pt_var.normal,
            nullptr, *plane, pending.footprint_corners_w);
      }
    }

    materialized.push_back(std::move(pending));
  }

  ctx.pending_new_points.swap(materialized);
  if (visual_geom_filter_log_en || visual_ref_post_ekf_build_en)
  {
    printf("[ VIO Ref Build ] camera_id=%d mode=%s post_ekf_materialized=%zu projection_reject=%d support_reject=%d\n",
           ctx.camera_id, virtual_fisheye_patch_en ? "virtual" : "raw",
           ctx.pending_new_points.size(), projection_reject, support_reject);
  }
}

void VIOManager::commitPendingNewPoints(
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  std::unordered_map<int, VisualPoint *> created_from_pg;
  std::vector<VisualPoint *> created_points;
  struct CreatedPendingSurfacePoint
  {
    VisualPoint *point = nullptr;
    int plane_id = -1;
    std::array<V3D, 4> footprint;
  };
  std::vector<CreatedPendingSurfacePoint> created_from_surface;
  for (PerCameraData &ctx : cameras_)
  {
    for (PendingNewPointObservation &pending : ctx.pending_new_points)
    {
      VisualPoint *point = nullptr;
      if (visual_map_manage_en && visual_point_footprint_redundancy_en && !pending.surface_valid)
      {
        const VoxelPlane *plane = nullptr;
        pending.surface_valid = associateVisualPointSurface(
            pending.pt_var.point_w, plane_map, pending.surface_voxel_x,
            pending.surface_voxel_y, pending.surface_voxel_z, plane);
        if (pending.surface_valid && plane != nullptr)
        {
          pending.surface_plane_id = plane->id_;
          pending.surface_revision = plane->revision_;
          V3D surface_normal = plane->normal_.normalized();
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = pending.pt_var.point_w - plane->center_;
          J_nq.block<1, 3>(0, 3) = -surface_normal;
          const double sigma = (J_nq * plane->plane_var_ * J_nq.transpose())(0, 0) +
                               surface_normal.dot(pending.pt_var.var * surface_normal);
          const double distance = std::fabs(surface_normal.dot(pending.pt_var.point_w - plane->center_));
          pending.geometry_chi2 = sigma > visual_geom_filter_min_sigma
                                      ? distance * distance / sigma
                                      : std::numeric_limits<double>::infinity();
          pending.footprint_valid = computeManagedFootprint(
              ctx, pending.T_f_w, pending.pt_var.point_w, pending.pt_var.normal,
              nullptr, *plane, pending.footprint_corners_w);
        }
      }
      VisualPoint *redundant_point = nullptr;
      double redundant_iou = 0.0;
      if (visual_map_manage_en && visual_point_footprint_redundancy_en)
      {
        redundant_point = findRedundantVisualPoint(pending, plane_map, redundant_iou);
        const bool admit_redundant = redundant_point != nullptr &&
                                      shouldReplaceRedundantPoint(pending, *redundant_point);
        if (redundant_point != nullptr && !admit_redundant)
        {
          ++visual_map_manage_stats_.redundant_rejected;
          if (!visual_map_manage_shadow_en) continue;
        }
        else if (admit_redundant && visual_point_information_prune_en)
        {
          ++visual_map_manage_stats_.information_admitted;
        }
      }
      if (visual_map_manage_en && !visual_map_manage_shadow_en &&
          pending.source_type == SOURCE_RAYCAST_PLANE &&
          pending.surface_valid && pending.footprint_valid)
      {
        const auto surface = plane_map.find(VOXEL_LOCATION(pending.surface_voxel_x,
                                                           pending.surface_voxel_y,
                                                           pending.surface_voxel_z));
        if (surface != plane_map.end() && surface->second != nullptr)
        {
          VoxelOctoTree *octo = surface->second->find_correspond(pending.pt_var.point_w);
          if (octo != nullptr && octo->plane_ptr_ != nullptr && octo->plane_ptr_->is_plane_)
          {
            for (const CreatedPendingSurfacePoint &created : created_from_surface)
            {
              if (created.point == nullptr || created.plane_id != pending.surface_plane_id) continue;
              const double iou = managedFootprintIoU(pending.footprint_corners_w,
                                                     created.footprint, *octo->plane_ptr_);
              if (iou >= visual_point_footprint_iou)
              {
                point = created.point;
                break;
              }
            }
          }
        }
      }
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
        if (visual_map_manage_en)
        {
          point->point_id_ = next_visual_point_id_++;
          point->state_ = visual_point_seed_validation_en
                              ? VisualPoint::State::SEED
                              : VisualPoint::State::CONFIRMED;
          point->surface_voxel_x_ = pending.surface_voxel_x;
          point->surface_voxel_y_ = pending.surface_voxel_y;
          point->surface_voxel_z_ = pending.surface_voxel_z;
          point->surface_plane_id_ = pending.surface_plane_id;
          point->surface_revision_ = pending.surface_revision;
          point->surface_valid_ = pending.surface_valid;
          point->geometry_chi2_ = pending.geometry_chi2;
          if (!visual_map_manage_shadow_en && visual_point_replacement_en &&
              redundant_point != nullptr &&
              shouldReplaceRedundantPoint(pending, *redundant_point))
          {
            point->challenger_of_ = redundant_point;
            ++visual_map_manage_stats_.replacement_suggested;
          }
        }
        created_points.push_back(point);
        if (pending.source_type == SOURCE_PG && pending.source_index >= 0)
          created_from_pg[pending.source_index] = point;
        if (pending.surface_valid && pending.footprint_valid)
          created_from_surface.push_back({point, pending.surface_plane_id,
                                          pending.footprint_corners_w});
      }

      float *patch = new float[pending.patch.size()];
      std::copy(pending.patch.begin(), pending.patch.end(), patch);
      Feature *feature = new Feature(point, patch, pending.px, pending.bearing, pending.T_f_w, pending.level,
                                     pending.camera_id, pending.capture_timestamp, pending.raw_timestamp,
                                     pending.corrected_timestamp, pending.td_used,
                                     pending.exposure_time_offset, pending.time_offset_group,
                                     pending.Rwi_ref, pending.Pwi_ref, pending.extrinsic_version);
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
      if (visual_map_manage_en)
      {
        initializeManagedReference(*feature, *point, ctx, true);
        feature->surface_plane_id_ = pending.surface_plane_id;
        feature->surface_revision_ = pending.surface_revision;
        feature->footprint_valid_ = pending.footprint_valid;
        feature->footprint_corners_w_ = pending.footprint_corners_w;
      }
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
  int level_gate_reject_num = 0;
  const SE3d pose_cur = ctx.new_frame->T_f_w_;
  for (int i = 0; i < ctx.total_points; ++i)
  {
    VisualPoint *pt = ctx.visual_submap->voxel_points[i];
    if (pt == nullptr || pt->pending_delete_) continue;
    if (pt->is_converged_ && (!visual_map_manage_en || visual_map_manage_shadow_en))
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
      refreshReferenceCalibration(*last_feature);
      const SE3d delta_pose = last_feature->T_f_w_ * pose_cur.inverse();
      const double delta_p = delta_pose.translation().norm();
      const double trace = delta_pose.rotationMatrix().trace();
      const double delta_theta = trace > 3.0 - 1e-6 ? 0.0 : std::acos(std::clamp(0.5 * (trace - 1.0), -1.0, 1.0));
      if (delta_p > 0.5 || delta_theta > 0.3 || (raw_px - last_feature->px_).norm() > 40.0) add_flag = true;
    }

    if (visual_map_manage_en && visual_ref_view_coverage_en && !visual_map_manage_shadow_en)
    {
      const Matrix2d affine = i < static_cast<int>(ctx.visual_submap->warp_affines.size())
                                  ? ctx.visual_submap->warp_affines[i]
                                  : Matrix2d::Identity();
      add_flag = shouldCreateManagedReference(ctx, *pt, raw_px, affine);
    }

    // Partial pyramid admission is useful for the current pose update, but it
    // must not add, replace, or evict persistent map observations.
    if (add_flag && ncc_en && !allPyramidLevelsActive(*ctx.visual_submap, i))
    {
      ++level_gate_reject_num;
      continue;
    }

    if ((!visual_map_manage_en || visual_map_manage_shadow_en) && pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(ctx.new_frame->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
    }
    if (!add_flag) continue;
    std::unique_ptr<float[]> patch(new float[patch_size_total]);
    cv::Mat virtual_support_img;
    cv::Point virtual_source_origin;
    SE3d T_v_w;
    M3D R_v_from_c, R_c_from_v;
    if (!createVirtualFeaturePatch(ctx, img, ctx.new_frame->T_f_w_, pt->pos_, patch.get(),
                                   virtual_support_img, virtual_source_origin, T_v_w, R_v_from_c, R_c_from_v))
    {
      ++rejected_virtual_support_oob_;
      continue;
    }

    const V3D bearing = ctx.cam->cam2world(raw_px);
    Feature *ftr_new = new Feature(pt, patch.release(), raw_px, bearing, ctx.new_frame->T_f_w_,
                                   ctx.visual_submap->search_levels[i], ctx.camera_id,
                                   ctx.new_frame->capture_timestamp_, ctx.new_frame->raw_timestamp_,
                                   ctx.new_frame->corrected_timestamp_, ctx.new_frame->td_used_,
                                   ctx.new_frame->exposure_time_offset_, ctx.new_frame->time_offset_group_,
                                   ctx.Rwi, ctx.Pwi, extrinsic_version_);
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
    initializeManagedReference(*ftr_new, *pt, ctx, false);
    pt->addFrameRef(ftr_new);
    manageReferenceBank(*pt);
    ctx.update_flag[i] = 1;
    ++update_num;
  }
  printf("[ VIO Virtual ] Update %d points in visual submap (all-level map gate rejected %d)\n",
         update_num, level_gate_reject_num);
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
  int level_gate_reject_num = 0;
  SE3 pose_cur = ctx.new_frame->T_f_w_;
  for (int i = 0; i < ctx.total_points; i++)
  {
    VisualPoint *pt = ctx.visual_submap->voxel_points[i];
    if (pt == nullptr || pt->pending_delete_) continue;
    if (pt->is_converged_ && (!visual_map_manage_en || visual_map_manage_shadow_en))
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(ctx.new_frame->w2c(pt->pos_));
    bool add_flag = false;
    
    float *patch_temp = nullptr;
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
      refreshReferenceCalibration(*last_feature);
      SE3 pose_ref = last_feature->T_f_w_;
      SE3 delta_pose = pose_ref * pose_cur.inverse();
      double delta_p = delta_pose.translation().norm();
      double delta_theta = (delta_pose.rotationMatrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotationMatrix().trace() - 1));
      if (delta_p > 0.5 || delta_theta > 0.3 || (pc - last_feature->px_).norm() > 40) add_flag = true;
    }

    if (visual_map_manage_en && visual_ref_view_coverage_en && !visual_map_manage_shadow_en)
    {
      const Matrix2d affine = i < static_cast<int>(ctx.visual_submap->warp_affines.size())
                                  ? ctx.visual_submap->warp_affines[i]
                                  : Matrix2d::Identity();
      add_flag = shouldCreateManagedReference(ctx, *pt, pc, affine);
    }

    if (add_flag && ncc_en && !allPyramidLevelsActive(*ctx.visual_submap, i))
    {
      ++level_gate_reject_num;
      continue;
    }

    // Maintain the size of 3D point observation features.
    if ((!visual_map_manage_en || visual_map_manage_shadow_en) && pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(ctx.new_frame->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      patch_temp = new float[patch_size_total];
      getImagePatch(ctx, img, pc, patch_temp, 0);
      update_num += 1;
      ctx.update_flag[i] = 1;
      Vector3d f = ctx.cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, ctx.new_frame->T_f_w_,
                                     ctx.visual_submap->search_levels[i], ctx.camera_id,
                                     ctx.new_frame->capture_timestamp_, ctx.new_frame->raw_timestamp_,
                                     ctx.new_frame->corrected_timestamp_, ctx.new_frame->td_used_,
                                     ctx.new_frame->exposure_time_offset_, ctx.new_frame->time_offset_group_,
                                     ctx.Rwi, ctx.Pwi, extrinsic_version_);
      ftr_new->img_ = img;
      ftr_new->id_ = ctx.new_frame->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time[ctx.camera_id];
      initializeManagedReference(*ftr_new, *pt, ctx, false);
      pt->addFrameRef(ftr_new);
      manageReferenceBank(*pt);
    }
    else
    {
      delete[] patch_temp;
    }
  }
  // Keep the raw path quiet by default, but retain the counters here for
  // symmetric debugging with the virtual path.
  (void)level_gate_reject_num;
}

void VIOManager::updateReferencePatch(PerCameraData &ctx, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (ctx.total_points == 0) return;

  for (int i = 0; i < static_cast<int>(ctx.visual_submap->voxel_points.size()); i++)
  {
    VisualPoint *pt = ctx.visual_submap->voxel_points[i];

    if (pt == nullptr || pt->pending_delete_ || !pt->is_normal_initialized_) continue;
    if (visual_map_manage_en)
    {
      if (pt->point_id_ == 0) pt->point_id_ = next_visual_point_id_++;
      pt->map_voxel_x_ = static_cast<int64_t>(std::floor(pt->pos_[0] / 0.5));
      pt->map_voxel_y_ = static_cast<int64_t>(std::floor(pt->pos_[1] / 0.5));
      pt->map_voxel_z_ = static_cast<int64_t>(std::floor(pt->pos_[2] / 0.5));
      for (Feature *feature : pt->obs_)
      {
        if (feature == nullptr || feature->pending_delete_) continue;
        refreshReferenceCalibration(*feature);
        if (feature->ref_id_ == 0)
        {
          feature->ref_id_ = next_visual_ref_id_++;
          feature->birth_frame_id_ = feature->id_;
          feature->ref_state_ = Feature::RefState::VALIDATED;
        }
        if (!feature->view_direction_w_.array().isFinite().all() ||
            feature->view_direction_w_.norm() <= 1.0e-9)
        {
          const V3D view = feature->pos() - pt->pos_;
          feature->view_range_ = view.norm();
          if (feature->view_range_ > 1.0e-9)
            feature->view_direction_w_ = view / feature->view_range_;
          else
            feature->view_direction_w_.setZero();
        }
      }
      const VoxelPlane *plane = nullptr;
      int64_t voxel_x = 0, voxel_y = 0, voxel_z = 0;
      const bool associated = visual_point_footprint_redundancy_en &&
                              associateVisualPointSurface(pt->pos_, plane_map,
                                                          voxel_x, voxel_y, voxel_z, plane);
      pt->surface_valid_ = associated;
      if (associated && plane != nullptr)
      {
        pt->surface_voxel_x_ = voxel_x;
        pt->surface_voxel_y_ = voxel_y;
        pt->surface_voxel_z_ = voxel_z;
        pt->surface_plane_id_ = plane->id_;
        pt->surface_revision_ = plane->revision_;
        V3D normal = plane->normal_.normalized();
        if (pt->normal_.dot(normal) < 0.0) normal = -normal;
        if (!visual_map_manage_shadow_en) pt->normal_ = normal;
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = pt->pos_ - plane->center_;
        J_nq.block<1, 3>(0, 3) = -normal;
        const double sigma = (J_nq * plane->plane_var_ * J_nq.transpose())(0, 0) +
                             normal.dot(pt->covariance_ * normal);
        const double distance = std::fabs(normal.dot(pt->pos_ - plane->center_));
        pt->geometry_chi2_ = sigma > visual_geom_filter_min_sigma
                                 ? distance * distance / sigma
                                 : std::numeric_limits<double>::infinity();
        if (visual_point_replacement_en &&
            std::isfinite(visual_geom_filter_max_chi2) && visual_geom_filter_max_chi2 > 0.0 &&
            pt->geometry_chi2_ > visual_geom_filter_max_chi2 && !visual_map_manage_shadow_en)
          pt->state_ = VisualPoint::State::SUSPECT;

        for (Feature *feature : pt->obs_)
        {
          if (feature == nullptr || feature->pending_delete_) continue;
          if (feature->surface_plane_id_ == plane->id_ &&
              feature->surface_revision_ == plane->revision_ && feature->footprint_valid_)
            continue;
          if (feature->camera_id_ < 0 || feature->camera_id_ >= numCameras()) continue;
          refreshReferenceCalibration(*feature);
          const PerCameraData &ref_ctx = cameras_[feature->camera_id_];
          feature->footprint_valid_ = computeManagedFootprint(
              ref_ctx, feature->T_f_w_, pt->pos_, normal, feature,
              *plane, feature->footprint_corners_w_);
          feature->surface_plane_id_ = plane->id_;
          feature->surface_revision_ = plane->revision_;
        }
      }
      manageReferenceBank(*pt);
    }
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
      refreshReferenceCalibration(*ref_patch_temp);
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
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time[0] * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time[0] * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time[0]);
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
        // Reuse the stored reference support instead of regenerating it every frame.
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
  if (state == nullptr || state_propagat == nullptr || state->stateDim() != kLegacyFixedStateDim)
  {
    printf("[ VIO Legacy ] updateStateInverseVirtual skipped: fixed legacy state dim=%d, current state dim=%d\n",
           kLegacyFixedStateDim, state != nullptr ? state->stateDim() : -1);
    return;
  }
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
      const MD(kLegacyFixedStateDim, kLegacyFixedStateDim) K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      const auto HTz = H_sub_T * z;
      const MD(kLegacyFixedStateDim, 1) vec = *state_propagat - *state;
      G.block<kLegacyFixedStateDim, 6>(0, 0) = K_1.block<kLegacyFixedStateDim, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      const MD(kLegacyFixedStateDim, 1) solution =
          -K_1.block<kLegacyFixedStateDim, 6>(0, 0) * HTz + vec - G.block<kLegacyFixedStateDim, 6>(0, 0) * vec.block<6, 1>(0, 0);
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
  if (state == nullptr || state_propagat == nullptr || state->stateDim() != kLegacyFixedStateDim)
  {
    printf("[ VIO Legacy ] updateStateInverse skipped: fixed legacy state dim=%d, current state dim=%d\n",
           kLegacyFixedStateDim, state != nullptr ? state->stateDim() : -1);
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
      MD(kLegacyFixedStateDim, kLegacyFixedStateDim) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<kLegacyFixedStateDim, 6>(0, 0) = K_1.block<kLegacyFixedStateDim, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<kLegacyFixedStateDim, 6>(0, 0) * HTz + vec - G.block<kLegacyFixedStateDim, 6>(0, 0) * vec.block<6, 1>(0, 0);
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
  if (state == nullptr || state_propagat == nullptr || state->stateDim() != kLegacyFixedStateDim)
  {
    printf("[ VIO Legacy ] updateStateVirtual skipped: fixed legacy state dim=%d, current state dim=%d\n",
           kLegacyFixedStateDim, state != nullptr ? state->stateDim() : -1);
    return;
  }
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
        Jimg *= state->inv_expo_time[0] * inv_scale;
        const MD(1, 3) Jimg_Jpi_R = Jimg * Jdpi * track.R_vcur_from_ccur_seed;
        const MD(1, 3) Jdphi = Jimg_Jpi_R * point_c_hat;
        const MD(1, 3) Jdp = -Jimg_Jpi_R;
        const MD(1, 3) JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
        const MD(1, 3) Jdt = Jdp * Jdp_dt;
        const double cur_value = current_values[patch_index];
        const double residual =
            state->inv_expo_time[0] * cur_value - inv_ref_expo * reference[patch_size_total * level + patch_index];
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
      const MD(kLegacyFixedStateDim, kLegacyFixedStateDim) K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      const auto HTz = H_sub_T * z;
      const MD(kLegacyFixedStateDim, 1) vec = *state_propagat - *state;
      G.block<kLegacyFixedStateDim, 7>(0, 0) = K_1.block<kLegacyFixedStateDim, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      const MD(kLegacyFixedStateDim, 1) solution =
          -K_1.block<kLegacyFixedStateDim, 7>(0, 0) * HTz + vec - G.block<kLegacyFixedStateDim, 7>(0, 0) * vec.block<7, 1>(0, 0);
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
  if (state == nullptr || state_propagat == nullptr || state->stateDim() != kLegacyFixedStateDim)
  {
    printf("[ VIO Legacy ] updateState skipped: fixed legacy state dim=%d, current state dim=%d\n",
           kLegacyFixedStateDim, state != nullptr ? state->stateDim() : -1);
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
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time[0]);

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
          Jimg = Jimg * state->inv_expo_time[0];
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time[0] * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

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
      MD(kLegacyFixedStateDim, kLegacyFixedStateDim) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.leftCols(6) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<kLegacyFixedStateDim, 7>(0, 0) = K_1.block<kLegacyFixedStateDim, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(kLegacyFixedStateDim, 1)
      solution = -K_1.block<kLegacyFixedStateDim, 7>(0, 0) * HTz + vec - G.block<kLegacyFixedStateDim, 7>(0, 0) * vec.block<7, 1>(0, 0);

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
  // if (state->inv_expo_time[0] < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time[0] = 0.0;}
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

  double frame_time = ctx.new_frame != nullptr ? ctx.new_frame->capture_timestamp_ : 0.0;
  double time_offset_delta = 0.0;
  int group_id = ctx.time_offset_group;
  if (ctx.new_frame != nullptr)
  {
    group_id = ctx.new_frame->time_offset_group_;
    double td_state = ctx.new_frame->td_used_;
    if (group_id >= 0 && group_id < state_value.num_time_offset_groups)
      td_state = state_value.time_offset[group_id];
    time_offset_delta = td_state - ctx.new_frame->td_used_;
    if (!std::isfinite(time_offset_delta)) time_offset_delta = 0.0;
    ctx.new_frame->time_offset_delta_ = time_offset_delta;
  }

  const V3D Vwi(state_value.vel_end);
  V3D gyro_i(current_unbiased_gyr_);
  if (!gyro_i.allFinite()) gyro_i.setZero();
  M3D Rwi = state_value.rot_end * Exp(gyro_i, time_offset_delta);
  V3D Pwi = state_value.pos_end + Vwi * time_offset_delta;
  ctx.Rwi = Rwi;
  ctx.Pwi = Pwi;
  ctx.Vwi = Vwi;
  ctx.gyro_i = gyro_i;
  ctx.frame_time = frame_time;
  ctx.time_offset_group = group_id;
  ctx.frame_time_offset_delta = time_offset_delta;
  ctx.Rcw = ctx.Rci * Rwi.transpose();
  ctx.Pcw = -ctx.Rci * Rwi.transpose() * Pwi + ctx.Pci;
  ctx.Jdp_dt = ctx.Rci * Rwi.transpose();
  ctx.dpc_dvel = -ctx.Rci * Rwi.transpose() * time_offset_delta;
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
            << std::fixed << std::setprecision(6)
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "
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

  ctx.new_frame.reset(new Frame(ctx.cam, cur_img, -1, 0, img_time, img_time, img_time));
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

    const double raw_timestamp = camera_id < static_cast<int>(mf.raw_timestamps.size())
                                     ? mf.raw_timestamps[camera_id]
                                     : mf.timestamp;
    const double corrected_timestamp = camera_id < static_cast<int>(mf.corrected_timestamps.size())
                                           ? mf.corrected_timestamps[camera_id]
                                           : mf.timestamp;
    const double capture_timestamp = camera_id < static_cast<int>(mf.capture_timestamps.size())
                                         ? mf.capture_timestamps[camera_id]
                                         : mf.timestamp;
    const double td_used = camera_id < static_cast<int>(mf.td_used.size()) ? mf.td_used[camera_id] : 0.0;
    const int time_group = camera_id < static_cast<int>(mf.time_offset_group.size())
                               ? mf.time_offset_group[camera_id]
                               : (camera_id < static_cast<int>(camera_time_offset_groups.size())
                                      ? camera_time_offset_groups[camera_id]
                                      : 0);
    ctx.new_frame.reset(new Frame(ctx.cam, image, mf.frame_id, camera_id, capture_timestamp,
                                  raw_timestamp, corrected_timestamp, td_used,
                                  capture_timestamp - corrected_timestamp, time_group));
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
  // printf("[ VIO Debug ] processMultiCameraFrame begin frame=%d cameras=%d pg=%zu feat_map=%zu plane_map=%zu virtual=%d cross_ref=%d normal=%d inverse=%d raycast=%d\n",
  //        mf.frame_id, numCameras(), pg.size(), feat_map.size(), plane_map.size(), virtual_fisheye_patch_en ? 1 : 0,
  //        cross_camera_reference_en ? 1 : 0, normal_en ? 1 : 0, inverse_composition_en ? 1 : 0, raycast_en ? 1 : 0);
  // fflush(stdout);
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
    const double raw_timestamp = camera_id < static_cast<int>(mf.raw_timestamps.size())
                                     ? mf.raw_timestamps[camera_id]
                                     : mf.timestamp;
    const double corrected_timestamp = camera_id < static_cast<int>(mf.corrected_timestamps.size())
                                           ? mf.corrected_timestamps[camera_id]
                                           : mf.timestamp;
    const double capture_timestamp = camera_id < static_cast<int>(mf.capture_timestamps.size())
                                         ? mf.capture_timestamps[camera_id]
                                         : mf.timestamp;
    const double td_used = camera_id < static_cast<int>(mf.td_used.size()) ? mf.td_used[camera_id] : 0.0;
    const int time_group = camera_id < static_cast<int>(mf.time_offset_group.size())
                               ? mf.time_offset_group[camera_id]
                               : (camera_id < static_cast<int>(camera_time_offset_groups.size())
                                      ? camera_time_offset_groups[camera_id]
                                      : 0);
    ctx.new_frame.reset(new Frame(ctx.cam, image, mf.frame_id, camera_id, capture_timestamp,
                                  raw_timestamp, corrected_timestamp, td_used,
                                  capture_timestamp - corrected_timestamp, time_group));
    updateFrameState(ctx, *state);
    resetGrid(ctx);
    // printf("[ VIO Debug ] frame setup camera_id=%d frame=%d image=%dx%d type=%d ns=%s\n",
    //        ctx.camera_id, mf.frame_id, image.cols, image.rows, image.type(), ctx.camera_namespace.c_str());
    // fflush(stdout);
  }
  flushVisualMapRetirements();
  const double frame_setup_end = omp_get_wtime();

  int raw_score_remaining_total = max_total_points;
  for (PerCameraData &ctx : cameras_)
  {
    int raw_score_point_quota = -1;
    if (virtual_raw_score_select_en)
    {
      const int camera_id = std::max(0, ctx.camera_id);
      const int cameras_left = std::max(0, numCameras() - camera_id - 1);
      const int min_per_camera = std::max(0, points_per_camera_min);
      const int max_per_camera = std::max(min_per_camera, points_per_camera_max);
      const int reserved_for_later = std::min(raw_score_remaining_total, min_per_camera * cameras_left);
      raw_score_point_quota = std::max(0, raw_score_remaining_total - reserved_for_later);
      raw_score_point_quota = std::min(raw_score_point_quota, max_per_camera);
    }
    // printf("[ VIO Debug ] retrieve begin camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
    retrieveFromVisualSparseMap(ctx, ctx.new_frame->img_, pg, plane_map, raw_score_point_quota);
    if (virtual_raw_score_select_en)
      raw_score_remaining_total = std::max(0, raw_score_remaining_total - ctx.total_points);
    // printf("[ VIO Debug ] retrieve end camera_id=%d frame=%d total_points=%d\n",
    //        ctx.camera_id, mf.frame_id, ctx.total_points);
    // fflush(stdout);
    // printf("[ VIO Debug ] generate begin camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
    generateVisualMapPoints(ctx, ctx.new_frame->img_, pg, plane_map);
    // printf("[ VIO Debug ] generate end camera_id=%d frame=%d pending=%zu\n",
    //        ctx.camera_id, mf.frame_id, ctx.pending_new_points.size());
    // fflush(stdout);
  }
  const double retrieve_end = omp_get_wtime();

  // printf("[ VIO Debug ] ekf begin frame=%d\n", mf.frame_id);
  // fflush(stdout);
  computeJacobianAndUpdateEKF();
  // printf("[ VIO Debug ] ekf end frame=%d\n", mf.frame_id);
  // fflush(stdout);
  const double ekf_end = omp_get_wtime();

  if (visual_map_manage_en)
    for (PerCameraData &ctx : cameras_) updateManagedObservationEvidence(ctx);

  if (visual_ref_post_ekf_build_en)
  {
    for (PerCameraData &ctx : cameras_)
      materializePendingNewPointObservations(ctx, ctx.new_frame->img_, plane_map);
  }

  if (ref_patch_dump_en && !cameras_.empty())
    processRefPatchDumpProbe(cameras_[kRefPatchDumpCameraId], cameras_[kRefPatchDumpCameraId].new_frame->img_);

  for (PerCameraData &ctx : cameras_)
  {
    if (!visual_ref_post_ekf_build_en)
    {
      for (PendingNewPointObservation &pending : ctx.pending_new_points)
      {
        pending.T_f_w = ctx.new_frame->T_f_w_;
        fillPendingObservationTiming(pending, ctx);
        if (pending.virtual_patch_valid) pending.T_v_w = composeVirtualPose(pending.R_v_from_c, pending.T_f_w);
      }
    }
    // printf("[ VIO Debug ] updateVisualMap begin camera_id=%d frame=%d pending=%zu\n",
    //        ctx.camera_id, mf.frame_id, ctx.pending_new_points.size());
    // fflush(stdout);
    updateVisualMapPoints(ctx, ctx.new_frame->img_);
    // printf("[ VIO Debug ] updateVisualMap end camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
  }
  const double map_update_end = omp_get_wtime();
  // printf("[ VIO Debug ] commitPendingNewPoints begin frame=%d\n", mf.frame_id);
  // fflush(stdout);
  commitPendingNewPoints(plane_map);
  // printf("[ VIO Debug ] commitPendingNewPoints end frame=%d feat_map=%zu\n", mf.frame_id, feat_map.size());
  // fflush(stdout);
  const double commit_end = omp_get_wtime();
  for (PerCameraData &ctx : cameras_)
  {
    // printf("[ VIO Debug ] updateReference begin camera_id=%d frame=%d total_points=%d\n",
    //        ctx.camera_id, mf.frame_id, ctx.total_points);
    // fflush(stdout);
    updateReferencePatch(ctx, plane_map);
    // printf("[ VIO Debug ] updateReference end camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
    plotTrackedPoints(ctx);
    if (plot_flag) projectPatchFromRefToCur(ctx, plane_map);
  }
  if (visual_map_manage_en && mf.frame_id % 20 == 0)
  {
    for (auto &voxel : feat_map)
    {
      if (voxel.second == nullptr) continue;
      for (VisualPoint *point : voxel.second->voxel_points)
      {
        if (point == nullptr || point->pending_delete_) continue;
        if (point->state_ == VisualPoint::State::SUSPECT &&
            point->last_success_frame_ >= 0 &&
            mf.frame_id - point->last_success_frame_ >= visual_point_stale_frames)
        {
          bool has_usable_reference = point->hasUsableReference(0, true, false);
          if (visual_map_manage_shadow_en && has_usable_reference)
          {
            has_usable_reference = false;
            for (Feature *feature : point->obs_)
            {
              if (feature != nullptr && feature->ref_state_ == Feature::RefState::VALIDATED &&
                  !feature->pending_delete_ && shadow_retired_ref_suggestions_.count(feature->ref_id_) == 0)
              {
                has_usable_reference = true;
                break;
              }
            }
          }
          if (!has_usable_reference) queuePointRetirement(*point);
        }
      }
    }
  }
  printVisualMapManageStats(mf.frame_id);
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
  printf("\033[1;32m| %-29s | %-27lld |\033[0m\n", "Linearized Residuals", vio_linearized_residual_count_);
  const double avg_time_per_residual_us =
      vio_linearized_residual_count_ > 0
          ? compute_jacobian_time * 1.0e6 / static_cast<double>(vio_linearized_residual_count_)
          : 0.0;
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Avg Time / Residual (us)", avg_time_per_residual_us);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Update Visual Map", map_update_end - ekf_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Commit New Points", commit_end - map_update_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Reference + Debug", reference_end - commit_end);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", elapsed);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printOnlineCalibrationStatsTable(mf.frame_id);
  maybePrintUsageStatsTable(mf.frame_id);
}
