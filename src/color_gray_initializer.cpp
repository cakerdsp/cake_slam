#include "color_gray_initializer.h"
#include "fixed_gray_weights.h"
#include "fixed_gray_image.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <numeric>
#include <utility>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#ifndef CAKE_HAVE_OPENCV_SIFT
#define CAKE_HAVE_OPENCV_SIFT 0
#endif

namespace color_gray_init
{
namespace
{
constexpr int kMaxPlaneCandidates = 24;
constexpr int kMaxRegionsPerPair = 12;
constexpr int kMaxFullCropsPerCamera = 12;
constexpr int kMaxCrop = 320;
constexpr int kMinCrop = 48;
constexpr int kMaxSamples = 4096;
constexpr int kGridSize = 64;
constexpr double kEps = 1.0e-9;

#if CAKE_HAVE_OPENCV_SIFT
using V2 = Eigen::Vector2d;
using V3 = Eigen::Vector3d;
using M2 = Eigen::Matrix2d;
using M3 = Eigen::Matrix3d;

bool cancelled(const std::atomic<bool> &cancel)
{
  return cancel.load(std::memory_order_relaxed);
}

bool dehom(const M3 &h, const V2 &p, V2 &out)
{
  const V3 v = h * V3(p.x(), p.y(), 1.0);
  if (!v.allFinite() || std::abs(v.z()) < kEps) return false;
  out = v.head<2>() / v.z();
  return out.allFinite();
}

struct Chart
{
  const CameraSnapshot *camera = nullptr;
  M3 R_v_c = M3::Identity();
  M3 R_v_w = M3::Identity();
  V3 t_v_w = V3::Zero();
  double focal = 0.0;

  bool toChart(const V2 &raw, V2 &q) const
  {
    const V3 ray = R_v_c * camera->unproject(raw);
    if (!ray.allFinite() || ray.z() <= kEps) return false;
    q = focal * ray.head<2>() / ray.z();
    return q.allFinite();
  }

  bool toRaw(const V2 &q, V2 &raw) const
  {
    const V3 ray = R_v_c.transpose() * V3(q.x() / focal, q.y() / focal, 1.0);
    if (!camera->project(ray, raw) || !raw.allFinite()) return false;
    return raw.x() >= 0.0 && raw.y() >= 0.0 &&
           raw.x() < camera->bgr.cols - 1.0 && raw.y() < camera->bgr.rows - 1.0;
  }
};

bool makeChart(const CameraSnapshot &camera, const PlaneSeed &plane, Chart &chart)
{
  const V3 p = camera.R_c_w * plane.point_w + camera.t_c_w;
  if (!p.allFinite() || p.norm() <= kEps) return false;
  V2 raw;
  if (!camera.project(p, raw) || !raw.allFinite()) return false;
  const V3 z = p.normalized();
  const V3 back = camera.unproject(raw);
  if (!back.allFinite() || back.norm() <= kEps || back.normalized().dot(z) < 0.999) return false;
  V3 x = V3::UnitX() - z.x() * z;
  if (x.norm() < 0.1) x = V3::UnitY() - z.y() * z;
  x.normalize();
  const V3 y = z.cross(x).normalized();
  chart.camera = &camera;
  chart.R_v_c.row(0) = x.transpose();
  chart.R_v_c.row(1) = y.transpose();
  chart.R_v_c.row(2) = z.transpose();
  chart.R_v_w = chart.R_v_c * camera.R_c_w;
  chart.t_v_w = chart.R_v_c * camera.t_c_w;
  chart.focal = camera.virtual_focal;
  return true;
}

// Restrict sampling to the estimated candidate support. Sparse LiDAR plane
// statistics are not interpreted as proof of absence of an occluder.
bool onPlaneSupport(const Chart &chart, const PlaneSeed &plane, const V2 &q)
{
  const V3 origin = -chart.camera->R_c_w.transpose() * chart.camera->t_c_w;
  const V3 ray_w = chart.R_v_w.transpose() * V3(q.x() / chart.focal, q.y() / chart.focal, 1.0);
  const double denominator = plane.normal_w.dot(ray_w);
  if (!std::isfinite(denominator) || std::abs(denominator) < kEps) return false;
  const double distance = plane.normal_w.dot(plane.point_w - origin) / denominator;
  if (!std::isfinite(distance) || distance <= 0.0) return false;
  const V3 point = origin + distance * ray_w;
  return point.allFinite() && (point - plane.point_w).norm() <= plane.radius;
}

struct CropFeatures
{
  cv::Rect roi;
  std::vector<cv::KeyPoint> keys;
  cv::Mat descriptors;
  cv::Mat support_mask;
  Chart chart;
};

// This inexpensive stage runs for both cameras before either spends a SIFT
// extraction budget. Invisible early seeds do not hide later shared seeds.
bool prepareRegion(const CameraSnapshot &camera, const PlaneSeed &plane, CropFeatures &out)
{
  if (!makeChart(camera, plane, out.chart)) return false;
  const V3 n = plane.normal_w.normalized();
  const V3 u = n.unitOrthogonal();
  const V3 v = n.cross(u);
  std::vector<cv::Point2f> boundary;
  V2 center;
  if (!camera.project(camera.R_c_w * plane.point_w + camera.t_c_w, center) ||
      !center.allFinite() || center.x() < 0.0 || center.y() < 0.0 ||
      center.x() >= camera.bgr.cols || center.y() >= camera.bgr.rows) return false;
  for (int i = 0; i < 16; ++i)
  {
    const double angle = i * (2.0 * 3.14159265358979323846 / 16.0);
    const V3 p = plane.point_w + plane.radius * (std::cos(angle) * u + std::sin(angle) * v);
    V2 raw;
    if (!camera.project(camera.R_c_w * p + camera.t_c_w, raw) || !raw.allFinite()) return false;
    // Guard conversion to OpenCV integer rectangles for malformed models.
    if (raw.cwiseAbs().maxCoeff() > 1.0e6) return false;
    boundary.emplace_back(static_cast<float>(raw.x()), static_cast<float>(raw.y()));
  }
  cv::Rect roi = cv::boundingRect(boundary) & cv::Rect(0, 0, camera.bgr.cols, camera.bgr.rows);
  if (roi.width < kMinCrop || roi.height < kMinCrop || !center.allFinite()) return false;
  const int width = std::min(kMaxCrop, roi.width), height = std::min(kMaxCrop, roi.height);
  const int x = std::clamp(static_cast<int>(std::floor(center.x() - width * 0.5)), roi.x, roi.x + roi.width - width);
  const int y = std::clamp(static_cast<int>(std::floor(center.y() - height * 0.5)), roi.y, roi.y + roi.height - height);
  out.roi = cv::Rect(x, y, width, height);
  return true;
}

bool extractFeatures(const CameraSnapshot &camera, const PlaneSeed &plane,
                     const cv::Ptr<cv::SIFT> &sift, CropFeatures &out,
                     const std::atomic<bool> &cancel)
{
  const int width = out.roi.width, height = out.roi.height;
  const int x = out.roi.x, y = out.roi.y;
  cv::Mat gray, mask(height, width, CV_8UC1, cv::Scalar(0));
  cv::cvtColor(camera.bgr(out.roi), gray, cv::COLOR_BGR2GRAY);
  // The mask is geometry only; using default grayscale for descriptors does
  // not assume that the final cross-camera color weights are known.
  for (int row = 0; row < height; ++row)
  {
    if (cancelled(cancel)) return false;
    for (int col = 0; col < width; ++col)
    {
      V2 q;
      if (out.chart.toChart(V2(x + col, y + row), q) && onPlaneSupport(out.chart, plane, q))
        mask.at<unsigned char>(row, col) = 255;
    }
  }
  out.support_mask = mask.clone();
  cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 4);
  if (cancelled(cancel)) return false;
  sift->detectAndCompute(gray, mask, out.keys, out.descriptors);
  if (cancelled(cancel)) return false;
  for (auto &key : out.keys) key.pt += cv::Point2f(static_cast<float>(x), static_cast<float>(y));
  return out.keys.size() >= 16 && !out.descriptors.empty();
}

struct CachedFeatures
{
  CropFeatures data;
  bool prepared = false;
  bool region_valid = false;
  bool extracted = false;
  bool features_valid = false;
};

struct FeatureCache
{
  const Snapshot &snapshot;
  const std::atomic<bool> &cancel;
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create(500);
  std::vector<std::vector<CachedFeatures>> entries;
  std::vector<int> extracted_pixels;

  FeatureCache(const Snapshot &input, const std::atomic<bool> &stop)
      : snapshot(input), cancel(stop),
        entries(input.cameras.size(), std::vector<CachedFeatures>(
            std::min(kMaxPlaneCandidates, static_cast<int>(input.planes.size())))),
        extracted_pixels(input.cameras.size(), 0) {}

  CachedFeatures &prepare(int camera, int plane)
  {
    CachedFeatures &entry = entries[camera][plane];
    if (!entry.prepared)
    {
      entry.prepared = true;
      entry.region_valid = prepareRegion(snapshot.cameras[camera], snapshot.planes[plane], entry.data);
    }
    return entry;
  }

  bool extract(int camera, int plane)
  {
    CachedFeatures &entry = prepare(camera, plane);
    if (entry.extracted) return entry.features_valid;
    entry.extracted = true;
    const int area = entry.data.roi.area();
    if (!entry.region_valid || cancelled(cancel) ||
        extracted_pixels[camera] + area > kMaxFullCropsPerCamera * kMaxCrop * kMaxCrop) return false;
    extracted_pixels[camera] += area;
    entry.features_valid = extractFeatures(snapshot.cameras[camera], snapshot.planes[plane], sift, entry.data, cancel);
    if (!entry.features_valid) entry.data.support_mask.release();
    return entry.features_valid;
  }
};

struct Correspondence
{
  V2 source, target;
  V2 source_raw, target_raw;
  float descriptor_distance = 0.0f;
};

std::vector<Correspondence> match(const CropFeatures &source, const CropFeatures &target)
{
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<std::vector<cv::DMatch>> forward, reverse;
  matcher.knnMatch(source.descriptors, target.descriptors, forward, 2);
  matcher.knnMatch(target.descriptors, source.descriptors, reverse, 2);
  std::vector<Correspondence> pairs;
  for (const auto &two : forward)
  {
    if (two.size() < 2 || !(two[0].distance < 0.75f * two[1].distance)) continue;
    const cv::DMatch &m = two[0];
    if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(reverse.size())) continue;
    const auto &back = reverse[m.trainIdx];
    if (back.size() < 2 || back[0].trainIdx != m.queryIdx || !(back[0].distance < 0.75f * back[1].distance)) continue;
    Correspondence p;
    p.descriptor_distance = m.distance;
    p.source_raw = V2(source.keys[m.queryIdx].pt.x, source.keys[m.queryIdx].pt.y);
    p.target_raw = V2(target.keys[m.trainIdx].pt.x, target.keys[m.trainIdx].pt.y);
    if (source.chart.toChart(p.source_raw, p.source) && target.chart.toChart(p.target_raw, p.target)) pairs.push_back(p);
  }
  // SIFT may report several orientations/octaves at one location. Keep the
  // strongest descriptor match per spatial neighborhood in BOTH images so
  // copies of one point cannot inflate support or leak into the holdout.
  std::stable_sort(pairs.begin(), pairs.end(), [](const Correspondence &a, const Correspondence &b) {
    return a.descriptor_distance < b.descriptor_distance;
  });
  std::vector<Correspondence> unique;
  unique.reserve(pairs.size());
  for (const Correspondence &p : pairs)
  {
    const bool duplicate = std::any_of(unique.begin(), unique.end(), [&p](const Correspondence &kept) {
      return (p.source_raw - kept.source_raw).squaredNorm() < 4.0 ||
             (p.target_raw - kept.target_raw).squaredNorm() < 4.0;
    });
    if (!duplicate) unique.push_back(p);
  }
  pairs.swap(unique);
  // Deterministic spatial ordering makes the holdout span the region rather
  // than selecting only the strongest descriptor responses.
  std::sort(pairs.begin(), pairs.end(), [](const Correspondence &a, const Correspondence &b) {
    return a.target.y() != b.target.y() ? a.target.y() < b.target.y() : a.target.x() < b.target.x();
  });
  return pairs;
}

bool planeHomography(const Chart &source, const Chart &target, const PlaneSeed &plane, M3 &h)
{
  const M3 rotation = target.R_v_w * source.R_v_w.transpose();
  const V3 translation = target.t_v_w - rotation * source.t_v_w;
  const V3 normal = source.R_v_w * plane.normal_w.normalized();
  const V3 point = source.R_v_w * plane.point_w + source.t_v_w;
  const double d = normal.dot(point);
  if (!std::isfinite(d) || std::abs(d) <= kEps) return false;
  M3 ka = M3::Identity(), kb = M3::Identity();
  ka(0, 0) = ka(1, 1) = source.focal;
  kb(0, 0) = kb(1, 1) = target.focal;
  h = kb * (rotation + translation * normal.transpose() / d) * ka.inverse();
  return h.allFinite() && std::abs(h.determinant()) > kEps;
}

bool fitResidual(const std::vector<Correspondence> &pairs, const M3 &prior,
                 bool reverse, M3 &corrected, std::vector<unsigned char> &inliers)
{
  std::vector<cv::Point2f> predicted, observed;
  for (size_t i = 0; i < pairs.size(); ++i)
  {
    if (i % 4 == 0) continue; // independent geometry holdout
    V2 q;
    if (!dehom(prior, reverse ? pairs[i].target : pairs[i].source, q)) return false;
    const V2 &target = reverse ? pairs[i].source : pairs[i].target;
    predicted.emplace_back(static_cast<float>(q.x()), static_cast<float>(q.y()));
    observed.emplace_back(static_cast<float>(target.x()), static_cast<float>(target.y()));
  }
  if (predicted.size() < 12) return false;
  cv::Mat mask;
  const cv::Mat affine = cv::estimateAffine2D(predicted, observed, mask, cv::RANSAC, 2.0, 1000, 0.995, 10);
  if (affine.empty() || affine.rows != 2 || affine.cols != 3) return false;
  M3 residual = M3::Identity();
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 3; ++col) residual(row, col) = affine.at<double>(row, col);
  if (!residual.allFinite() || residual.topLeftCorner<2, 2>().determinant() <= 0.0) return false;
  const Eigen::JacobiSVD<M2> svd(residual.topLeftCorner<2, 2>());
  if (svd.singularValues().minCoeff() < 0.8 || svd.singularValues().maxCoeff() > 1.25) return false;
  corrected = residual * prior;
  if (!corrected.allFinite() || std::abs(corrected.determinant()) <= kEps) return false;
  inliers.assign(pairs.size(), 0);
  int index = 0, count = 0;
  for (size_t i = 0; i < pairs.size(); ++i)
  {
    if (i % 4 == 0) continue;
    const bool good = mask.at<unsigned char>(index++) != 0;
    inliers[i] = good ? 1 : 0;
    count += good;
  }
  return count >= 8;
}

bool rawError(const Chart &target, const M3 &h, const V2 &source,
              const V2 &target_raw, double &error)
{
  V2 q, raw;
  if (!dehom(h, source, q) || !target.toRaw(q, raw)) return false;
  error = (raw - target_raw).norm();
  return std::isfinite(error);
}

bool verifyGeometry(const CropFeatures &source, const CropFeatures &target,
                    const std::vector<Correspondence> &pairs, const M3 &prior,
                    M3 &forward, M3 &reverse, std::vector<cv::Point2f> &hull)
{
  if (pairs.size() < 16) return false;
  std::vector<unsigned char> inlier_a, inlier_b;
  if (!fitResidual(pairs, prior, false, forward, inlier_a) ||
      !fitResidual(pairs, prior.inverse(), true, reverse, inlier_b)) return false;
  std::vector<cv::Point2f> support;
  int validation_count = 0;
  double validation_error = 0.0;
  for (size_t i = 0; i < pairs.size(); ++i)
  {
    const auto &p = pairs[i];
    double ea = 0.0, eb = 0.0;
    if (!rawError(target.chart, forward, p.source, p.target_raw, eb) ||
        !rawError(source.chart, reverse, p.target, p.source_raw, ea)) return false;
    if (i % 4 == 0)
    {
      // A held-out point is not silently dropped because it fits poorly.
      if (ea > 2.0 || eb > 2.0) return false;
      validation_error += 0.5 * (ea + eb);
      ++validation_count;
    }
    else if (inlier_a[i] && inlier_b[i] && ea <= 2.0 && eb <= 2.0)
    {
      V2 predicted, raw;
      if (!dehom(prior, p.source, predicted) || !target.chart.toRaw(predicted, raw) ||
          (raw - p.target_raw).norm() > 12.0) return false;
      support.emplace_back(static_cast<float>(p.target.x()), static_cast<float>(p.target.y()));
    }
  }
  if (validation_count < 4 || validation_error / validation_count > 1.25 || support.size() < 8) return false;
  cv::convexHull(support, hull);
  if (hull.size() < 3 || std::abs(cv::contourArea(hull)) < 100.0) return false;
  const cv::RotatedRect box = cv::minAreaRect(hull);
  if (std::min(box.size.width, box.size.height) < 8.0 ||
      std::max(box.size.width, box.size.height) > 8.0 * std::min(box.size.width, box.size.height)) return false;
  return true;
}

bool sourceRaw(const Chart &chart, const M3 &from_common, const V2 &q, V2 &raw)
{
  V2 local;
  return dehom(from_common, q, local) && chart.toRaw(local, raw);
}

bool rawJacobian(const Chart &chart, const M3 &from_common, const V2 &q, M2 &j)
{
  for (int axis = 0; axis < 2; ++axis)
  {
    V2 delta = V2::Zero(), plus, minus;
    delta[axis] = 0.25;
    if (!sourceRaw(chart, from_common, q + delta, plus) || !sourceRaw(chart, from_common, q - delta, minus)) return false;
    j.col(axis) = (plus - minus) / 0.5;
  }
  return j.allFinite() && std::abs(j.determinant()) > kEps;
}

double maximumScale(const M2 &j)
{
  return Eigen::JacobiSVD<M2>(j).singularValues().maxCoeff();
}

bool prepareImage(const cv::Mat &input, double minification, cv::Mat &output, int &halo)
{
  // Engineering low-pass rule for minification, not a measured camera PSF.
  // A large required footprint is rejected rather than clipped silently.
  const double sigma = 0.8 * std::sqrt(std::max(0.0, minification * minification - 1.0));
  if (!std::isfinite(sigma) || sigma > 6.0) return false;
  halo = static_cast<int>(std::ceil(3.0 * sigma)) + 2;
  input.convertTo(output, CV_32F);
  if (sigma > 0.05)
  {
    const int size = 2 * static_cast<int>(std::ceil(3.0 * sigma)) + 1;
    cv::GaussianBlur(output, output, cv::Size(size, size), sigma, sigma, cv::BORDER_REFLECT_101);
  }
  return true;
}

cv::Mat rawSampleMask(const CropFeatures &features, int halo)
{
  cv::Mat mask;
  // Test original channels before averaging: clipping must not disappear
  // inside a blurred mean. Geometry is the estimated LiDAR candidate mask.
  cv::inRange(features.chart.camera->bgr(features.roi), cv::Scalar(2, 2, 2), cv::Scalar(253, 253, 253), mask);
  cv::bitwise_and(mask, features.support_mask, mask);
  cv::erode(mask, mask, cv::Mat::ones(2 * halo + 1, 2 * halo + 1, CV_8U),
            cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
  return mask;
}

bool validBilinearSupport(const cv::Mat &mask, const V2 &p)
{
  if (!p.allFinite() || p.x() < 0.0 || p.y() < 0.0 ||
      p.x() >= mask.cols - 1.0 || p.y() >= mask.rows - 1.0) return false;
  const int x = static_cast<int>(std::floor(p.x())), y = static_cast<int>(std::floor(p.y()));
  return mask.at<unsigned char>(y, x) && mask.at<unsigned char>(y, x + 1) &&
         mask.at<unsigned char>(y + 1, x) && mask.at<unsigned char>(y + 1, x + 1);
}

bool overlappingRawSupport(const CropFeatures &a, const CropFeatures &b)
{
  const cv::Rect overlap = a.roi & b.roi;
  if (overlap.empty()) return false;
  // These masks are in distinct crop coordinates but refer to the same
  // camera image. Reject any shared candidate pixels conservatively; every
  // accepted color/filter tap is already restricted to its candidate mask.
  const cv::Rect local_a(overlap.x - a.roi.x, overlap.y - a.roi.y, overlap.width, overlap.height);
  const cv::Rect local_b(overlap.x - b.roi.x, overlap.y - b.roi.y, overlap.width, overlap.height);
  cv::Mat shared;
  cv::bitwise_and(a.support_mask(local_a), b.support_mask(local_b), shared);
  return cv::countNonZero(shared) != 0;
}

void takeSamples(const CropFeatures &source, const CropFeatures &target,
                 const PlaneSeed &plane, const M3 &forward, const M3 &reverse,
                 const std::vector<cv::Point2f> &hull, const V3 &target_weights,
                 bool target_is_anchor, int region, std::vector<ColorSample> &samples,
                 const std::atomic<bool> &cancel)
{
  const cv::Rect bounds = cv::boundingRect(hull);
  const V2 center(bounds.x + bounds.width * 0.5, bounds.y + bounds.height * 0.5);
  const M3 inverse = forward.inverse();
  M2 ja, jb;
  if (!rawJacobian(source.chart, inverse, center, ja) || !rawJacobian(target.chart, M3::Identity(), center, jb)) return;
  const Eigen::JacobiSVD<M2> sa(ja), sb(jb);
  if (sa.singularValues().minCoeff() <= kEps || sb.singularValues().minCoeff() <= kEps ||
      sa.singularValues().maxCoeff() / sa.singularValues().minCoeff() > 2.0 ||
      sb.singularValues().maxCoeff() / sb.singularValues().minCoeff() > 2.0) return;
  const double native_step = std::max(1.0 / sa.singularValues().minCoeff(), 1.0 / sb.singularValues().minCoeff());
  const double step = std::max(native_step, std::max(bounds.width, bounds.height) / static_cast<double>(kGridSize - 1));
  if (!std::isfinite(step) || step <= 0.0) return;
  const int width = std::min(kGridSize, static_cast<int>(std::floor(bounds.width / step)) + 1);
  const int height = std::min(kGridSize, static_cast<int>(std::floor(bounds.height / step)) + 1);
  if (width < 16 || height < 16) return;
  // A single local blur is inappropriate when the warp varies strongly.
  for (int corner = 0; corner < 4; ++corner)
  {
    const V2 q(bounds.x + ((corner & 1) ? bounds.width : 0), bounds.y + ((corner & 2) ? bounds.height : 0));
    M2 a, b;
    if (!rawJacobian(source.chart, inverse, q, a) || !rawJacobian(target.chart, M3::Identity(), q, b) ||
        (a - ja).norm() > 0.2 * ja.norm() || (b - jb).norm() > 0.2 * jb.norm()) return;
  }

  cv::Mat source_float, target_float, target_gray;
  int halo_a = 0, halo_b = 0;
  if (!prepareImage(source.chart.camera->bgr(source.roi), maximumScale(ja) * step, source_float, halo_a)) return;
  if (cancelled(cancel)) return;
  const bool legacy_gray = target_is_anchor || (target_weights - defaultWeights()).squaredNorm() == 0.0;
  if (legacy_gray)
    target_gray = fixed_gray_image::convertFixedGray(target.chart.camera->bgr(target.roi));
  else
  {
    const auto lut = fixed_gray_image::prepareFixedGrayLut(target_weights);
    target_gray = fixed_gray_image::convertFixedGray(target.chart.camera->bgr(target.roi), &lut);
  }
  if (!prepareImage(target_gray, maximumScale(jb) * step, target_float, halo_b)) return;
  const cv::Mat mask_a = rawSampleMask(source, halo_a), mask_b = rawSampleMask(target, halo_b);
  if (cancelled(cancel)) return;
  cv::Mat map_a(height, width, CV_32FC2), map_b(height, width, CV_32FC2);
  cv::Mat valid(height, width, CV_8UC1, cv::Scalar(0));
  for (int row = 0; row < height; ++row)
  {
    if (cancelled(cancel)) return;
    for (int col = 0; col < width; ++col)
    {
      const V2 q(bounds.x + col * step, bounds.y + row * step);
      V2 a, b, local_a, cycle, cycle_raw;
      bool ok = cv::pointPolygonTest(hull, cv::Point2f(static_cast<float>(q.x()), static_cast<float>(q.y())), false) >= 0.0;
      ok = ok && dehom(inverse, q, local_a) && onPlaneSupport(source.chart, plane, local_a) &&
           onPlaneSupport(target.chart, plane, q) && source.chart.toRaw(local_a, a) && target.chart.toRaw(q, b);
      // Reverse fit is independent; inverse(forward) alone is not a test.
      ok = ok && dehom(reverse, q, cycle) && source.chart.toRaw(cycle, cycle_raw) && (cycle_raw - a).norm() <= 1.5;
      if (ok)
      {
        a -= V2(source.roi.x, source.roi.y);
        b -= V2(target.roi.x, target.roi.y);
        ok = a.x() >= halo_a && a.y() >= halo_a && a.x() < source.roi.width - halo_a && a.y() < source.roi.height - halo_a &&
             b.x() >= halo_b && b.y() >= halo_b && b.x() < target.roi.width - halo_b && b.y() < target.roi.height - halo_b;
        ok = ok && validBilinearSupport(mask_a, a) && validBilinearSupport(mask_b, b);
      }
      map_a.at<cv::Vec2f>(row, col) = ok ? cv::Vec2f(static_cast<float>(a.x()), static_cast<float>(a.y())) : cv::Vec2f(-1000.0f, -1000.0f);
      map_b.at<cv::Vec2f>(row, col) = ok ? cv::Vec2f(static_cast<float>(b.x()), static_cast<float>(b.y())) : cv::Vec2f(-1000.0f, -1000.0f);
      if (ok) valid.at<unsigned char>(row, col) = 255;
    }
  }
  cv::Mat rgb, gray;
  cv::remap(source_float, rgb, map_a, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  cv::remap(target_float, gray, map_b, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  cv::Mat rgb_mean, rgb_square_mean, gray_mean, gray_square_mean;
  cv::GaussianBlur(rgb, rgb_mean, cv::Size(11, 11), 1.5, 1.5);
  cv::GaussianBlur(rgb.mul(rgb), rgb_square_mean, cv::Size(11, 11), 1.5, 1.5);
  cv::GaussianBlur(gray, gray_mean, cv::Size(11, 11), 1.5, 1.5);
  cv::GaussianBlur(gray.mul(gray), gray_square_mean, cv::Size(11, 11), 1.5, 1.5);
  // All taps must be valid; BORDER_CONSTANT never becomes an observation.
  cv::erode(valid, valid, cv::Mat::ones(11, 11, CV_8U), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
  for (int row = 5; row + 5 < height; row += 3)
  {
    if (cancelled(cancel)) return;
    for (int col = 5; col + 5 < width && samples.size() < kMaxSamples; col += 3)
    {
      if (!valid.at<unsigned char>(row, col)) continue;
      const cv::Vec3f mean = rgb_mean.at<cv::Vec3f>(row, col);
      const cv::Vec3f square = rgb_square_mean.at<cv::Vec3f>(row, col);
      const float y = gray_mean.at<float>(row, col);
      bool ok = std::isfinite(y) && y > 4.0f && y < 251.0f && gray_square_mean.at<float>(row, col) - y * y <= 36.0f;
      for (int channel = 0; channel < 3; ++channel)
        ok = ok && std::isfinite(mean[channel]) && mean[channel] > 4.0f && mean[channel] < 251.0f &&
             square[channel] - mean[channel] * mean[channel] <= 36.0f;
      if (!ok) continue;
      ColorSample sample;
      sample.rgb = V3(mean[2], mean[1], mean[0]);
      sample.target = y;
      sample.region = region;
      samples.push_back(sample);
    }
  }
}

std::vector<ColorSample> pairSamples(const Snapshot &snapshot, FeatureCache &cache,
                                    int source_id, int target_id, const V3 &target_weights,
                                    const std::atomic<bool> &cancel)
{
  std::vector<ColorSample> samples;
  samples.reserve(kMaxSamples);
  const int count = std::min(kMaxPlaneCandidates, static_cast<int>(snapshot.planes.size()));
  int regions_considered = 0;
  std::vector<std::pair<const CropFeatures *, const CropFeatures *>> used_regions;
  for (int i = 0; i < count && samples.size() < kMaxSamples; ++i)
  {
    if (cancelled(cancel)) break;
    const PlaneSeed &plane = snapshot.planes[i];
    if (!plane.point_w.allFinite() || !plane.normal_w.allFinite() || plane.normal_w.norm() < kEps ||
        !std::isfinite(plane.radius) || plane.radius <= 0.0) continue;
    const CachedFeatures &source_entry = cache.prepare(source_id, i);
    const CachedFeatures &target_entry = cache.prepare(target_id, i);
    if (!source_entry.region_valid || !target_entry.region_valid || cancelled(cancel)) continue;
    if (regions_considered >= kMaxRegionsPerPair) break;
    ++regions_considered;
    if (!cache.extract(source_id, i) || !cache.extract(target_id, i)) continue;
    const CropFeatures &source = source_entry.data, &target = target_entry.data;
    // Distinct world-plane labels alone do not guarantee independent image
    // data (e.g. different depths project onto the same raw image support).
    // Keep region-level color validation free of shared filter/input pixels
    // on BOTH sides. Cache entries have stable addresses for this snapshot.
    const bool overlaps_used_region = std::any_of(used_regions.begin(), used_regions.end(),
        [&source, &target](const std::pair<const CropFeatures *, const CropFeatures *> &used) {
          return overlappingRawSupport(source, *used.first) || overlappingRawSupport(target, *used.second);
        });
    if (overlaps_used_region) continue;
    const auto pairs = match(source, target);
    M3 prior, forward, reverse;
    std::vector<cv::Point2f> hull;
    if (!planeHomography(source.chart, target.chart, plane, prior) ||
        !verifyGeometry(source, target, pairs, prior, forward, reverse, hull)) continue;
    if (cancelled(cancel)) break;
    const size_t previous_samples = samples.size();
    takeSamples(source, target, plane, forward, reverse, hull, target_weights, target_id == 0, i, samples, cancel);
    if (samples.size() > previous_samples) used_regions.emplace_back(&source, &target);
  }
  return samples;
}
#endif
} // namespace

Result estimate(const Snapshot &snapshot, const std::atomic<bool> &cancel)
{
  Result result;
  const size_t count = snapshot.cameras.size();
  result.weights.assign(count, defaultWeights());
  result.cameras.resize(count);
#if !CAKE_HAVE_OPENCV_SIFT
  (void)cancel;
  result.reason = "OpenCV SIFT is unavailable in this build";
  return result;
#else
  try
  {
    if (count < 2 || count > 16 || snapshot.planes.empty())
    {
      result.reason = "initialization requires 2..16 cameras and LiDAR plane supports";
      return result;
    }
    for (const CameraSnapshot &camera : snapshot.cameras)
      if (camera.bgr.empty() || camera.bgr.type() != CV_8UC3 || !camera.project || !camera.unproject ||
          !camera.R_c_w.allFinite() || !camera.t_c_w.allFinite() ||
          !std::isfinite(camera.virtual_focal) || camera.virtual_focal <= 0.0 ||
          (camera.R_c_w.transpose() * camera.R_c_w - Eigen::Matrix3d::Identity()).norm() > 1.0e-5 ||
          camera.R_c_w.determinant() <= 0.0)
      {
        result.reason = "invalid immutable camera snapshot";
        return result;
      }
    result.cameras[0].calibrated = true;
    result.cameras[0].reason = "anchor keeps legacy BGR2GRAY";
    FeatureCache feature_cache(snapshot, cancel);
    std::vector<std::vector<unsigned char>> tried(count, std::vector<unsigned char>(count, 0));
    int attempts = 0;
    const int attempt_limit = std::min(32, static_cast<int>(count) * 4);
    bool progress = true;
    while (progress && attempts < attempt_limit)
    {
      progress = false;
      for (size_t source = 1; source < count && attempts < attempt_limit; ++source)
      {
        if (result.cameras[source].calibrated) continue;
        for (size_t target = 0; target < count && attempts < attempt_limit; ++target)
        {
          if (cancelled(cancel)) { result.reason = "initialization cancelled"; return result; }
          if (!result.cameras[target].calibrated || tried[source][target]) continue;
          tried[source][target] = 1;
          ++attempts;
          const auto samples = pairSamples(snapshot, feature_cache, static_cast<int>(source), static_cast<int>(target), result.weights[target], cancel);
          if (cancelled(cancel)) { result.reason = "initialization cancelled"; return result; }
          const WeightFit fit = fitWeights(samples);
          CameraReport &report = result.cameras[source];
          report.parent_camera = static_cast<int>(target);
          report.samples = static_cast<int>(samples.size());
          std::vector<int> regions;
          for (const auto &sample : samples) regions.push_back(sample.region);
          std::sort(regions.begin(), regions.end());
          regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
          report.regions = static_cast<int>(regions.size());
          report.validation_rmse = fit.validation_rmse;
          report.baseline_rmse = fit.baseline_rmse;
          report.reason = fit.reason;
          if (fit.accepted)
          {
            report.calibrated = true;
            result.weights[source] = fit.weights;
            progress = true;
            break;
          }
        }
      }
    }
    if (cancelled(cancel)) { result.reason = "initialization cancelled"; return result; }
    result.accepted = std::all_of(result.cameras.begin(), result.cameras.end(), [](const CameraReport &c) { return c.calibrated; });
    result.reason = result.accepted ? "all fixed grayscale weights accepted" : "not all cameras have validated color correspondences and weights";
  }
  catch (const cv::Exception &error)
  {
    result.accepted = false;
    result.reason = std::string("OpenCV initialization failure: ") + error.what();
  }
  catch (const std::exception &error)
  {
    result.accepted = false;
    result.reason = std::string("initialization failure: ") + error.what();
  }
  return result;
#endif
}

} // namespace color_gray_init
