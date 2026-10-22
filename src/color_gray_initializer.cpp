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
#include <sstream>
#include <utility>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#ifndef CAKE_HAVE_OPENCV_SIFT
#define CAKE_HAVE_OPENCV_SIFT 0
#endif
#ifndef CAKE_HAVE_OPENCV_XFEATURES2D_SIFT
#define CAKE_HAVE_OPENCV_XFEATURES2D_SIFT 0
#endif
#if CAKE_HAVE_OPENCV_XFEATURES2D_SIFT
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

namespace color_gray_init
{
namespace
{
constexpr int kMaxPlaneCandidates = 24;
constexpr int kMaxRegionsPerPair = 12;
constexpr int kMaxFullCropsPerCamera = 12;
constexpr int kMaxCrop = 320;
constexpr int kMinFeatureContext = 96;
constexpr int kMinGeometryMatches = 6;
constexpr int kMinGeometryTraining = 4;
constexpr int kMinGeometryHoldout = 2;
constexpr int kGeometryHoldoutStride = 3;
constexpr int kMaxSamples = 4096;
constexpr int kGridSize = 16;
constexpr int kColorKernelSize = 5;
constexpr int kColorHalo = kColorKernelSize / 2;
constexpr int kColorSampleStride = 1;
constexpr double kColorSigma = 1.0;
constexpr double kEps = 1.0e-9;
static_assert(kMinGeometryMatches - (kMinGeometryMatches + kGeometryHoldoutStride - 1) /
                  kGeometryHoldoutStride >= kMinGeometryTraining, "insufficient geometry training points");
static_assert((kMinGeometryMatches + kGeometryHoldoutStride - 1) / kGeometryHoldoutStride >=
                  kMinGeometryHoldout, "insufficient geometry holdout points");
static_assert(kColorKernelSize % 2 == 1 && kGridSize >= kColorKernelSize, "invalid color support");

#if CAKE_HAVE_OPENCV_SIFT || CAKE_HAVE_OPENCV_XFEATURES2D_SIFT
enum Stage
{
  Seed, Roi, SiftSource, SiftTarget, Independent, Geometry, Sampling,
  Ratio, Mutual, ChartMapping, Unique, Grid, Color, Fit
};

PairReport makePairReport(int source, int target)
{
  PairReport report;
  report.source_camera = source;
  report.target_camera = target;
  const auto add = [&report](const char *name, const char *unit) {
    StageReport stage;
    stage.name = name;
    stage.unit = unit;
    report.stages.push_back(std::move(stage));
  };
  for (const char *name : {"seed", "roi", "sift_src", "sift_dst", "independent", "geometry", "rgb_region"})
    add(name, "regions");
  for (const char *name : {"ratio", "mutual", "tangent", "unique"}) add(name, "matches");
  add("grid", "pixels");
  add("color", "samples");
  add("fit", "pairs");
  return report;
}

bool reject(std::string &reason, const char *value)
{
  reason = value;
  return false;
}

void record(StageReport &stage, bool passed, const std::string &reason)
{
  ++stage.entered;
  if (passed) ++stage.passed;
  else ++stage.rejected[reason];
}

cv::Ptr<cv::Feature2D> createSift()
{
#if CAKE_HAVE_OPENCV_SIFT
  return cv::SIFT::create(500);
#else
  return cv::xfeatures2d::SIFT::create(500);
#endif
}

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

bool makeChart(const CameraSnapshot &camera, const PlaneSeed &plane, Chart &chart, std::string &reason)
{
  const V3 p = camera.R_c_w * plane.point_w + camera.t_c_w;
  if (!p.allFinite() || p.norm() <= kEps) return reject(reason, "invalid_camera_point");
  V2 raw;
  if (!camera.project(p, raw) || !raw.allFinite()) return reject(reason, "center_projection_failed");
  const V3 z = p.normalized();
  const V3 back = camera.unproject(raw);
  if (!back.allFinite() || back.norm() <= kEps || back.normalized().dot(z) < 0.999)
    return reject(reason, "projection_unprojection_inconsistent");
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
bool prepareRegion(const CameraSnapshot &camera, const PlaneSeed &plane, CropFeatures &out, std::string &reason)
{
  if (!makeChart(camera, plane, out.chart, reason)) return false;
  const V3 n = plane.normal_w.normalized();
  const V3 u = n.unitOrthogonal();
  const V3 v = n.cross(u);
  std::vector<cv::Point2f> boundary;
  V2 center;
  if (!camera.project(camera.R_c_w * plane.point_w + camera.t_c_w, center) ||
      !center.allFinite() || center.x() < 0.0 || center.y() < 0.0 ||
      center.x() >= camera.bgr.cols || center.y() >= camera.bgr.rows) return reject(reason, "center_outside_image");
  for (int i = 0; i < 16; ++i)
  {
    const double angle = i * (2.0 * 3.14159265358979323846 / 16.0);
    const V3 p = plane.point_w + plane.radius * (std::cos(angle) * u + std::sin(angle) * v);
    V2 raw;
    if (!camera.project(camera.R_c_w * p + camera.t_c_w, raw) || !raw.allFinite())
      return reject(reason, "plane_boundary_projection_failed");
    // Guard conversion to OpenCV integer rectangles for malformed models.
    if (raw.cwiseAbs().maxCoeff() > 1.0e6) return reject(reason, "plane_boundary_overflow");
    boundary.emplace_back(static_cast<float>(raw.x()), static_cast<float>(raw.y()));
  }
  const cv::Rect plane_roi = cv::boundingRect(boundary) & cv::Rect(0, 0, camera.bgr.cols, camera.bgr.rows);
  if (plane_roi.empty()) return reject(reason, "empty_plane_projection");
  // Context is image storage only. The plane radius and the masks used for
  // geometry/color sampling do not grow with this padded crop.
  const int width = std::min(camera.bgr.cols, std::min(kMaxCrop, std::max(kMinFeatureContext, plane_roi.width)));
  const int height = std::min(camera.bgr.rows, std::min(kMaxCrop, std::max(kMinFeatureContext, plane_roi.height)));
  const int x = std::clamp(static_cast<int>(std::floor(center.x() - width * 0.5)), 0, camera.bgr.cols - width);
  const int y = std::clamp(static_cast<int>(std::floor(center.y() - height * 0.5)), 0, camera.bgr.rows - height);
  out.roi = cv::Rect(x, y, width, height);
  return true;
}

bool extractFeatures(const CameraSnapshot &camera, const PlaneSeed &plane,
                     const cv::Ptr<cv::Feature2D> &sift, CropFeatures &out,
                     const std::atomic<bool> &cancel, std::string &reason)
{
  const int width = out.roi.width, height = out.roi.height;
  const int x = out.roi.x, y = out.roi.y;
  cv::Mat gray, mask(height, width, CV_8UC1, cv::Scalar(0));
  cv::cvtColor(camera.bgr(out.roi), gray, cv::COLOR_BGR2GRAY);
  // The mask is geometry only; using default grayscale for descriptors does
  // not assume that the final cross-camera color weights are known.
  for (int row = 0; row < height; ++row)
  {
    if (cancelled(cancel)) return reject(reason, "cancelled");
    for (int col = 0; col < width; ++col)
    {
      V2 q;
      if (out.chart.toChart(V2(x + col, y + row), q) && onPlaneSupport(out.chart, plane, q))
        mask.at<unsigned char>(row, col) = 255;
    }
  }
  out.support_mask = mask.clone();
  // This mask selects feature CENTERS. SIFT descriptors may read the padded
  // image context; they need not fit wholly inside a tiny planar support.
  if (cancelled(cancel)) return reject(reason, "cancelled");
  sift->detectAndCompute(gray, mask, out.keys, out.descriptors);
  if (cancelled(cancel)) return reject(reason, "cancelled");
  // SIFT's integer mask lookup is not an exact subpixel support test. Filter
  // descriptor rows together with keypoints before any pairing/affine fit.
  std::vector<cv::KeyPoint> supported_keys;
  cv::Mat supported_descriptors;
  for (size_t i = 0; i < out.keys.size(); ++i)
  {
    cv::KeyPoint key = out.keys[i];
    key.pt += cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    V2 q;
    if (!out.chart.toChart(V2(key.pt.x, key.pt.y), q) || !onPlaneSupport(out.chart, plane, q)) continue;
    supported_keys.push_back(key);
    if (!out.descriptors.empty()) supported_descriptors.push_back(out.descriptors.row(static_cast<int>(i)));
  }
  out.keys.swap(supported_keys);
  out.descriptors = std::move(supported_descriptors);
  if (out.keys.size() < kMinGeometryMatches)
    return reject(reason, cv::countNonZero(mask) == 0 ? "empty_sift_mask" : "supported_keypoints_lt_6");
  if (out.descriptors.empty()) return reject(reason, "empty_descriptors");
  return true;
}

struct CachedFeatures
{
  CropFeatures data;
  bool prepared = false;
  bool region_valid = false;
  bool extracted = false;
  bool features_valid = false;
  std::string region_reason;
  std::string feature_reason;
};

struct FeatureCache
{
  const Snapshot &snapshot;
  const std::atomic<bool> &cancel;
  cv::Ptr<cv::Feature2D> sift = createSift();
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
      entry.region_valid = prepareRegion(snapshot.cameras[camera], snapshot.planes[plane], entry.data, entry.region_reason);
    }
    return entry;
  }

  bool extract(int camera, int plane)
  {
    CachedFeatures &entry = prepare(camera, plane);
    if (entry.extracted) return entry.features_valid;
    entry.extracted = true;
    const int area = entry.data.roi.area();
    if (!entry.region_valid) return reject(entry.feature_reason, "invalid_region");
    if (cancelled(cancel)) return reject(entry.feature_reason, "cancelled");
    if (extracted_pixels[camera] + area > kMaxFullCropsPerCamera * kMaxCrop * kMaxCrop)
      return reject(entry.feature_reason, "camera_sift_pixel_budget");
    extracted_pixels[camera] += area;
    entry.features_valid = extractFeatures(snapshot.cameras[camera], snapshot.planes[plane], sift, entry.data, cancel, entry.feature_reason);
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

std::vector<Correspondence> match(const CropFeatures &source, const CropFeatures &target, PairReport &report)
{
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<std::vector<cv::DMatch>> forward, reverse;
  matcher.knnMatch(source.descriptors, target.descriptors, forward, 2);
  matcher.knnMatch(target.descriptors, source.descriptors, reverse, 2);
  std::vector<Correspondence> pairs;
  for (const auto &two : forward)
  {
    const bool ratio_ok = two.size() >= 2 && two[0].distance < 0.75f * two[1].distance;
    record(report.stages[Ratio], ratio_ok, two.size() < 2 ? "fewer_than_2_neighbors" : "ratio_ge_0.75");
    if (!ratio_ok) continue;
    const cv::DMatch &m = two[0];
    if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(reverse.size()))
    {
      record(report.stages[Mutual], false, "invalid_reverse_index");
      continue;
    }
    const auto &back = reverse[m.trainIdx];
    const bool mutual_ok = back.size() >= 2 && back[0].trainIdx == m.queryIdx && back[0].distance < 0.75f * back[1].distance;
    record(report.stages[Mutual], mutual_ok, back.size() < 2 ? "fewer_than_2_reverse_neighbors" :
           (back[0].trainIdx != m.queryIdx ? "not_mutual_best" : "reverse_ratio_ge_0.75"));
    if (!mutual_ok) continue;
    Correspondence p;
    p.descriptor_distance = m.distance;
    p.source_raw = V2(source.keys[m.queryIdx].pt.x, source.keys[m.queryIdx].pt.y);
    p.target_raw = V2(target.keys[m.trainIdx].pt.x, target.keys[m.trainIdx].pt.y);
    const bool chart_ok = source.chart.toChart(p.source_raw, p.source) && target.chart.toChart(p.target_raw, p.target);
    record(report.stages[ChartMapping], chart_ok, "tangent_mapping_failed");
    if (chart_ok) pairs.push_back(p);
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
    record(report.stages[Unique], !duplicate, "duplicate_within_2px");
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

bool isGeometryHoldout(size_t index)
{
  return index % kGeometryHoldoutStride == 0;
}

bool hasAffineSpan(const std::vector<cv::Point2f> &points)
{
  // Centering makes the rank check independent of tangent-plane origin;
  // normalization by trace^2 makes it independent of coordinate scale.
  if (points.size() < 3) return false;
  V2 mean = V2::Zero();
  for (const auto &p : points) mean += V2(p.x, p.y);
  mean /= static_cast<double>(points.size());
  M2 scatter = M2::Zero();
  for (const auto &p : points)
  {
    const V2 delta = V2(p.x, p.y) - mean;
    scatter += delta * delta.transpose();
  }
  const double trace = scatter.trace();
  return scatter.allFinite() && trace > kEps && scatter.determinant() > kEps * trace * trace;
}

bool fitResidual(const std::vector<Correspondence> &pairs, const M3 &prior,
                 bool reverse, M3 &corrected, std::vector<unsigned char> &inliers, std::string &reason)
{
  std::vector<cv::Point2f> predicted, observed;
  for (size_t i = 0; i < pairs.size(); ++i)
  {
    if (isGeometryHoldout(i)) continue; // six matches -> four training, two holdout
    V2 q;
    if (!dehom(prior, reverse ? pairs[i].target : pairs[i].source, q)) return reject(reason, "prior_projection_failed");
    const V2 &target = reverse ? pairs[i].source : pairs[i].target;
    predicted.emplace_back(static_cast<float>(q.x()), static_cast<float>(q.y()));
    observed.emplace_back(static_cast<float>(target.x()), static_cast<float>(target.y()));
  }
  if (predicted.size() < kMinGeometryTraining) return reject(reason, "training_matches_lt_4");
  if (!hasAffineSpan(predicted) || !hasAffineSpan(observed)) return reject(reason, "training_points_degenerate");
  cv::Mat mask;
  const cv::Mat affine = cv::estimateAffine2D(predicted, observed, mask, cv::RANSAC, 2.0, 1000, 0.995, 10);
  if (affine.empty() || affine.rows != 2 || affine.cols != 3) return reject(reason, "ransac_no_affine");
  if (mask.total() != predicted.size()) return reject(reason, "ransac_invalid_inlier_mask");
  M3 residual = M3::Identity();
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 3; ++col) residual(row, col) = affine.at<double>(row, col);
  if (!residual.allFinite() || residual.topLeftCorner<2, 2>().determinant() <= 0.0)
    return reject(reason, "invalid_affine_orientation");
  const Eigen::JacobiSVD<M2> svd(residual.topLeftCorner<2, 2>());
  if (svd.singularValues().minCoeff() < 0.8 || svd.singularValues().maxCoeff() > 1.25)
    return reject(reason, "affine_scale_outside_0.8_1.25");
  corrected = residual * prior;
  if (!corrected.allFinite() || std::abs(corrected.determinant()) <= kEps) return reject(reason, "singular_corrected_warp");
  inliers.assign(pairs.size(), 0);
  int index = 0, count = 0;
  std::vector<cv::Point2f> inlier_predicted, inlier_observed;
  for (size_t i = 0; i < pairs.size(); ++i)
  {
    if (isGeometryHoldout(i)) continue;
    const bool good = mask.at<unsigned char>(index) != 0;
    if (good)
    {
      inlier_predicted.push_back(predicted[index]);
      inlier_observed.push_back(observed[index]);
    }
    ++index;
    inliers[i] = good ? 1 : 0;
    count += good;
  }
  if (count < kMinGeometryTraining) return reject(reason, "ransac_inliers_lt_4");
  if (!hasAffineSpan(inlier_predicted) || !hasAffineSpan(inlier_observed))
    return reject(reason, "ransac_inliers_degenerate");
  return true;
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
                    M3 &forward, M3 &reverse, std::vector<cv::Point2f> &hull, std::string &reason)
{
  if (pairs.size() < kMinGeometryMatches) return reject(reason, "unique_matches_lt_6");
  std::vector<unsigned char> inlier_a, inlier_b;
  if (!fitResidual(pairs, prior, false, forward, inlier_a, reason))
  { reason = "forward_" + reason; return false; }
  if (!fitResidual(pairs, prior.inverse(), true, reverse, inlier_b, reason))
  { reason = "reverse_" + reason; return false; }
  std::vector<cv::Point2f> support;
  int validation_count = 0;
  double validation_error = 0.0;
  for (size_t i = 0; i < pairs.size(); ++i)
  {
    const auto &p = pairs[i];
    double ea = 0.0, eb = 0.0;
    if (!rawError(target.chart, forward, p.source, p.target_raw, eb) ||
        !rawError(source.chart, reverse, p.target, p.source_raw, ea)) return reject(reason, "raw_reprojection_failed");
    if (isGeometryHoldout(i))
    {
      // A held-out point is not silently dropped because it fits poorly.
      if (ea > 2.0 || eb > 2.0) return reject(reason, "holdout_error_gt_2px");
      validation_error += 0.5 * (ea + eb);
      ++validation_count;
    }
    else if (inlier_a[i] && inlier_b[i] && ea <= 2.0 && eb <= 2.0)
    {
      V2 predicted, raw;
      if (!dehom(prior, p.source, predicted) || !target.chart.toRaw(predicted, raw) ||
          (raw - p.target_raw).norm() > 12.0) return reject(reason, "prior_projection_invalid_or_error_gt_12px");
      support.emplace_back(static_cast<float>(p.target.x()), static_cast<float>(p.target.y()));
    }
  }
  if (validation_count < kMinGeometryHoldout) return reject(reason, "holdout_count_lt_2");
  if (validation_error / validation_count > 1.25) return reject(reason, "holdout_mean_gt_1.25px");
  if (support.size() < kMinGeometryTraining) return reject(reason, "bidirectional_support_lt_4");
  cv::convexHull(support, hull);
  if (hull.size() < 3 || std::abs(cv::contourArea(hull)) < 100.0) return reject(reason, "hull_area_lt_100_or_degenerate");
  const cv::RotatedRect box = cv::minAreaRect(hull);
  if (std::min(box.size.width, box.size.height) < 8.0) return reject(reason, "hull_width_lt_8");
  if (std::max(box.size.width, box.size.height) > 8.0 * std::min(box.size.width, box.size.height))
    return reject(reason, "hull_aspect_gt_8");
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
                 const std::atomic<bool> &cancel, PairReport &report, std::string &reason)
{
  const size_t initial_samples = samples.size();
  const cv::Moments moments = cv::moments(hull);
  if (!std::isfinite(moments.m00) || std::abs(moments.m00) <= kEps)
  { reason = "invalid_hull_centroid"; return; }
  const V2 center(moments.m10 / moments.m00, moments.m01 / moments.m00);
  if (!center.allFinite()) { reason = "invalid_hull_centroid"; return; }
  const M3 inverse = forward.inverse();
  M2 ja, jb;
  if (!rawJacobian(source.chart, inverse, center, ja) || !rawJacobian(target.chart, M3::Identity(), center, jb))
  { reason = "center_jacobian_failed"; return; }
  const Eigen::JacobiSVD<M2> sa(ja), sb(jb);
  if (sa.singularValues().minCoeff() <= kEps || sb.singularValues().minCoeff() <= kEps ||
      sa.singularValues().maxCoeff() / sa.singularValues().minCoeff() > 2.0 ||
      sb.singularValues().maxCoeff() / sb.singularValues().minCoeff() > 2.0)
  { reason = "singular_footprint_or_anisotropy_gt_2"; return; }
  const double step = std::max(1.0 / sa.singularValues().minCoeff(), 1.0 / sb.singularValues().minCoeff());
  if (!std::isfinite(step) || step <= 0.0) { reason = "invalid_grid_step"; return; }
  // A local 16x16 window, not a resize of the entire feature-support hull.
  // Keep the native-footprint pitch; shrinking the grid must not force more
  // minification or require both sides of a thin hull to span 16 samples.
  const int width = kGridSize, height = kGridSize;
  const V2 origin = center - V2::Constant(0.5 * (kGridSize - 1) * step);
  // A single local blur is inappropriate when the warp varies strongly.
  for (int corner = 0; corner < 4; ++corner)
  {
    const V2 q = origin + V2((corner & 1) ? (width - 1) * step : 0.0,
                             (corner & 2) ? (height - 1) * step : 0.0);
    M2 a, b;
    if (!rawJacobian(source.chart, inverse, q, a) || !rawJacobian(target.chart, M3::Identity(), q, b))
    { reason = "corner_jacobian_failed"; return; }
    if ((a - ja).norm() > 0.2 * ja.norm() || (b - jb).norm() > 0.2 * jb.norm())
    { reason = "jacobian_variation_gt_20pct"; return; }
  }

  cv::Mat source_float, target_float, target_gray;
  int halo_a = 0, halo_b = 0;
  if (!prepareImage(source.chart.camera->bgr(source.roi), maximumScale(ja) * step, source_float, halo_a))
  { reason = "source_blur_sigma_invalid_or_gt_6"; return; }
  if (cancelled(cancel)) { reason = "cancelled"; return; }
  const bool legacy_gray = target_is_anchor || (target_weights - defaultWeights()).squaredNorm() == 0.0;
  if (legacy_gray)
    target_gray = fixed_gray_image::convertFixedGray(target.chart.camera->bgr(target.roi));
  else
  {
    const auto lut = fixed_gray_image::prepareFixedGrayLut(target_weights);
    target_gray = fixed_gray_image::convertFixedGray(target.chart.camera->bgr(target.roi), &lut);
  }
  if (!prepareImage(target_gray, maximumScale(jb) * step, target_float, halo_b))
  { reason = "target_blur_sigma_invalid_or_gt_6"; return; }
  const cv::Mat mask_a = rawSampleMask(source, halo_a), mask_b = rawSampleMask(target, halo_b);
  if (cancelled(cancel)) { reason = "cancelled"; return; }
  // Fixed counters avoid allocating strings or updating maps per grid pixel.
  enum GridReject { Hull, Inverse, SourcePlane, TargetPlane, SourceRaw, TargetRaw,
                    CycleProjection, CycleError, Halo, SourceMask, TargetMask, GridRejectCount };
  const char *grid_reasons[] = {"outside_hull", "inverse_projection", "source_plane_support", "target_plane_support",
      "source_raw_projection", "target_raw_projection", "cycle_projection", "cycle_error_gt_1.5px",
      "filter_halo_outside_crop", "source_raw_mask", "target_raw_mask"};
  std::array<int, GridRejectCount> grid_rejected{};
  const auto grid_check = [&grid_rejected](bool ok, GridReject cause) {
    if (!ok) ++grid_rejected[cause];
    return ok;
  };
  cv::Mat map_a(height, width, CV_32FC2), map_b(height, width, CV_32FC2);
  cv::Mat valid(height, width, CV_8UC1, cv::Scalar(0));
  for (int row = 0; row < height; ++row)
  {
    if (cancelled(cancel)) { reason = "cancelled"; return; }
    for (int col = 0; col < width; ++col)
    {
      const V2 q = origin + V2(col * step, row * step);
      V2 a, b, local_a, cycle, cycle_raw;
      ++report.stages[Grid].entered;
      bool ok = grid_check(cv::pointPolygonTest(hull, cv::Point2f(static_cast<float>(q.x()), static_cast<float>(q.y())), false) >= 0.0, Hull);
      ok = ok && grid_check(dehom(inverse, q, local_a), Inverse) &&
           grid_check(onPlaneSupport(source.chart, plane, local_a), SourcePlane) &&
           grid_check(onPlaneSupport(target.chart, plane, q), TargetPlane) &&
           grid_check(source.chart.toRaw(local_a, a), SourceRaw) && grid_check(target.chart.toRaw(q, b), TargetRaw);
      // Reverse fit is independent; inverse(forward) alone is not a test.
      ok = ok && grid_check(dehom(reverse, q, cycle) && source.chart.toRaw(cycle, cycle_raw), CycleProjection) &&
           grid_check((cycle_raw - a).norm() <= 1.5, CycleError);
      if (ok)
      {
        a -= V2(source.roi.x, source.roi.y);
        b -= V2(target.roi.x, target.roi.y);
        ok = grid_check(a.x() >= halo_a && a.y() >= halo_a && a.x() < source.roi.width - halo_a && a.y() < source.roi.height - halo_a &&
             b.x() >= halo_b && b.y() >= halo_b && b.x() < target.roi.width - halo_b && b.y() < target.roi.height - halo_b, Halo);
        ok = ok && grid_check(validBilinearSupport(mask_a, a), SourceMask) && grid_check(validBilinearSupport(mask_b, b), TargetMask);
      }
      map_a.at<cv::Vec2f>(row, col) = ok ? cv::Vec2f(static_cast<float>(a.x()), static_cast<float>(a.y())) : cv::Vec2f(-1000.0f, -1000.0f);
      map_b.at<cv::Vec2f>(row, col) = ok ? cv::Vec2f(static_cast<float>(b.x()), static_cast<float>(b.y())) : cv::Vec2f(-1000.0f, -1000.0f);
      if (ok) { valid.at<unsigned char>(row, col) = 255; ++report.stages[Grid].passed; }
    }
  }
  for (int i = 0; i < GridRejectCount; ++i)
    if (grid_rejected[i]) report.stages[Grid].rejected[grid_reasons[i]] += grid_rejected[i];
  cv::Mat rgb, gray;
  cv::remap(source_float, rgb, map_a, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  cv::remap(target_float, gray, map_b, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  cv::Mat rgb_mean, rgb_square_mean, gray_mean, gray_square_mean;
  const cv::Size color_kernel(kColorKernelSize, kColorKernelSize);
  cv::GaussianBlur(rgb, rgb_mean, color_kernel, kColorSigma, kColorSigma);
  cv::GaussianBlur(rgb.mul(rgb), rgb_square_mean, color_kernel, kColorSigma, kColorSigma);
  cv::GaussianBlur(gray, gray_mean, color_kernel, kColorSigma, kColorSigma);
  cv::GaussianBlur(gray.mul(gray), gray_square_mean, color_kernel, kColorSigma, kColorSigma);
  // All taps must be valid; BORDER_CONSTANT never becomes an observation.
  cv::erode(valid, valid, cv::Mat::ones(kColorKernelSize, kColorKernelSize, CV_8U),
            cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
  enum ColorReject { FilterSupport, TargetFinite, TargetRange, TargetVariance,
                     SourceFinite, SourceRange, SourceVariance, ColorRejectCount };
  const char *color_reasons[] = {"invalid_5x5_support", "target_nonfinite", "target_mean_outside_4_251",
      "target_variance_gt_36_or_nan", "source_nonfinite", "source_mean_outside_4_251", "source_variance_gt_36_or_nan"};
  std::array<int, ColorRejectCount> color_rejected{};
  const auto color_check = [&color_rejected](bool ok, ColorReject cause) {
    if (!ok) ++color_rejected[cause];
    return ok;
  };
  // Visit each valid center in the small grid. Adjacent local means may
  // overlap, but keep the same physical-region ID and train/validation split.
  for (int row = kColorHalo; row + kColorHalo < height; row += kColorSampleStride)
  {
    if (cancelled(cancel)) { reason = "cancelled"; return; }
    for (int col = kColorHalo; col + kColorHalo < width && samples.size() < kMaxSamples; col += kColorSampleStride)
    {
      ++report.stages[Color].entered;
      if (!color_check(valid.at<unsigned char>(row, col) != 0, FilterSupport)) continue;
      const cv::Vec3f mean = rgb_mean.at<cv::Vec3f>(row, col);
      const cv::Vec3f square = rgb_square_mean.at<cv::Vec3f>(row, col);
      const float y = gray_mean.at<float>(row, col);
      bool ok = color_check(std::isfinite(y), TargetFinite) && color_check(y > 4.0f && y < 251.0f, TargetRange) &&
                color_check(gray_square_mean.at<float>(row, col) - y * y <= 36.0f, TargetVariance);
      for (int channel = 0; channel < 3; ++channel)
        ok = ok && color_check(std::isfinite(mean[channel]), SourceFinite) &&
             color_check(mean[channel] > 4.0f && mean[channel] < 251.0f, SourceRange) &&
             color_check(square[channel] - mean[channel] * mean[channel] <= 36.0f, SourceVariance);
      if (!ok) continue;
      ColorSample sample;
      sample.rgb = V3(mean[2], mean[1], mean[0]);
      sample.target = y;
      sample.region = region;
      samples.push_back(sample);
      ++report.stages[Color].passed;
    }
  }
  for (int i = 0; i < ColorRejectCount; ++i)
    if (color_rejected[i]) report.stages[Color].rejected[color_reasons[i]] += color_rejected[i];
  if (samples.size() == initial_samples) reason = "no_valid_color_samples";
}

std::vector<ColorSample> pairSamples(const Snapshot &snapshot, FeatureCache &cache,
                                    int source_id, int target_id, const V3 &target_weights,
                                    const std::atomic<bool> &cancel, PairReport &report)
{
  std::vector<ColorSample> samples;
  samples.reserve(kMaxSamples);
  const int count = std::min(kMaxPlaneCandidates, static_cast<int>(snapshot.planes.size()));
  report.plane_candidates = count;
  int regions_considered = 0;
  std::vector<std::pair<const CropFeatures *, const CropFeatures *>> used_regions;
  for (int i = 0; i < count && samples.size() < kMaxSamples; ++i)
  {
    if (cancelled(cancel)) { report.stop_reason = "cancelled"; break; }
    const PlaneSeed &plane = snapshot.planes[i];
    const bool invalid_seed = !plane.point_w.allFinite() || !plane.normal_w.allFinite() || plane.normal_w.norm() < kEps ||
                              !std::isfinite(plane.radius) || plane.radius <= 0.0;
    record(report.stages[Seed], !invalid_seed, "invalid_plane_seed");
    if (invalid_seed) continue;
    const CachedFeatures &source_entry = cache.prepare(source_id, i);
    const CachedFeatures &target_entry = cache.prepare(target_id, i);
    const bool roi_ok = source_entry.region_valid && target_entry.region_valid;
    record(report.stages[Roi], roi_ok, !source_entry.region_valid ? "source_" + source_entry.region_reason : "target_" + target_entry.region_reason);
    if (!roi_ok || cancelled(cancel)) continue;
    if (regions_considered >= kMaxRegionsPerPair) { report.stop_reason = "pair_region_budget_12"; break; }
    ++regions_considered;
    // Preserve the original short circuit: do not extract target if source failed.
    ++report.stages[SiftSource].entered;
    const bool source_ok = cache.extract(source_id, i);
    report.source_keypoints_max = std::max(report.source_keypoints_max, static_cast<int>(source_entry.data.keys.size()));
    if (!source_ok) { ++report.stages[SiftSource].rejected[source_entry.feature_reason]; continue; }
    ++report.stages[SiftSource].passed;
    ++report.stages[SiftTarget].entered;
    const bool target_ok = cache.extract(target_id, i);
    report.target_keypoints_max = std::max(report.target_keypoints_max, static_cast<int>(target_entry.data.keys.size()));
    if (!target_ok) { ++report.stages[SiftTarget].rejected[target_entry.feature_reason]; continue; }
    ++report.stages[SiftTarget].passed;
    const CropFeatures &source = source_entry.data, &target = target_entry.data;
    // Distinct world-plane labels alone do not guarantee independent image
    // data (e.g. different depths project onto the same raw image support).
    // Keep region-level color validation free of shared filter/input pixels
    // on BOTH sides. Cache entries have stable addresses for this snapshot.
    const bool overlaps_used_region = std::any_of(used_regions.begin(), used_regions.end(),
        [&source, &target](const std::pair<const CropFeatures *, const CropFeatures *> &used) {
          return overlappingRawSupport(source, *used.first) || overlappingRawSupport(target, *used.second);
        });
    record(report.stages[Independent], !overlaps_used_region, "raw_support_overlaps_used_region");
    if (overlaps_used_region) continue;
    const auto pairs = match(source, target, report);
    report.matches_max = std::max(report.matches_max, static_cast<int>(pairs.size()));
    M3 prior, forward, reverse;
    std::vector<cv::Point2f> hull;
    ++report.stages[Geometry].entered;
    std::string reason;
    if (!planeHomography(source.chart, target.chart, plane, prior))
    { ++report.stages[Geometry].rejected["plane_homography_failed"]; continue; }
    if (!verifyGeometry(source, target, pairs, prior, forward, reverse, hull, reason))
    { ++report.stages[Geometry].rejected[reason]; continue; }
    ++report.stages[Geometry].passed;
    if (cancelled(cancel)) { report.stop_reason = "cancelled"; break; }
    const size_t previous_samples = samples.size();
    ++report.stages[Sampling].entered;
    takeSamples(source, target, plane, forward, reverse, hull, target_weights, target_id == 0, i, samples, cancel, report, reason);
    if (samples.size() > previous_samples)
    {
      used_regions.emplace_back(&source, &target);
      ++report.stages[Sampling].passed;
    }
    else ++report.stages[Sampling].rejected[reason];
  }
  report.samples = static_cast<int>(samples.size());
  report.regions = static_cast<int>(used_regions.size());
  if (cancelled(cancel)) report.stop_reason = "cancelled";
  else if (samples.size() >= kMaxSamples) report.stop_reason = "sample_budget_4096";
  else if (report.stop_reason.empty()) report.stop_reason = "candidate_list_exhausted";
  return samples;
}
#endif
} // namespace

const char *siftBackend()
{
#if CAKE_HAVE_OPENCV_SIFT
  return "cv::SIFT";
#elif CAKE_HAVE_OPENCV_XFEATURES2D_SIFT
  return "cv::xfeatures2d::SIFT";
#else
  return "unavailable";
#endif
}

bool siftAvailable(std::string &reason)
{
  reason.clear();
#if CAKE_HAVE_OPENCV_SIFT || CAKE_HAVE_OPENCV_XFEATURES2D_SIFT
  try
  {
    if (!createSift().empty()) return true;
    reason = std::string(siftBackend()) + " returned an empty detector";
  }
  catch (const std::exception &error)
  {
    reason = std::string(siftBackend()) + " cannot be created: " + error.what();
  }
#else
  reason = "this OpenCV build has neither supported cv::SIFT nor cv::xfeatures2d::SIFT";
#endif
  return false;
}

Result estimate(const Snapshot &snapshot, const std::atomic<bool> &cancel)
{
  Result result;
  const size_t count = snapshot.cameras.size();
  result.weights.assign(count, defaultWeights());
  result.cameras.resize(count);
#if !CAKE_HAVE_OPENCV_SIFT && !CAKE_HAVE_OPENCV_XFEATURES2D_SIFT
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
          result.pairs.push_back(makePairReport(static_cast<int>(source), static_cast<int>(target)));
          PairReport &pair_report = result.pairs.back();
          const auto samples = pairSamples(snapshot, feature_cache, static_cast<int>(source), static_cast<int>(target), result.weights[target], cancel, pair_report);
          if (cancelled(cancel)) { result.reason = "initialization cancelled"; return result; }
          ++pair_report.stages[Fit].entered;
          const WeightFit fit = fitWeights(samples);
          if (fit.accepted) ++pair_report.stages[Fit].passed;
          else ++pair_report.stages[Fit].rejected[fit.reason];
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

std::vector<std::string> diagnosticLines(const Result &result)
{
  std::vector<std::string> lines;
  if (!result.pairs.empty())
  {
    std::ostringstream settings;
    settings << "small-region settings: feature_context_min=" << kMinFeatureContext
             << " matches_min=" << kMinGeometryMatches << " train_min=" << kMinGeometryTraining
             << " holdout_min=" << kMinGeometryHoldout << " grid=" << kGridSize << 'x' << kGridSize
             << " color_kernel=" << kColorKernelSize << 'x' << kColorKernelSize
             << " color_stride=" << kColorSampleStride;
    lines.push_back(settings.str());
  }
  for (const PairReport &pair : result.pairs)
  {
    const std::string prefix = "camera" + std::to_string(pair.source_camera) + "->camera" + std::to_string(pair.target_camera);
    std::ostringstream summary;
    summary << prefix << " candidates=" << pair.plane_candidates << " regions=" << pair.regions
            << " samples=" << pair.samples << " scan_stop="
            << (pair.stop_reason.empty() ? "interrupted" : pair.stop_reason);
    lines.push_back(summary.str());
    for (const char *unit : {"regions", "matches", "pixels", "samples", "pairs"})
    {
      std::ostringstream counts;
      counts << prefix << ' ' << unit << " pass/entered:";
      int entered = 0;
      for (const StageReport &stage : pair.stages)
      {
        if (stage.unit != unit) continue;
        counts << ' ' << stage.name << '=' << stage.passed << '/' << stage.entered;
        entered += stage.entered;
      }
      if (std::string(unit) == "regions")
        counts << " keypoints_max(src/dst)=" << pair.source_keypoints_max << '/' << pair.target_keypoints_max;
      if (std::string(unit) == "matches") counts << " unique_max_per_region=" << pair.matches_max;
      // Region and fitting counts always show whether the pipeline was reached.
      if (entered || std::string(unit) == "regions" || std::string(unit) == "pairs") lines.push_back(counts.str());
    }
    for (const StageReport &stage : pair.stages)
    {
      int rejected = 0;
      for (const auto &entry : stage.rejected) rejected += entry.second;
      const int unfinished = stage.entered - stage.passed - rejected;
      if (stage.rejected.empty() && unfinished == 0) continue;
      std::ostringstream causes;
      causes << prefix << ' ' << stage.name << " reject (" << stage.unit << "):";
      for (const auto &entry : stage.rejected) causes << ' ' << entry.first << '=' << entry.second;
      if (unfinished) causes << " interrupted=" << unfinished;
      lines.push_back(causes.str());
    }
  }
  return lines;
}

} // namespace color_gray_init
