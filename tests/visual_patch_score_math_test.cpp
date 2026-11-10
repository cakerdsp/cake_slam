#include "visual_patch_score.h"

#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

static Eigen::Vector2d project(const Eigen::Vector3d &point)
{
  return point.head<2>() / point.z();
}

static Eigen::Matrix<double, 1, 6> planeJacobian(const Eigen::Vector3d &point_world,
                                               const Eigen::Vector3d &normal_world,
                                               const Eigen::Matrix3d &rotation,
                                               const Eigen::Vector3d &position)
{
  Eigen::Matrix<double, 1, 6> row;
  const Eigen::Vector3d point_body = rotation.transpose() * (point_world - position);
  const Eigen::Vector3d normal_body = rotation.transpose() * normal_world;
  row.head<3>() = point_body.cross(normal_body).transpose();
  row.tail<3>() = normal_world.transpose();
  return row;
}

int main()
{
  using visual_patch_score::PoseMatrix;
  using visual_patch_score::PoseProjection;
  visual_patch_score::Settings settings;
  require(settings.valid(), "invalid default regularization");
  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(.5, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  const Eigen::Matrix3d camera_rotation = Eigen::AngleAxisd(-.3, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d time_rotation = Eigen::AngleAxisd(.12, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d position(.5, -.2, .3), camera_translation(.2, .1, -.04), time_shift(.02, -.01, .03);
  const Eigen::Vector3d point_camera(.8, -.4, 4.0);
  const Eigen::Vector3d point_world = rotation * time_rotation * camera_rotation.transpose() *
      (point_camera - camera_translation) + position + time_shift;
  Eigen::Matrix<double, 2, 3> projection;
  projection << 1.0 / point_camera.z(), 0, -point_camera.x() / std::pow(point_camera.z(), 2),
                0, 1.0 / point_camera.z(), -point_camera.y() / std::pow(point_camera.z(), 2);
  const Eigen::Vector3d camera_origin = -camera_rotation.transpose() * camera_translation;
  const PoseProjection pose_projection = visual_patch_score::poseProjection(
      projection, point_camera, camera_rotation * time_rotation.transpose(),
      -camera_rotation * visual_patch_score::crossMatrix(camera_origin) * time_rotation.transpose(),
      camera_rotation * (rotation * time_rotation).transpose());

  // Independent finite differences include the extrinsic lever arm and a
  // nonzero camera time offset, with right rotation and world translation.
  constexpr double epsilon = 1e-6;
  for (int axis = 0; axis < 6; ++axis)
  {
    auto perturbedPixel = [&](double sign) -> Eigen::Vector2d {
      Eigen::Matrix3d perturbed_rotation = rotation;
      Eigen::Vector3d perturbed_position = position;
      if (axis < 3)
        perturbed_rotation = rotation * Eigen::AngleAxisd(sign * epsilon,
                                                          Eigen::Vector3d::Unit(axis)).toRotationMatrix();
      else
        perturbed_position[axis - 3] += sign * epsilon;
      return project(camera_rotation * (perturbed_rotation * time_rotation).transpose() *
                     (point_world - perturbed_position - time_shift) + camera_translation);
    };
    const Eigen::Vector2d numerical = (perturbedPixel(1) - perturbedPixel(-1)) / (2 * epsilon);
    require((numerical - pose_projection.col(axis)).norm() < 1e-8, "pose projection finite difference");
  }

  Eigen::Matrix<double, 6, 1> weak = Eigen::Matrix<double, 6, 1>::Zero();
  weak[0] = weak[4] = 1.0 / std::sqrt(2.0);
  const PoseMatrix normalized = PoseMatrix::Identity() - weak * weak.transpose();
  visual_patch_score::GeometrySummary geometry;
  geometry.information = normalized;
  geometry.displacement_metric.setIdentity();
  geometry.weight_sum = 1.0;
  geometry.world_from_body = rotation;
  geometry.body_position = position;
  geometry.count = 100;
  geometry.stamp = 10;
  geometry.valid = true;
  const auto directional = visual_patch_score::directionalWeight(geometry, rotation, position, settings);
  require(directional.valid, "valid geometry rejected");
  require((directional.weight * weak - weak).norm() < 1e-10, "coupled weak mode lost");
  require(std::abs(directional.weight(2, 2) - 1.0 / 11.0) < 1e-10, "strong mode weight");
  PoseProjection weak_response = PoseProjection::Zero(), strong_response = PoseProjection::Zero();
  weak_response.row(0) = weak.transpose();
  strong_response(0, 2) = 1;
  require(visual_patch_score::sensitivity(Eigen::Matrix2d::Identity(), weak_response, directional.weight, 1, 1) >
          visual_patch_score::sensitivity(Eigen::Matrix2d::Identity(), strong_response, directional.weight, 1, 1),
          "complementary motion not prioritized");
  auto duplicated = geometry;
  duplicated.information *= 10;
  duplicated.displacement_metric *= 10;
  duplicated.weight_sum *= 10;
  duplicated.count *= 10;
  require((visual_patch_score::directionalWeight(duplicated, rotation, position, settings).weight -
           directional.weight).norm() < 1e-10, "uniform measurement duplication changes directional weights");

  constexpr int width = 24, height = 20, stride = 31, u = 12, v = 10;
  std::vector<uint8_t> image(stride * height, 0);
  std::mt19937 random(9);
  for (auto &value : image) value = random() % 256;
  const auto tensor = visual_patch_score::gradientTensor8x8(image.data(), stride, width, height, u, v);
  const double exposure = 1.3, variance = 1000.0;
  PoseMatrix exact = PoseMatrix::Zero();
  for (int y = v - 4; y < v + 4; ++y)
    for (int x = u - 4; x < u + 4; ++x)
    {
      Eigen::RowVector2d gradient;
      gradient << .5 * (image[y * stride + x + 1] - image[y * stride + x - 1]),
                  .5 * (image[(y + 1) * stride + x] - image[(y - 1) * stride + x]);
      const Eigen::Matrix<double, 1, 6> jacobian = exposure * gradient * pose_projection;
      exact += jacobian.transpose() * jacobian / (64 * variance);
    }
  const double score = visual_patch_score::sensitivity(tensor, pose_projection, directional.weight, exposure, variance);
  require(std::abs(score - (directional.weight * exact).trace()) < 1e-10, "six-DoF pixel energy mismatch");
  require(score >= 0, "negative PSD score");
  require(std::abs(score - visual_patch_score::sensitivity(9 * tensor, pose_projection, directional.weight,
                                                        exposure / 3, variance)) < 1e-10, "exposure invariance");
  require(visual_patch_score::sensitivity(tensor, pose_projection, PoseMatrix::Identity(), exposure, variance) >= score,
          "directional weight exceeds isotropic weight");
  require(visual_patch_score::gradientTensor8x8(image.data(), stride, width, height, 0, 0).isZero(), "border guard");
  std::fill(image.begin(), image.end(), 42);
  require(visual_patch_score::gradientTensor8x8(image.data(), stride, width, height, u, v).isZero(), "flat patch");

  // Snapshot transport must equal directly rebuilt plane Jacobians at the
  // new pose, including the change of rotation centre.
  const Eigen::Matrix3d current_rotation = rotation * time_rotation;
  const Eigen::Vector3d current_position = position + Eigen::Vector3d(.8, -.5, .2);
  geometry.information.setZero();
  geometry.displacement_metric.setZero();
  geometry.weight_sum = 0.0;
  PoseMatrix rebuilt = PoseMatrix::Zero();
  PoseMatrix rebuilt_metric = PoseMatrix::Zero();
  for (int i = 0; i < 30; ++i)
  {
    const Eigen::Vector3d point(.1 * i, std::sin(i), 2 + std::cos(i));
    const Eigen::Vector3d normal = Eigen::Vector3d(std::cos(i), .3, std::sin(i)).normalized();
    const auto before = planeJacobian(point, normal, rotation, position);
    const auto after = planeJacobian(point, normal, current_rotation, current_position);
    const double weight = 1.0 + .05 * i;
    geometry.information += weight * before.transpose() * before;
    rebuilt += weight * after.transpose() * after;
    visual_patch_score::accumulateDisplacementMetric(geometry, rotation.transpose() * (point - position), rotation, weight);
    Eigen::Matrix<double, 3, 6> point_jacobian;
    point_jacobian.leftCols<3>() = -visual_patch_score::crossMatrix(point - current_position) * current_rotation;
    point_jacobian.rightCols<3>().setIdentity();
    rebuilt_metric += weight * point_jacobian.transpose() * point_jacobian;
  }
  const PoseMatrix transport = visual_patch_score::currentToSnapshot(geometry, current_rotation, current_position);
  require((transport.transpose() * geometry.information * transport - rebuilt).norm() < 1e-10,
          "snapshot rotation/origin transport mismatch");
  require((transport.transpose() * geometry.displacement_metric * transport - rebuilt_metric).norm() < 1e-10,
          "point metric transport mismatch");
  const auto current_weight = visual_patch_score::directionalWeight(geometry, current_rotation, current_position, settings);
  require(current_weight.valid && current_weight.lambda_min >= 0 && current_weight.lambda_max <= 1,
          "generalized spectrum outside [0,1]");
  const PoseMatrix mean_metric = rebuilt_metric / geometry.weight_sum;
  const PoseMatrix mean_information = rebuilt / geometry.weight_sum;
  // Independent closed form: C alpha(N+alpha I)^-1 C^T = alpha(H+alpha M)^-1.
  const PoseMatrix expected_weight = current_weight.alpha *
      (mean_information + current_weight.alpha * mean_metric).inverse();
  require((current_weight.weight - expected_weight).norm() < 1e-9, "generalized soft weight mismatch");
  const auto isotropic = visual_patch_score::directionalWeight(geometry, current_rotation, current_position, settings, false);
  require(isotropic.valid && (isotropic.weight * mean_metric - PoseMatrix::Identity()).norm() < 1e-9,
          "isotropic mode must use inverse displacement metric");
  require(visual_patch_score::sensitivity(tensor, pose_projection, isotropic.weight, exposure, variance) + 1e-10 >=
          visual_patch_score::sensitivity(tensor, pose_projection, current_weight.weight, exposure, variance),
          "directional score exceeds metric-isotropic score");
  const Eigen::Vector3d origin_change(100, -50, 20);
  auto transformed = geometry;
  PoseMatrix coordinate_change = PoseMatrix::Identity();
  coordinate_change.bottomRightCorner<3, 3>() = camera_rotation.transpose();
  transformed.information = coordinate_change.transpose() * geometry.information * coordinate_change;
  transformed.displacement_metric = coordinate_change.transpose() * geometry.displacement_metric * coordinate_change;
  transformed.world_from_body = camera_rotation * rotation;
  transformed.body_position = camera_rotation * position + origin_change;
  const auto transformed_weight = visual_patch_score::directionalWeight(
      transformed, camera_rotation * current_rotation, camera_rotation * current_position + origin_change, settings);
  require(transformed_weight.valid, "world-frame transformed geometry invalid");
  PoseProjection transformed_projection = pose_projection;
  transformed_projection.rightCols<3>() = pose_projection.rightCols<3>() * camera_rotation.transpose();
  require(std::abs(visual_patch_score::sensitivity(tensor, pose_projection, current_weight.weight, exposure, variance) -
                   visual_patch_score::sensitivity(tensor, transformed_projection, transformed_weight.weight, exposure, variance)) < 1e-9,
          "score changes under world-frame rotation/translation");

  // Represent the same frozen geometry at another body origin. Both forms and
  // the visual Jacobian must transform together; the physical score is fixed.
  auto recentered = geometry;
  recentered.information = rebuilt;
  recentered.displacement_metric = rebuilt_metric;
  recentered.world_from_body = current_rotation;
  recentered.body_position = current_position;
  const auto old_weight = visual_patch_score::directionalWeight(geometry, rotation, position, settings);
  const auto new_weight = visual_patch_score::directionalWeight(recentered, current_rotation, current_position, settings);
  require(old_weight.valid && new_weight.valid, "origin shift rejected");
  require(std::abs(old_weight.lambda_min - new_weight.lambda_min) < 1e-10 &&
          std::abs(old_weight.lambda_max - new_weight.lambda_max) < 1e-10, "origin-dependent spectrum");
  require(std::abs(visual_patch_score::sensitivity(tensor, pose_projection, old_weight.weight, exposure, variance) -
          visual_patch_score::sensitivity(tensor, pose_projection * transport, new_weight.weight, exposure, variance)) < 1e-9,
          "origin-dependent physical score");

  // A uniform length-unit conversion preserves eigenvalues and rankings;
  // sensitivity per squared length changes by the common factor 1/scale^2.
  constexpr double length_scale = 1000.0;
  PoseMatrix units = PoseMatrix::Identity();
  units.bottomRightCorner<3, 3>() /= length_scale;
  auto scaled = geometry;
  scaled.information = length_scale * length_scale * units.transpose() * geometry.information * units;
  scaled.displacement_metric = length_scale * length_scale * units.transpose() * geometry.displacement_metric * units;
  scaled.body_position *= length_scale;
  const auto scaled_weight = visual_patch_score::directionalWeight(scaled, rotation, scaled.body_position, settings);
  require(scaled_weight.valid && std::abs(scaled_weight.lambda_min - old_weight.lambda_min) < 1e-10 &&
          std::abs(scaled_weight.lambda_max - old_weight.lambda_max) < 1e-10, "unit-dependent spectrum");
  require(std::abs(visual_patch_score::sensitivity(tensor, pose_projection * units, scaled_weight.weight, exposure, variance) *
          length_scale * length_scale - visual_patch_score::sensitivity(tensor, pose_projection, old_weight.weight, exposure, variance)) < 1e-9,
          "unit conversion changes ranking scale inconsistently");

  // Coplanar points still define a full displacement metric; collinear points
  // do not. Do not manufacture a length scale by adding epsilon*I to M.
  visual_patch_score::GeometrySummary plane;
  plane.valid = true; plane.count = 36;
  for (int y = 0; y < 6; ++y)
    for (int x = 0; x < 6; ++x)
    {
      const Eigen::Vector3d point(x - 2.5, y - 2.5, 3);
      const auto row = planeJacobian(point, Eigen::Vector3d::UnitZ(), Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
      plane.information += row.transpose() * row;
      visual_patch_score::accumulateDisplacementMetric(plane, point, Eigen::Matrix3d::Identity(), 1);
    }
  require(visual_patch_score::directionalWeight(plane, plane.world_from_body, plane.body_position, settings).valid,
          "coplanar points incorrectly rejected");
  auto line = plane;
  line.information.setZero(); line.displacement_metric.setZero(); line.weight_sum = 0;
  for (int i = 0; i < 36; ++i)
    visual_patch_score::accumulateDisplacementMetric(line, Eigen::Vector3d(i, 0, 0), Eigen::Matrix3d::Identity(), 1);
  require(!visual_patch_score::directionalWeight(line, line.world_from_body, line.body_position, settings).valid,
          "collinear metric accepted");
  line.displacement_metric.setZero(); line.weight_sum = 0;
  for (int i = 0; i < 36; ++i)
    visual_patch_score::accumulateDisplacementMetric(line,
        Eigen::Vector3d(i, 1e-7 * std::sin(i), 1e-7 * std::cos(i)), Eigen::Matrix3d::Identity(), 1);
  require(!visual_patch_score::directionalWeight(line, line.world_from_body, line.body_position, settings).valid,
          "nearly collinear metric accepted");

  require(visual_patch_score::fresh(geometry, 10.1) && !visual_patch_score::fresh(geometry, 11), "stale geometry");
  require(!visual_patch_score::fresh(geometry, 9.9), "future geometry");
  geometry.count = 29;
  const auto fallback = visual_patch_score::directionalWeight(geometry, rotation, position, settings);
  require(!fallback.valid && fallback.weight.isZero(), "insufficient geometry must request score fallback");
  geometry.count = 100;
  geometry.information.setZero();
  require(visual_patch_score::directionalWeight(geometry, rotation, position, settings).valid, "all-zero information handling");
  geometry.information.setIdentity();
  geometry.information(0, 0) = -1;
  require(!visual_patch_score::directionalWeight(geometry, rotation, position, settings).valid, "indefinite information");
  geometry.information(0, 0) = std::numeric_limits<double>::quiet_NaN();
  require(!visual_patch_score::directionalWeight(geometry, rotation, position, settings).valid, "nonfinite information");
  geometry.information = 2 * geometry.displacement_metric;
  require(!visual_patch_score::directionalWeight(geometry, rotation, position, settings).valid, "H greater than M accepted");
  settings.regularization_ratio = 0;
  require(!settings.valid(), "zero regularization accepted");
  std::cout << "point metric, generalized spectrum, six-DoF score and fallback checks passed\n";
}
