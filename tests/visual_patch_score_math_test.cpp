#include "visual_patch_score.h"

#include <iostream>
#include <random>
#include <stdexcept>

static void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

int main()
{
  constexpr int width = 24, height = 20, stride = 31, u = 12, v = 10;
  std::vector<uint8_t> image(stride * height, 0);
  std::mt19937 random(9);
  for (auto &value : image) value = random() % 256;
  const Eigen::Matrix2d tensor = visual_patch_score::gradientTensor8x8(image.data(), stride, width, height, u, v);
  Eigen::Matrix<double, 2, 3> projection;
  projection << .7, .2, -.15, -.1, .6, .3;
  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(.5, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  const Eigen::Matrix3d world_weight = (Eigen::Vector3d(1, .1, .4)).asDiagonal();
  const Eigen::Matrix3d weight_camera = rotation * world_weight * rotation.transpose();
  const double exposure = 1.3, variance = 1000.0;
  Eigen::Matrix3d exact = Eigen::Matrix3d::Zero();
  for (int y = v - 4; y < v + 4; ++y)
    for (int x = u - 4; x < u + 4; ++x)
    {
      Eigen::RowVector2d gradient;
      gradient << .5 * (image[y * stride + x + 1] - image[y * stride + x - 1]),
                  .5 * (image[(y + 1) * stride + x] - image[(y - 1) * stride + x]);
      const Eigen::RowVector3d jacobian = exposure * gradient * projection * rotation;
      exact += jacobian.transpose() * jacobian / (64 * variance);
    }
  const double score = visual_patch_score::sensitivity(tensor, projection, weight_camera, exposure, variance);
  require(std::abs(score - (world_weight * exact).trace()) < 1e-10, "score differs from explicit 64-pixel Jacobian energy");
  require(score >= 0, "negative PSD score");
  const double scaled = visual_patch_score::sensitivity(9 * tensor, projection, weight_camera, exposure / 3, variance);
  require(std::abs(score - scaled) < 1e-10, "brightness/exposure scale invariance");
  const double iso = visual_patch_score::sensitivity(tensor, projection, Eigen::Matrix3d::Identity(), exposure, variance);
  require(iso >= score, "directional weight bounded by identity");
  require(visual_patch_score::gradientTensor8x8(image.data(), stride, width, height, 0, 0).isZero(), "border guard");
  std::fill(image.begin(), image.end(), 42);
  require(visual_patch_score::gradientTensor8x8(image.data(), stride, width, height, u, v).isZero(), "flat patch");
  std::vector<visual_patch_score::GeometrySample> normals(100);
  for (auto &sample : normals) visual_patch_score::consider(&sample, Eigen::Vector3d(0, 0, 2), .1);
  auto geometry = visual_patch_score::summarize(normals);
  require(geometry.valid && geometry.count == 100, "geometric sample count");
  require(std::abs(geometry.weight(0, 0) - 1) < 1e-12 && std::abs(geometry.weight(2, 2) - 1.0 / 11) < 1e-12,
          "plane normal did not identify weak tangential translations");
  for (auto &sample : normals) sample.normal = rotation * sample.normal;
  const auto rotated = visual_patch_score::summarize(normals);
  require((rotated.weight - rotation * geometry.weight * rotation.transpose()).norm() < 1e-12, "world frame invariance");
  geometry.stamp = 10;
  require(visual_patch_score::fresh(geometry, 10.1) && !visual_patch_score::fresh(geometry, 11), "stale geometry");
  normals.resize(29);
  require(!visual_patch_score::summarize(normals).valid, "insufficient geometry must use isotropic fallback");
  visual_patch_score::GeometrySample sample;
  visual_patch_score::consider(&sample, Eigen::Vector3d::UnitX(), .4);
  require(!sample.valid, "geometric cutoff");
  visual_patch_score::consider(&sample, Eigen::Vector3d::UnitX(), .1);
  visual_patch_score::consider(&sample, Eigen::Vector3d::UnitY(), .2);
  require(sample.normal == Eigen::Vector3d::UnitX(), "closest geometric plane must be retained");
  std::cout << "explicit Jacobian, exposure/frame invariance and P-free geometry tests passed\n";
}
