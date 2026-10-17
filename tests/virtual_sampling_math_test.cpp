#include "virtual_sampling.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace vs = virtual_sampling;
using Eigen::Matrix2d;

namespace
{

void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, const char *message)
{
  require(std::isfinite(actual) && std::isfinite(expected) &&
              std::abs(actual - expected) <= 1.0e-10 * std::max(1.0, std::abs(expected)),
          message);
}

Matrix2d rotation(double angle)
{
  Matrix2d result;
  result << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
  return result;
}

Matrix2d footprint(const Matrix2d &B, double additional_sigma)
{
  const Matrix2d inverse = B.inverse();
  return (vs::kPixelSigmaRaw * vs::kPixelSigmaRaw +
          additional_sigma * additional_sigma) * inverse * inverse.transpose();
}

void checkBoundsAndAntialiasing(const Matrix2d &D_ref, const Matrix2d &D_cur,
                              const Matrix2d &A)
{
  const vs::Plan plan = vs::makePlan(D_ref, D_cur, A);
  require(plan.valid, "valid sampling geometry must produce a plan");
  require(plan.pitch >= vs::kMinPitch && plan.pitch <= vs::kMaxPitch,
          "bounded pitch");
  const Matrix2d matrices[] = {D_ref * A.inverse(), D_cur};
  const double sigmas[] = {plan.sigma_ref, plan.sigma_cur};
  const double pi = std::acos(-1.0);
  const double nyquist_variance =
      2.0 * plan.pitch * plan.pitch * std::log(1.0 / vs::kNyquistAttenuation) / (pi * pi);
  for (int index = 0; index < 2; ++index)
  {
    require(std::isfinite(sigmas[index]) && sigmas[index] >= 0.0,
            "finite additional Gaussian sigma");
    const Matrix2d native = footprint(matrices[index], 0.0);
    const Matrix2d filtered = footprint(matrices[index], sigmas[index]);
    const Eigen::SelfAdjointEigenSolver<Matrix2d> added(filtered - native);
    require(added.eigenvalues().minCoeff() >= -1.0e-10,
            "additional filtering covariance is PSD");
    const Eigen::SelfAdjointEigenSolver<Matrix2d> spectrum(filtered);
    require(spectrum.eigenvalues().minCoeff() >= nyquist_variance - 1.0e-10,
            "actual clamped pitch satisfies Gaussian Nyquist attenuation");
    const double attenuation = std::exp(-0.5 * spectrum.eigenvalues().minCoeff() *
                                         pi * pi / (plan.pitch * plan.pitch));
    require(attenuation <= vs::kNyquistAttenuation + 1.0e-10,
            "Nyquist amplitude bound");
  }
}

void testIdentityAndForwardBackwardScale()
{
  const Matrix2d identity = Matrix2d::Identity();
  const vs::Plan base = vs::makePlan(identity, identity, identity);
  require(base.valid, "identity plan");
  near(base.pitch, 1.0, "native identity pitch");
  near(base.sigma_ref, base.sigma_cur, "identity has equal filtering");
  require(base.sigma_ref > 0.0, "identity includes target antialiasing blur");

  for (double magnification : {0.5, 1.0, 2.0})
  {
    const Matrix2d A = magnification * identity;
    const vs::Plan plan = vs::makePlan(identity, identity, A);
    require(plan.valid, "forward/backward scale plan");
    near(plan.pitch, std::max(1.0, magnification), "coarsest common pitch");
    const Matrix2d ref_footprint = footprint(identity / magnification, plan.sigma_ref);
    const Matrix2d cur_footprint = footprint(identity, plan.sigma_cur);
    require((ref_footprint - cur_footprint).norm() < 1.0e-10,
            "isotropic forward/backward footprints match");
    checkBoundsAndAntialiasing(identity, identity, A);
  }
}

void testReferenceCurrentSwap()
{
  const Matrix2d D_ref = 0.8 * Matrix2d::Identity();
  const Matrix2d D_cur = 1.2 * Matrix2d::Identity();
  const Matrix2d A = 1.5 * rotation(0.4);
  const vs::Plan original = vs::makePlan(D_ref, D_cur, A);
  const vs::Plan swapped = vs::makePlan(D_cur, D_ref, A.inverse());
  require(original.valid && swapped.valid, "both reference assignments valid");
  near(original.sigma_ref, swapped.sigma_cur, "swap retains reference raw kernel");
  near(original.sigma_cur, swapped.sigma_ref, "swap retains current raw kernel");
  near(original.pitch, 1.5 * swapped.pitch, "pitch transforms with common coordinates");
}

void testCoordinateRescaleAndClamps()
{
  const Matrix2d identity = Matrix2d::Identity();
  const vs::Plan base = vs::makePlan(identity, identity, identity);
  for (double coordinate_scale : {0.5, 2.0})
  {
    const vs::Plan scaled = vs::makePlan(identity / coordinate_scale,
                                      identity / coordinate_scale, identity);
    require(scaled.valid, "coordinate rescale plan");
    near(scaled.pitch, coordinate_scale * base.pitch, "unclamped coordinate pitch");
    near(scaled.sigma_ref, base.sigma_ref, "rescaling coordinates leaves raw filtering unchanged");
    near(scaled.sigma_cur, base.sigma_cur, "rescaling coordinates leaves raw filtering unchanged");
  }
  // Bounds are in canonical virtual-coordinate units. Once active they do
  // change the representation, but must not violate the antialiasing bound.
  near(vs::makePlan(10.0 * identity, 10.0 * identity, identity).pitch,
       vs::kMinPitch, "minimum pitch clamp");
  near(vs::makePlan(0.1 * identity, 0.1 * identity, identity).pitch,
       vs::kMaxPitch, "maximum pitch clamp");
  checkBoundsAndAntialiasing(10.0 * identity, 10.0 * identity, identity);
  checkBoundsAndAntialiasing(0.1 * identity, 0.1 * identity, identity);
}

void testAnisotropyAndInvalidGeometry()
{
  Matrix2d A;
  A << 1.3, 0.2, -0.1, 0.9;
  const Matrix2d D_ref = rotation(0.3) * Eigen::Vector2d(2.0, 0.8).asDiagonal();
  const Matrix2d D_cur = rotation(-0.2) * Eigen::Vector2d(1.5, 0.6).asDiagonal();
  const vs::Plan anisotropic = vs::makePlan(D_ref, D_cur, A);
  require(anisotropic.valid && anisotropic.anisotropy_ref > 1.0 &&
              anisotropic.anisotropy_cur > 1.0,
          "anisotropy is reported, not rejected");
  checkBoundsAndAntialiasing(D_ref, D_cur, A);

  const Matrix2d identity = Matrix2d::Identity();
  require(!vs::makePlan(identity, identity, Matrix2d::Zero()).valid,
          "singular affine map is unsupported");
  require(!vs::makePlan(Matrix2d::Zero(), identity, identity).valid,
          "singular reference projection is unsupported");
  require(!vs::makePlan(identity, Matrix2d::Zero(), identity).valid,
          "singular current projection is unsupported");
  Matrix2d invalid = identity;
  invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
  require(!vs::makePlan(identity, identity, invalid).valid, "NaN affine map");
  invalid(0, 0) = std::numeric_limits<double>::infinity();
  require(!vs::makePlan(invalid, identity, identity).valid, "infinite projection");
  invalid = Eigen::Vector2d(1.0, 1.0e-12).asDiagonal();
  require(!vs::makePlan(identity, invalid, identity).valid,
          "numerically singular projection cannot define a stable sampling plan");
}

void testSymmetricRangeCoverage()
{
  require(vs::distanceScaleCovered(2.0, 2.0), "equal range covered");
  near(vs::scalePenalty(2.0, 2.0), 0.0, "equal range has zero ranking penalty");
  for (double ratio : {1.0 / std::sqrt(2.0), std::sqrt(2.0)})
  {
    require(vs::distanceScaleCovered(2.0, 2.0 * ratio), "coverage includes boundary");
    near(vs::scalePenalty(2.0, 2.0 * ratio), 1.0, "normalized boundary penalty");
  }
  require(!vs::distanceScaleCovered(2.0, 4.0) && !vs::distanceScaleCovered(4.0, 2.0),
          "large scale change is uncovered in either direction");
  near(vs::scalePenalty(2.0, 4.0), vs::scalePenalty(4.0, 2.0), "symmetric ranking");
  near(vs::scalePenalty(1.0e300, 2.0e300), 2.0, "large finite ranges avoid ratio overflow");
  require(!vs::distanceScaleCovered(0.0, 1.0), "zero range uncovered");
  require(!vs::distanceScaleCovered(-1.0, 1.0), "negative range uncovered");
  require(std::isinf(vs::scalePenalty(0.0, 1.0)), "invalid range has infinite ranking penalty");
}

}  // namespace

int main()
{
  testIdentityAndForwardBackwardScale();
  testReferenceCurrentSwap();
  testCoordinateRescaleAndClamps();
  testAnisotropyAndInvalidGeometry();
  testSymmetricRangeCoverage();
  std::cout << "virtual_sampling_math_test passed\n";
  return 0;
}
