#include "vio_fisheye.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  VIOManager vio;
  vio.virtual_fisheye_patch_en = true;
  vio.virtual_focal_length = 300.0;
  vio.virtual_min_z = 1e-6;
  vio.virtual_support_radius = 64;

  const V3D directions[] = {
      V3D(0.0, 0.0, 1.0),
      V3D(0.1, -0.2, -1.0).normalized(),
      V3D(1.0, 0.01, 0.01).normalized(),
      V3D(-0.4, 0.8, -0.3).normalized(),
  };

  for (const V3D &direction : directions)
  {
    M3D R_v_from_c, R_c_from_v;
    assert(vio.buildVirtualFrameRotation(direction, R_v_from_c, R_c_from_v));
    assert((R_v_from_c * direction.normalized() - V3D::UnitZ()).norm() < 1e-9);
    assert((R_v_from_c * R_v_from_c.transpose() - M3D::Identity()).norm() < 1e-9);
    assert(std::fabs(R_v_from_c.determinant() - 1.0) < 1e-9);
    assert((R_c_from_v * vio.virtualCam2World(V2D(64.0, 64.0)) - direction.normalized()).norm() < 1e-9);
  }

  Matrix2d affine;
  assert(vio.getWarpMatrixAffineVirtual(V3D(0.0, 0.0, 2.0), SE3<double>(), 0, 0, 4, affine));
  assert((affine - Matrix2d::Identity()).norm() < 1e-9);

  MD(2, 3) analytic;
  const V3D point_v(0.2, -0.1, 2.0);
  vio.computeVirtualProjectionJacobian(point_v, analytic);
  MD(2, 3) numeric;
  const double epsilon = 1e-7;
  for (int axis = 0; axis < 3; ++axis)
  {
    V3D plus = point_v;
    V3D minus = point_v;
    plus[axis] += epsilon;
    minus[axis] -= epsilon;
    numeric.col(axis) = (vio.virtualProject(plus) - vio.virtualProject(minus)) / (2.0 * epsilon);
  }
  assert((analytic - numeric).cwiseAbs().maxCoeff() < 1e-5);
  std::cout << "vio_fisheye_math_test passed\n";
  return 0;
}
