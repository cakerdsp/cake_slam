/*
 * equidistant_camera.cpp
 *
 *  Created on: January 26, 2023
 *      Author: xuankuzcr
 */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vikit/equidistant_camera.h>
#include <vikit/math_utils.h>

namespace vk {

EquidistantCamera::
EquidistantCamera(double width, double height, double scale,
              double fx, double fy,
              double cx, double cy,
              double k1, double k2, double k3, double k4) :
              AbstractCamera(width * scale , height * scale, scale),
              fx_(fx * scale), fy_(fy * scale), cx_(cx * scale), cy_(cy * scale),
              // distortion_(fabs(k1) > 0.0000001)
              distortion_(fabs(k1) > 1e-7 || fabs(k2) > 1e-7 || fabs(k3) > 1e-7 || fabs(k4) > 1e-7)
{
  cout << "scale: " << scale << endl;
  k1_ = k1; k2_ = k2; k3_ = k3; k4_ = k4;
}

EquidistantCamera::
~EquidistantCamera()
{}

Vector3d EquidistantCamera::
cam2world(const double& u, const double& v) const
{
  // Vector3d xyz;
  // if(!distortion_)
  // {
  //   xyz[0] = (u - cx_)/fx_;
  //   xyz[1] = (v - cy_)/fy_;
  //   xyz[2] = 1.0;
  // }
  // else
  // {
  //   double x = (u - cx_)/fx_;
  //   double y = (v - cy_)/fy_;
  //   const double thetad = std::sqrt(x * x + y * y);
  //   double theta = thetad;
  //   for (int i = 0; i < 5; ++i)
  //   {
  //     const double theta2 = theta * theta;
  //     const double theta4 = theta2 * theta2;
  //     const double theta6 = theta4 * theta2;
  //     const double theta8 = theta4 * theta4;
  //     theta = thetad /
  //             (1.0 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);
  //   }
  //   const double scaling = std::tan(theta) / thetad;
  //   x *= scaling;
  //   y *= scaling;
  //   xyz[0] = x;
  //   xyz[1] = y;
  //   xyz[2] = 1.0;
  // }
  // return xyz.normalized();
  if (!distortion_)
  {
    return Vector3d(
      (u - cx_) / fx_,
      (v - cy_) / fy_,
      1.0).normalized();
  }

  const double x = (u - cx_) / fx_;
  const double y = (v - cy_) / fy_;
  const double thetad = std::sqrt(x * x + y * y);

  if (thetad < 1e-12)
  {
    return Vector3d::UnitZ();
  }

  double theta = thetad;
  for (int i = 0; i < 10; ++i)
  {
    const double derivative = deriv_thetad_from_theta(theta);
    if (!std::isfinite(derivative) || std::abs(derivative) < 1e-12)
    {
      return Vector3d::Zero();
    }

    const double step =
      (thetad_from_theta(theta) - thetad) / derivative;
    theta -= step;

    if (std::abs(step) < 1e-12)
    {
      break;
    }
  }

  if (!std::isfinite(theta) ||
      theta < 0.0 ||
      theta > 3.14159265358979323846 ||
      deriv_thetad_from_theta(theta) <= 0.0 ||
      std::abs(thetad_from_theta(theta) - thetad) > 1e-8)
  {
    return Vector3d::Zero();
  }

  const double scale = std::sin(theta) / thetad;

  return Vector3d(
    x * scale,
    y * scale,
    std::cos(theta));
}

Vector3d EquidistantCamera::
cam2world (const Vector2d& uv) const
{
  return cam2world(uv[0], uv[1]);
}

Vector2d EquidistantCamera::
world2cam(const Vector3d& xyz) const
{
  // return world2cam(project2d(xyz));
  const double norm = xyz.norm();
  if (!std::isfinite(norm) || norm < 1e-12)
  {
    return Vector2d(-1.0, -1.0);
  }

  if (!distortion_)
  {
    if (xyz[2] <= 1e-12)
    {
      return Vector2d(-1.0, -1.0);
    }

    return Vector2d(
      fx_ * xyz[0] / xyz[2] + cx_,
      fy_ * xyz[1] / xyz[2] + cy_);
  }

  const double radius =
    std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);

  if (radius < 1e-12)
  {
    return xyz[2] > 0.0
      ? Vector2d(cx_, cy_)
      : Vector2d(-1.0, -1.0);
  }

  const double theta = std::atan2(radius, xyz[2]);
  const double thetad = thetad_from_theta(theta);

  if (!std::isfinite(thetad) ||
      thetad < 0.0 ||
      deriv_thetad_from_theta(theta) <= 0.0)
  {
    return Vector2d(-1.0, -1.0);
  }

  const double scale = thetad / radius;

  return Vector2d(
    fx_ * xyz[0] * scale + cx_,
    fy_ * xyz[1] * scale + cy_);
}

Vector2d EquidistantCamera::
world2cam(const Vector2d& uv) const
{
  Vector2d px;
  if(!distortion_)
  {
    px[0] = fx_*uv[0] + cx_;
    px[1] = fy_*uv[1] + cy_;
  }
  else
  {
    double xd, yd;
    const double r = uv.norm();
    if (r < 1e-8)
    {
      // return uv;
      return Vector2d(cx_, cy_);
    }
    const double theta = std::atan(r);
    const double thetad = thetad_from_theta(theta);
    const double scaling = thetad / r;
    xd = uv[0] * scaling;
    yd = uv[1] * scaling;
    px[0] = xd*fx_ + cx_;
    px[1] = yd*fy_ + cy_;
  }
  return px;
}

} // end namespace vk
