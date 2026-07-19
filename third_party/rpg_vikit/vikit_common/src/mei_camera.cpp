#include <stdio.h>
#include <cmath>
#include <vikit/mei_camera.h>

namespace vk {

MEICamera::MEICamera(double width, double height, double scale,
          double fx, double fy, double cx, double cy,
          double xi, double k1, double k2, double p1, double p2) :
          AbstractCamera(width * scale, height * scale, scale),
          fx_(fx * scale), fy_(fy * scale), cx_(cx * scale), cy_(cy * scale),
          xi_(xi), k1_(k1), k2_(k2), p1_(p1), p2_(p2),
          distortion_(std::abs(k1) > 1e-9 || std::abs(k2) > 1e-9 || std::abs(p1) > 1e-9 || std::abs(p2) > 1e-9)
{
  w_ = (xi_ <= 1.0) ? xi_ : (1.0 / xi_);
  printf("MEICamera Initialized (Scalar Mode): Res=%dx%d, fx=%.2f, xi=%.4f\n", (int)width_, (int)height_, fx_, xi_);
}

MEICamera::~MEICamera() {

}

Eigen::Vector3d MEICamera::cam2world(const double& u, const double& v) const {
  // 1. 鏉烆剚宕查崚鏉跨秺娑撯偓閸栨牕閽╅棃?(閸掓繂鈧?
  double mx = (u - cx_) / fx_;
  double my = (v - cy_) / fy_;
  
  // 2. 缁?C 閺嶅洭鍣烘潻顓濆敩閸樿崵鏆╅崣?
  removeDistortion(mx, my);
  
  double r2 = mx*mx + my*my;
  double term = 1.0 + (1.0 - xi_*xi_) * r2;
  
  if (term < 0) return Eigen::Vector3d(0, 0, 0);

  double scale = (xi_ + std::sqrt(term)) / (1.0 + r2);
  
  // 3. 閺嬪嫬缂撶紒鎾寸亯楠炶泛缍婃稉鈧崠?
  Eigen::Vector3d res(scale * mx, scale * my, scale - xi_);
  // double norm = std::sqrt(res[0]*res[0] + res[1]*res[1] + res[2]*res[2]);
  // if (norm > 1e-10) {
  //     res /= norm;
  // }
  return res;
}

Eigen::Vector3d MEICamera::cam2world(const Eigen::Vector2d& px) const {
  return cam2world(px[0], px[1]);
}

Eigen::Vector2d MEICamera::world2cam(const Eigen::Vector3d& xyz) const {
  double x = xyz[0], y = xyz[1], z = xyz[2];
  double d = std::sqrt(x*x + y*y + z*z); 
  
  if (z <= -w_ * d + 1e-6) {
    return Eigen::Vector2d(-1, -1);
  }

  double common_div = z + xi_ * d;

  // if (common_div <= 1e-6 || (xi_ * z + d <= 0)) {
  //   return Eigen::Vector2d(-1, -1);
  // }

  // 瑜版帊绔撮崠鏍ч挬闂堛垹娼楅弽?
  double mx = x / common_div;
  double my = y / common_div;

  // 閹靛濮╂惔鏃傛暏閻ｇ褰?
  if (distortion_) {
    double r2 = mx*mx + my*my;
    double r4 = r2*r2;
    double radial = 1.0 + k1_*r2 + k2_*r4;
    double mx_dist = mx * radial + 2.0*p1_*mx*my + p2_*(r2 + 2.0*mx*mx);
    double my_dist = my * radial + p1_*(r2 + 2.0*my*my) + 2.0*p2_*mx*my;
    mx = mx_dist;
    my = my_dist;
  }

  return Eigen::Vector2d(mx * fx_ + cx_, my * fy_ + cy_);
}

Eigen::Vector2d MEICamera::world2cam(const Eigen::Vector2d& uv) const {
  double mx = uv[0];
  double my = uv[1];

  if (distortion_) {
    double r2 = mx*mx + my*my;
    double r4 = r2*r2;
    double radial = 1.0 + k1_*r2 + k2_*r4;
    double mx_dist = mx * radial + 2.0*p1_*mx*my + p2_*(r2 + 2.0*mx*mx);
    double my_dist = my * radial + p1_*(r2 + 2.0*my*my) + 2.0*p2_*mx*my;
    mx = mx_dist;
    my = my_dist;
  }
  return Eigen::Vector2d(mx * fx_ + cx_, my * fy_ + cy_);
}

Eigen::Matrix<double, 2, 3> MEICamera::errorJac(const Eigen::Vector3d& p) const {
  double x = p[0], y = p[1], z = p[2];
  double d2 = x*x + y*y + z*z;
  double d = std::sqrt(d2);
  double r_xi = z + xi_ * d;

  // if (r_xi <= 1e-6 || (xi_ * z + d <= 0)) {
  //   return Eigen::Matrix<double, 2, 3>::Zero();
  // }
  if (z <= -w_ * d + 1e-6) {
    return Eigen::Matrix<double, 2, 3>::Zero();
  }

  Eigen::Matrix<double, 2, 3> J;
  double r_xi2_inv = 1.0 / (r_xi * r_xi);
  double d_inv = 1.0 / d;
  
  double dr_xi_dx = xi_ * x * d_inv;
  double dr_xi_dy = xi_ * y * d_inv;
  double dr_xi_dz = 1.0 + xi_ * z * d_inv;

  // 缁绢垱鐖ｉ柌蹇氱ゴ閸婄》绱濇穱婵囧瘮妤傛ɑ鈧嗗厴
  J(0,0) = fx_ * (r_xi - x * dr_xi_dx) * r_xi2_inv;
  J(0,1) = fx_ * (-x * dr_xi_dy) * r_xi2_inv;
  J(0,2) = fx_ * (-x * dr_xi_dz) * r_xi2_inv;

  J(1,0) = fy_ * (-y * dr_xi_dx) * r_xi2_inv;
  J(1,1) = fy_ * (r_xi - y * dr_xi_dy) * r_xi2_inv;
  J(1,2) = fy_ * (-y * dr_xi_dz) * r_xi2_inv;

  return J;
}

} // namespace vk