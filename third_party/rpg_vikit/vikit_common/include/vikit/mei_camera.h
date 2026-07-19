#ifndef MEI_CAMERA_H_
#define MEI_CAMERA_H_

#include <stdlib.h>
#include <string>
#include <Eigen/Eigen>
#include <vikit/abstract_camera.h>

namespace vk {

class MEICamera : public AbstractCamera {
private:
  // 娴ｈ法鏁ら崢鐔烘晸 double 鐎涙ê鍋嶉敍灞戒氦鎼存洝顫夐柆?Eigen 閸愬懎鐡ㄧ€靛綊缍堥梾鎰亝
  const double fx_, fy_;
  const double cx_, cy_;
  const double xi_;
  bool distortion_;
  double k1_, k2_, p1_, p2_;
  double w_;
public:
  // 閾忕晫鍔ч悳鏉挎躬濞屸剝婀?Eigen 閹存劕鎲抽敍灞肩稻娣囨繄鏆€濮濄倕鐣禒銉╂Щ娑撳洣绔撮張顏呮降濞ｈ濮為崶鍝勭暰鐏忓搫顕幋鎰喅
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MEICamera(double width, double height, double scale,
            double fx, double fy, double cx, double cy,
            double xi,
            double k1=0.0, double k2=0.0,
            double p1=0.0, double p2=0.0);

  virtual ~MEICamera();

  virtual Eigen::Vector3d cam2world(const double& u, const double& v) const override;
  virtual Eigen::Vector3d cam2world(const Eigen::Vector2d& px) const override;
  virtual Eigen::Vector2d world2cam(const Eigen::Vector3d& xyz_c) const override;
  virtual Eigen::Vector2d world2cam(const Eigen::Vector2d& uv) const override;

  // 缁?C 妞嬪孩鐗搁惃鍕箵閻ｇ褰夐敍灞炬￥ Eigen 娑撳瓨妞傜€电钖?
  inline void removeDistortion(double& x, double& y) const {
    if (!distortion_) return;
    double x_dist = x;
    double y_dist = y;
    for (int i = 0; i < 10; ++i) {
      double r2 = x*x + y*y;
      double r4 = r2*r2;
      double rad = 1.0 + k1_*r2 + k2_*r4;
      
      // f_uv: 瑜版挸澧犳导鎷岊吀閸婂吋濮囪ぐ鍗炴倵閻ㄥ嫪缍呯純?
      double f_x = x * rad + 2.0*p1_*x*y + p2_*(r2 + 2.0*x*x);
      double f_y = y * rad + p1_*(r2 + 2.0*y*y) + 2.0*p2_*x*y;
      
      double err_x = f_x - x_dist;
      double err_y = f_y - y_dist;
      if ((err_x*err_x + err_y*err_y) < 1e-22) break; // 1e-11 閻ㄥ嫬閽╅弬?

      // 閹靛濮╃拋锛勭暬闂嗗懎褰插В鏃傜叐闂?J 閻ㄥ嫬娲撴稉顏勫瀻闁?
      double r_tmp = k1_ + 2.0*k2_*r2;
      double drad_dx = 2.0 * x * r_tmp;
      double drad_dy = 2.0 * y * r_tmp;
      
      double j00 = rad + x*drad_dx + 2.0*p1_*y + 6.0*p2_*x;
      double j01 = x*drad_dy + 2.0*p1_*x + 2.0*p2_*y;
      double j10 = y*drad_dx + 2.0*p1_*x + 2.0*p2_*y;
      double j11 = rad + y*drad_dy + 6.0*p1_*y + 2.0*p2_*x;
      
      // 閹靛濮╃拋锛勭暬 2x2 闁棛鐓╅梼鍏哥濞? delta = J^-1 * err
      double det = j00 * j11 - j01 * j10;
      if (std::abs(det) < 1e-10) break;
      double inv_det = 1.0 / det;
      x -= inv_det * (j11 * err_x - j01 * err_y);
      y -= inv_det * (-j10 * err_x + j00 * err_y);
    }
  }

  virtual Eigen::Matrix<double, 2, 3> errorJac(const Eigen::Vector3d& p) const;

  virtual double errorMultiplier2() const override { return std::abs(fx_); }
  virtual double errorMultiplier() const override { return std::abs(4.0*fx_*fy_); }
  virtual double fx() const override { return fx_; }
  virtual double fy() const override { return fy_; }
  virtual double cx() const override { return cx_; }
  virtual double cy() const override { return cy_; }
  double k1() const  { return k1_; }
  double k2() const  { return k2_; }
  double p1() const  { return p1_; }
  double p2() const  { return p2_; }
  double xi() const { return xi_; }
};

} // namespace vk
#endif