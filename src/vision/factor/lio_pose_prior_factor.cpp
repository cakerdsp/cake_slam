// 模块功能：LIO 位姿先验因子实现，
// 把 IESKF 位姿与协方差转化为 Ceres 残差。

#include "lio_pose_prior_factor.h"

namespace cake_slam {

namespace {

struct LioPosePriorResidual
{
  explicit LioPosePriorResidual(const LioPosePrior &prior)
      : q_prior(prior.R_WB),
        p_prior(prior.p_WB),
        R_BW_prior(prior.R_WB.transpose()),
        sqrt_info(prior.sqrt_information)
  {
  }

  template <typename T>
  bool operator()(const T *const pose, T *residuals) const
  {
    // Parameter order follows VINS: [p_x,p_y,p_z,q_x,q_y,q_z,q_w].
    // The residual is expressed as a local SE(3) error and then whitened by the
    // square-root information exported from the IESKF covariance.
    const Eigen::Matrix<T, 3, 1> p(pose[0], pose[1], pose[2]);
    const Eigen::Quaternion<T> q(pose[6], pose[3], pose[4], pose[5]);
    const Eigen::Quaternion<T> q_ref(T(q_prior.w()), T(q_prior.x()), T(q_prior.y()), T(q_prior.z()));
    const Eigen::Quaternion<T> dq = q_ref.conjugate() * q;

    Eigen::Matrix<T, 6, 1> raw;
    raw.template segment<3>(0) = T(2.0) * Eigen::Matrix<T, 3, 1>(dq.x(), dq.y(), dq.z());
    raw.template segment<3>(3) =
        R_BW_prior.cast<T>() * (p - p_prior.cast<T>());

    const Eigen::Matrix<T, 6, 1> whitened = sqrt_info.cast<T>() * raw;
    for (int i = 0; i < 6; ++i) {
      residuals[i] = whitened(i);
    }
    return true;
  }

  Eigen::Quaterniond q_prior;
  Eigen::Vector3d p_prior;
  Eigen::Matrix3d R_BW_prior;
  Eigen::Matrix<double, 6, 6> sqrt_info;
};

} // namespace

ceres::CostFunction *LioPosePriorFactor::Create(const LioPosePrior &prior)
{
  return new ceres::AutoDiffCostFunction<LioPosePriorResidual, 6, 7>(
      new LioPosePriorResidual(prior));
}

} // namespace cake_slam
