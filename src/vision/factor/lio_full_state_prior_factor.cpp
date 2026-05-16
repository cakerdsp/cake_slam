// Full-state LIO prior factor implementation.
#include "lio_full_state_prior_factor.h"

namespace cake_slam {

namespace {

struct LioFullStatePriorResidual
{
  explicit LioFullStatePriorResidual(const LioFullStatePrior &prior)
      : q_prior(prior.R_WB),
        p_prior(prior.p_WB),
        v_prior(prior.v_WB),
        ba_prior(prior.ba),
        bg_prior(prior.bg),
        sqrt_info(prior.sqrt_information)
  {
  }

  template <typename T>
  bool operator()(const T *const pose, const T *const speed_bias, T *residuals) const
  {
    const Eigen::Matrix<T, 3, 1> p(pose[0], pose[1], pose[2]);
    const Eigen::Quaternion<T> q(pose[6], pose[3], pose[4], pose[5]);
    const Eigen::Quaternion<T> q_ref(T(q_prior.w()), T(q_prior.x()), T(q_prior.y()), T(q_prior.z()));
    const Eigen::Quaternion<T> dq = q_ref.conjugate() * q;

    Eigen::Matrix<T, 15, 1> raw;
    raw.template segment<3>(0) = p - p_prior.cast<T>();
    raw.template segment<3>(3) = T(2.0) * Eigen::Matrix<T, 3, 1>(dq.x(), dq.y(), dq.z());
    raw.template segment<3>(6) =
        Eigen::Matrix<T, 3, 1>(speed_bias[0], speed_bias[1], speed_bias[2]) - v_prior.cast<T>();
    raw.template segment<3>(9) =
        Eigen::Matrix<T, 3, 1>(speed_bias[3], speed_bias[4], speed_bias[5]) - ba_prior.cast<T>();
    raw.template segment<3>(12) =
        Eigen::Matrix<T, 3, 1>(speed_bias[6], speed_bias[7], speed_bias[8]) - bg_prior.cast<T>();

    const Eigen::Matrix<T, 15, 1> whitened = sqrt_info.cast<T>() * raw;
    for (int i = 0; i < 15; ++i) {
      residuals[i] = whitened(i);
    }
    return true;
  }

  Eigen::Quaterniond q_prior;
  Eigen::Vector3d p_prior;
  Eigen::Vector3d v_prior;
  Eigen::Vector3d ba_prior;
  Eigen::Vector3d bg_prior;
  Eigen::Matrix<double, 15, 15> sqrt_info;
};

} // namespace

ceres::CostFunction *LioFullStatePriorFactor::Create(const LioFullStatePrior &prior)
{
  return new ceres::AutoDiffCostFunction<LioFullStatePriorResidual, 15, 7, 9>(
      new LioFullStatePriorResidual(prior));
}

} // namespace cake_slam
