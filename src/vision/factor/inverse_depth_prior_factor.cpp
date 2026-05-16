// 模块功能：逆深度先验因子实现，
// 将 LiDAR 深度先验转化为优化残差。

#include "inverse_depth_prior_factor.h"

#include "cake_slam/vision/estimator/parameters.h"

namespace cake_slam {

InverseDepthPriorFactor::InverseDepthPriorFactor(double prior_inv_depth, double inv_depth_var)
    : prior_inv_depth_(prior_inv_depth)
{
  const double var = std::max(inv_depth_var, MIN_INV_DEPTH_VAR);
  sqrt_info_ = 1.0 / std::sqrt(var);
}

bool InverseDepthPriorFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
  // One-dimensional whitening: r = (lambda - lambda_lidar) / sigma_lambda.
  // lambda is the VINS inverse-depth parameter [1/m].
  residuals[0] = sqrt_info_ * (parameters[0][0] - prior_inv_depth_);
  if (jacobians && jacobians[0]) {
    jacobians[0][0] = sqrt_info_;
  }
  return true;
}

} // namespace cake_slam
