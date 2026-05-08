#pragma once

#include <algorithm>
#include <cmath>

#include <ceres/ceres.h>

namespace cake_slam {

/**
 * @brief One-dimensional inverse-depth prior for LiDAR-initialized landmarks.
 *
 * Parameter block: lambda [1/m]. Residual: whitened lambda-lambda_prior.
 * The variance is propagated from the LiDAR 3D point covariance into camera z.
 */
class InverseDepthPriorFactor : public ceres::SizedCostFunction<1, 1>
{
public:
  InverseDepthPriorFactor(double prior_inv_depth, double inv_depth_var);
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

private:
  double prior_inv_depth_ = 1.0;
  double sqrt_info_ = 1.0;
};

} // namespace cake_slam
