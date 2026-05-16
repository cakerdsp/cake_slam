#pragma once

// Full-state LIO prior factor for the VIO sliding-window optimizer.
#include <ceres/ceres.h>
#include <Eigen/Dense>

#include "cake_slam/lidar_visual_types.h"

namespace cake_slam {

/**
 * @brief Ceres prior over pose, velocity, and IMU biases from LIO.
 *
 * Parameter blocks:
 * - pose: VINS order [p_x,p_y,p_z,q_x,q_y,q_z,q_w]
 * - speed_bias: [v_x,v_y,v_z,ba_x,ba_y,ba_z,bg_x,bg_y,bg_z]
 *
 * Residual order:
 * [p - p_hat, 2 * (q_hat^-1 * q).xyz, v - v_hat, ba - ba_hat, bg - bg_hat].
 */
class LioFullStatePriorFactor
{
public:
  static ceres::CostFunction *Create(const cake_slam::LioFullStatePrior &prior);
};

} // namespace cake_slam
