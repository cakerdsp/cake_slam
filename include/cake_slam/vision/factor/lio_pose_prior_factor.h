#pragma once

// 模块功能：LIO 位姿先验因子接口，
// 将 LiDAR-IMU 估计的位姿与协方差注入视觉优化。

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include "cake_slam/lidar_visual_types.h"

namespace cake_slam {

/**
 * @brief Ceres SE(3) prior from LiDAR-inertial IESKF pose/covariance.
 *
 * Parameter block: VINS pose [p_x,p_y,p_z,q_x,q_y,q_z,q_w].
 * Residual order: [delta theta(rad), delta p(m)] in the prior/body frame.
 */
class LioPosePriorFactor
{
public:
  static ceres::CostFunction *Create(const cake_slam::LioPosePrior &prior);
};

} // namespace cake_slam
