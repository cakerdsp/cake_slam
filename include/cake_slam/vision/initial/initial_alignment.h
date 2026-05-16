/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

// 模块功能：视觉-IMU 对齐初始化接口，
// 用于估计重力方向与陀螺偏置等初始状态。
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <rcutils/logging_macros.h>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <rclcpp/rclcpp.hpp>
#include <map>
#include "../estimator/feature_manager.h"
using namespace std;

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR


class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        IntegrationBase *pre_integration;
        bool is_key_frame;
};
void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, StatesGroup states_[]);
bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, StatesGroup states_[], Eigen::Vector3d &g, Eigen::VectorXd &x);
