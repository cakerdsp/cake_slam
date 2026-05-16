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

// 模块功能：IMU-相机外参旋转初始化工具，
// 用于在未知外参情况下标定初始旋转。

#include <vector>
#include "../estimator/parameters.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
#include <rcutils/logging_macros.h>
// #include <ros/console.h>


#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialEXRotation
{
public:
	InitialEXRotation();
    bool CalibrationExRotation(vector<pair<Eigen::Vector3d, Eigen::Vector3d>> corres, Eigen::Quaterniond delta_q_imu, Eigen::Matrix3d &calib_ric_result);
private:
	Eigen::Matrix3d solveRelativeR(const vector<pair<Eigen::Vector3d, Eigen::Vector3d>> &corres);

    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count;

    vector< Eigen::Matrix3d > Rc;
    vector< Eigen::Matrix3d > Rimu;
    vector< Eigen::Matrix3d > Rc_g;
    Eigen::Matrix3d ric;
};


