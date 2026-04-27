/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR

// 用于把归一化坐标误差近似换算到像素量级的参考焦距。
const double FOCAL_LENGTH = 460.0;
// 滑动窗口大小，实际窗口帧数为 WINDOW_SIZE + 1。
const int WINDOW_SIZE = 10;
// 允许同时维护的最大特征数量。
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

// 以下变量由 readParameters 从配置文件中统一加载。
extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern int USE_GPU;
extern int USE_GPU_ACC_FLOW;
extern int USE_GPU_CERES;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern std::string WORLD_FRAME_ID;
extern std::string BODY_FRAME_ID;
extern std::string CAMERA_FRAME_ID;

// 加载视觉模块相关参数，并同步写入上述全局变量。
// 输入要求：config_file 应指向统一格式的 YAML / OpenCV 配置文件。
void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    // 位姿参数块：平移 3 + 四元数 4。
    SIZE_POSE = 7,
    // 速度与偏置参数块：速度 3 + 加计偏置 3 + 陀螺偏置 3。
    SIZE_SPEEDBIAS = 9,
    // 路标参数块：单目场景下通常使用 1 维逆深度。
    SIZE_FEATURE = 1
};

enum StateOrder
{
    // 各误差状态在 15 维 IMU 误差状态向量中的排列顺序。
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    // 各噪声块在连续/离散噪声向量中的排列顺序。
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
