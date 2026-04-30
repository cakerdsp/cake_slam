/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
#include <rcpputils/asserts.hpp>

#include "parameters.h"
#include "cake_slam/common_lib.h"
#include "../utility/tic_toc.h"


#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR


class FeaturePerFrame
{
  public:
    // _point 的含义：
    // [0:2] 归一化相机坐标，
    // [3:4] 原始像素坐标，
    // [5:6] 像素速度。
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
        is_stereo = false;
    }
    // 写入右目观测，使该特征可以参与双目三角化。
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        is_stereo = true;
    }
    double cur_td; // 当前观测对应的时间延迟补偿值。
    Vector3d point, pointRight; // 左右目归一化相机坐标。
    Vector2d uv, uvRight; // 左右目像素坐标。
    Vector2d velocity, velocityRight; // 左右目像素速度。
    bool is_stereo; // 当前帧是否存在右目观测。
};

class FeaturePerId
{
  public:
    const int feature_id; // 全局唯一特征 id。
    int start_frame; // 该特征第一次出现时所在的滑窗帧号。
    vector<FeaturePerFrame> feature_per_frame; // 该特征在多个帧中的观测序列。
    int used_num; // 当前优化中被实际使用的观测数量。
    double estimated_depth; // 以首帧相机为参考的逆深度或深度估计值。
    int solve_flag; // 0 未求解；1 求解成功；2 求解失败。

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    // 返回该特征最后一次被观测到的帧号。
    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager();

    // 设置相机到 IMU/body 的旋转外参。
    void setRic(Matrix3d _ric[]);
    // 清空所有跟踪特征和统计量。
    void clearState();
    // 统计当前可以进入优化的特征数量。
    int getFeatureCount();
    // 向管理器中插入一帧观测，并检查是否满足关键帧视差条件。
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    // 获取两帧之间的匹配归一化坐标对。
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    // 用优化变量回写每个特征的深度。
    void setDepth(const VectorXd &x);
    // 删除求解失败的特征。
    void removeFailures();
    // 清空所有特征深度。
    void clearDepth();
    // 导出当前所有待优化特征的深度向量。
    VectorXd getDepthVector();
    // 为滑窗中的特征执行三角化。
    void triangulate(int frameCnt, StatesGroup states_[], Vector3d tic[], Matrix3d ric[]);
    // 用两帧位姿和两个观测恢复单个三维点。
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    // 利用已知三维点，通过 PnP 初始化某一帧位姿。
    void initFramePoseByPnP(int frameCnt, StatesGroup states_[], Vector3d tic[], Matrix3d ric[]);
    // 求解一帧的位姿，输入 2D-3D 对应关系至少应满足 PnP 最低要求。
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                            vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    // 滑窗左移时，同时对特征深度参考系做变换。
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    // 删除最老帧对应的观测。
    void removeBack();
    // 删除最新帧之外的前部观测。
    void removeFront(int frame_count);
    // 根据外点 id 删除异常特征。
    void removeOutlier(set<int> &outlierIndex);
    list<FeaturePerId> feature; // 当前滑窗内所有特征轨迹。
    int last_track_num; // 上一帧成功跟踪的特征数。
    double last_average_parallax; // 上一轮统计得到的平均视差。
    int new_feature_num; // 本帧新加入特征数。
    int long_track_num; // 长轨迹特征数量。

  private:
    // 计算补偿了相机外参和时间延迟后的视差。
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    Matrix3d ric[2]; // 相机到 IMU/body 的旋转外参。
};

#endif
