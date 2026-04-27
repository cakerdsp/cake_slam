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

// #define GPU_MODE 1


#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#ifdef GPU_MODE
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;


#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

// 判断像素点是否仍位于可跟踪图像边界内。
bool inBorder(const cv::Point2f &pt);
// 根据状态数组原地筛掉无效点。
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
// 根据状态数组原地筛掉无效 id。
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    // 输入一帧图像并输出“特征 id -> 多相机观测”的统一数据结构。
    // 输入要求：
    // 1. _cur_time 单位为秒，且应单调递增；
    // 2. _img 为左目/单目图像；
    // 3. _img1 仅在双目模式下使用。
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    // 根据已有跟踪次数设置特征检测掩膜，优先保留长轨迹特征。
    void setMask();
    // 在掩膜允许的区域补充新的特征点。
    void addPoints();
    // 读取相机模型和内参文件。
    void readIntrinsicParameter(const vector<string> &calib_file);
    // 调试函数：显示去畸变结果。
    void showUndistortion(const string &name);
    // 利用基础矩阵剔除几何外点。
    void rejectWithF();
    // 将当前特征点从像素坐标转换为归一化平面坐标。
    void undistortedPoints();
    // 对任意点集执行去畸变并输出归一化坐标。
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    // 根据相邻两帧位置计算特征速度。
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    // 调试函数：并排显示双目图像和对应点。
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    // 绘制跟踪可视化图。
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    // 设置来自后端预测的特征落点，提高光流初值质量。
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    // 计算两点欧氏距离。
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    // 根据外点 id 集合删除当前跟踪点。
    void removeOutliers(set<int> &removePtsIds);
    // 返回最近一次绘制的跟踪图像。
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);

    // 图像尺寸。
    int row, col;
    // 调试用轨迹可视化图像。
    cv::Mat imTrack;
    // 特征检测掩膜。
    cv::Mat mask;
    // 鱼眼有效区域掩膜。
    cv::Mat fisheye_mask;
    // 上一帧与当前帧灰度图。
    cv::Mat prev_img, cur_img;
    // 新检测出来等待加入的点。
    vector<cv::Point2f> n_pts;
    // 后端预测点及其调试副本。
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    // 左目上一帧/当前帧、右目当前帧跟踪点。
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    // 去畸变后的归一化平面点。
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    // 特征像素速度。
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    // 左右目特征 id。
    vector<int> ids, ids_right;
    // 每个特征已连续跟踪的帧数。
    vector<int> track_cnt;
    // 当前/上一帧归一化坐标映射，便于按 id 查找。
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    // 相机模型对象。
    vector<camodocal::CameraPtr> m_camera;
    // 当前和上一帧时间戳。
    double cur_time;
    double prev_time;
    // 是否工作在双目模式。
    bool stereo_cam;
    // 下一个可分配的特征 id。
    int n_id;
    // 当前是否持有后端预测。
    bool hasPrediction;
};
