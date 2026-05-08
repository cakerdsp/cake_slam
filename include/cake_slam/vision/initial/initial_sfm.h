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
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <rcutils/logging_macros.h>
using namespace std;

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

// SFM 初始化阶段使用的特征结构。
// 每个特征保存其跨帧观测与最终求得的三维位置。
struct SFMFeature
{
    bool state; // 是否已经成功恢复三维位置。
    int id; // 特征 id。
    vector<pair<int,Eigen::Vector2d>> observation; // <帧号, 归一化坐标> 观测序列。
    double position[3]; // 三维点坐标。
    double depth; // 深度估计值。
};

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	// 在视觉初始化阶段执行全局 SFM，恢复多帧相机位姿和稀疏路标。
	// 输入要求：
	// 1. q / T 数组长度至少为 frame_num；
	// 2. l 为参考帧索引；
	// 3. relative_R / relative_T 为参考两帧的相对位姿初值。
	bool construct(int frame_num, Eigen::Quaterniond* q, Eigen::Vector3d* T, int l,
			  const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points);

private:
	// 使用 PnP 估计单帧位姿。
	bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	// 由两帧观测三角化单个三维点。
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
	// 对 frame0 与 frame1 之间可见的全部特征做批量三角化。
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num; // 当前进入 SFM 的特征总数。
};
