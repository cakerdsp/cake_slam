#pragma once

#include <eigen3/Eigen/Dense>

// 统一的轻量级 SLAM 状态输出结构。
// 该结构用于把内部实现各不相同的前端/后端状态，整理成对外一致的接口格式。
struct SlamState
{
  // 状态对应的时间戳，单位为秒，通常对应当前融合结果所代表的传感器时刻。
  double stamp = 0.0;
  // 世界坐标系到机体坐标系的旋转结果，采用 3x3 旋转矩阵表示。
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  // 机体在世界坐标系下的位置，单位为米。
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  // 机体在世界坐标系下的速度，单位为米每秒。
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  // 加速度计偏置估计值，单位为米每二次方秒。
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  // 陀螺仪偏置估计值，单位为弧度每秒。
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  // 重力向量估计值，方向与大小由系统初始化/估计结果给出。
  Eigen::Vector3d g = Eigen::Vector3d::Zero();
};
