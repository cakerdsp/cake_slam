#pragma once

// 模块功能：统一 IMU 采样数据结构与消息转换工具，
// 为 LIO/VIO 前后端提供一致的惯导输入格式。

#include <eigen3/Eigen/Dense>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/rclcpp.hpp>

// 统一的 IMU 原始测量样本。
// LIO 与视觉估计器都通过该结构读取惯导输入，避免不同模块各自维护一套格式。
struct ImuSample
{
  // IMU 消息采样时间，单位为秒。
  double stamp = 0.0;
  // 三轴线加速度测量值，坐标系为 IMU 本体坐标系，单位为米每二次方秒。
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  // 三轴角速度测量值，坐标系为 IMU 本体坐标系，单位为弧度每秒。
  Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
};

// 将 ROS IMU 消息转换为项目统一的 IMU 样本格式。
// 要求输入消息中的时间戳、线加速度和角速度字段已经按驱动约定正确填写。
inline ImuSample ImuSampleFromMsg(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
{
  ImuSample sample;
  sample.stamp = rclcpp::Time(msg->header.stamp).seconds();
  sample.acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  sample.gyr << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  return sample;
}
