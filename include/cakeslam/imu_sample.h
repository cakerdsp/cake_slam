#pragma once

#include <eigen3/Eigen/Dense>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/rclcpp.hpp>

struct ImuSample
{
  double stamp = 0.0;
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
};

inline ImuSample ImuSampleFromMsg(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
{
  ImuSample sample;
  sample.stamp = rclcpp::Time(msg->header.stamp).seconds();
  sample.acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  sample.gyr << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  return sample;
}
