#pragma once

#include <deque>
#include <mutex>
#include <vector>

#include <sensor_msgs/msg/imu.hpp>

namespace cake_slam {

class ImuHub
{
public:
  void Push(const sensor_msgs::msg::Imu::ConstSharedPtr &msg);

  // Extract IMU samples in [t0, t1]. Old samples before t0 are dropped.
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> GetRange(double t0, double t1);

  void Clear();

private:
  std::mutex mutex_;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> buffer_;
};

} // namespace cake_slam
