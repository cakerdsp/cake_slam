#include "cake_slam/imu_hub.h"

#include <rclcpp/rclcpp.hpp>

namespace cake_slam {

void ImuHub::Push(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.push_back(msg);
}

std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> ImuHub::GetRange(double t0, double t1)
{
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> out;
  std::lock_guard<std::mutex> lock(mutex_);
  while (!buffer_.empty()) {
    double t = rclcpp::Time(buffer_.front()->header.stamp).seconds();
    if (t < t0) {
      buffer_.pop_front();
      continue;
    }
    if (t > t1) {
      break;
    }
    out.push_back(buffer_.front());
    buffer_.pop_front();
  }
  return out;
}

void ImuHub::Clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.clear();
}

} // namespace cake_slam
