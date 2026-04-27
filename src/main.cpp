#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv)
{
  // 目前这里只保留 ROS2 进程生命周期骨架。
  // 真实的订阅、同步、LIO/VIO 节点装配还没有在这里接入。
  rclcpp::init(argc, argv);
  // TODO: Wire up cake_slam modules here.
  rclcpp::shutdown();
  return 0;
}
