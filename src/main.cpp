// 模块功能：ROS2 进程入口占位实现，
// 保持最小化启动与退出流程。

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv)
{
  // The installed node entrypoint is src/slam_node.cpp. This file is kept only as a
  // minimal standalone lifecycle stub and is not listed in CMakeLists.txt.
  rclcpp::init(argc, argv);
  rclcpp::shutdown();
  return 0;
}
