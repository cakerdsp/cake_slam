#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv)
{
  // The installed node entrypoint is src/slam_node.cpp. This file is kept only as a
  // minimal standalone lifecycle stub and is not listed in CMakeLists.txt.
  rclcpp::init(argc, argv);
  rclcpp::shutdown();
  return 0;
}
