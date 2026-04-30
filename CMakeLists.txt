cmake_minimum_required(VERSION 3.8)
project(cake_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_models REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

include_directories(
  include
  include/cake_slam
  include/cake_slam/utils
  include/cake_slam/vision/estimator
  include/cake_slam/vision/featureTracker
  include/cake_slam/vision/factor
  include/cake_slam/vision/initial
  include/cake_slam/vision/utility
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

file(GLOB_RECURSE VISION_SOURCES
  src/vision/estimator/*.cpp
  src/vision/featureTracker/*.cpp
  src/vision/factor/*.cpp
  src/vision/initial/*.cpp
  src/vision/utility/*.cpp
)

set(cake_slam_SOURCES
  src/config.cpp
  src/imu_processor.cpp
  src/lidar_preprocess.cpp
  src/lio_core.cpp
  src/utils.cpp
  src/voxel_map.cpp
  ${VISION_SOURCES}
)

add_library(cake_slam_core ${cake_slam_SOURCES})
ament_target_dependencies(cake_slam_core
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  message_filters
  pcl_conversions
  pcl_ros
  Eigen3
  OpenCV
  cv_bridge
  image_transport
  camera_models
  Ceres
  Sophus
  livox_ros_driver2
)
target_link_libraries(cake_slam_core ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(cake_slam_node src/slam.cpp)
ament_target_dependencies(cake_slam_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  pcl_conversions
  cv_bridge
  livox_ros_driver2
)
target_link_libraries(cake_slam_node cake_slam_core)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  cake_slam_core
  cake_slam_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  message_filters
  PCL
  pcl_conversions
  pcl_ros
  Eigen3
  OpenCV
  cv_bridge
  image_transport
  camera_models
  Ceres
  Sophus
  livox_ros_driver2
)

ament_package()
