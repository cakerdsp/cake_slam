cmake_minimum_required(VERSION 3.8)
project(cake_slam)

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    set(CAKE_SLAM_LIO_OPT_FLAGS -O3 -march=native -mtune=native -funroll-loops)
    message(STATUS "Using x86 LIO-only optimizations: ${CAKE_SLAM_LIO_OPT_FLAGS}")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64)$")
    set(CAKE_SLAM_LIO_OPT_FLAGS -O3 -mcpu=native -mtune=native)
    message(STATUS "Using aarch64 LIO-only optimizations: ${CAKE_SLAM_LIO_OPT_FLAGS}")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARM)")
    set(CAKE_SLAM_LIO_OPT_FLAGS -O3 -mcpu=native -mtune=native -mfpu=neon)
    message(STATUS "Using ARM LIO-only optimizations: ${CAKE_SLAM_LIO_OPT_FLAGS}")
  else()
    set(CAKE_SLAM_LIO_OPT_FLAGS -O3)
    message(STATUS "Using generic LIO-only optimization: ${CAKE_SLAM_LIO_OPT_FLAGS}")
  endif()
endif()

include(ProcessorCount)
ProcessorCount(CAKE_SLAM_PROC_COUNT)
if(NOT CAKE_SLAM_PROC_COUNT)
  set(CAKE_SLAM_PROC_COUNT 1)
endif()
if(CAKE_SLAM_PROC_COUNT GREATER 4)
  set(CAKE_SLAM_MP_PROC_NUM 4)
elseif(CAKE_SLAM_PROC_COUNT GREATER 1)
  set(CAKE_SLAM_MP_PROC_NUM ${CAKE_SLAM_PROC_COUNT})
else()
  set(CAKE_SLAM_MP_PROC_NUM 1)
endif()
message(STATUS "Processor count: ${CAKE_SLAM_PROC_COUNT}; LIO OpenMP threads: ${CAKE_SLAM_MP_PROC_NUM}")

# 1. 查找依赖
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
find_package(PCL REQUIRED COMPONENTS common io filters)
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
find_package(OpenMP REQUIRED)

# 2. QHULL 补丁（防止 PCL 报错找不到目标）
find_package(Qhull REQUIRED)
if(NOT TARGET QHULL::QHULL)
    add_library(QHULL::QHULL UNKNOWN IMPORTED)
    set_target_properties(QHULL::QHULL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${QHULL_INCLUDE_DIRS}"
        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libqhull_r.so"
    )
endif()

# 3. 包含路径
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
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# 4. 收集源码
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
  src/lidar_visual_selector.cpp
  src/lidar_preprocess.cpp
  src/local_voxel_map.cpp
  src/lio_core.cpp
  src/vision/factor/lio_full_state_prior_factor.cpp
  src/utils.cpp
  src/voxel_map.cpp
  ${VISION_SOURCES}
)

# 5. 构建核心库
add_library(cake_slam_core ${cake_slam_SOURCES})
# Keep aggressive optimization on the LiDAR hot path only. Optimizing the whole
# VIO stack with these flags has exposed memory issues in the fisheye camera path.
if(CAKE_SLAM_LIO_OPT_FLAGS)
  set_source_files_properties(
    src/lio_core.cpp
    src/voxel_map.cpp
    src/local_voxel_map.cpp
    PROPERTIES COMPILE_OPTIONS "${CAKE_SLAM_LIO_OPT_FLAGS}"
  )
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(src/vision/featureTracker/feature_tracker.cpp PROPERTIES
    COMPILE_OPTIONS "-O0;-fno-strict-aliasing"
  )
endif()
if(CAKE_SLAM_MP_PROC_NUM GREATER 1)
  set_source_files_properties(src/voxel_map.cpp PROPERTIES
    COMPILE_DEFINITIONS "MP_EN;MP_PROC_NUM=${CAKE_SLAM_MP_PROC_NUM}"
  )
else()
  set_source_files_properties(src/voxel_map.cpp PROPERTIES
    COMPILE_DEFINITIONS "MP_PROC_NUM=1"
  )
endif()

# ROS2 依赖链接（底层使用 plain signature）
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

# 【核心修正】：手动链接其他非 ROS2 包，全部删掉 PUBLIC/PRIVATE 关键字
target_link_libraries(cake_slam_core
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
  QHULL::QHULL
)

if(TARGET Sophus::Sophus)
  target_link_libraries(cake_slam_core Sophus::Sophus)
endif()

# 6. 构建节点
add_executable(cake_slam_node src/slam_node.cpp)

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

# 7. 安装与导出
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

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
