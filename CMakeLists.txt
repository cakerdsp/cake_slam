cmake_minimum_required(VERSION 3.0.2)
project(cake_slam)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops")
  add_definitions(-DX86_ARCH)
endif()

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

include(ProcessorCount)
ProcessorCount(N)
if(N GREATER 4)
  set(PROC_NUM 4)
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
elseif(N GREATER 1)
  set(PROC_NUM ${N})
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
endif()

find_package(mimalloc QUIET)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  livox_ros_driver2
  cv_bridge
  image_transport
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
# Keep older ROS/OpenCV installations buildable. The optional initializer uses
# the core SIFT API (OpenCV >= 4.4), only when it compiles and links here.
include(CheckCXXSourceCompiles)
set(_cake_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(_cake_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${OpenCV_LIBRARIES})
check_cxx_source_compiles("#include <opencv2/features2d.hpp>
int main() { auto detector = cv::SIFT::create(500); return detector.empty(); }"
  CAKE_HAVE_OPENCV_SIFT)
set(CMAKE_REQUIRED_INCLUDES "${_cake_saved_required_includes}")
set(CMAKE_REQUIRED_LIBRARIES "${_cake_saved_required_libraries}")
unset(_cake_saved_required_includes)
unset(_cake_saved_required_libraries)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

set(cake_slam_SOPHUS_LIBS ${Sophus_LIBRARIES})
if(TARGET Sophus::Sophus)
  set(cake_slam_SOPHUS_LIBS Sophus::Sophus)
endif()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp sensor_msgs geometry_msgs nav_msgs visualization_msgs
    pcl_ros pcl_conversions tf2 tf2_ros tf2_geometry_msgs
    livox_ros_driver2 cv_bridge image_transport
  DEPENDS Eigen3 PCL OpenCV Sophus fmt Boost
)

include_directories(
  include
  third_party/rpg_vikit/vikit_common/include
  third_party/rpg_vikit/vikit_ros/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

add_library(vikit_common_vendor SHARED
  third_party/rpg_vikit/vikit_common/src/atan_camera.cpp
  third_party/rpg_vikit/vikit_common/src/omni_camera.cpp
  third_party/rpg_vikit/vikit_common/src/math_utils.cpp
  third_party/rpg_vikit/vikit_common/src/vision.cpp
  third_party/rpg_vikit/vikit_common/src/performance_monitor.cpp
  third_party/rpg_vikit/vikit_common/src/robust_cost.cpp
  third_party/rpg_vikit/vikit_common/src/user_input_thread.cpp
  third_party/rpg_vikit/vikit_common/src/pinhole_camera.cpp
  third_party/rpg_vikit/vikit_common/src/equidistant_camera.cpp
  third_party/rpg_vikit/vikit_common/src/polynomial_camera.cpp
  third_party/rpg_vikit/vikit_common/src/mei_camera.cpp
  third_party/rpg_vikit/vikit_common/src/homography.cpp
  third_party/rpg_vikit/vikit_common/src/img_align.cpp
)
target_link_libraries(vikit_common_vendor
  ${OpenCV_LIBRARIES}
  ${cake_slam_SOPHUS_LIBS}
  fmt::fmt
)

add_library(vikit_ros_vendor SHARED
  third_party/rpg_vikit/vikit_ros/src/output_helper.cpp
)
target_link_libraries(vikit_ros_vendor
  vikit_common_vendor
  ${catkin_LIBRARIES}
  ${cake_slam_SOPHUS_LIBS}
)

add_library(utils src/utils.cpp)
add_library(color_gray_initializer src/color_gray_initializer.cpp)
target_link_libraries(color_gray_initializer ${OpenCV_LIBRARIES})
if(CAKE_HAVE_OPENCV_SIFT)
  target_compile_definitions(color_gray_initializer PUBLIC CAKE_HAVE_OPENCV_SIFT=1)
else()
  target_compile_definitions(color_gray_initializer PUBLIC CAKE_HAVE_OPENCV_SIFT=0)
  message(STATUS "cv::SIFT unavailable: color_gray_init_en will retain legacy grayscale")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # This new module rejects non-finite/degenerate calibration input. Preserve
  # those checks even on ARM configurations that enable project-wide fast-math.
  target_compile_options(color_gray_initializer PRIVATE -fno-fast-math)
endif()
add_library(vio_multi_cam
  src/vio_multi_cam.cpp
  src/frame_multi_cam.cpp
  src/visual_point_multi_cam.cpp
)
add_library(lio_multi_cam src/voxel_map_multi_cam.cpp)
add_library(pre_multi_cam src/preprocess_multi_cam.cpp)
add_library(imu_proc_multi_cam src/IMU_Processing_multi_cam.cpp)
add_library(laser_mapping_multi_cam src/LIVMapper_multi_cam.cpp)

add_dependencies(vikit_common_vendor ${catkin_EXPORTED_TARGETS})
add_dependencies(vikit_ros_vendor ${catkin_EXPORTED_TARGETS})
add_dependencies(utils ${catkin_EXPORTED_TARGETS})
add_dependencies(vio_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(lio_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(pre_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(imu_proc_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(laser_mapping_multi_cam ${catkin_EXPORTED_TARGETS})

set(COMMON_LIBS
  ${catkin_LIBRARIES}
  vikit_common_vendor
  vikit_ros_vendor
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${cake_slam_SOPHUS_LIBS}
  ${Boost_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  list(APPEND COMMON_LIBS OpenMP::OpenMP_CXX)
endif()

target_link_libraries(utils ${COMMON_LIBS})
target_link_libraries(vio_multi_cam color_gray_initializer ${COMMON_LIBS})
target_link_libraries(lio_multi_cam utils ${COMMON_LIBS})
target_link_libraries(pre_multi_cam ${COMMON_LIBS})
target_link_libraries(imu_proc_multi_cam ${COMMON_LIBS})
target_link_libraries(laser_mapping_multi_cam
  vio_multi_cam
  lio_multi_cam
  pre_multi_cam
  imu_proc_multi_cam
  utils
  ${COMMON_LIBS}
)

add_executable(cake_slam_mapping_multi_cam src/main_multi_cam.cpp)
add_dependencies(cake_slam_mapping_multi_cam ${catkin_EXPORTED_TARGETS})
target_link_libraries(cake_slam_mapping_multi_cam
  laser_mapping_multi_cam
  ${COMMON_LIBS}
)

if(mimalloc_FOUND)
  target_link_libraries(cake_slam_mapping_multi_cam mimalloc)
endif()

install(TARGETS
  color_gray_initializer
  vikit_common_vendor
  vikit_ros_vendor
  utils
  vio_multi_cam
  lio_multi_cam
  pre_multi_cam
  imu_proc_multi_cam
  laser_mapping_multi_cam
  cake_slam_mapping_multi_cam
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY config launch rviz_cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
