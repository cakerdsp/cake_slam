cmake_minimum_required(VERSION 3.0.2)
project(fast_livo)

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
  livox_ros_driver
  cv_bridge
  image_transport
  vikit_common
  vikit_ros
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp sensor_msgs geometry_msgs nav_msgs visualization_msgs
    pcl_ros pcl_conversions tf2 tf2_ros tf2_geometry_msgs
    livox_ros_driver cv_bridge image_transport vikit_common vikit_ros
  DEPENDS Eigen3 PCL OpenCV Sophus Boost
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

add_library(utils src/utils.cpp)
add_library(vio_multi_cam
  src/vio_multi_cam.cpp
  src/frame_multi_cam.cpp
  src/visual_point_multi_cam.cpp
)
add_library(lio_multi_cam src/voxel_map_multi_cam.cpp)
add_library(pre_multi_cam src/preprocess_multi_cam.cpp)
add_library(imu_proc_multi_cam src/IMU_Processing_multi_cam.cpp)
add_library(laser_mapping_multi_cam src/LIVMapper_multi_cam.cpp)

add_dependencies(utils ${catkin_EXPORTED_TARGETS})
add_dependencies(vio_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(lio_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(pre_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(imu_proc_multi_cam ${catkin_EXPORTED_TARGETS})
add_dependencies(laser_mapping_multi_cam ${catkin_EXPORTED_TARGETS})

set(COMMON_LIBS
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  list(APPEND COMMON_LIBS OpenMP::OpenMP_CXX)
endif()

target_link_libraries(utils ${COMMON_LIBS})
target_link_libraries(vio_multi_cam ${COMMON_LIBS})
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

add_executable(fastlivo_mapping_multi_cam src/main_multi_cam.cpp)
add_dependencies(fastlivo_mapping_multi_cam ${catkin_EXPORTED_TARGETS})
target_link_libraries(fastlivo_mapping_multi_cam
  laser_mapping_multi_cam
  ${COMMON_LIBS}
)

if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping_multi_cam mimalloc)
endif()

install(TARGETS
  utils
  vio_multi_cam
  lio_multi_cam
  pre_multi_cam
  imu_proc_multi_cam
  laser_mapping_multi_cam
  fastlivo_mapping_multi_cam
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
