cmake_minimum_required(VERSION 3.0.2)
project(vikit_ros)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  vikit_common
)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

set(VIKIT_ROS_SOPHUS_LIBS ${Sophus_LIBRARIES})
if(TARGET Sophus::Sophus)
  set(VIKIT_ROS_SOPHUS_LIBS Sophus::Sophus)
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp tf2_ros tf2_geometry_msgs visualization_msgs vikit_common
  DEPENDS Eigen3 Sophus
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Sophus_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/output_helper.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${VIKIT_ROS_SOPHUS_LIBS}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
