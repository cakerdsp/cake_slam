cmake_minimum_required(VERSION 3.0.2)
project(vikit_common)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core calib3d features2d highgui imgproc)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)

set(VIKIT_SOPHUS_LIBS ${Sophus_LIBRARIES})
if(TARGET Sophus::Sophus)
  set(VIKIT_SOPHUS_LIBS Sophus::Sophus)
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  DEPENDS OpenCV Sophus fmt
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/atan_camera.cpp
  src/omni_camera.cpp
  src/math_utils.cpp
  src/vision.cpp
  src/performance_monitor.cpp
  src/robust_cost.cpp
  src/user_input_thread.cpp
  src/pinhole_camera.cpp
  src/equidistant_camera.cpp
  src/polynomial_camera.cpp
  src/mei_camera.cpp
  src/homography.cpp
  src/img_align.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${VIKIT_SOPHUS_LIBS}
  fmt::fmt
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
