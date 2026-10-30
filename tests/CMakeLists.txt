cmake_minimum_required(VERSION 3.10)
project(cake_covariance_math_tests LANGUAGES CXX)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
enable_testing()

# Exercise the production lifecycle methods with small owning fixtures, without
# requiring ROS/PCL. Keep the generated bodies synchronized with source edits.
find_package(PythonInterp 3 REQUIRED)
set(retirement_impl ${CMAKE_CURRENT_BINARY_DIR}/visual_map_retirement_impl.inc)
add_custom_command(
  OUTPUT ${retirement_impl}
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/extract_visual_map_retirement.py
          ${CMAKE_CURRENT_SOURCE_DIR}/.. ${retirement_impl}
  DEPENDS extract_visual_map_retirement.py ../src/vio_multi_cam.cpp ../src/visual_point_multi_cam.cpp
  VERBATIM)
add_executable(visual_map_retirement_test visual_map_retirement_test.cpp ${retirement_impl})
target_compile_features(visual_map_retirement_test PRIVATE cxx_std_17)
target_include_directories(visual_map_retirement_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME visual_map_retirement_test COMMAND visual_map_retirement_test)

foreach(test estimator_covariance estimator_update_workspace directional_update virtual_sampling fixed_gray_weights)
  if(test STREQUAL "fixed_gray_weights")
    add_executable(${test}_math_test ${test}_test.cpp)
  else()
    add_executable(${test}_math_test ${test}_math_test.cpp)
  endif()
  target_compile_features(${test}_math_test PRIVATE cxx_std_17)
  target_include_directories(${test}_math_test PRIVATE ../include)
  target_link_libraries(${test}_math_test PRIVATE Eigen3::Eigen)
  # The older directional test uses assert; keep it active in Release too.
  if(MSVC)
    target_compile_options(${test}_math_test PRIVATE /UNDEBUG)
  else()
    target_compile_options(${test}_math_test PRIVATE -UNDEBUG)
  endif()
  add_test(NAME ${test}_math_test COMMAND ${test}_math_test)
endforeach()
target_compile_definitions(estimator_update_workspace_math_test PRIVATE EIGEN_RUNTIME_NO_MALLOC)

# Optional image tests do not make the existing Eigen-only tests depend on OpenCV.
find_package(OpenCV QUIET COMPONENTS core imgproc)
if(OpenCV_FOUND)
  find_package(Threads REQUIRED)
  add_executable(adaptive_image_pyramid_test adaptive_image_pyramid_test.cpp)
  target_compile_features(adaptive_image_pyramid_test PRIVATE cxx_std_17)
  target_include_directories(adaptive_image_pyramid_test PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(adaptive_image_pyramid_test PRIVATE ${OpenCV_LIBS} Threads::Threads)
  add_test(NAME adaptive_image_pyramid_test COMMAND adaptive_image_pyramid_test)
  add_executable(fixed_gray_image_test fixed_gray_image_test.cpp)
  target_compile_features(fixed_gray_image_test PRIVATE cxx_std_17)
  target_include_directories(fixed_gray_image_test PRIVATE ../include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(fixed_gray_image_test PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
  add_test(NAME fixed_gray_image_test COMMAND fixed_gray_image_test)
else()
  message(STATUS "OpenCV core/imgproc unavailable: image tests not configured")
endif()
