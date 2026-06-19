function(configure_fast_livo_multi_cam_targets)
  if(TARGET fastlivo_mapping_multi_cam)
    return()
  endif()

  add_library(vio_multi_cam
    ${CMAKE_SOURCE_DIR}/src/vio_multi_cam.cpp
    ${CMAKE_SOURCE_DIR}/src/frame_multi_cam.cpp
    ${CMAKE_SOURCE_DIR}/src/visual_point_multi_cam.cpp)
  add_library(lio_multi_cam ${CMAKE_SOURCE_DIR}/src/voxel_map_multi_cam.cpp)
  add_library(pre_multi_cam ${CMAKE_SOURCE_DIR}/src/preprocess_multi_cam.cpp)
  add_library(imu_proc_multi_cam ${CMAKE_SOURCE_DIR}/src/IMU_Processing_multi_cam.cpp)
  add_library(laser_mapping_multi_cam ${CMAKE_SOURCE_DIR}/src/LIVMapper_multi_cam.cpp)

  foreach(target vio_multi_cam lio_multi_cam pre_multi_cam imu_proc_multi_cam laser_mapping_multi_cam)
    ament_target_dependencies(${target} ${dependencies})
    target_link_libraries(${target} ${COMMON_DEPENDENCIES})
  endforeach()
  target_link_libraries(lio_multi_cam utils)

  add_executable(fastlivo_mapping_multi_cam ${CMAKE_SOURCE_DIR}/src/main_multi_cam.cpp)
  ament_target_dependencies(fastlivo_mapping_multi_cam ${dependencies})
  target_link_libraries(fastlivo_mapping_multi_cam
    laser_mapping_multi_cam
    vio_multi_cam
    lio_multi_cam
    pre_multi_cam
    imu_proc_multi_cam
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${Boost_LIBRARIES})

  if(mimalloc_FOUND)
    target_link_libraries(fastlivo_mapping_multi_cam mimalloc)
  endif()

  install(TARGETS fastlivo_mapping_multi_cam DESTINATION lib/${PROJECT_NAME})
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL configure_fast_livo_multi_cam_targets)
