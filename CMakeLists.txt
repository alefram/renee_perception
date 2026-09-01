cmake_minimum_required(VERSION 3.8)
project(renee_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

ament_python_install_package(renception)

add_library(rgbd_dataset_writer
  src/rgbd_dataset_writer.cpp
)
target_compile_features(rgbd_dataset_writer PUBLIC cxx_std_17)
target_include_directories(rgbd_dataset_writer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(rgbd_dataset_writer
  builtin_interfaces
  cv_bridge
  geometry_msgs
  sensor_msgs
)

install(TARGETS
  rgbd_dataset_writer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(PROGRAMS
  tools/build_pointcloud.py
  tools/extract_video_frames.py
  tools/record_data.py
  tools/visualize.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(rgbd_dataset_writer)
ament_export_dependencies(builtin_interfaces cv_bridge geometry_msgs sensor_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(renee_perception_pytest test)
endif()

ament_package()
