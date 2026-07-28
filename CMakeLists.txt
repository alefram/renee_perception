cmake_minimum_required(VERSION 3.8)
project(renee_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

ament_python_install_package(renception)

set(interface_files
  "msg/CapturedRgbdFrame.msg"
  "srv/CaptureRGBD.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${interface_files}
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

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

add_executable(rgbd_capture_node
  src/main.cpp
  src/rgbd_capture_node.cpp
)
target_compile_features(rgbd_capture_node PUBLIC cxx_std_17)
target_include_directories(rgbd_capture_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(rgbd_capture_node
  builtin_interfaces
  cv_bridge
  geometry_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
)
target_link_libraries(rgbd_capture_node
  rgbd_dataset_writer
  "${cpp_typesupport_target}"
)

install(TARGETS
  rgbd_dataset_writer
  rgbd_capture_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  tools/build_pointcloud.py
  tools/extract_video_frames.py
  tools/record_data.py
  tools/visualize.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(renee_perception_pytest test)
endif()

ament_package()
