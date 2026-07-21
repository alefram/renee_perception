cmake_minimum_required(VERSION 3.8)
project(renee_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(perception PACKAGE_DIR ../src/perception)

install(PROGRAMS
  ../src/perception/camera_capture_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME camera_capture_node
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(renee_perception_pytest ../tests)
endif()

ament_package()
