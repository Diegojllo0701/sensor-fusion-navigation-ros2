cmake_minimum_required(VERSION 3.8)
project(sensor-fusion-navigation-ros2)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# add_executable(cv_node src/cv_node.cpp)
# ament_target_dependencies(cv_node rclcpp)

# install(TARGETS
#   cv_node
#   DESTINATION lib/${PROJECT_NAME})

# install(PROGRAMS
#   scripts/navigation_node.py
#   DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()