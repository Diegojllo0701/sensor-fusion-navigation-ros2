cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(sensor-fusion-navigation-ros2)

# General Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Point Cloud Clustering Dependencies
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)


add_executable(point_cloud_clustering nodes/point_cloud_clustering.cpp)

ament_target_dependencies(point_cloud_clustering
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  geometry_msgs
)

target_link_libraries(point_cloud_clustering ${PCL_LIBRARIES})

# install(TARGETS
#   cv_node
#   DESTINATION lib/${PROJECT_NAME})

# install(PROGRAMS
#   scripts/navigation_node.py
#   DESTINATION lib/${PROJECT_NAME})


install(TARGETS
  point_cloud_clustering
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()