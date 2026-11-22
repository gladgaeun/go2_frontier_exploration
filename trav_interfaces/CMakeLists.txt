cmake_minimum_required(VERSION 3.8)
project(trav_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detections.msg"
  "msg/ImageScore.msg"
  "srv/GeoCosts.srv"
  "srv/FrontierTargets.srv"
  "srv/TargetGoal.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
