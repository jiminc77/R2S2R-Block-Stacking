cmake_minimum_required(VERSION 3.0.2)
project(digital_twin)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  actionlib
  geometry_msgs
  tf2_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/robot_motion_action_server.py
  scripts/aruco_multiview_scanner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
