cmake_minimum_required(VERSION 3.0.2)
project(digital_twin)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  actionlib
  actionlib_msgs
  geometry_msgs
  tf2_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# No C++ nodes to compile, this is a python-based package.
# But we should install the scripts/nodes
catkin_install_python(PROGRAMS
  client/robot_motion_action_server.py
  client/aruco_multiview_scanner.py
  control/gripper_action_server.py
  control/robotiq_gripper.py
  nodes/publish_handeye_tf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
