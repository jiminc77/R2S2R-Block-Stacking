#!/usr/bin/env python3
import sys
import rospy
import actionlib
import tf2_ros
import moveit_commander
import numpy as np

from move_base_msgs.msg import MoveBaseAction, MoveBaseResult

class RobotMotionActionServer:

    def __init__(self, name: str):
        self._action_name = name
        
        # 1. Initialize MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        
        # Retry loop for MoveGroup connection
        self.group = None
        for i in range(10):
            try:
                self.group = moveit_commander.MoveGroupCommander("manipulator", wait_for_servers=5.0)
                break
            except RuntimeError:
                rospy.logwarn(f"Waiting for MoveGroup... ({i+1}/10)")
                rospy.sleep(2.0)
        
        if not self.group:
            rospy.logerr("Critical: Could not connect to MoveGroup.")
            sys.exit(1)
        
        # 2. Configure Safety Parameters
        self.group.set_planning_time(5.0)
        self.group.set_max_velocity_scaling_factor(0.3)      # 30% Speed
        self.group.set_max_acceleration_scaling_factor(0.3)  # 30% Accel
        
        # 3. TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 4. Start Action Server
        self._as = actionlib.SimpleActionServer(
            self._action_name, 
            MoveBaseAction, 
            execute_cb=self.execute_cb, 
            auto_start=False
        )
        self._as.start()
        print(f"✅ Action Server '{self._action_name}' Ready.")

    def execute_cb(self, goal):
        result = MoveBaseResult()
        
        if self._as.is_preempt_requested():
            self._as.set_preempted()
            return


        PRESET_JOINTS = {
            "HOME":   [-180, -90, -90, -90, 90, -180],
            "PREP_1": [-180, -90, -90, -90, 180, -180],
            "PREP_2": [-180, -90, -90, -180, 180, -180],
            "PREP_3": [-180, -90, -90, -180, 270, -180],
            "CENTER": [-180, -50, -146, -187, -90, 0],
            "RIGHT":  [-109, -121, -83, -131, 48, -19],
            "LEFT":   [-230, -129, -69, -165, -197, 180],
        }
        
        target_frame = goal.target_pose.header.frame_id
        
        if target_frame in PRESET_JOINTS:
            rospy.loginfo(f"Motion Request: {target_frame}")
            joints_rad = np.deg2rad(PRESET_JOINTS[target_frame]).tolist()
            
            try:
                success = self.group.go(joints_rad, wait=True)
                self.group.stop()
                
                if success:
                    self._as.set_succeeded(result)
                else:
                    self._as.set_aborted(result)
            except Exception as e:
                rospy.logerr(f"Motion Failed: {e}")
                self._as.set_aborted(result)
            return

        # ---------------------------------------------------------
        # Fallback: Cartesian Move (Only if needed)
        # ---------------------------------------------------------
        # (This part is preserved as a fallback, but primary operation uses Presets)
        rospy.logwarn(f"Unknown target '{target_frame}', aborting for safety.")
        self._as.set_aborted(result)

if __name__ == "__main__":
    rospy.init_node('robot_motion_server')
    server = RobotMotionActionServer('robot_motion')
    rospy.spin()