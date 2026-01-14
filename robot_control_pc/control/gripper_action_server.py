#!/usr/bin/env python3
import os
import sys
import threading
import time

# Ensure local modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import rospy
import actionlib
from robotiq_gripper import RobotiqGripperUSB
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

try:
    from control_msgs.msg import GripperCommandAction, GripperCommandResult, GripperCommandFeedback
except ImportError:
    # Fallback for systems where control_msgs is not in the default path
    sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
    from control_msgs.msg import GripperCommandAction, GripperCommandResult, GripperCommandFeedback

class GripperActionServer:
    """
    Action Server for Robotiq 140 Gripper.
    - Implements control_msgs/GripperCommandAction
    - Publishes continuous /gripper/joint_states
    """
    def __init__(self, name, port="/dev/ttyUSB0", slave_id=9):
        self._action_name = name
        self.port = port
        self._lock = threading.Lock()
        
        # Initialize Hardware
        self.gripper = RobotiqGripperUSB(portname=self.port, slave_id=slave_id)
        
        print(f"🔌 Connecting to gripper on {self.port}...")
        if not self.gripper.connect():
            rospy.logerr(f"Failed to connect to gripper on {self.port}")
            sys.exit(1)
            
        print("🔄 Resetting and Activating gripper...")
        if not self.gripper.activate():
             rospy.logerr("Failed to activate gripper")
             sys.exit(1)

        # Joint State Publisher
        self.joint_pub = rospy.Publisher('/gripper/joint_states', JointState, queue_size=1)
        self.output_thread = threading.Thread(target=self._publish_state_loop)
        self.output_thread.daemon = True
        self.output_thread.start()

        # Action Server
        self._as = actionlib.SimpleActionServer(
            self._action_name, 
            GripperCommandAction, 
            execute_cb=self.execute_cb, 
            auto_start=False
        )
        self._as.start()
        print(f"✅ Gripper Action Server '{self._action_name}' started")

    def _publish_state_loop(self):
        """Continuous loop to publish current gripper state"""
        rate = rospy.Rate(5) # 5Hz for stability
        while not rospy.is_shutdown():
            status = None
            with self._lock:
                status = self.gripper.get_status()
            
            if status:
                cur_pos_reg = status['position']
                # Map 0..255 (Register) to 0.0..0.7 (Radians)
                # Note: Register 0 is FULL OPEN (0.0 rad), 255 is CLOSED (0.7 rad approx)
                cur_pos_rad = (cur_pos_reg / 255.0) * 0.7
                
                msg = JointState()
                msg.header = Header()
                msg.header.stamp = rospy.Time.now()
                msg.name = ['finger_joint']
                msg.position = [cur_pos_rad]
                msg.velocity = []
                msg.effort = []
                
                self.joint_pub.publish(msg)
            
            rate.sleep()

    def execute_cb(self, goal):
        result = GripperCommandResult()
        feedback = GripperCommandFeedback()
        
        if self._as.is_preempt_requested():
            self._as.set_preempted()
            return

        target_width = goal.command.position
        
        # Convert Meters (0.0=Closed, 0.140=Open) to Register (255=Closed, 0=Open)
        req_pos_m = max(0.0, min(0.140, target_width))
        req_pos_reg = int((1.0 - (req_pos_m / 0.140)) * 255)
        req_pos_reg = max(0, min(255, req_pos_reg))
        
        rospy.loginfo(f"Gripper Goal: {target_width:.4f}m -> Register {req_pos_reg}")
        
        try:
            with self._lock:
                # 0x0900 (Act=1, GTO=1) | req_pos_reg | 0xFF80 (Max Speed/50% Force)
                self.gripper.write_regs(0x03E8, [0x0900, req_pos_reg, 0xFF80])
            
            # Monitor until reached or timeout (5s)
            start_time = rospy.Time.now()
            while (rospy.Time.now() - start_time).to_sec() < 5.0:
                if self._as.is_preempt_requested():
                    self._as.set_preempted()
                    return
                
                status = None
                with self._lock:
                    status = self.gripper.get_status()
                
                if status:
                    # gOBJ: 0=Moving, 1=Stopped(Object), 2=Stopped(Open), 3=Completed
                    obj_status = status['object']
                    cur_pos_reg = status['position']
                    cur_pos_m = 0.140 * (1.0 - (cur_pos_reg / 255.0))
                    
                    feedback.position = cur_pos_m
                    feedback.stalled = (obj_status == 1 or obj_status == 2)
                    feedback.reached_goal = (obj_status == 3)
                    self._as.publish_feedback(feedback)
                    
                    if obj_status != 0: # Stopped or Reached
                        result.position = cur_pos_m
                        result.stalled = feedback.stalled
                        result.reached_goal = feedback.reached_goal
                        self._as.set_succeeded(result)
                        return
                
                time.sleep(0.1)
                
            rospy.logwarn("Gripper action timed out")
            self._as.set_aborted(result)

        except Exception as e:
            rospy.logerr(f"Gripper exception: {e}")
            self._as.set_aborted(result)

    def shutdown(self):
        self.gripper.disconnect()

if __name__ == "__main__":
    rospy.init_node('gripper_action_server')
    port = rospy.get_param('~port', "/dev/ttyUSB0")
    
    server = GripperActionServer('gripper_controller/gripper_cmd', port=port)
    rospy.on_shutdown(server.shutdown)
    rospy.spin()
