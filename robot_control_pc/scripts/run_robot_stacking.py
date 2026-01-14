#!/usr/bin/env python3
"""
Real Robot Block Stacking Controller

Subscribes to optimal stacking results and executes the stacking motion
using a real UR5e robot through ROS control.
"""

import json
import sys
import time
import traceback
from enum import Enum

import rospy
import actionlib
import moveit_commander
import numpy as np

from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from scipy.spatial.transform import Rotation as R


class StackingConfig:
    """Configuration parameters for robot stacking"""
    
    # Dimensions (meters)
    BLOCK_SIZE = 0.06
    GRIPPER_LENGTH = 0.20
    
    # Motion Parameters
    SAFETY_HEIGHT_ABOVE_TOWER = 0.05
    APPROACH_HEIGHT_ABOVE_BLOCK = 0.05
    GRASP_OFFSET = 0.03
    PLACE_OFFSET = 0.03
    RETREAT_DISTANCE = 0.10
    
    # Gripper
    GRIPPER_OPEN_VALUE = 0.3
    GRIPPER_CLOSE_VALUE = 0.0
    GRIPPER_TIMEOUT = 2.0
    
    # Motion Planning
    MAX_VELOCITY_SCALE = 0.2
    MAX_ACCELERATION_SCALE = 0.2
    PLANNING_TIME = 5.0
    PLANNING_ATTEMPTS = 10
    
    # Timing
    STABILIZE_MOTION = 0.5
    STABILIZE_GRASP = 1.0


class StackingState(Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    DESCEND_TO_PLACE = "DESCEND_TO_PLACE"
    PLACE = "PLACE"
    RETREAT = "RETREAT"
    DONE = "DONE"
    ERROR = "ERROR"


class RobotStackingController:
    """Controls real UR5e robot to execute block stacking operations"""

    def __init__(self):
        rospy.init_node('robot_stacking_controller', anonymous=True)
        rospy.loginfo("Initializing Robot Stacking Controller...")

        # State & Data
        self.state = StackingState.IDLE
        self.task_received = False
        self.target_pos = None
        self.target_rot = None
        self.new_block_pos = None
        self.new_block_rot = None
        self.tower_poses = []
        self.success_rate = 0.0

        # Initialize Components
        self._init_moveit()
        self._init_gripper()
        self._init_motion_client()

        # Subscribers
        self.result_sub = rospy.Subscriber(
            '/optimal_stacking/result', String, self._result_callback, queue_size=1
        )

        rospy.loginfo("✅ Robot Stacking Controller Ready. Waiting for task...")

    def _init_moveit(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")
        
        self.arm_group.set_max_velocity_scaling_factor(StackingConfig.MAX_VELOCITY_SCALE)
        self.arm_group.set_max_acceleration_scaling_factor(StackingConfig.MAX_ACCELERATION_SCALE)
        self.arm_group.set_planning_time(StackingConfig.PLANNING_TIME)
        self.arm_group.set_num_planning_attempts(StackingConfig.PLANNING_ATTEMPTS)
        
        rospy.loginfo(f"✅ MoveIt initialized (Frame: {self.arm_group.get_planning_frame()})")

    def _init_gripper(self):
        self.gripper_client = actionlib.SimpleActionClient('gripper_controller/gripper_cmd', GripperCommandAction)
        if self.gripper_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("✅ Gripper Client Connected")
        else:
            rospy.logwarn("❌ Gripper Server NOT found")

    def _init_motion_client(self):
        self.motion_client = actionlib.SimpleActionClient('robot_motion', MoveBaseAction)
        if self.motion_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("✅ Robot Motion Client Connected")
        else:
            rospy.logwarn("❌ Robot Motion Server NOT found")

    def _result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            
            # Parse Target
            t_data = data.get('target_pose', data.get('target_pos'))
            if isinstance(t_data, dict):
                self.target_pos = np.array(t_data['pos'])
                # Input [w, x, y, z] -> Scipy [x, y, z, w]
                rot = t_data.get('rot', [1, 0, 0, 0])
                self.target_rot = np.array([rot[1], rot[2], rot[3], rot[0]])
            else:
                self.target_pos = np.array(t_data)
                self.target_rot = np.array([0, 0, 0, 1])

            # Parse Block
            b_data = data.get('new_block_pose', data.get('new_block_pos'))
            if isinstance(b_data, dict):
                self.new_block_pos = np.array(b_data['pos'])
                rot = b_data.get('rot', [1, 0, 0, 0])
                self.new_block_rot = np.array([rot[1], rot[2], rot[3], rot[0]])
            else:
                self.new_block_pos = np.array(b_data)
                self.new_block_rot = np.array([0, 0, 0, 1])

            self.tower_poses = data.get('tower_poses', [])
            self.success_rate = data.get('success_rate', 0.0)

            rospy.loginfo(f"📦 Task Received: Rate={self.success_rate*100:.1f}%")
            self.task_received = True

        except Exception as e:
            rospy.logerr(f"Failed to parse stacking result: {e}")

    def _compute_safety_height(self):
        """Calculate max safe height to clear all obstacles"""
        candidates = [
            self.new_block_pos[2] + StackingConfig.APPROACH_HEIGHT_ABOVE_BLOCK,
            self.target_pos[2] + StackingConfig.APPROACH_HEIGHT_ABOVE_BLOCK
        ]
        if self.tower_poses:
            max_tower = max(p['pos'][2] for p in self.tower_poses)
            candidates.append(max_tower + StackingConfig.SAFETY_HEIGHT_ABOVE_TOWER)
            
        return max(candidates)

    def _get_downward_orientation(self, target_quat):
        """Compute optimal gripper orientation (yaw)"""
        try:
            # 1. Get Object Yaw
            r = R.from_quat(target_quat)
            yaw = r.as_euler('zyx', degrees=True)[0]
            
            # 2. Candidates (Normal vs Flipped 180)
            yaw_a = yaw + 90.0
            yaw_b = yaw_a + 180.0
            
            qa = R.from_euler('xyz', [180, 0, yaw_a], degrees=True).as_quat()
            qb = R.from_euler('xyz', [180, 0, yaw_b], degrees=True).as_quat()
            
            # 3. Minimize Rotation from Current Pose
            current_pose = self.arm_group.get_current_pose().pose
            q_curr = [current_pose.orientation.x, current_pose.orientation.y, 
                      current_pose.orientation.z, current_pose.orientation.w]
            
            if abs(np.dot(qb, q_curr)) > abs(np.dot(qa, q_curr)):
                return Quaternion(*qb)
            return Quaternion(*qa)
            
        except Exception:
            # Fallback
            yaw_def = R.from_quat(target_quat).as_euler('zyx', degrees=True)[0] + 90.0
            q_def = R.from_euler('xyz', [180, 0, yaw_def], degrees=True).as_quat()
            return Quaternion(*q_def)

    def _create_pose(self, position, orientation):
        pose = Pose()
        pose.position = Point(*position)
        pose.orientation = orientation
        return pose

    def _move_to_pose(self, pose, description="Target"):
        rospy.loginfo(f"🤖 Moving to {description}...")
        self.arm_group.set_pose_target(pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        
        if success:
            time.sleep(StackingConfig.STABILIZE_MOTION)
        else:
            rospy.logerr(f"❌ Failed to reach {description}")
        return success

    def _set_gripper(self, value):
        if not self.gripper_client:
            return False
            
        goal = GripperCommandGoal()
        goal.command.position = value
        goal.command.max_effort = 100.0
        
        self.gripper_client.send_goal(goal)
        finished = self.gripper_client.wait_for_result(rospy.Duration(StackingConfig.GRIPPER_TIMEOUT))
        
        if finished:
            return True
        rospy.logwarn("⚠️ Gripper timeout")
        return False

    def open_gripper(self):
        return self._set_gripper(StackingConfig.GRIPPER_OPEN_VALUE)

    def close_gripper(self):
        success = self._set_gripper(StackingConfig.GRIPPER_CLOSE_VALUE)
        if success:
            time.sleep(StackingConfig.STABILIZE_GRASP)
        return success

    def move_to_home(self):
        if not self.motion_client: return False
        
        rospy.loginfo("🏠 Home...")
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "HOME"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        self.motion_client.send_goal(goal)
        if self.motion_client.wait_for_result(rospy.Duration(10.0)):
            return self.motion_client.get_state() == actionlib.GoalStatus.SUCCEEDED
        return False

    def execute_stacking_sequence(self):
        rospy.loginfo("🚀 Executing Stacking Sequence")
        self.state = StackingState.APPROACH
        
        while self.state not in [StackingState.DONE, StackingState.ERROR] and not rospy.is_shutdown():
            if self.state == StackingState.APPROACH:
                self._state_approach()
            elif self.state == StackingState.DESCEND:
                self._state_descend()
            elif self.state == StackingState.GRASP:
                self._state_grasp()
            elif self.state == StackingState.LIFT:
                self._state_lift()
            elif self.state == StackingState.TRANSPORT:
                self._state_transport()
            elif self.state == StackingState.DESCEND_TO_PLACE:
                self._state_descend_to_place()
            elif self.state == StackingState.PLACE:
                self._state_place()
            elif self.state == StackingState.RETREAT:
                self._state_retreat()
                
        if self.state == StackingState.DONE:
            rospy.loginfo("✅ Sequence Complete")
        else:
            rospy.logerr("❌ Sequence Failed")

    # --- State Handlers ---

    def _state_approach(self):
        if not self.open_gripper():
            self.state = StackingState.ERROR
            return

        pos = self.new_block_pos.copy()
        pos[2] += StackingConfig.APPROACH_HEIGHT_ABOVE_BLOCK + StackingConfig.GRIPPER_LENGTH
        pose = self._create_pose(pos, self._get_downward_orientation(self.new_block_rot))
        
        if self._move_to_pose(pose, "Approach Block"):
            self.state = StackingState.DESCEND
        else:
            self.state = StackingState.ERROR

    def _state_descend(self):
        pos = self.new_block_pos.copy()
        pos[2] += StackingConfig.GRASP_OFFSET + StackingConfig.GRIPPER_LENGTH
        pose = self._create_pose(pos, self._get_downward_orientation(self.new_block_rot))
        
        if self._move_to_pose(pose, "Grasp Position"):
            self.state = StackingState.GRASP
        else:
            self.state = StackingState.ERROR

    def _state_grasp(self):
        if self.close_gripper():
            self.state = StackingState.LIFT
        else:
            self.state = StackingState.ERROR

    def _state_lift(self):
        pos = self.new_block_pos.copy()
        pos[2] = self._compute_safety_height() + StackingConfig.GRIPPER_LENGTH
        pose = self._create_pose(pos, self._get_downward_orientation(self.new_block_rot))
        
        if self._move_to_pose(pose, "Lift"):
            self.state = StackingState.TRANSPORT
        else:
            self.state = StackingState.ERROR

    def _state_transport(self):
        safety_z = self._compute_safety_height()
        pos = self.target_pos.copy()
        pos[2] = safety_z + StackingConfig.GRIPPER_LENGTH
        pose = self._create_pose(pos, self._get_downward_orientation(self.target_rot))
        
        if self._move_to_pose(pose, "Transport"):
            self.state = StackingState.DESCEND_TO_PLACE
        else:
            self.state = StackingState.ERROR

    def _state_descend_to_place(self):
        pos = self.target_pos.copy()
        pos[2] += StackingConfig.PLACE_OFFSET + StackingConfig.GRIPPER_LENGTH
        pose = self._create_pose(pos, self._get_downward_orientation(self.target_rot))
        
        if self._move_to_pose(pose, "Place Position"):
            self.state = StackingState.PLACE
        else:
            self.state = StackingState.ERROR

    def _state_place(self):
        if self.open_gripper():
            self.state = StackingState.RETREAT
        else:
            self.state = StackingState.ERROR

    def _state_retreat(self):
        pos = self.target_pos.copy()
        pos[2] += StackingConfig.RETREAT_DISTANCE + StackingConfig.GRIPPER_LENGTH
        pose = self._create_pose(pos, self._get_downward_orientation(self.target_rot))
        
        if self._move_to_pose(pose, "Retreat"):
            self.state = StackingState.DONE
        else:
            self.state = StackingState.ERROR

    def run(self):
        """Main Loop"""
        self.move_to_home()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            if self.task_received and self.state == StackingState.IDLE:
                self.execute_stacking_sequence()
                self.move_to_home()
                
                self.task_received = False
                self.state = StackingState.IDLE
                rospy.loginfo("⏳ Waiting for next task...")
            rate.sleep()


def main():
    try:
        controller = RobotStackingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal Error: {e}")
        traceback.print_exc()
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
