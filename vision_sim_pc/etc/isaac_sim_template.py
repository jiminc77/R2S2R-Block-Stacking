import argparse
import sys
import os
import json
import traceback
import numpy as np

# 1. Initialize SimulationApp
from omni.isaac.kit import SimulationApp

# Parse arguments
parser = argparse.ArgumentParser(description="Digital Twin Simulation Template")
parser.add_argument("--width", type=int, default=1280, help="Window width")
parser.add_argument("--height", type=int, default=720, help="Window height")
parser.add_argument("--headless", action="store_true", help="Run in headless mode") # Added headless arg support
args_cli, unknown = parser.parse_known_args()

config = {
    "width": args_cli.width,
    "height": args_cli.height,
    "headless": args_cli.headless, 
}
simulation_app = SimulationApp(config)

# 2. Imports (Must be after SimulationApp)
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
import omni.usd
from pxr import UsdPhysics, UsdShade, Gf

# Enable ROS Bridge
enable_extension("omni.isaac.ros_bridge")
import rospy
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState

# Import Custom Classes (Example)
# from dt_utils.robots.ur5e_handeye import UR5eHandeye

# Project Settings
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------------------------------------------------------
# Helper Classes
# ----------------------------------------------------------------------------

class JointStateListener:
    """Subscribes to /joint_states. Useful for syncing robot."""
    def __init__(self):
        self.joint_positions = {}
        
        try:
             rospy.init_node('isaac_sim_template_node', anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSInitException:
             pass

        self.sub_arm = rospy.Subscriber("/joint_states", JointState, self._callback)
        print("[ROS] Subscribed to /joint_states")

    def _callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

# ----------------------------------------------------------------------------
# Main System Class
# ----------------------------------------------------------------------------

class SimulationTemplate:
    def __init__(self):
        # Initialize World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane() # Default Ground Plane
        
        # Setup Scene
        self._setup_scene()
        
        # Setup Robot (Optional)
        # self._setup_robot()

        self.world.reset()
        
        # Listener hook
        self.joint_listener = None

    def _setup_scene(self):
        """
        Set up the environment (Tables, Obstacles, etc.)
        """
        # Example: Light
        # self.world.scene.add_default_ground_plane() # Already added in init

        # Example: Table
        # table_pos = np.array([0.5, 0.0, 0.1])
        # self.table = FixedCuboid(
        #     prim_path="/World/Table",
        #     name="table",
        #     position=table_pos,
        #     scale=np.array([0.4, 0.4, 0.2]),
        #     color=np.array([0.8, 0.2, 0.2]),
        # )
        # self.world.scene.add(self.table)
        
        print("[Sim] Scene setup complete.")

    def _setup_robot(self):
        """
        Load and setup the robot.
        """
        # robot_usd_path = os.path.join(PROJECT_ROOT, "assets/ur5e_handeye_gripper.usd")
        # self.robot = UR5eHandeye(
        #     prim_path="/World/Robot",
        #     name="ur5e",
        #     usd_path=robot_usd_path,
        #     activate_camera=True
        # )
        # self.world.scene.add(self.robot)
        pass

    def set_joint_listener(self, listener):
        self.joint_listener = listener

    def _sync_robot_joints(self):
        """
        Example method to sync robot joints from ROS
        """
        if not self.joint_listener: return
        
        # Example logic:
        # target_positions = self.robot.get_joint_positions()
        # ros_joints = self.joint_listener.joint_positions
        # ... map ros_joints to target_positions ...
        # self.robot.set_joint_positions(target_positions)
        pass

    def run(self):
        print("[Sim] Simulation Started...")
        while simulation_app.is_running():
            self.world.step(render=True)
            if self.world.is_playing():
                # self._sync_robot_joints()
                pass


def main():
    try:
        # Initialize Listeners
        # joint_listener = JointStateListener()
        
        # System Setup
        sim = SimulationTemplate()
        
        # Connect Listener
        # sim.set_joint_listener(joint_listener)
        
        # Run
        sim.run()
        
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
