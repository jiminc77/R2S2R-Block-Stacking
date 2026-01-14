import argparse
import sys
import os
import json
import traceback
import numpy as np

# 1. Initialize SimulationApp
from omni.isaac.kit import SimulationApp

# Parse arguments
parser = argparse.ArgumentParser(description="Digital Twin Sync")
parser.add_argument("--width", type=int, default=1280, help="Window width")
parser.add_argument("--height", type=int, default=720, help="Window height")
args_cli, unknown = parser.parse_known_args()

config = {
    "width": args_cli.width,
    "height": args_cli.height,
    "headless": False,
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

# Import Custom Classes
from dt_utils.robots.ur5e_handeye import UR5eHandeye

# Project Settings
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------------------------------------------------------
# Helper Classes
# ----------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

class PhysicsParams:
    """Manages physical properties for simulation from calibration files."""
    def __init__(self):
        self.block_masses = {} # {id: mass}
        self.friction_block_block = 0.45
        self.friction_block_table = 0.40
        self._load()

    def _load(self):
        # 1. Masses
        m_data = load_json(os.path.join(PROJECT_ROOT, "calibration/block_masses.json"))
        for key, val in m_data.items():
            bid = val.get("block_id", -1)
            mass = val.get("mean_kg", 0.082)
            if bid >= 0: self.block_masses[bid] = mass
        print(f"[Physics] Loaded {len(self.block_masses)} block masses.")

        # 2. Block-Block Friction
        o_data = load_json(os.path.join(PROJECT_ROOT, "calibration/overhang_test_optimized.json"))
        val = o_data.get("friction_block_block", 0.45)
        self.friction_block_block = val
        print(f"[Physics] Block-Block Friction: {self.friction_block_block}")

        # 3. Block-Table Friction
        s_data = load_json(os.path.join(PROJECT_ROOT, "calibration/sliding_test_results.json"))
        frictions = [v["friction_coefficient"] for k, v in s_data.items() if "friction_coefficient" in v]
        if frictions:
            self.friction_block_table = sum(frictions) / len(frictions)
        print(f"[Physics] Block-Table Friction: {self.friction_block_table}")

    def get_mass(self, block_id):
        return self.block_masses.get(block_id, 0.082)

class PoseListener:
    """Subscribes to /block_poses and latches the first valid message."""
    def __init__(self):
        try:
            rospy.init_node('digital_twin_sync_node', anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSInitException:
            pass    

        self.poses = None
        print("[ROS] Waiting for /block_poses...")
        try:
            msg = rospy.wait_for_message("/block_poses", PoseArray, timeout=60.0)
            self._process_msg(msg)
        except rospy.ROSException:
            print("[ROS] Timeout waiting for /block_poses!")

    def _process_msg(self, msg):
        self.poses = []
        for i, p in enumerate(msg.poses):
            self.poses.append({
                "id": i,
                "pos": [p.position.x, p.position.y, p.position.z],
                "rot": [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            })
        print(f"[ROS] Received {len(self.poses)} block poses.")

class JointStateListener:
    """Subscribes to /joint_states and /gripper/joint_states."""
    def __init__(self):
        self.joint_positions = {}
        self.gripper_val = 0.0
        
        # 1. Robot Arm Subscriber
        self.sub_arm = rospy.Subscriber("/joint_states", JointState, self._callback)
        
        print("[ROS] Subscribed to /joint_states")

    def _callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos


# ----------------------------------------------------------------------------
# Main System Class
# ----------------------------------------------------------------------------

class DigitalTwinSystem:
    def __init__(self, poses, physics_params):
        self.poses = poses
        self.params = physics_params
        
        # Initialize World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Geometry Analysis for Table Placement
        self._setup_geometry()
        
        # Setup Scene
        self._setup_scene()
        
        # Setup Robot
        self._setup_robot()

        self.world.reset()
        
        # Listener hook
        self.joint_listener = None

    def _setup_geometry(self):
        min_z = min([p["pos"][2] for p in self.poses]) if self.poses else 0.0
        bottom_layer = [p for p in self.poses if (p["pos"][2] - min_z) < 0.03]
        
        if bottom_layer:
            avg_x = sum([p["pos"][0] for p in bottom_layer]) / len(bottom_layer)
            avg_y = sum([p["pos"][1] for p in bottom_layer]) / len(bottom_layer)
        else:
            avg_x, avg_y = 0.0, 0.0
            
        self.table_center_xy = (avg_x, avg_y)
        self.z_offset = 0.235 - min_z
        print(f"[Sim] Table Center: ({avg_x:.3f}, {avg_y:.3f})")

        # Set Camera
        self.world.get_physics_context().set_gravity(-9.81)
        # We can set camera via viewport API if needed, but World class doesn't expose it directly conveniently like SimContext
        # simulation_app.set_camera_view ... logic equivalent
        
    def _setup_scene(self):
        # 1. Table
        table_pos = np.array([self.table_center_xy[0], self.table_center_xy[1], 0.1])
        self.table = FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=table_pos,
            scale=np.array([0.4, 0.4, 0.2]),
            color=np.array([0.8, 0.2, 0.2]),
        )
        self.world.scene.add(self.table)
        
        # Table Material
        table_mat = PhysicsMaterial(
            prim_path="/World/Materials/TableMat",
            static_friction=self.params.friction_block_table,
            dynamic_friction=self.params.friction_block_table,
        )
        self.table.apply_physics_material(table_mat)

        # 2. Blocks
        block_mat = PhysicsMaterial(
            prim_path="/World/Materials/BlockMat",
            static_friction=self.params.friction_block_block,
            dynamic_friction=self.params.friction_block_block,
        )
        
        for i, pose in enumerate(self.poses):
            usd_path = os.path.join(PROJECT_ROOT, "assets/block_0.usd")
            prim_path = f"/World/Blocks/Block_{i}"
            
            # Use add_reference_to_stage to load USD
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            
            # Wrap in RigidPrim to control physics
            block = RigidPrim(
                prim_path=prim_path,
                name=f"block_{i}",
                position=np.array([pose["pos"][0], pose["pos"][1], pose["pos"][2] + self.z_offset]),
                orientation=np.array(pose["rot"]), # (w, x, y, z)
            )
            
            # Set Mass
            block.set_mass(self.params.get_mass(i))
            GeometryPrim(prim_path=prim_path).apply_physics_material(block_mat)
            
            self.world.scene.add(block)

    def _setup_robot(self):
        robot_usd_path = os.path.join(PROJECT_ROOT, "assets/ur5e_handeye_gripper.usd")
        
        self.robot = UR5eHandeye(
            prim_path="/World/Robot",
            name="ur5e",
            usd_path=robot_usd_path,
            activate_camera=True
        )
        self.world.scene.add(self.robot)
        
    def set_joint_listener(self, listener):
        self.joint_listener = listener

    def _sync_robot_joints(self):
        if not self.joint_listener: return
        
        target_positions = self.robot.get_joint_positions() # Get current to modify
        dof_names = self.robot.dof_names
        
        ros_joints = self.joint_listener.joint_positions
        
        # Update Arm
        for name, pos in ros_joints.items():
            if name in dof_names:
                idx = self.robot.get_dof_index(name)
                target_positions[idx] = pos
                
        # Apply strict position target
        self.robot.set_joint_positions(target_positions)


    def run(self):
        print("[Sim] Simulation Started...")
        while simulation_app.is_running():
            self.world.step(render=True)
            if self.world.is_playing():
                self._sync_robot_joints()


def main():
    try:
        # Data Loading
        physics = PhysicsParams()
        listener = PoseListener()

        if not listener.poses:
            print("[Warn] No poses received. Starting empty scene.")
            # return # Optional: Exit or continue empty

        # System Setup
        system = DigitalTwinSystem(listener.poses or [], physics)
        
        # ROS Listener
        joint_listener = JointStateListener()
        system.set_joint_listener(joint_listener)
        
        # Run
        system.run()
        
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
