import argparse
import sys
import os
import json
import traceback
import numpy as np
import torch

# ----------------------------------------------------------------------------
# 1. App Launcher & Configuration
# ----------------------------------------------------------------------------
from omni.isaac.lab.app import AppLauncher

# Argument Parsing
parser = argparse.ArgumentParser(description="Digital Twin Parallel Spawn (Isaac Lab)")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments to spawn")
parser.add_argument("--spacing", type=float, default=2.0, help="Spacing between environments")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------
# 2. Imports (Must be after App Launch)
# ----------------------------------------------------------------------------
import rospy
from geometry_msgs.msg import PoseArray

import omni.usd
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils

# Project Settings
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------------------------------------------------------
# 3. Helper Classes (Data & ROS)
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
        
        # 3. Block-Table Friction
        s_data = load_json(os.path.join(PROJECT_ROOT, "calibration/sliding_test_results.json"))
        frictions = [v["friction_coefficient"] for k, v in s_data.items() if "friction_coefficient" in v]
        if frictions:
            self.friction_block_table = sum(frictions) / len(frictions)

    def get_mass(self, block_id):
        return self.block_masses.get(block_id, 0.082)

class PoseListener:
    """Subscribes to /block_poses and latches the first valid message."""
    def __init__(self):
        # Note: In Isaac Lab, usually allow rospy to init if not already
        try:
            rospy.init_node('digital_twin_lab_spawn', anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSInitException:
            pass    

        self.poses = None
        print("[ROS] Waiting for /block_poses to configure scene...")
        try:
            msg = rospy.wait_for_message("/block_poses", PoseArray, timeout=60.0)
            self._process_msg(msg)
        except rospy.ROSException:
            print("[ROS] Timeout waiting for /block_poses! Creating empty scene.")

    def _process_msg(self, msg):
        self.poses = []
        for i, p in enumerate(msg.poses):
            self.poses.append({
                "id": i,
                "pos": [p.position.x, p.position.y, p.position.z],
                "rot": [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            })
        print(f"[ROS] Received {len(self.poses)} block poses. Config will spawn {len(self.poses)} blocks per env.")

# ----------------------------------------------------------------------------
# 4. Scene Configuration
# ----------------------------------------------------------------------------

@configclass
class DigitalTwinSceneCfg(InteractiveSceneCfg):
    """Configuration for the Digital Twin scene."""
    
    # 1. Table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.2), # Standard size
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)) # Center at origin, z adjusted for size
    )

    # 3. Light & Ground (Global assets usually spawned separately or via Utils)
    # Note: Lab handles ground/light via direct spawn calls or configs in Scene

# ----------------------------------------------------------------------------
# 5. Main Execution
# ----------------------------------------------------------------------------

def main():
    try:
        # Load Data
        physics = PhysicsParams()
        listener = PoseListener()

        # Initialize Simulation Context
        sim_cfg = SimulationCfg(dt=1/60.0, device=args_cli.device)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])

        # Setup Scene Config
        scene_cfg = DigitalTwinSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.spacing)
        
        # Dynamically add blocks relative to received poses
        # We spawn N blocks *per environment*.
        # Note: We will set their positions in the reset loop.
        initial_block_defs = []
        
        if listener.poses:
            
            # Just add N RigidObjects to the config
            for i, _ in enumerate(listener.poses):
                block_name = f"block_{i}"
                block_cfg = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Blocks/Block_{i}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=os.path.join(PROJECT_ROOT, "assets/block_0.usd"),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                        mass_props=sim_utils.MassPropertiesCfg(mass=physics.get_mass(i))
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, -10.0), # Spawn far away initially, reset later
                    )
                )
                # Dynamically attach to config instance
                setattr(scene_cfg, block_name, block_cfg)
                initial_block_defs.append(block_name)

        # Create Scene
        print("[Sim] Creating Interactive Scene...")
        scene = InteractiveScene(scene_cfg)
        
        # Spawn Light & Ground
        sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9)).func("/World/Light", sim_utils.DomeLightCfg(intensity=1000.0))
        sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

        # Reset
        print("[Sim] Resetting Simulation...")
        sim.reset()

        # Apply Initial State from ROS (Replicate to all Envs)
        print(f"[Sim] Applying ROS poses to {args_cli.num_envs} environments...")
        if listener.poses:
             # Calculate offset to center on table
            min_z = min([p["pos"][2] for p in listener.poses])
            z_offset = 0.201 - min_z # 0.2 is table top

            for i, p_info in enumerate(listener.poses):
                block_name = f"block_{i}"
                if hasattr(scene, block_name): # 'scene[name]' access usually works if defined in config
                    # Construct Tensor for N environments
                    # Local pose relative to environment origin
                    # ROS pose is likely global-ish or relative to camera.
                    # Assuming ROS pose is close to (0,0,0) center.
                    
                    pos_tensor = torch.zeros((args_cli.num_envs, 3), device=sim.device)
                    rot_tensor = torch.zeros((args_cli.num_envs, 4), device=sim.device)
                    
                    # Fill with the single ROS pose
                    raw_pos = torch.tensor(p_info["pos"], device=sim.device)
                    raw_rot = torch.tensor(p_info["rot"], device=sim.device) # (w, x, y, z)
                    
                    # Adjust Height
                    # We assume ROS Z is relative to some ground, we map it to table top
                    # If ROS pose is raw camera frame, this might need more math.
                    # For now, using logic from original script:
                    final_pos = raw_pos.clone()
                    final_pos[2] += z_offset
                    
                    pos_tensor[:] = final_pos
                    rot_tensor[:] = raw_rot
                    
                    # Apply
                    # Note: accessing scene entities dynamically
                    entity = getattr(scene, block_name) # This works because InteractiveScene creates attributes
                    entity.write_root_pose_to_sim(torch.cat([pos_tensor, rot_tensor], dim=-1))
                    entity.reset()

        print("[Sim] Simulation Ready. Running loop...")
        
        while simulation_app.is_running():
            # Step the simulation
            sim.step()
            
            # Here you can implement parallel logic, RL inference, etc.
            # scene.update(dt=sim.get_physics_dt()) # Update scene buffers if needed
            
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
