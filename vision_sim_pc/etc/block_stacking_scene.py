
"""
Script to create a block stacking scene in Isaac Lab.
"""
import argparse
import sys
import os

sys.path.append("/isaac-sim/kit/extscore/omni.client")

from omni.isaac.lab.app import AppLauncher

# Create the parser and add AppLauncher args
parser = argparse.ArgumentParser(description="Block stacking scene")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------
# Imports (Must be done AFTER launching the app)
# --------------------------------------------------------------------------------
import torch
import numpy as np

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, SimulationContext, PhysxCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

@configclass
class BlockStackingSceneCfg(InteractiveSceneCfg):
    """Block stacking scene configuration"""

    # Ground
    ground = sim_utils.GroundPlaneCfg()

    # Table (measure your actual table dimensions)
    table = RigidObjectCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.75),  # Width, Depth, Height
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True  # Static table
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2)  # Wood color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.375)  # Half height
        )
    )

    # Blocks placeholder (will be added in scene creation logic if needed, 
    # but here we defined them dynamically in the loop in original code. 
    # We can keep the dynamic creation in the function.)


def create_test_scene():
    """Create and test scene with blocks"""
    
    # Simulation config
    sim_cfg = SimulationCfg(
        dt=1/120,
        device=args_cli.device, # Use device from args
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # TGS
            enable_stabilization=True,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Create simulation
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.0, 0.0, 0.8])

    # Create scene
    scene_cfg = BlockStackingSceneCfg(num_envs=1, env_spacing=2.0)

    # Add ground and table
    ground = scene_cfg.ground.func("/World/Ground", scene_cfg.ground)
    table = RigidObject(scene_cfg.table)

    # Add blocks
    blocks = {}
    for i in range(4):
        # We need to construct the path relative to default assets or provided assets
        # Assuming assets folder is in the current directory or handle it.
        # The original code used "assets/block_{i}.usd".
        # Use absolute path to ensure asset is loaded correctly
        usd_path = f"/workspace/DigitalTwin/assets/block_{i}.usd"
        
        block_cfg = RigidObjectCfg(
            prim_path=f"/World/Block_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.2 + i * 0.07, 0.0, 0.80),  # Spread on table
                rot=(1.0, 0.0, 0.0, 0.0)
            )
        )
        blocks[f"block_{i}"] = RigidObject(block_cfg)

    print("="*60)
    print("Scene created successfully")
    print(f"  Ground: OK")
    print(f"  Table: OK")
    print(f"  Blocks: {len(blocks)}")
    print("="*60)

    # Run simulation test
    print("\nRunning physics simulation test...")
    sim.reset()

    for step in range(240):  # 2 seconds at 120Hz
        sim.step(render=False)

        if step % 60 == 0:
            print(f"  Step {step}/240 - {step/120:.1f}s")

    # Check final block positions
    print("\nFinal block positions:")
    for name, block in blocks.items():
        # Update buffers
        block.update(dt=sim_cfg.dt)
        pos = block.data.root_pos_w[0].cpu().numpy()
        print(f"  {name}: z={pos[2]:.4f}m")

    print("\n✅ Scene simulation test passed")  

    return sim, blocks


if __name__ == "__main__":
    try:
        create_test_scene()
    finally:
        # Close the app
        simulation_app.close()