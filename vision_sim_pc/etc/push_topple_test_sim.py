
import argparse
import sys
import os
import traceback
import torch
import time
import numpy as np

# Isaac Sim / Lab Imports
sys.path.append("/isaac-sim/kit/extscore/omni.client")
from omni.isaac.lab.app import AppLauncher

# Argument Parsing
parser = argparse.ArgumentParser(description="Push Topple Verification Simulation")
parser.add_argument("--width", type=int, default=1280, help="Window width (GUI only)")
parser.add_argument("--height", type=int, default=720, help="Window height (GUI only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# GUI Mode Info
if not args_cli.headless:
    print(f"[INFO] Running in GUI mode. Target Resolution: {args_cli.width}x{args_cli.height}")

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Window Resizing
if not args_cli.headless:
    try:
        import omni.appwindow
        app_window = omni.appwindow.get_default_app_window()
        if app_window:
            app_window.resize(args_cli.width, args_cli.height)
    except Exception as e:
        print(f"[WARN] Failed to resize window: {e}")

# Isaac Lab & USD Imports
from pxr import Usd, UsdPhysics, UsdShade, Gf
import omni.usd
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils

# Utilities
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

try:
    from dt_utils.json_io import save_json, load_json
except ImportError:
    import json
    def save_json(path, data): json.dump(data, open(path, 'w'), indent=4)
    def load_json(path): return json.load(open(path, 'r'))


@configclass
class PushToppleSceneCfg(InteractiveSceneCfg):
    """Configuration for the Push Topple Test Scene."""
    
    # Table (Red)
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(  
            size=(0.4, 0.4, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1))
    )

    # Bottom Block (Dynamic)
    bottom_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BottomBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/block_0.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, 
                disable_gravity=False
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.231),
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )

    # Top Block (Dynamic)
    top_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TopBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/block_1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.292), 
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )

    # Pusher (Kinematic)
    pusher = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pusher",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.017, 0.025), # Length=10cm, Thickness=1.7cm, Height=2.5cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.2, 0.0, 0.5), # Safe start position
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )


class PushToppleVerifier:
    def __init__(self):
        print("[INFO] Initializing Simulation Context...", flush=True)
        
        # Load Data
        self.real_data = load_json("calibration/push_topple_real.json")
        self.results_list = self.real_data.get('results', [])
        self.test_direction = self.real_data.get('test_direction', '-X')
        
        # Determine Batch Size
        self.num_envs = min(len(self.results_list), 128)
        print(f"[INFO] Loaded {len(self.results_list)} test cases. Simulating batch of {self.num_envs}.", flush=True)

        self.sim_cfg = SimulationCfg(dt=1/240.0, device=args_cli.device)
        self.sim = SimulationContext(self.sim_cfg)
        self.sim.set_camera_view([0.8, 0.8, 0.6], [0.0, 0.0, 0.2])

        # Spawn Environment
        self._spawn_global_assets()
        scene_cfg = PushToppleSceneCfg(num_envs=self.num_envs, env_spacing=1.0)
        self.scene = InteractiveScene(scene_cfg)
        self.cached_origins = self.scene.env_origins.clone()
        
        # Setup Physics & Materials
        self._setup_usd_structure()
        self._apply_calibrated_physics()
        
        # Warm Start
        print("[INFO] Warm Starting Simulation...", flush=True)
        self.sim.reset()
        for _ in range(10):
            self.sim.step(render=False)
        print("[INFO] Simulation Ready.", flush=True)

    def _spawn_global_assets(self):
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)

    def _setup_usd_structure(self):
        stage = omni.usd.get_context().get_stage()
        self.mat_block_prims = []
        self.mat_table_prims = []
        self.block_mass_apis = []
        
        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            
            # 1. Material for Blocks
            mat_block_path = f"{env_path}/MaterialBlock"
            mat_block = stage.DefinePrim(mat_block_path, "PhysicsMaterial")
            UsdPhysics.MaterialAPI.Apply(mat_block)
            self.mat_block_prims.append(mat_block)

            # 2. Material for Table
            mat_table_path = f"{env_path}/MaterialTable"
            mat_table = stage.DefinePrim(mat_table_path, "PhysicsMaterial")
            UsdPhysics.MaterialAPI.Apply(mat_table)
            self.mat_table_prims.append(mat_table)

            # Bind Table
            prim_table = stage.GetPrimAtPath(f"{env_path}/Table")
            if prim_table.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim_table).Bind(
                    UsdShade.Material(mat_table), materialPurpose="physics"
                )

            # Bind Blocks
            for obj_name in ["BottomBlock", "TopBlock"]:
                prim_path = f"{env_path}/{obj_name}"
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        UsdShade.Material(mat_block), materialPurpose="physics"
                    )
                    
                    # Apply Mass API
                    if not prim.HasAPI(UsdPhysics.MassAPI):
                        UsdPhysics.MassAPI.Apply(prim)
                    self.block_mass_apis.append(UsdPhysics.MassAPI(prim))

    def _apply_calibrated_physics(self):
        # 1. Block-Block Friction
        if os.path.exists("calibration/overhang_test_optimized.json"):
            opt_params = load_json("calibration/overhang_test_optimized.json")
            f_block_block = opt_params.get('friction_block_block', 0.45)
            print(f"[INFO] Loaded Block-Block Friction: {f_block_block:.4f}")
        else:
            f_block_block = 0.45
            print("[WARN] Using default Block-Block Friction: 0.45")

        # 2. Block-Table Friction
        if os.path.exists("calibration/sliding_test_results.json"):
            slide_data = load_json("calibration/sliding_test_results.json")
            frictions = [v["friction_coefficient"] for k, v in slide_data.items() if "friction_coefficient" in v]
            
            if frictions:
                f_block_table = sum(frictions) / len(frictions)
                print(f"[INFO] Loaded Block-Table Friction (Avg): {f_block_table:.4f}")
            else:
                f_block_table = 0.40
                print("[WARN] File found but no friction data. Using default: 0.40")
        else:
            f_block_table = 0.40
            print("[WARN] Using default Block-Table Friction: 0.40")
            
        # 3. Mass
        mass_kg = 0.082
        if os.path.exists("calibration/block_masses.json"):
            mass_data = load_json("calibration/block_masses.json")
            if "Block_1" in mass_data:
                mass_kg = mass_data["Block_1"].get("mean_kg", 0.082)
            print(f"[INFO] Loaded Block Mass: {mass_kg:.4f} kg")

        # 4. Calculation for Materials
        mat_block_friction = f_block_block
        mat_table_friction = max(0.0, 2.0 * f_block_table - f_block_block)
        
        print(f"[INFO] Derived Material Properties: Block={mat_block_friction:.4f}, Table={mat_table_friction:.4f}")

        # Apply
        for i in range(self.num_envs):
            # Block Material
            mb = UsdPhysics.MaterialAPI(self.mat_block_prims[i])
            mb.CreateStaticFrictionAttr().Set(mat_block_friction)
            mb.CreateDynamicFrictionAttr().Set(mat_block_friction)
            mb.CreateRestitutionAttr().Set(0.0)

            # Table Material
            mt = UsdPhysics.MaterialAPI(self.mat_table_prims[i])
            mt.CreateStaticFrictionAttr().Set(mat_table_friction)
            mt.CreateDynamicFrictionAttr().Set(mat_table_friction)
            mt.CreateRestitutionAttr().Set(0.0)
            
            # Mass: Apply to All
            for mass_api in self.block_mass_apis:
                 mass_api.CreateMassAttr().Set(mass_kg)


    def run(self):
        print("\n>>> Setting up Test Cases...", flush=True)
        
        pusher_start_pos = torch.zeros((self.num_envs, 3), device=self.sim.device)
        push_vectors = torch.zeros((self.num_envs, 3), device=self.sim.device)
        push_distances = torch.zeros(self.num_envs, device=self.sim.device)
        push_speeds = torch.zeros(self.num_envs, device=self.sim.device)
        
        top_block_z_sim = 0.292 
        
        for i in range(self.num_envs):
            case = self.results_list[i]
            h_mm = case['push_height_mm']
            speed = case.get('push_speed', 0.01)
            dist = case.get('push_distance', 0.08)
            direction = case.get('push_direction', self.test_direction)

            # Reset Block Positions
            self.scene["top_block"].data.default_root_state[i, :3] = self.cached_origins[i] + torch.tensor([0, 0, top_block_z_sim], device=self.sim.device)
            self.scene["bottom_block"].data.default_root_state[i, :3] = self.cached_origins[i] + torch.tensor([0, 0, 0.231], device=self.sim.device)

            # Pusher Calculation
            pusher_z = 0.2 + (h_mm / 1000.0) # Height relative to table surface (0.2)
            start_offset = 0.10 # 5cm gap + 5cm half-length
            
            if direction == '-X':
                pusher_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.sim.device)
                pusher_pos_offset = torch.tensor([start_offset, 0.0, 0.0], device=self.sim.device)
                push_vec = torch.tensor([-1.0, 0.0, 0.0], device=self.sim.device)
            elif direction == '+X':
                pusher_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.sim.device)
                pusher_pos_offset = torch.tensor([-start_offset, 0.0, 0.0], device=self.sim.device)
                push_vec = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device)
            elif direction == '+Y':
                pusher_quat = torch.tensor([0.707, 0.0, 0.0, 0.707], device=self.sim.device)
                pusher_pos_offset = torch.tensor([0.0, -start_offset, 0.0], device=self.sim.device)
                push_vec = torch.tensor([0.0, 1.0, 0.0], device=self.sim.device)
            elif direction == '-Y':
                pusher_quat = torch.tensor([0.707, 0.0, 0.0, 0.707], device=self.sim.device)
                pusher_pos_offset = torch.tensor([0.0, start_offset, 0.0], device=self.sim.device)
                push_vec = torch.tensor([0.0, -1.0, 0.0], device=self.sim.device)
            else:
                # Default -X
                pusher_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.sim.device)
                pusher_pos_offset = torch.tensor([start_offset, 0.0, 0.0], device=self.sim.device)
                push_vec = torch.tensor([-1.0, 0.0, 0.0], device=self.sim.device)

            abs_pusher_pos = self.cached_origins[i] + pusher_pos_offset
            abs_pusher_pos[2] = pusher_z 
            
            self.scene["pusher"].data.default_root_state[i, :3] = abs_pusher_pos
            self.scene["pusher"].data.default_root_state[i, 3:7] = pusher_quat
            
            pusher_start_pos[i] = abs_pusher_pos
            push_vectors[i] = push_vec
            push_distances[i] = dist
            push_speeds[i] = speed

        self.scene.reset()
        
        # -------------------------------------------------------------------
        # User Wait Loop
        # -------------------------------------------------------------------
        print(f"\n>>> Simulating {self.num_envs} cases...", flush=True)
        print("[INFO] Waiting 5 seconds before starting...", flush=True)
        start_wait = time.time()
        while time.time() - start_wait < 5.0:
             self.sim.step(render=True)
        
        # Initial settle
        for _ in range(60): 
            self.sim.step(render=False)
        
        # Simulation Loop
        max_dist = push_distances.max().item()
        min_speed = push_speeds.min().item()
        total_time = (max_dist + 0.08) / min_speed 
        total_steps = int(total_time * 240)
        
        print(f"Total Steps: {total_steps} ({total_time:.2f}s)", flush=True)

        for step in range(total_steps):
            current_time = step * (1/240.0)
            dists_so_far = push_speeds * current_time
            active_mask = dists_so_far < (push_distances + 0.05)
            
            if active_mask.any():
                clamped_dists = torch.min(dists_so_far, push_distances + 0.05)
                final_target_pos = pusher_start_pos + push_vectors * clamped_dists.unsqueeze(1)
                
                self.scene["pusher"].data.root_pos_w[:, :] = final_target_pos
                self.scene["pusher"].write_root_state_to_sim(self.scene["pusher"].data.root_state_w)
            
            self.sim.step(render=True)
            
            if step % 600 == 0:
                print(f"Step {step}/{total_steps}", flush=True)

        # Post-push settle
        for i in range(120): # 0.5s
            self.sim.step(render=True)


        # -------------------------------------------------------------------
        # Analysis
        # -------------------------------------------------------------------
        print("\n>>> Analysis...", flush=True)
        self.scene["top_block"].update(dt=0.0)
        self.scene["bottom_block"].update(dt=0.0)
        
        final_top_pos = self.scene["top_block"].data.root_pos_w.clone()
        final_bot_pos = self.scene["bottom_block"].data.root_pos_w.clone()
        
        init_top_pos = self.cached_origins + torch.tensor([0,0, top_block_z_sim], device=self.sim.device)
        init_bot_pos = self.cached_origins + torch.tensor([0,0, 0.231], device=self.sim.device)
        
        matches = 0
        results_out = []
        
        for i in range(self.num_envs):
            # Calculate movements
            top_move_xy = torch.norm(final_top_pos[i, :2] - init_top_pos[i, :2]).item()
            bot_move_xy = torch.norm(final_bot_pos[i, :2] - init_bot_pos[i, :2]).item()
            
            top_z = final_top_pos[i, 2].item()
            bot_z = final_bot_pos[i, 2].item()
            
            # Outcome Logic
            thresh_top_stable = 0.28
            thresh_bot_move = 0.01 # 1cm
            thresh_bot_table = 0.20 # Table height
            
            outcome = "unknown"
            
            if top_z > thresh_top_stable:
                outcome = "stable"
            else:
                # Top fell (Z <= 0.28)
                bot_moved = (bot_move_xy > thresh_bot_move)
                bot_fell = (bot_z < thresh_bot_table)
                
                if bot_moved or bot_fell:
                    outcome = "both_topple"
                else:
                    outcome = "top_slides"

            real_outcome = self.results_list[i]['outcome']
            sim_outcome = outcome
            match = (sim_outcome == real_outcome)
            matches += int(match)
            
            results_out.append({
                "push_height_mm": self.results_list[i]['push_height_mm'],
                "real_outcome": real_outcome,
                "sim_outcome": sim_outcome,
                "values": {
                    "bot_move_xy": bot_move_xy,
                    "top_z": top_z,
                    "bot_z": bot_z
                },
                "match": match
            })
            
            if i < 10: # Print first 10
                print(f"Env {i}: H={self.results_list[i]['push_height_mm']}mm | Real={real_outcome} vs Sim={sim_outcome} | {'MATCH' if match else 'MISS'}")

        accuracy = matches / self.num_envs
        print(f"\nTotal Accuracy: {matches}/{self.num_envs} ({accuracy*100:.1f}%)")
        
        save_json("calibration/push_topple_sim_results.json", {
            "accuracy": accuracy,
            "results": results_out
        })
        print("[INFO] Results saved to calibration/push_topple_sim_results.json")

    def close(self):
        if self.sim:
            self.sim.stop()
            self.sim.clear_instance()

def main():
    verifier = None
    try:
        verifier = PushToppleVerifier()
        verifier.run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        traceback.print_exc()
    finally:
        if verifier: verifier.close()
        simulation_app.close()

if __name__ == "__main__":
    main()