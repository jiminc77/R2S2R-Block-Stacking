
import argparse
import sys
import os
import traceback
import torch
import numpy as np

# Isaac Sim / Lab Imports
sys.path.append("/isaac-sim/kit/extscore/omni.client")
from omni.isaac.lab.app import AppLauncher

# Argument Parsing
parser = argparse.ArgumentParser(description="Overhang Calibration Simulation")
parser.add_argument("--width", type=int, default=1280, help="Window width (GUI only)")
parser.add_argument("--height", type=int, default=720, help="Window height (GUI only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# If not headless, force low resolution for performance
if not args_cli.headless:
    print(f"[INFO] Running in GUI mode. Target Resolution: {args_cli.width}x{args_cli.height}")

# Livestream is no longer forced. Use --livestream 2 if needed.

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Resize Window if GUI
if not args_cli.headless:
    try:
        import omni.appwindow
        app_window = omni.appwindow.get_default_app_window()
        if app_window:
            app_window.resize(args_cli.width, args_cli.height)
            print(f"[INFO] Resized window to {args_cli.width}x{args_cli.height}")
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
class CalibrationSceneCfg(InteractiveSceneCfg):
    """Configuration for the Overhang Test Scene."""
    
    # Table (Red)
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1))
    )

    # Bottom Block (Fixed Base)
    bottom_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BottomBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/block_0.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                disable_gravity=True
            ),
            # Precision contact offset baked into asset, enforcing here for clarity
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.23),
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )

    # Top Block (Dynamic Test Subject)
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
            pos=(0.0, 0.0, 0.261), 
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )


class OverhangCalibrator:
    def __init__(self):
        print("[INFO] Initializing Simulation Context...", flush=True)
        
        self.sim_cfg = SimulationCfg(dt=0.00833, device=args_cli.device)
        self.sim = SimulationContext(self.sim_cfg)
        self.sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.2])

        # Spawn Environment
        self._spawn_global_assets()
        
        self.num_envs = 64
        scene_cfg = CalibrationSceneCfg(num_envs=self.num_envs, env_spacing=1.5)
        self.scene = InteractiveScene(scene_cfg)

        self.cached_origins = self.scene.env_origins.clone()
        
        # Setup Physics & Materials
        self._setup_usd_structure()
        self.friction_range = (0.35, 0.55)
        self.com_range_mm = (-1, 1) # Range for search
        self._create_grid()
        self._apply_static_physics()
        
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

    def _create_grid(self):
        steps = int(np.sqrt(self.num_envs))
        f_vals = torch.linspace(self.friction_range[0], self.friction_range[1], steps)
        c_vals = torch.linspace(self.com_range_mm[0], self.com_range_mm[1], steps)
        grid_f, grid_c = torch.meshgrid(f_vals, c_vals, indexing='ij')
        self.grid_friction = grid_f.flatten().to(self.sim.device)
        self.grid_com_mm = grid_c.flatten().to(self.sim.device)

    def _setup_usd_structure(self):
        stage = omni.usd.get_context().get_stage()
        self.material_prims = []
        self.top_block_mass_apis = []

        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            
            # Create & Bind Material
            mat_path = f"{env_path}/CalibrationMaterial"
            mat_prim = stage.DefinePrim(mat_path, "PhysicsMaterial")
            UsdPhysics.MaterialAPI.Apply(mat_prim)
            self.material_prims.append(mat_prim)

            for block_name in ["BottomBlock", "TopBlock"]:
                prim_path = f"{env_path}/{block_name}"
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        UsdShade.Material(mat_prim), materialPurpose="physics"
                    )
            
            # Cache MassAPI for runtime CoM updates
            top_prim = stage.GetPrimAtPath(f"{env_path}/TopBlock")
            if top_prim.IsValid():
                if not top_prim.HasAPI(UsdPhysics.MassAPI):
                    UsdPhysics.MassAPI.Apply(top_prim)
                self.top_block_mass_apis.append(UsdPhysics.MassAPI(top_prim))

    def _apply_static_physics(self):
        for i in range(self.num_envs):
            f_val = self.grid_friction[i].item()
            mat_api = UsdPhysics.MaterialAPI(self.material_prims[i])
            mat_api.CreateStaticFrictionAttr().Set(f_val)
            mat_api.CreateDynamicFrictionAttr().Set(f_val)
            mat_api.CreateRestitutionAttr().Set(0.0)
            
            # Initialize CoM (Will be updated at runtime)
            mass_api = self.top_block_mass_apis[i]
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def run(self):
        # Load Real Data
        try:
            real_data = load_json("calibration/overhang_test_real.json")
        except:
            print("[WARN] Using dummy real data.")
            real_data = {d: {"mean_m": 0.030} for d in ["+X", "-X", "+Y", "-Y"]}

        directions = ["+X", "-X", "+Y", "-Y"]
        env_scores = torch.zeros(self.num_envs, device=self.sim.device)
        results_detail = {}
        
        # Configuration
        start_z = 0.292 
        
        for d in directions:
            if d not in real_data: continue
            
            target_dist = real_data[d]["mean_m"]
            print(f"\n>>> Testing {d} @ {target_dist*1000:.1f}mm", flush=True)

            # -------------------------------------------------------------------
            # 1. State Setup (Position)
            # -------------------------------------------------------------------
            origins = self.cached_origins
            offset = torch.zeros((self.num_envs, 3), device=self.sim.device)
            if d == '+X': offset[:, 0] = target_dist
            elif d == '-X': offset[:, 0] = -target_dist
            elif d == '+Y': offset[:, 1] = target_dist
            elif d == '-Y': offset[:, 1] = -target_dist

            self.scene["bottom_block"].data.default_root_state[:, :3] = origins + torch.tensor([0,0,0.23], device=self.sim.device)
            self.scene["top_block"].data.default_root_state[:, :3] = origins + offset + torch.tensor([0,0,start_z], device=self.sim.device)
            
            # Force State Update & Reset
            self.scene["top_block"].write_root_state_to_sim(self.scene["top_block"].data.default_root_state)
            self.scene.reset()

            # -------------------------------------------------------------------
            # 2. Runtime Physics (CoM)
            # -------------------------------------------------------------------
            com_poses = torch.zeros((self.num_envs, 7), device=self.sim.device)
            c_vals_m = self.grid_com_mm / 1000.0
            
            # Align CoM offset: Positive Offset = Destabilizing (Towards Void)
            if d == '+X':   com_poses[:, 0] = c_vals_m
            elif d == '-X': com_poses[:, 0] = -c_vals_m
            elif d == '+Y': com_poses[:, 1] = c_vals_m
            elif d == '-Y': com_poses[:, 1] = -c_vals_m
            com_poses[:, 3] = 1.0 # Identity Rotation

            self.scene["top_block"].root_physx_view.set_coms(com_poses.cpu(), indices=torch.arange(self.num_envs, device="cpu"))
            
            # -------------------------------------------------------------------
            # 3. Wake Up (Velocity Kick)
            # -------------------------------------------------------------------
            # Apply tiny downward velocity (-0.05 m/s) to ensure solver activation
            vels = torch.zeros((self.num_envs, 6), device=self.sim.device)
            vels[:, 2] = -0.05
            self.scene["top_block"].write_root_velocity_to_sim(vels)
            
            # -------------------------------------------------------------------
            # 4. Simulation Loop
            # -------------------------------------------------------------------
            render = True 
            for step in range(240): # 2 Seconds
                self.sim.step(render=render)
                if step % 60 == 0:
                    self.scene["top_block"].update(dt=0.01)

            # -------------------------------------------------------------------
            # 5. Analysis
            # -------------------------------------------------------------------
            self.scene["top_block"].update(dt=0.01)
            final_z = self.scene["top_block"].data.root_pos_w[:, 2]
            
            passed = final_z > (start_z - 0.03) # Stable if it hasn't dropped >3cm
            stable_count = passed.sum().item()
            
            print(f"    Stable: {stable_count}/{self.num_envs}")
            
            env_scores += passed.int()
            results_detail[d] = passed.cpu().numpy().tolist()

        # Final Report
        best_idx = torch.argmax(env_scores).item()
        best_score = env_scores[best_idx].item()

        print("\n" + "="*50)
        print(f"CALIBRATION RESULT")
        print(f"Best Env Index: {best_idx}")
        print(f"Friction : {self.grid_friction[best_idx].item():.4f}")
        print(f"CoM Offset: {self.grid_com_mm[best_idx].item():.3f} mm")
        print(f"Score: {best_score}/4 directions")
        print("="*50)

        save_json("calibration/overhang_test_optimized.json", {
            "friction_block_block": float(self.grid_friction[best_idx].item()),
            "com_offset_mm": float(self.grid_com_mm[best_idx].item()),
            "score": int(best_score),
            "results": results_detail
        })
        print("[INFO] Saved results to calibration/overhang_test_optimized.json")

    def close(self):
        if self.sim:
            self.sim.stop()
            self.sim.clear_instance()

def main():
    calibrator = None
    try:
        calibrator = OverhangCalibrator()
        calibrator.run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        traceback.print_exc()
    finally:
        if calibrator: calibrator.close()
        simulation_app.close()

if __name__ == "__main__":
    main()