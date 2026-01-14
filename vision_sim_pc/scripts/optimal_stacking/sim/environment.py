import os
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.assets import RigidObjectCfg

from optimal_stacking.sim.scene import DigitalTwinSceneCfg, TABLE_POS


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def quat_mul(q1, q2):
    """Multiply two quaternions (wxyz format). q1, q2: (N, 4)"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=1)


class DomainRandomization:
    """Handles domain randomization logic (Pose & Velocity)."""
    def __init__(self):
        self.pos_std = 0.002
        self.z_drop_std = 0.003
        self.rot_noise_yaw_std = 5.0
        self.rot_noise_tilt_std = 2.0
        self.lin_vel_std = 0.01
        self.ang_vel_std = 0.05

    def apply_pose_noise(self, position, num_envs, device="cuda:0"):
        pos_arr = np.array(position) if isinstance(position, (list, tuple, np.ndarray)) else np.array([0,0,0])

        # Position Noise
        pos_noise = np.random.normal(0, self.pos_std, size=(num_envs, 3))
        pos_noise[:, 2] = np.random.normal(0, self.z_drop_std, size=num_envs)
        noisy_positions = pos_arr + pos_noise
        
        # Rotation Noise
        yaw = np.random.normal(0, self.rot_noise_yaw_std, size=num_envs)
        pitch = np.random.normal(0, self.rot_noise_tilt_std, size=num_envs)
        roll = np.random.normal(0, self.rot_noise_tilt_std, size=num_envs)
        
        rotations = R.from_euler('xyz', np.stack([roll, pitch, yaw], axis=1), degrees=True)
        quats_xyzw = rotations.as_quat()
        quats_wxyz = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]])
        
        return torch.tensor(noisy_positions, dtype=torch.float32, device=device), torch.tensor(quats_wxyz, dtype=torch.float32, device=device)

    def sample_velocity_noise(self, num_envs, device="cuda:0"):
        lin_vel = torch.randn(num_envs, 3, device=device) * self.lin_vel_std
        ang_vel = torch.randn(num_envs, 3, device=device) * self.ang_vel_std
        return lin_vel, ang_vel


class StackingVerificationEnv:
    def __init__(self, sim_context, tower_poses, physics_params, num_envs=128):
        self.sim = sim_context
        self.num_envs = num_envs
        self.physics = physics_params
        self.tower_poses = tower_poses
        self.device = self.sim.device
        
        self.scene_cfg = DigitalTwinSceneCfg(num_envs=num_envs, env_spacing=2.0)
        self.block_names = []

        # Calculate Table Center
        min_z = min([p["pos"][2] for p in self.tower_poses]) if self.tower_poses else 0.0
        bottom_layer = [p for p in self.tower_poses if (p["pos"][2] - min_z) < 0.03]
        
        if bottom_layer:
            avg_x = sum([p["pos"][0] for p in bottom_layer]) / len(bottom_layer)
            avg_y = sum([p["pos"][1] for p in bottom_layer]) / len(bottom_layer)
        else:
            avg_x, avg_y = 0.0, 0.0
            
        print(f"[Sim] Table Center: ({avg_x:.3f}, {avg_y:.3f})")
        self.table_center_xy = torch.tensor([avg_x, avg_y], device=self.device, dtype=torch.float32)

        self.scene_cfg.table.init_state.pos = (avg_x, avg_y, TABLE_POS[2])
        
        # Add existing tower blocks
        for i, pose_info in enumerate(tower_poses):
            bname = f"block_{i}"
            self.block_names.append(bname)
            
            usd_path = os.path.join(self.physics.project_root, "assets/block_0.usd")
            mass = self.physics.get_mass(i)
            
            b_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{bname}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass)
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0))
            )
            setattr(self.scene_cfg, bname, b_cfg)

        # Add New Block (Candidate)
        self.new_block_name = "new_block"
        new_b_mass = self.physics.get_mass(-1)
        usd_path_new = os.path.join(self.physics.project_root, "assets/block_0.usd")
        
        b_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{self.new_block_name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path_new,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=new_b_mass)
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0))
        )
        setattr(self.scene_cfg, self.new_block_name, b_cfg)
        
        self.scene = InteractiveScene(self.scene_cfg)
        self.randomizer = DomainRandomization()
        
        sim_utils.DomeLightCfg(intensity=1000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1000.0))
        sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
        
        self.sim.reset()

    def verify_batch(self, candidates):
        """Runs physics verification for a BATCH of candidates in parallel."""
        num_cands = len(candidates)
        if num_cands == 0:
            return []
            
        envs_per_cand = self.num_envs // num_cands
        if envs_per_cand == 0:
            print(f"[Sim] Warning: Too many candidates ({num_cands}) for {self.num_envs} envs. truncating.")
            num_cands = self.num_envs
            candidates = candidates[:num_cands]
            envs_per_cand = 1
            
        if not self.sim.is_playing():
            self.sim.play()
        
        env_origins = self.scene.env_origins
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        # Reset Fixed Tower Poses
        for i, pose_info in enumerate(self.tower_poses):
            bname = self.block_names[i]
            entity = self.scene[bname]
            p = pose_info['pos']
            r = pose_info.get('rot', [1, 0, 0, 0])
            
            pos_t = torch.tensor(p, device=self.device).repeat(self.num_envs, 1)
            pos_w = pos_t + env_origins
            rot_t = torch.tensor(r, device=self.device).repeat(self.num_envs, 1)

            zeros_vel = torch.zeros((self.num_envs, 6), device=self.device)
            entity.write_root_velocity_to_sim(zeros_vel, env_ids=all_env_ids)
            entity.write_root_pose_to_sim(torch.cat([pos_w, rot_t], dim=-1), env_ids=all_env_ids)
            entity.reset()

        # Apply New Block
        nb_entity = self.scene[self.new_block_name]
        
        all_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        all_rot_w = torch.zeros((self.num_envs, 4), device=self.device)
        
        for i, cand in enumerate(candidates):
            start_idx = i * envs_per_cand
            end_idx = start_idx + envs_per_cand
            
            c_pos = cand['pos']
            c_rot = cand['rot']
            
            noisy_pos, noise_quat = self.randomizer.apply_pose_noise(c_pos, envs_per_cand, self.device)
            
            chunk_origins = env_origins[start_idx:end_idx]
            noisy_pos_w = noisy_pos + chunk_origins
            
            c_rot_t = torch.tensor(c_rot, device=self.device).repeat(envs_per_cand, 1)
            final_rot = quat_mul(noise_quat, c_rot_t)
            
            all_pos_w[start_idx:end_idx] = noisy_pos_w
            all_rot_w[start_idx:end_idx] = final_rot

        lin_vel_noise, ang_vel_noise = self.randomizer.sample_velocity_noise(self.num_envs, self.device)
        velocity_input = torch.cat([lin_vel_noise, ang_vel_noise], dim=-1)

        nb_entity.write_root_velocity_to_sim(velocity_input, env_ids=all_env_ids)
        nb_entity.write_root_pose_to_sim(torch.cat([all_pos_w, all_rot_w], dim=-1), env_ids=all_env_ids)
        nb_entity.reset()

        self.scene.write_data_to_sim()
        self.sim.step() 
        self.scene.update(dt=self.sim.get_physics_dt())

        initial_zs = torch.tensor([p['pos'][2] for p in self.tower_poses], device=self.device)
        steps = 60
        failed_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        if not self.sim.is_playing():
             self.sim.play()

        for step in range(steps):
            self.sim.step()
            self.scene.update(dt=self.sim.get_physics_dt())
            
            for i, bname in enumerate(self.block_names):
                entity = self.scene[bname]
                z = entity.data.root_pos_w[:, 2]
                z_fail = z < (initial_zs[i] - 0.03)
                failed_envs |= z_fail
                
                pos_local = entity.data.root_pos_w - env_origins
                pos_xy = pos_local[:, :2]
                dist_from_center = torch.norm(pos_xy - self.table_center_xy, dim=1)
                failed_envs |= (dist_from_center > 0.3)
                
            nb_pose = nb_entity.data.root_pos_w
            nb_local_xy = (nb_pose - env_origins)[:, :2]
            dist_nb = torch.norm(nb_local_xy - self.table_center_xy, dim=1)
            failed_envs |= (dist_nb > 0.3)
            failed_envs |= (nb_pose[:, 2] < 0.1)

        success_rates = []
        for i in range(num_cands):
            start_idx = i * envs_per_cand
            end_idx = start_idx + envs_per_cand
            
            chunk_fails = failed_envs[start_idx:end_idx]
            success_count = (~chunk_fails).sum().item()
            rate = success_count / envs_per_cand
            success_rates.append(rate)
            
        return success_rates