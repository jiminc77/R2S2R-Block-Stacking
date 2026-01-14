import argparse
import os
import sys
import numpy as np
import random
import json
import time
import traceback
from omni.isaac.lab.app import AppLauncher

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description="Optimal Block Stacking Search")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of environments for verification")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    # Launch Isaac Sim app (Must be done before importing ROS or Isaac Sim modules that need Kit)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Enable ROS Bridge extension
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.isaac.ros_bridge")

    # Late imports after Kit startup
    import rospy
    import torch
    from std_msgs.msg import String
    from omni.isaac.lab.sim import SimulationContext, SimulationCfg
    
    from optimal_stacking.core.optimizer import GeometricOptimizer
    from optimal_stacking.sim.scene import PhysicsParams
    from optimal_stacking.sim.environment import StackingVerificationEnv
    from optimal_stacking.utils.ros_utils import PoseListener
    from optimal_stacking.core.clustering import BlockClustering

    try:
        rospy.init_node('optimal_stacking_policy_search', anonymous=True)
        result_pub = rospy.Publisher('/optimal_stacking/result', String, queue_size=1, latch=True)

        # 1. Retrieve Block Poses from ROS
        listener = PoseListener()
        raw_poses = listener.wait_for_poses()
        if not raw_poses:
            print("[Main] No poses received. Exiting.")
            return

        z_offset = 0.001
        tower_poses = [
            {'pos': [p[0], p[1], p[2] + z_offset], 'rot': p[3:] if len(p) > 3 else [1, 0, 0, 0]}
            for p in raw_poses
        ]
        
        print(f"[Main] Received {len(tower_poses)} blocks.")

        # 2. Cluster Blocks
        tower_positions = [p['pos'] for p in tower_poses]
        tower_indices, other_indices = BlockClustering.identify_tower_and_new_block(tower_positions)

        new_block_data = None
        if tower_indices:
            print(f"[Main] Main Tower: {len(tower_indices)} blocks.")
            if other_indices:
                idx = other_indices[0]
                new_block_data = tower_poses[idx]
                print(f"[Main] Target Block Position: {new_block_data['pos']}")
            
            # Filter to keep only tower blocks
            all_poses = tower_poses
            tower_poses = [all_poses[i] for i in tower_indices]
        else:
            print("[Main] Warning: Clustering failed. Using all blocks.")

        # 3. Geometric Optimization (Candidate Generation)
        params = PhysicsParams(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        optimizer = GeometricOptimizer(block_mass=params.get_mass(0))
        candidates = optimizer.find_candidates(tower_poses)

        if not candidates:
            print("[Main] No candidates found.")
            return

        # Cap candidates for performance
        if len(candidates) > 400:
            random.shuffle(candidates)
            candidates = candidates[:400]

        print(f"[Main] Verifying {len(candidates)} candidates.")

        # 4. Physics Verification
        sim_cfg = SimulationCfg(dt=1/60.0, device=args_cli.device)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([1.2, 1.2, 1.2], [0.0, 0.0, 0.2])

        env = StackingVerificationEnv(sim, tower_poses, params, num_envs=args_cli.num_envs)

        # Calculate Tower Center of Mass (XY) for tie-breaking
        tower_com = np.mean([p['pos'] for p in tower_poses], axis=0) if tower_poses else np.zeros(3)

        all_results = []
        target_envs_per_cand = 128
        batch_size = max(1, args_cli.num_envs // target_envs_per_cand)

        for i in range(0, len(candidates), batch_size):
            batch_cands = candidates[i : i + batch_size]
            batch_rates = env.verify_batch(batch_cands)
            for j, rate in enumerate(batch_rates):
                all_results.append((batch_cands[j], rate))

        # Sort: Primary = Success Rate (desc), Secondary = Dist to CoM (asc)
        all_results.sort(key=lambda item: (
            -item[1], 
            np.linalg.norm(np.array(item[0]['pos'][:2]) - tower_com[:2])
        ))

        best_candidate, best_rate = all_results[0] if all_results else (None, 0.0)

        print(f"\n[Result] Best Candidate Success: {best_rate*100:.1f}%")
        if best_rate < 0.95:
            print("[Warning] Success rate < 95%")

        # 5. Publish Result
        if best_candidate and new_block_data:
            result_data = {
                'target_pose': best_candidate,
                'new_block_pose': new_block_data,
                'tower_poses': [{'pos': p['pos'], 'rot': p['rot']} for p in tower_poses],
                'success_rate': float(best_rate),
                'generated_at': time.time()
            }
            result_pub.publish(json.dumps(convert_to_native(result_data)))
            print("[Main] Result published. Waiting for shutdown...")
            rospy.spin()
        else:
            print("[Main] Error: Missing candidate or new block data.")

    except Exception:
        traceback.print_exc()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

