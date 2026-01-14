import numpy as np
from .geometry import TowerStabilityAnalyzer
from .candidates import CandidateGenerator
from .clustering import BlockClustering

class GeometricOptimizer:
    """Phase 1: Geometric Filtering to find promising block placement candidates."""

    def __init__(self, block_mass=0.082, block_size=0.060):
        self.block_mass = block_mass
        self.block_size = block_size
        self.analyzer = TowerStabilityAnalyzer()

    def find_candidates(self, all_block_poses):
        """
        Main pipeline for Phase 1.
        Returns stable_candidates: List of dicts with 'pos', 'rot', 'score'
        """
        poses_pos = []
        poses_rot = []
        
        if len(all_block_poses) > 0:
            first = all_block_poses[0]
            if isinstance(first, (list, np.ndarray, tuple)) and len(first) == 3:
                poses_pos = all_block_poses
                poses_rot = [[1, 0, 0, 0]] * len(all_block_poses)
            elif isinstance(first, dict):
                poses_pos = [f['pos'] for f in all_block_poses]
                poses_rot = [f.get('rot', [1, 0, 0, 0]) for f in all_block_poses]
        
        if not poses_pos:
            return []

        # 1. Clustering
        tower_indices, _ = BlockClustering.identify_tower_and_new_block(poses_pos)
        if not tower_indices:
            print("[GeometricOptimizer] No tower found.")
            return []
            
        tower_poses_pos = [poses_pos[i] for i in tower_indices]
        tower_poses_rot = [poses_rot[i] for i in tower_indices]
        
        # 2. Analyze Tower Base
        tower_full = list(zip(tower_poses_pos, tower_poses_rot))
        tower_full.sort(key=lambda p: p[0][2])
        lowest_z = tower_full[0][0][2]
        
        base_blocks = [blk for blk in tower_full if (blk[0][2] - lowest_z) < 0.003]
        
        support_polygon = self.analyzer.compute_multi_block_support_polygon(base_blocks, self.block_size)
        
        base_positions = np.array([b[0][:2] for b in base_blocks])
        base_center_xy = np.mean(base_positions, axis=0)

        highest_block = max(tower_full, key=lambda p: p[0][2]) 
        highest_block_pos = highest_block[0]
        highest_block_rot = highest_block[1]
        
        current_com = self.analyzer.compute_tower_com(tower_poses_pos, self.block_mass)
        
        # 3. Generate Candidates
        candidates = CandidateGenerator.generate_candidates(
            highest_block_pos, 
            top_block_orientation=highest_block_rot,
            grid_resolution=0.005, 
            search_range=0.02, 
            block_height=self.block_size,
            reference_xy=current_com[:2]
        )

        # 4. Fast Filtering
        stable_candidates = []
        for cand_pos, cand_rot in candidates:
            # Hypothetical tower with new block
            new_tower_positions = tower_poses_pos + [cand_pos]
            new_com = self.analyzer.compute_tower_com(new_tower_positions, self.block_mass)
            
            if self.analyzer.is_stable(new_com, support_polygon):
                new_com_xy = np.array(new_com[:2])
                dist_score = np.linalg.norm(new_com_xy - base_center_xy)
                
                stable_candidates.append({
                    'pos': cand_pos, 
                    'rot': cand_rot,
                    'score': dist_score 
                })

        stable_candidates.sort(key=lambda x: x['score'])

        print(f"[GeometricOptimizer] {len(stable_candidates)} candidates passed geometric filtering.")
        return stable_candidates
