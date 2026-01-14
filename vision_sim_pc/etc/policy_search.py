import numpy as np
import torch
from src.domain_randomization import ParallelRandomizedSimulation, DomainRandomizationCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
import omni.isaac.lab.sim as sim_utils


class TowerStabilityAnalyzer:
    """Analyze tower stability and compute center of mass"""

    @staticmethod
    def compute_tower_com(block_poses: list, block_mass: float) -> np.ndarray:
        """
        Compute center of mass of stacked blocks

        Args:
            block_poses: List of block positions [[x, y, z], ...]
            block_mass: Mass of each block (assumed uniform)

        Returns:
            com_position: Center of mass [x, y, z]
        """

        positions = np.array(block_poses)
        total_mass = len(positions) * block_mass

        # Weighted average (uniform mass)
        com = positions.mean(axis=0)

        return com

    @staticmethod
    def compute_support_polygon(base_block_pos: np.ndarray, block_size: float = 0.060) -> np.ndarray:
        """
        Compute support polygon vertices for base block

        Args:
            base_block_pos: Position of base block [x, y, z]
            block_size: Block size (60mm cube)

        Returns:
            vertices: 4 corner vertices [[x, y], ...]
        """

        half_size = block_size / 2.0
        cx, cy = base_block_pos[0], base_block_pos[1]

        vertices = np.array([
            [cx - half_size, cy - half_size],  # Bottom-left
            [cx + half_size, cy - half_size],  # Bottom-right
            [cx + half_size, cy + half_size],  # Top-right
            [cx - half_size, cy + half_size],  # Top-left
        ])

        return vertices

    @staticmethod
    def is_stable(tower_com: np.ndarray, support_polygon: np.ndarray) -> bool:
        """
        Check if tower COM is within support polygon

        Args:
            tower_com: Tower center of mass [x, y, z]
            support_polygon: Support polygon vertices [[x, y], ...]

        Returns:
            stable: True if COM within support polygon
        """

        from matplotlib.path import Path

        com_xy = tower_com[:2]
        polygon = Path(support_polygon)

        return polygon.contains_point(com_xy)


class RobustPolicySearcher:
    """Search for robust block placement policy"""

    def __init__(self, cfg: DomainRandomizationCfg = None):
        if cfg is None:
            cfg = DomainRandomizationCfg(num_envs=256)

        self.cfg = cfg
        self.sim = ParallelRandomizedSimulation(cfg)
        self.analyzer = TowerStabilityAnalyzer()

    def generate_placement_candidates(
        self,
        tower_com: np.ndarray,
        tower_height: float,
        grid_resolution: float = 0.005  # 5mm resolution
    ) -> list:
        """
        Generate candidate placement positions around tower COM

        Args:
            tower_com: Current tower center of mass
            tower_height: Current tower height
            grid_resolution: Grid spacing for candidates (5mm)

        Returns:
            candidates: List of candidate positions [[x, y, z], ...]
        """

        # Search range: ±15mm around tower COM in X and Y
        search_range = 0.015  # 15mm

        # Z position: on top of tower
        z = tower_height + 0.060  # Block size

        candidates = []

        # Grid search in X-Y plane
        x_range = np.arange(
            tower_com[0] - search_range,
            tower_com[0] + search_range,
            grid_resolution
        )
        y_range = np.arange(
            tower_com[1] - search_range,
            tower_com[1] + search_range,
            grid_resolution
        )

        for x in x_range:
            for y in y_range:
                candidates.append([x, y, z])

        print(f"Generated {len(candidates)} placement candidates")
        return candidates

    def evaluate_placement(
        self,
        candidate_pos: np.ndarray,
        tower_block_poses: list,
        block_mass: float
    ) -> float:
        """
        Evaluate placement candidate across all randomized environments

        Args:
            candidate_pos: Candidate placement position [x, y, z]
            tower_block_poses: Existing tower block positions
            block_mass: Block mass

        Returns:
            success_rate: Success rate across all environments (0-1)
        """

        # Compute new tower COM with candidate block
        new_tower_poses = tower_block_poses + [candidate_pos]
        new_com = self.analyzer.compute_tower_com(new_tower_poses, block_mass)

        # Get base block position (bottom of tower)
        base_block_pos = tower_block_poses[0]
        support_polygon = self.analyzer.compute_support_polygon(base_block_pos)

        # Check stability analytically (fast approximation)
        analytical_stable = self.analyzer.is_stable(new_com, support_polygon)

        if not analytical_stable:
            # If analytically unstable, skip simulation
            return 0.0

        # Simulate in randomized environments
        success_count = 0

        for env_idx in range(self.cfg.num_envs):
            # Get randomized parameters
            params = self.sim.get_env_parameters(env_idx)

            # Apply perception noise to all block positions
            noisy_tower_poses = []
            for pos in new_tower_poses:
                noisy_pos, _ = self.sim.envs[env_idx].apply_perception_noise(
                    pos, np.array([1, 0, 0, 0])  # Dummy rotation
                )
                noisy_tower_poses.append(noisy_pos)

            # Recompute COM with noisy positions
            noisy_com = self.analyzer.compute_tower_com(noisy_tower_poses, params['mass'])

            # Check stability with noisy COM
            stable = self.analyzer.is_stable(noisy_com, support_polygon)

            if stable:
                success_count += 1

        success_rate = success_count / self.cfg.num_envs
        return success_rate

    def search_best_placement(
        self,
        tower_block_poses: list,
        success_threshold: float = 0.95
    ) -> dict:
        """
        Search for best placement position

        Args:
            tower_block_poses: List of existing tower block positions
            success_threshold: Minimum required success rate (95%)

        Returns:
            result: dict with 'position', 'success_rate', 'found'
        """

        print("\n" + "="*60)
        print("Robust Placement Policy Search")
        print("="*60)

        # Compute current tower properties
        tower_com = self.analyzer.compute_tower_com(tower_block_poses, self.sim.base_mass)
        tower_height = max([pos[2] for pos in tower_block_poses])

        print(f"Tower height: {tower_height*1000:.1f}mm")
        print(f"Tower CoM: [{tower_com[0]:.4f}, {tower_com[1]:.4f}, {tower_com[2]:.4f}]")

        # Generate candidates
        candidates = self.generate_placement_candidates(tower_com, tower_height)

        # Evaluate all candidates
        best_position = None
        best_success_rate = 0.0

        print("\nEvaluating candidates...")
        for i, candidate in enumerate(candidates):
            success_rate = self.evaluate_placement(
                np.array(candidate),
                tower_block_poses,
                self.sim.base_mass
            )

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_position = candidate

            # Early stopping if found good enough solution
            if success_rate >= success_threshold:
                print(f"  Candidate {i+1}/{len(candidates)}: {success_rate*100:.1f}% ✅")
                break

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{len(candidates)} candidates, best: {best_success_rate*100:.1f}%")

        print("\n" + "="*60)
        print("Search Results:")
        print("="*60)

        if best_success_rate >= success_threshold:
            print(f"✅ Found robust placement!")
            print(f"  Position: [{best_position[0]:.4f}, {best_position[1]:.4f}, {best_position[2]:.4f}]")
            print(f"  Success rate: {best_success_rate*100:.1f}%")
            found = True
        else:
            print(f"⚠️ Could not find placement with >{success_threshold*100:.0f}% success")
            print(f"  Best found: {best_success_rate*100:.1f}%")
            print(f"  Using conservative fallback (centered on tower COM)")
            # Fallback: place directly above tower COM
            best_position = [tower_com[0], tower_com[1], tower_height + 0.060]
            found = False

        return {
            'position': np.array(best_position),
            'success_rate': best_success_rate,
            'found': found
        }


if __name__ == "__main__":
    # Test robust policy search

    # Example: 2 blocks already stacked
    tower_block_poses = [
        [0.30, 0.00, 0.78],  # Block 0 (on table at 0.75m + 0.03m)
        [0.30, 0.00, 0.84],  # Block 1 (on Block 0)
    ]

    searcher = RobustPolicySearcher()
    result = searcher.search_best_placement(tower_block_poses)

    print(f"\nRecommended placement: {result['position']}")
    print(f"Expected success rate: {result['success_rate']*100:.1f}%")