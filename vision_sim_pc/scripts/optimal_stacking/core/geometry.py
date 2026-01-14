import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull

class TowerStabilityAnalyzer:
    """Analyze tower stability and compute center of mass."""

    @staticmethod
    def compute_tower_com(block_poses: list, block_mass: float) -> np.ndarray:
        """Compute center of mass of stacked blocks."""
        positions = np.array(block_poses)
        if len(positions) == 0:
            return np.array([0, 0, 0])
        return positions.mean(axis=0)

    @staticmethod
    def compute_support_polygon(base_block_pos: np.ndarray, block_quat: list = None, block_size: float = 0.060) -> np.ndarray:
        """Compute support polygon vertices for base block."""
        half_size = block_size / 2.0
        cx, cy = base_block_pos[0], base_block_pos[1]

        local_verts = np.array([
            [-half_size, -half_size],
            [ half_size, -half_size],
            [ half_size,  half_size],
            [-half_size,  half_size]
        ])

        yaw = 0.0
        if block_quat is not None and len(block_quat) == 4:
            # Assume [w, x, y, z]
            w, x, y, z = block_quat
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array(((c, -s), (s, c)))

        vertices = []
        for v in local_verts:
            v_rot = R.dot(v)
            vertices.append([cx + v_rot[0], cy + v_rot[1]])
            
        return np.array(vertices)

    @staticmethod
    def compute_multi_block_support_polygon(block_poses_full: list, block_size: float = 0.060) -> np.ndarray:
        """Compute the Convex Hull of all base blocks."""
        all_corners = []
        
        for pos, rot in block_poses_full:
            corners = TowerStabilityAnalyzer.compute_support_polygon(pos, rot, block_size)
            all_corners.extend(corners)
        
        all_corners = np.array(all_corners)
        
        if len(all_corners) < 3:
            return all_corners 
            
        hull = ConvexHull(all_corners)
        return all_corners[hull.vertices]

    @staticmethod
    def is_stable(tower_com: np.ndarray, support_polygon: np.ndarray) -> bool:
        """Check if tower COM is within support polygon."""
        com_xy = tower_com[:2]
        polygon = Path(support_polygon)
        return polygon.contains_point(com_xy)
