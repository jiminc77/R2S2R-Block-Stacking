import numpy as np
from scipy.spatial.transform import Rotation as R

class CandidateGenerator:
    """Generate candidate positions and orientations for block stacking."""

    @staticmethod
    def generate_candidates(tower_highest_point, top_block_orientation=None, 
                            grid_resolution=0.005, search_range=0.025, 
                            block_height=0.060, reference_xy=None,
                            yaw_offsets=None):
        """
        Generate candidates with specific yaw variations relative to the top block.
        
        Args:
            tower_highest_point: Reference point [x, y, z] for height calculation.
            top_block_orientation: Quaternion [w, x, y, z] of the top block.
            grid_resolution: Grid spacing in meters.
            search_range: Search range in meters.
            block_height: Height of the block to be stacked.
            reference_xy: Optional [x, y] center for grid search. 
            yaw_offsets: List of degrees to rotate relative to the top block.
                          
        Returns:
            candidates: List of tuples (position, orientation)
        """
        center_x, center_y, top_z = tower_highest_point
        
        if yaw_offsets is None:
            yaw_offsets = [0, 15, -15, 30, -30]

        if reference_xy is not None:
            center_x, center_y = reference_xy
            
        new_z = top_z + block_height + 0.002 
        
        x_vals = np.arange(center_x - search_range, center_x + search_range + 0.0001, grid_resolution)
        y_vals = np.arange(center_y - search_range, center_y + search_range + 0.0001, grid_resolution)
        
        if top_block_orientation is None:
            base_r = R.from_quat([0, 0, 0, 1])
        else:
            w, x, y, z = top_block_orientation
            base_r = R.from_quat([x, y, z, w])

        candidates = []

        for x in x_vals:
            for y in y_vals:
                for yaw_deg in yaw_offsets:
                    
                    # Calculate Orientation (Local Rotation)
                    rot_z = R.from_euler('z', yaw_deg, degrees=True)
                    final_r = base_r * rot_z  
                    
                    rx, ry, rz, rw = final_r.as_quat()
                    new_orientation = [rw, rx, ry, rz]
                    
                    new_position = [x, y, new_z]
                    candidates.append((new_position, new_orientation))
                
        return candidates