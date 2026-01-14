import os
import json
import numpy as np
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils

TABLE_SIZE = (0.8, 0.8, 0.067)
TABLE_POS = (0.0, 0.0, 0.0335)

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

class PhysicsParams:
    """Manages physical properties for simulation from calibration files."""
    def __init__(self, project_root):
        self.project_root = project_root
        self.block_masses = {} 
        self.friction_block_block = 0.45
        self.friction_block_table = 0.40
        self._load()

    def _load(self):
        m_path = os.path.join(self.project_root, "calibration/block_masses.json")
        m_data = load_json(m_path)
        for key, val in m_data.items():
            bid = val.get("block_id", -1)
            mass = val.get("mean_kg", 0.082)
            if bid >= 0: self.block_masses[bid] = mass
        
        o_data = load_json(os.path.join(self.project_root, "calibration/overhang_test_optimized.json"))
        self.friction_block_block = o_data.get("friction_block_block", 0.45)
        
        s_data = load_json(os.path.join(self.project_root, "calibration/sliding_test_results.json"))
        frictions = [v["friction_coefficient"] for k, v in s_data.items() if "friction_coefficient" in v]
        if frictions:
            self.friction_block_table = sum(frictions) / len(frictions)

    def get_mass(self, block_id):
        return self.block_masses.get(block_id, 0.082)

@configclass
class DigitalTwinSceneCfg(InteractiveSceneCfg):
    """Configuration for the Digital Twin scene."""

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE, 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS)
    )
