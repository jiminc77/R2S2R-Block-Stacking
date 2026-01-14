import torch
import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.utils import configclass


@configclass
class DomainRandomizationCfg:
    """Configuration for domain randomization"""

    # Number of parallel environments
    num_envs: int = 256

    # Physics randomization
    friction_range: tuple = (0.95, 1.05)  # ±5% of calibrated value
    mass_std: float = 0.01  # ±1% Gaussian noise
    com_std: float = 0.001  # ±1mm Gaussian noise in meters

    # Perception randomization
    position_std: float = 0.002  # ±2mm Gaussian noise
    rotation_std_deg: float = 2.0  # ±2 degrees Gaussian noise

    # Restitution randomization
    restitution_range: tuple = (0.05, 0.15)  # Bounce coefficient


class DomainRandomizedEnv:
    """Single environment with randomized parameters"""

    def __init__(
        self,
        env_idx: int,
        cfg: DomainRandomizationCfg,
        base_friction: float,
        base_mass: float,
        device: str = "cuda:0"
    ):
        self.env_idx = env_idx
        self.cfg = cfg
        self.device = device

        # Randomize physics parameters
        self.friction = self._randomize_friction(base_friction)
        self.mass = self._randomize_mass(base_mass)
        self.com_offset = self._randomize_com()
        self.restitution = self._randomize_restitution()

    def _randomize_friction(self, base_value: float) -> float:
        """Randomize friction coefficient (uniform ±5%)"""
        low, high = self.cfg.friction_range
        factor = np.random.uniform(low, high)
        return base_value * factor

    def _randomize_mass(self, base_value: float) -> float:
        """Randomize mass (Gaussian ±1%)"""
        noise = np.random.normal(0, self.cfg.mass_std)
        return base_value * (1 + noise)

    def _randomize_com(self) -> np.ndarray:
        """Randomize center of mass offset (Gaussian ±1mm)"""
        offset = np.random.normal(0, self.cfg.com_std, size=3)
        return offset

    def _randomize_restitution(self) -> float:
        """Randomize restitution coefficient"""
        return np.random.uniform(*self.cfg.restitution_range)

    def apply_perception_noise(self, position: np.ndarray, rotation: np.ndarray) -> tuple:
        """
        Apply perception noise to detected block pose

        Args:
            position: Ground truth position [x, y, z]
            rotation: Ground truth rotation (quaternion [w, x, y, z])

        Returns:
            noisy_position, noisy_rotation
        """

        # Position noise (Gaussian ±2mm)
        pos_noise = np.random.normal(0, self.cfg.position_std, size=3)
        noisy_position = position + pos_noise

        # Rotation noise (Gaussian ±2 degrees around each axis)
        from scipy.spatial.transform import Rotation as R

        rot_noise_deg = np.random.normal(0, self.cfg.rotation_std_deg, size=3)
        rot_noise_rad = np.deg2rad(rot_noise_deg)

        base_rot = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])  # xyzw format
        noise_rot = R.from_rotvec(rot_noise_rad)

        noisy_rot = noise_rot * base_rot
        noisy_quat_xyzw = noisy_rot.as_quat()
        noisy_rotation = np.array([noisy_quat_xyzw[3], noisy_quat_xyzw[0], noisy_quat_xyzw[1], noisy_quat_xyzw[2]])  # wxyz

        return noisy_position, noisy_rotation


class ParallelRandomizedSimulation:
    """Parallel Isaac Lab simulation with domain randomization"""

    def __init__(self, cfg: DomainRandomizationCfg):
        self.cfg = cfg

        # Load calibrated physics parameters
        friction_params = np.load("calibration/friction_final.npy", allow_pickle=True).item()
        self.base_friction_block_block = friction_params['friction_block_block']
        self.base_friction_block_table = friction_params['friction_block_table']

        mass_data = np.load("calibration/block_masses.npy", allow_pickle=True).item()
        self.base_mass = mass_data["Block_1"]["mean_kg"]

        # Create simulation
        self._setup_simulation()

        # Create randomized environments
        self.envs = []
        for i in range(cfg.num_envs):
            env = DomainRandomizedEnv(
                env_idx=i,
                cfg=cfg,
                base_friction=self.base_friction_block_block,
                base_mass=self.base_mass
            )
            self.envs.append(env)

        print(f"✅ Created {len(self.envs)} randomized environments")

    def _setup_simulation(self):
        """Setup Isaac Lab simulation"""

        sim_cfg = SimulationCfg(
            dt=1/120,  # 120Hz for faster simulation
            device="cuda:0",
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                enable_stabilization=True,
                bounce_threshold_velocity=0.2,
            ),
        )

        self.sim = SimulationContext(sim_cfg)
        print("✅ Parallel simulation initialized")

    def get_env_parameters(self, env_idx: int) -> dict:
        """Get randomized parameters for specific environment"""
        env = self.envs[env_idx]
        return {
            'friction': env.friction,
            'mass': env.mass,
            'com_offset': env.com_offset,
            'restitution': env.restitution
        }


if __name__ == "__main__":
    # Test domain randomization
    cfg = DomainRandomizationCfg(num_envs=256)
    sim = ParallelRandomizedSimulation(cfg)

    # Print sample of randomized parameters
    print("\nSample of randomized parameters:")
    for i in range(5):
        params = sim.get_env_parameters(i)
        print(f"\nEnv {i}:")
        print(f"  Friction: {params['friction']:.4f}")
        print(f"  Mass: {params['mass']:.6f} kg")
        print(f"  CoM offset: {params['com_offset']*1000} mm")
        print(f"  Restitution: {params['restitution']:.3f}")