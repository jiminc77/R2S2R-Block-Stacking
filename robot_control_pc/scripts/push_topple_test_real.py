import sys
import os
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rtde_control
import rtde_receive
from dt_utils import json_io
from pymodbus.client import ModbusSerialClient
from control.robotiq_gripper import RobotiqGripperUSB

class PushToppleTestReal:
    """
    Push to topple test for 2-block tower
    Tests dynamic friction and validates stability
    """

    def __init__(self, robot_ip: str = "172.27.190.178", gripper_port: str = "/dev/ttyUSB0"):
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port
        self.rtde_c = None
        self.rtde_r = None
        self.gripper = None

    def connect_robot(self):
        """Connect to UR5e via RTDE"""
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            print(f"✅ Connected to UR5e at {self.robot_ip}")

            # Connect and activate gripper
            self.gripper = RobotiqGripperUSB(self.gripper_port)
            if self.gripper.connect():
                if not self.gripper.activate():
                     print("❌ Failed to activate gripper, stopping connection.")
                     return False
                self.gripper.close() # Ensure gripper is closed for pushing
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    def move_to_safe_position(self):
        """Move robot to safe starting position"""
        # Home position above table
        safe_joints = [-3.1416, -1.0472, -1.9467, -3.2877, -1.5708, 0.0]
        self.rtde_c.moveJ(safe_joints, speed=0.3, acceleration=0.3)
        print("✅ Moved to safe position")

    def execute_push_test(
        self,
        tower_position: np.ndarray,  # [x, y, z] of tower base
        push_height_mm: float,       # Height from table to push point
        push_direction: str,         # '+X', '-X', '+Y', '-Y'
        push_speed: float = 0.01,    # m/s
        push_distance: float = 0.08  # m (8cm)
    ) -> dict:
        """
        Execute push test at specific height

        Returns:
            result: dict with outcome ('both_topple', 'top_slides', 'stable')
        """

        print(f"\n{'='*60}")
        print(f"Push Test: Height={push_height_mm:.1f}mm, Direction={push_direction}")
        print(f"{'='*60}")

        # Compute push start position
        push_height_m = push_height_mm / 1000.0
        push_start = tower_position.copy()
        push_start[2] = tower_position[2] + push_height_m

        # Offset behind tower in push direction
        offset_vector = self._direction_to_vector(push_direction)
        push_start[:2] -= offset_vector[:2] * 0.05  # 5cm behind

        # Move to push start
        print(f"Moving to push start: {push_start}")
        start_tcp_pose = self._compute_tcp_pose(push_start, push_direction)
        self.rtde_c.moveL(start_tcp_pose, speed=0.3, acceleration=0.3)
        time.sleep(1.0)

        # Execute controlled push
        print(f"Executing push: {push_distance*100:.1f}cm at {push_speed*100:.1f}cm/s")

        push_data = {
            'time': [],
            'tcp_force': [],
            'joint_torques': []
        }

        # Compute push trajectory
        push_end = push_start.copy()
        push_end[:2] += offset_vector[:2] * push_distance

        # Use servoL for force-controlled push
        dt = 0.01  # 100Hz
        n_steps = int(push_distance / push_speed / dt)
        print(f"Pushing for {n_steps} steps ({n_steps*dt:.2f}s)")
        for i in range(n_steps):
            t = i * dt
            alpha = (i / n_steps)
            current_pos = push_start + (push_end - push_start) * alpha

            # Calculate target pose for this step
            step_tcp_pose = self._compute_tcp_pose(current_pos, push_direction)
            self.rtde_c.servoL(step_tcp_pose, 0, 0, dt, 0.1, 300)

            # Record force data
            push_data['time'].append(t)
            push_data['tcp_force'].append(self.rtde_r.getActualTCPForce())
            push_data['joint_torques'].append(self.rtde_r.getTargetMoment())

            time.sleep(dt)

        self.rtde_c.servoStop()
        time.sleep(0.5)

        # Retract logic: Move back to start, then to safe position
        print("Retracting to start position...")
        self.rtde_c.moveL(start_tcp_pose, speed=0.1, acceleration=0.3)
        
        print("Moving away to safe position...")
        self.move_to_safe_position()

        # Human observation of outcome
        print("\nObserve the tower behavior:")
        print("  1 = Both blocks toppled together")
        print("  2 = Top block slid/toppled, bottom stayed")
        print("  3 = Tower remained stable")

        outcome_code = int(input("Enter outcome (1/2/3): "))

        outcome_map = {
            1: 'both_topple',
            2: 'top_slides',
            3: 'stable'
        }

        outcome = outcome_map.get(outcome_code, 'unknown')

        result = {
            'tower_position': tower_position.tolist(),
            'push_height_mm': push_height_mm,
            'push_direction': push_direction,
            'push_speed': push_speed,
            'push_distance': push_distance,
            'outcome': outcome,
            'force_data': push_data
        }

        print(f"Outcome recorded: {outcome}")

        return result

    def _direction_to_vector(self, direction: str) -> np.ndarray:
        """Convert direction string to unit vector"""
        if direction == '+X':
            return np.array([1.0, 0.0, 0.0])
        elif direction == '-X':
            return np.array([-1.0, 0.0, 0.0])
        elif direction == '+Y':
            return np.array([0.0, 1.0, 0.0])
        elif direction == '-Y':
            return np.array([0.0, -1.0, 0.0])
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def _compute_tcp_pose(self, position: np.ndarray, direction: str) -> list:
        """
        Compute TCP pose for pushing
        Gripper oriented to push horizontally
        """
        fixed_rotation = [1.214, -1.209, -1.212]
        # TCP pose: [x, y, z, rx, ry, rz]
        return list(position) + fixed_rotation

    def run_full_test_protocol(self):
        """
        Run complete push topple test protocol
        Tests at 3 different heights: low, mid, high
        """

        os.makedirs("calibration", exist_ok=True)

        print("="*60)
        print("Push to Topple Test - Real World Protocol")
        print("="*60)
        print()
        print("Setup:")
        print("  1. Stack 2 blocks on table (centered)")
        print("  2. Measure tower base position")
        print("  3. Robot will push at different heights")
        print()

        # Manually measure tower position
        print("Measure tower base position (center of bottom block):")
        tower_x, tower_y, tower_z = -0.7, 0.095, 0.175
        tower_position = np.array([tower_x, tower_y, tower_z])

        # Connect to robot
        if not self.connect_robot():
            return None

        self.move_to_safe_position()

        # Test heights: 50mm, 80mm, 110mm from table
        test_heights_mm = [40.0, 75.0, 110.0]

        # Test direction (can extend to all 4 directions)
        test_direction = '-X'

        all_results = []

        for height_mm in test_heights_mm:
            input(f"\nReady to push at {height_mm:.1f}mm height. Press Enter to continue...")

            result = self.execute_push_test(
                tower_position=tower_position,
                push_height_mm=height_mm,
                push_direction=test_direction,
                push_speed=0.01,  # 1cm/s slow push
                push_distance=0.08  # 8cm
            )

            all_results.append(result)

            input("Reset tower (re-stack blocks), then press Enter...")

        # Save results
        test_data = {
            'tower_position': tower_position.tolist(),
            'test_direction': test_direction,
            'results': all_results
        }

        json_io.save_json("calibration/push_topple_real.json", test_data)

        print("\n" + "="*60)
        print("Test Results Summary:")
        print("="*60)
        for result in all_results:
            print(f"Height {result['push_height_mm']:.1f}mm: {result['outcome']}")

        print("\n✅ Results saved to: calibration/push_topple_real.json")

        return test_data


if __name__ == "__main__":
    tester = PushToppleTestReal()
    tester.run_full_test_protocol()