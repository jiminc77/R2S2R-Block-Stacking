import numpy as np
import os, sys
sys.path.append(os.getcwd())
from dt_utils.json_io import save_json

def measure_sliding_angles():
    """
    Measure sliding angle for Block-Table friction

    Protocol:
    1. Use flat board with same surface as table
    2. Place block on board
    3. Slowly tilt until block slides
    4. Measure angle
    5. Repeat 3 times per block
    """

    print("="*60)
    print("Sliding Angle Measurement Protocol")
    print("="*60)
    print("Equipment needed:")
    print("  - Flat board (same material as table)")
    print("  - Protractor or digital angle meter")
    print("  - 4 blocks")
    print()
    print("Protocol:")
    print("  1. Place block on board")
    print("  2. Slowly tilt board")
    print("  3. Record angle when block starts sliding")
    print("  4. Repeat 3 times per block")
    print()

    results = {}

    for block_id in range(4):
        block_name = f"Block_{block_id+1}"
        print(f"\n{block_name} (ID {block_id}):")
        print("-"*40)

        angles = []
        for trial in range(3):
            angle = float(input(f"  Trial {trial+1} - Sliding angle (degrees): "))
            angles.append(angle)

        mean_angle = np.mean(angles)
        std_angle = np.std(angles)

        # Calculate friction coefficient: μ = tan(θ)
        friction = np.tan(np.deg2rad(mean_angle))

        results[block_name] = {
            "block_id": block_id,
            "angles_deg": angles,
            "mean_angle_deg": float(mean_angle),
            "std_angle_deg": float(std_angle),
            "friction_coefficient": float(friction)
        }

        print(f"  Mean angle: {mean_angle:.2f}° ± {std_angle:.2f}°")
        print(f"  Friction coefficient: {friction:.3f}")

    # Calculate average
    avg_friction = np.mean([r["friction_coefficient"] for r in results.values()])

    print("\n" + "="*60)
    print(f"Average Block-Table friction: {avg_friction:.3f}")
    print("="*60)

    # Save
    # Save
    save_json("calibration/sliding_test_results.json", results)
    print("✅ Results saved to: calibration/sliding_test_results.json")

    return results, avg_friction


if __name__ == "__main__":
    measure_sliding_angles()