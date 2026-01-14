import numpy as np
import os
import sys
sys.path.append(os.getcwd()) # Ensure utils can be imported
from dt_utils.json_io import save_json

def measure_maximum_overhang():
    """
    Measure maximum stable overhang distance in 4 directions

    Protocol:
    1. Stack 2 blocks vertically (Block 0 bottom, Block 1 top)
    2. Slowly slide top block horizontally until it topples
    3. Measure overhang distance with calipers (±0.1mm accuracy)
    4. Test all 4 directions: +X, -X, +Y, -Y
    5. Repeat 3 times per direction for statistical reliability
    """

    os.makedirs("calibration", exist_ok=True)

    print("="*60)
    print("Maximum Overhang Test - Real World Protocol")
    print("="*60)
    print("Equipment:")
    print("  - 2 blocks (Block 0 as base, Block 1 as top)")
    print("  - Digital calipers (±0.1mm)")
    print("  - Flat, stable surface")
    print()
    print("Directions:")
    print("  +X: Front of block")
    print("  -X: Back of block")
    print("  +Y: Right side of block")
    print("  -Y: Left side of block")
    print()

    directions = ['+X', '-X', '+Y', '-Y']
    results = {}

    for direction in directions:
        print(f"\n{direction} Direction:")
        print("-"*40)
        print("1. Stack Block 1 on Block 0 (centered)")
        print(f"2. Slowly slide Block 1 in {direction} direction")
        print("3. Stop when Block 1 starts to topple")
        print("4. Measure overhang distance from edge alignment")
        print()

        measurements_mm = []
        for trial in range(3):
            overhang_mm = float(input(f"  Trial {trial+1} - Overhang (mm): "))
            measurements_mm.append(overhang_mm)

        mean_mm = np.mean(measurements_mm)
        std_mm = np.std(measurements_mm)
        mean_m = mean_mm / 1000.0

        results[direction] = {
            "measurements_mm": measurements_mm,
            "mean_mm": float(mean_mm),
            "std_mm": float(std_mm),
            "mean_m": float(mean_m)
        }

        print(f"  Mean: {mean_mm:.2f}mm ± {std_mm:.2f}mm ({mean_m:.6f}m)")

    # Theoretical maximum (no friction): 30mm (half of 60mm block)
    # Actual overhang is less due to friction and CoM uncertainty

    avg_overhang_mm = np.mean([r["mean_mm"] for r in results.values()])

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for direction, data in results.items():
        print(f"  {direction}: {data['mean_mm']:.2f}mm ± {data['std_mm']:.2f}mm")
    print(f"\nAverage overhang: {avg_overhang_mm:.2f}mm")
    print("Theoretical max (no friction): 30mm")
    print(f"Measured percentage: {avg_overhang_mm/30*100:.1f}%")

    # Save results
    # Save results
    save_json("calibration/overhang_test_real.json", results)
    print("\n✅ Results saved to: calibration/overhang_test_real.json")

    return results


if __name__ == "__main__":
    measure_maximum_overhang()