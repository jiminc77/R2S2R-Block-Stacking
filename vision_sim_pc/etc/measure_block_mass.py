import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from dt_utils.json_io import save_json
import sys
sys.path.append(os.getcwd())
from dt_utils.json_io import save_json

def measure_block_masses():
   """Measure precise mass for each block"""

   os.makedirs("calibration", exist_ok=True)

   print("="*60)
   print("Block Mass Measurement Protocol")
   print("="*60)
   print("Equipment: Precision scale (±0.1g accuracy)")
   print("Protocol: Measure each block 3 times")
   print()

   blocks_data = {}

   for block_num in range(1, 5):
       block_name = f"Block_{block_num}"
       print(f"\n{block_name} (ID {block_num-1}):")

       measurements = []
       for trial in range(3):
           mass_g = float(input(f"  Trial {trial+1} mass (grams): "))
           measurements.append(mass_g)

       mean_g = np.mean(measurements)
       std_g = np.std(measurements)
       mean_kg = mean_g / 1000.0

       blocks_data[block_name] = {
           "block_id": block_num - 1,
           "measurements_g": measurements,
           "mean_g": float(mean_g),
           "std_g": float(std_g),
           "mean_kg": float(mean_kg)
       }

       print(f"  Mean: {mean_g:.2f}g ± {std_g:.2f}g ({mean_kg:.6f} kg)")

   # Save
   # Save
   save_json("calibration/block_masses.json", blocks_data)

   print("\n" + "="*60)
   print("✅ Mass measurements saved to: calibration/block_masses.json")
   print("\nSummary:")
   for name, data in blocks_data.items():
       print(f"  {name}: {data['mean_kg']:.6f} kg")

   return blocks_data

if __name__ == "__main__":
   measure_block_masses()