import numpy as np
import os, sys
sys.path.append(os.getcwd())
from dt_utils.json_io import save_json
import os, sys
sys.path.append(os.getcwd())
from dt_utils.json_io import save_json

def measure_geometry_offset():
   """
   Measure offset from ArUco marker center to block geometric center

   Coordinate system:
   - Marker surface is z=0 reference
   - x, y: horizontal offset from marker center to block center
   - z: depth from marker surface to block geometric center
   """

   print("="*60)
   print("Geometry Offset Measurement Protocol")
   print("="*60)
   print("Use calipers (±0.1mm accuracy)")
   print()

   # Measure dimensions
   print("Block Dimensions:")
   block_size_mm = float(input("  Block size (mm) [expected ~60mm]: "))
   marker_size_mm = float(input("  Marker size (mm) [expected ~40mm]: "))

   # Theoretical offset (if marker perfectly centered)
   theoretical_xy = 0.0
   theoretical_z = block_size_mm / 2.0

   print(f"\nTheoretical offset (perfectly centered marker):")
   print(f"  x, y: {theoretical_xy:.2f} mm")
   print(f"  z: {theoretical_z:.2f} mm")

   # Actual measurements
   print("\nActual measurements:")
   print("(Measure from marker center to block center)")
   offset_x_mm = float(input("  Offset X (mm): "))
   offset_y_mm = float(input("  Offset Y (mm): "))
   offset_z_mm = float(input("  Offset Z (mm): "))

   # Convert to meters
   offset_m = np.array([
       offset_x_mm / 1000.0,
       offset_y_mm / 1000.0,
       offset_z_mm / 1000.0
   ])

   # Save
   geometry_data = {
       "block_size_m": block_size_mm / 1000.0,
       "marker_size_m": marker_size_mm / 1000.0,
       "offset_x_m": float(offset_m[0]),
       "offset_y_m": float(offset_m[1]),
       "offset_z_m": float(offset_m[2]),
       "offset_xyz_m": offset_m.tolist()
   }

   save_json("calibration/geometry_offset.json", geometry_data)

   print("\n" + "="*60)
   print("✅ Geometry offset saved to: calibration/geometry_offset.json")
   print(f"Offset (m): [{offset_m[0]:.6f}, {offset_m[1]:.6f}, {offset_m[2]:.6f}]")

   return geometry_data

if __name__ == "__main__":
   measure_geometry_offset()