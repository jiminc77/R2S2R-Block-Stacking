import numpy as np
import os
import sys
import argparse

# [FIX] Use AppLauncher to setup environment/python paths
sys.path.append("/isaac-sim/kit/extscore/omni.client")
from omni.isaac.lab.app import AppLauncher

# App Launch
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, PhysxSchema

# Add project root to sys.path to find dt_utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

def convert_block_to_usd(
    stl_path: str,
    output_usd_path: str,
    mass_kg: float,
    friction_static: float = 0.5,
    friction_dynamic: float = 0.4,
    restitution: float = 0.1
):
    """
    Convert block STL to USD with physics properties

    Args:
        stl_path: Path to input STL file
        output_usd_path: Path to output USD file
        mass_kg: Block mass in kilograms
        friction_static: Static friction coefficient
        friction_dynamic: Dynamic friction coefficient
        restitution: Restitution coefficient
    """

    # Load mesh
    mesh = trimesh.load(stl_path)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Create USD stage
    stage = Usd.Stage.CreateNew(output_usd_path)

    # Set metadata
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root
    root_prim = stage.DefinePrim("/Block", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create mesh geometry
    mesh_prim = UsdGeom.Mesh.Define(stage, "/Block/mesh")

    # Scale vertices to meters (assuming STL is in mm)
    vertices_m = mesh.vertices * 0.001
    
    # [FIX] Center the mesh geometry at (0,0,0)
    # The original STL might be offset (e.g. min -26, max +34 -> center +4)
    # We want min -30, max +30
    min_v = vertices_m.min(axis=0)
    max_v = vertices_m.max(axis=0)
    center = (min_v + max_v) / 2.0
    print(f"  Original Center: {center}")
    print(f"  Centering mesh...")
    vertices_m = vertices_m - center
    print(f"  New Bounds: {vertices_m.min(axis=0)} to {vertices_m.max(axis=0)}")

    points = [Gf.Vec3f(*v) for v in vertices_m]
    mesh_prim.GetPointsAttr().Set(points)

    min_bound = Gf.Vec3f(*vertices_m.min(axis=0))
    max_bound = Gf.Vec3f(*vertices_m.max(axis=0))
    mesh_prim.CreateExtentAttr().Set([min_bound, max_bound])
    mesh_prim.GetDisplayColorAttr().Set([(0.5, 0.5, 0.5)])

    # Set faces
    face_indices = mesh.faces.flatten().tolist()
    mesh_prim.GetFaceVertexIndicesAttr().Set(face_indices)

    face_counts = [3] * len(mesh.faces)
    mesh_prim.GetFaceVertexCountsAttr().Set(face_counts)

    # Set normals
    if hasattr(mesh, 'vertex_normals'):
        normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals]
        mesh_prim.GetNormalsAttr().Set(normals)

    # Apply rigid body physics to ROOT
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)

    # Apply mass to ROOT
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.GetMassAttr().Set(mass_kg)

    # Collision API
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    
    # [FIX] Apply PhysxCollisionAPI for precise contact offset (1mm vs default 20mm)
    # This prevents the "phantom stability" observed in overhang tests.
    physx_coll_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim.GetPrim())
    physx_coll_api.CreateContactOffsetAttr().Set(0.001)
    physx_coll_api.CreateRestOffsetAttr().Set(0.0)

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
    mesh_collision_api.GetApproximationAttr().Set("convexHull")

    # Create physics material
    material_path = "/Block/PhysicsMaterial"
    material_prim = stage.DefinePrim(material_path, "Material")
    physics_mat = UsdPhysics.MaterialAPI.Apply(material_prim)

    physics_mat.CreateStaticFrictionAttr(friction_static)
    physics_mat.CreateDynamicFrictionAttr(friction_dynamic)
    physics_mat.CreateRestitutionAttr(restitution)

    # Bind material to mesh
    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    shade_material = UsdShade.Material(material_prim)
    binding_api.Bind(shade_material, materialPurpose="physics")

    # Save
    stage.Save()
    print(f"✅ USD saved: {output_usd_path}")
    print(f"   Mass: {mass_kg:.6f} kg")
    print(f"   Friction (static/dynamic): {friction_static:.2f}/{friction_dynamic:.2f}")

    return output_usd_path


def batch_convert_blocks():
    """Convert all blocks using measured masses"""

    # Load mass data
    # Load mass data
    from dt_utils.json_io import load_json
    json_path = os.path.join(project_root, "calibration", "block_masses.json")
    mass_data = load_json(json_path)

    # Create assets directory
    assets_dir = os.path.join(project_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Assume we have a single STL file for the block shape
    stl_path = os.path.join(assets_dir, "block_60mm.stl")

    if not os.path.exists(stl_path):
        print(f"❌ STL file not found: {stl_path}")
        print("Please provide block STL file")
        return

    print("="*60)
    print("Converting blocks to USD")
    print("="*60)

    for block_name, data in mass_data.items():
        block_id = data['block_id']
        mass_kg = data['mean_kg']

        output_path = os.path.join(assets_dir, f"block_{block_id}.usd")

        print(f"\n{block_name} (ID {block_id}):")
        convert_block_to_usd(
            stl_path=stl_path,
            output_usd_path=output_path,
            mass_kg=mass_kg,
            friction_static=0.5,  # Initial value, will be calibrated
            friction_dynamic=0.4,
            restitution=0.1
        )

    print("\n" + "="*60)
    print("✅ All blocks converted to USD")


if __name__ == "__main__":
    batch_convert_blocks()