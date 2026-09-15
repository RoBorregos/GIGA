#!/usr/bin/env python3
"""Convert a folder of object meshes into GIGA-format per-object URDFs.

generate_data_parallel.py's --object-set points at a directory of .urdf
files (self.urdf_root / self.object_set in simulation.py), one per object,
each with mass/inertia -- not raw meshes. This script writes one alongside
each mesh, computing mass/inertia from the mesh's own volume via trimesh.

Usage:
    python3 scripts/mesh_to_object_urdf.py data/urdfs/frida/train --density 1000

Not committed anywhere by this script: this is a local, throwaway
conversion step -- rerun it if you replace a mesh.
"""
import argparse
import sys
from pathlib import Path

import trimesh

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="{com_x:.6f} {com_y:.6f} {com_z:.6f}" rpy="0 0 0" />
      <mass value="{mass:.6f}" />
      <inertia ixx="{ixx:.8f}" ixy="{ixy:.8f}" ixz="{ixz:.8f}"
               iyy="{iyy:.8f}" iyz="{iyz:.8f}" izz="{izz:.8f}" />
    </inertial>
    <visual>
      <geometry><mesh filename="{mesh_file}" /></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{mesh_file}" /></geometry>
    </collision>
  </link>
</robot>
"""


def convert(mesh_path: Path, density: float) -> None:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not mesh.is_watertight:
        print(
            f"  warning: {mesh_path.name} is not watertight -- "
            "mass/inertia/COM below may be inaccurate (pybullet convex-hulls "
            "collision anyway, but mass errors still affect grasp physics)",
            file=sys.stderr,
        )
    mesh.density = density
    mass = mesh.mass
    com = mesh.center_mass
    inertia = mesh.moment_inertia
    urdf = URDF_TEMPLATE.format(
        name=mesh_path.stem,
        com_x=com[0], com_y=com[1], com_z=com[2],
        mass=mass,
        ixx=inertia[0][0], ixy=inertia[0][1], ixz=inertia[0][2],
        iyy=inertia[1][1], iyz=inertia[1][2], izz=inertia[2][2],
        mesh_file=mesh_path.name,
    )
    out_path = mesh_path.with_suffix(".urdf")
    out_path.write_text(urdf)
    print(f"  wrote {out_path.name}  (mass={mass:.3f} kg)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_dir", type=Path, help="directory containing .obj/.stl meshes")
    parser.add_argument(
        "--density", type=float, default=1000.0,
        help="kg/m^3, rough default (water/plastic-ish); pass the object's "
             "real density if you know it -- a Rubik's cube and a foam cup "
             "are very different, and mass drives grasp-success physics",
    )
    args = parser.parse_args()

    meshes = sorted(
        p for ext in ("*.obj", "*.stl", "*.STL", "*.dae")
        for p in args.mesh_dir.glob(ext)
    )
    if not meshes:
        sys.exit(f"no .obj/.stl/.dae files found in {args.mesh_dir}")

    for mesh_path in meshes:
        convert(mesh_path, args.density)


if __name__ == "__main__":
    main()
