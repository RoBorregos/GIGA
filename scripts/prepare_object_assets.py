#!/usr/bin/env python3
"""Turn raw object scans into simulation-ready GIGA object sets.

The FRIDA scans come out of the scanner at a resolution nothing in this
pipeline can use: Apple.stl is 2.2M triangles for a 41mm fruit, while GIGA
voxelizes a 30cm workspace into a 40^3 grid -- 7.5mm per voxel. That
resolution costs real time (a 5-object `sim.reset()` measured at 42.6s,
almost all of it PyBullet building convex hulls out of millions of
vertices) and makes the asset folder too big to version.

So each mesh gets split into the two things the simulator actually asks
for, at the resolution each one needs:

  visual     decimated to --max-faces. This is what the depth camera
             renders, and therefore what the TSDF -- the network's input --
             is built from. At 10k faces the surface deviates <0.2mm from
             the original scan, i.e. under 1/35 of a voxel.

  collision  an explicit convex hull, or a VHACD decomposition when the
             hull would swallow a concave feature (a mug's handle, the
             gap in a fork). PyBullet already replaces a dynamic body's
             collision mesh with its convex hull at load time, so the hull
             path changes nothing about the physics -- it just stops the
             hull being recomputed from scratch on every load. The VHACD
             path is a genuine improvement over that default.

Mass and inertia are recomputed from the decimated mesh's own volume at
--density, in the scaled (meter) frame the URDF is loaded in.

Usage:
    python3 scripts/prepare_object_assets.py \
        /workspace/giga_data/mesh_sources/train data/urdfs/frida/train
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

# Everything this script writes is in meters, loaded with scale="1 1 1".
# The raw scans are not consistent about this -- the FRIDA scans are in
# millimeters while the YCB meshes are already in meters -- and the old
# object sets applied a blanket scale="0.001" to both, which loaded every
# YCB object at 1/1000 of its size (an 81mm apple became 81 microns).
# Normalizing at preparation time is what stops that class of bug: after
# this script runs there is exactly one unit in the repo.
MESH_SCALE = 1.0

# A tabletop object is somewhere between 1cm and 1m. Anything whose largest
# extent lands above this is being measured in millimeters; nothing that is
# genuinely meters-scale comes close to it from below.
MM_DETECT_THRESHOLD = 10.0

# ClutterRemovalSim builds its workspace as 6 * finger_depth = 0.30m.
# generate_packed_scene() re-scales each object by a random 0.7-0.9 before
# placing it, so the size that actually has to fit is 0.7 * extent -- an
# object only becomes unplaceable above WORKSPACE_SIZE / 0.7.
WORKSPACE_SIZE = 0.30
PACKED_MIN_SCALE = 0.7

# A grasp label is "did the gripper lift it", so an object with an absurd
# mass gets every one of its grasps labelled negative and quietly poisons
# the set. Open meshes are where this comes from: 029_plate.stl is a shell
# with no thickness, and the convex hull that its mass falls back to is the
# solid cone the shell encloses -- 15.7kg of dinner plate.
DEFAULT_MAX_MASS = 2.0

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0" />
      <mass value="{mass:.6f}" />
      <inertia ixx="{I[0][0]:.8f}" ixy="{I[0][1]:.8f}" ixz="{I[0][2]:.8f}"
               iyy="{I[1][1]:.8f}" iyz="{I[1][2]:.8f}" izz="{I[2][2]:.8f}" />
    </inertial>
    <visual>
      <geometry><mesh filename="{visual}" scale="{s} {s} {s}" /></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{collision}" scale="{s} {s} {s}" /></geometry>
    </collision>
  </link>
</robot>
"""


def to_meters(mesh: trimesh.Trimesh, units: str):
    """Return (mesh_in_meters, detected_units).

    Scales in place rather than deferring to the URDF's scale attribute, so
    the file on disk and the simulation agree about size.
    """
    if units == "auto":
        units = "mm" if mesh.extents.max() > MM_DETECT_THRESHOLD else "m"
    if units == "mm":
        mesh = mesh.copy()
        mesh.apply_scale(0.001)
    return mesh, units


def mass_properties(mesh: trimesh.Trimesh, density: float):
    """Mass, center of mass and inertia, with a fallback for open meshes.

    trimesh integrates volume over a closed surface; on a mesh with holes
    (029_plate.stl, for one) that integral can come out negative and the
    resulting URDF carries a negative mass. Repair first, and if the mesh
    still is not watertight fall back to its convex hull, which is closed
    by construction and overestimates rather than inverts.
    """
    m = mesh
    if not m.is_watertight:
        m = m.copy()
        m.fill_holes()
        m.fix_normals()
    fallback = not m.is_watertight or m.volume <= 0
    if fallback:
        m = mesh.convex_hull
    m.density = density
    return m.mass, m.center_mass, m.moment_inertia, fallback


def decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Quadric-decimate to max_faces, or pass through if already under it."""
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3 = o3.simplify_quadric_decimation(target_number_of_triangles=max_faces)
    o3.remove_duplicated_vertices()
    o3.remove_degenerate_triangles()
    return trimesh.Trimesh(
        np.asarray(o3.vertices), np.asarray(o3.triangles), process=False
    )


def surface_deviation(reference: trimesh.Trimesh, candidate: trimesh.Trimesh, n=20000):
    """Max/mean distance from points sampled on `reference` to `candidate`.

    Uses Open3D's raycasting scene rather than trimesh.proximity, which
    needs rtree -- not installed in the training image and not worth a
    dependency for a reporting number.
    """
    pts = reference.sample(n).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(candidate.vertices),
                o3d.utility.Vector3iVector(candidate.faces),
            )
        )
    )
    d = scene.compute_distance(o3d.core.Tensor(pts)).numpy()
    return float(d.mean()), float(d.max())


def run_vhacd(mesh: trimesh.Trimesh, out_path: Path, resolution: int) -> bool:
    """Decompose `mesh` into convex parts via PyBullet's VHACD.

    Returns False (leaving out_path untouched) if VHACD is unavailable or
    produces nothing usable, so the caller can fall back to a plain hull.
    """
    import pybullet

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.obj"
        log = Path(tmp) / "vhacd.log"
        mesh.export(src)
        try:
            pybullet.vhacd(
                str(src), str(out_path), str(log), resolution=resolution
            )
        except Exception as exc:  # noqa: BLE001 - fall back, don't abort the set
            print(f"      VHACD failed ({exc}), falling back to convex hull")
            return False
    if not out_path.exists() or out_path.stat().st_size == 0:
        print("      VHACD produced no output, falling back to convex hull")
        return False
    return True


def prepare(src_mesh: Path, out_dir: Path, args) -> dict:
    name = src_mesh.stem
    raw = trimesh.load(src_mesh, force="mesh")
    original, units = to_meters(raw, args.units)
    visual = decimate(original, args.max_faces)

    # Merge duplicate vertices so the hull and the volume integral see a
    # closed surface -- raw scan STLs store every triangle independently.
    visual.merge_vertices()
    visual.fix_normals()

    visual_path = out_dir / f"{name}.stl"
    visual.export(visual_path)

    hull = decimate(visual.convex_hull, args.hull_faces)
    # A hull much bigger than the mesh means a concave feature (handle,
    # gap between tines) is being filled in -- exactly where grasp labels
    # would go wrong, so those objects get a real decomposition instead.
    bloat = hull.volume / visual.volume if visual.volume > 0 else 1.0
    collision_path = out_dir / f"{name}_col.obj"
    used_vhacd = False
    if args.vhacd and bloat > args.concave_threshold:
        used_vhacd = run_vhacd(visual, collision_path, args.vhacd_resolution)
    if not used_vhacd:
        hull.export(collision_path)

    mass, com, inertia, hull_fallback = mass_properties(visual, args.density)
    clamped = args.max_mass > 0 and mass > args.max_mass
    if clamped:
        # Scale the inertia tensor with the mass so the two stay consistent;
        # the geometry it was integrated over has not changed.
        inertia = inertia * (args.max_mass / mass)
        mass = args.max_mass
    (out_dir / f"{name}.urdf").write_text(
        URDF_TEMPLATE.format(
            name=name,
            com=com,
            mass=mass,
            I=inertia,
            visual=visual_path.name,
            collision=collision_path.name,
            s=MESH_SCALE,
        )
    )

    mean_dev, max_dev = surface_deviation(original, visual)
    return {
        "name": name,
        "units": units,
        "faces_in": len(raw.faces),
        "faces_out": len(visual.faces),
        # Deviations are computed in meters (post-normalization); report in
        # millimeters, which is the scale a voxel is discussed in.
        "mean_dev": mean_dev * 1000.0,
        "max_dev": max_dev * 1000.0,
        "size_cm": visual.extents.max() * 100.0,
        "oversize": visual.extents.max() * PACKED_MIN_SCALE > WORKSPACE_SIZE,
        "clamped": clamped,
        "mass": mass,
        "hull_fallback": hull_fallback,
        "collision": "vhacd" if used_vhacd else "hull",
        "kb_in": src_mesh.stat().st_size / 1024,
        "kb_out": (visual_path.stat().st_size + collision_path.stat().st_size) / 1024,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="folder of source meshes (.stl/.obj/.ply)")
    p.add_argument("dst", type=Path, help="output object-set folder")
    p.add_argument("--units", choices=["auto", "mm", "m"], default="auto",
                   help="units of the source meshes; auto-detects per mesh "
                        "from its size (default: auto)")
    p.add_argument("--max-faces", type=int, default=10000,
                   help="visual mesh face budget; 0 disables decimation (default: 10000)")
    p.add_argument("--hull-faces", type=int, default=256,
                   help="face budget for the convex-hull collision mesh (default: 256)")
    p.add_argument("--density", type=float, default=1000.0,
                   help="kg/m^3 used for mass and inertia (default: 1000, water)")
    p.add_argument("--max-mass", type=float, default=DEFAULT_MAX_MASS,
                   help="clamp computed mass to this many kg; 0 disables "
                        f"(default: {DEFAULT_MAX_MASS})")
    p.add_argument("--concave-threshold", type=float, default=1.10,
                   help="hull/mesh volume ratio above which VHACD is used (default: 1.10)")
    p.add_argument("--vhacd-resolution", type=int, default=200000)
    p.add_argument("--no-vhacd", dest="vhacd", action="store_false",
                   help="always use a plain convex hull for collision")
    args = p.parse_args()

    meshes = sorted(
        f for f in args.src.iterdir()
        if f.suffix.lower() in {".stl", ".obj", ".ply"} and not f.stem.endswith("_col")
    )
    if not meshes:
        print(f"no meshes found in {args.src}", file=sys.stderr)
        return 1

    if args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)

    rows = []
    for i, m in enumerate(meshes, 1):
        print(f"[{i}/{len(meshes)}] {m.name}", flush=True)
        rows.append(prepare(m, args.dst, args))

    print(f"\n{'object':22s} {'src':>4s} {'faces in':>9s} {'out':>6s} "
          f"{'mean dev':>9s} {'max dev':>9s} {'size':>7s} {'mass':>9s} "
          f"{'collision':>10s} {'KB in':>8s} {'KB out':>7s}")
    for r in rows:
        flags = "!" if r["hull_fallback"] else " "
        flags += "X" if r["oversize"] else " "
        flags += "M" if r["clamped"] else " "
        print(f"{r['name'][:22]:22s} {r['units']:>4s} {r['faces_in']:9d} "
              f"{r['faces_out']:6d} {r['mean_dev']:7.4f}mm {r['max_dev']:7.4f}mm "
              f"{r['size_cm']:5.1f}cm {r['mass']:7.3f}kg {r['collision']:>10s} "
              f"{r['kb_in']:7.0f}K {r['kb_out']:6.0f}K {flags}")

    voxel_mm = WORKSPACE_SIZE / 40 * 1000
    print(f"\n{len(rows)} objects: {sum(r['kb_in'] for r in rows)/1024:.1f} MB in -> "
          f"{sum(r['kb_out'] for r in rows)/1024:.1f} MB out")
    print(f"worst surface deviation {max(r['max_dev'] for r in rows):.3f} mm "
          f"= {max(r['max_dev'] for r in rows)/voxel_mm:.1%} of a voxel "
          f"({voxel_mm:.1f} mm at 40^3 over {WORKSPACE_SIZE}m)")

    # Both of these silently produce bad training data rather than failing,
    # so they are called out explicitly instead of left in the table.
    oversize = [r["name"] for r in rows if r["oversize"]]
    if oversize:
        print(f"\nX  too large to be placed even at the packed scene's "
              f"minimum {PACKED_MIN_SCALE} scaling: {', '.join(oversize)}")
    clamped = [r["name"] for r in rows if r["clamped"]]
    if clamped:
        print(f"\nM  mass clamped to {args.max_mass}kg -- the source mesh is "
              f"open, so its volume is not physical. Re-scan or drop these "
              f"rather than trusting their grasp labels: {', '.join(clamped)}")
    patched = [r["name"] for r in rows if r["hull_fallback"]]
    if patched:
        print(f"\n!  not watertight, mass/inertia taken from the convex hull: "
              f"{', '.join(patched)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
