"""Convert mesh_pose_list poses from the COM frame to the URDF link frame.

Data generated before the fix in vgn.utils.implicit.get_mesh_pose_list_from_world
stored the PyBullet base pose, which is the centre of mass, while the meshes
live in the link frame. Everything built from mesh_pose_list (occupancy
labels, scene reconstruction) was offset by each object's COM offset.

For every raw root given this rewrites mesh_pose_list/*.npz in place, keeps
the originals in mesh_pose_list_com/, and leaves a .mesh_pose_link_frame
marker so running it twice is a no-op. Re-run save_occ_data_parallel.py on
the root afterwards (the old occ/ files are wrong).

    python3 scripts/fix_mesh_pose_frames.py /workspace/giga_data/v6/raw_train /workspace/giga_data/v6/raw_test
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pybullet

from vgn.utils.implicit import com_to_link

MARKER = ".mesh_pose_link_frame"


def main(args):
    p = pybullet.connect(pybullet.DIRECT)
    cache = {}

    def link_pose(mesh_path, scale, T):
        urdf = Path(mesh_path).with_suffix(".urdf")
        key = (str(urdf), round(float(scale), 6))
        if key not in cache:
            uid = pybullet.loadURDF(str(urdf), globalScaling=float(scale), physicsClientId=p)
            cache[key] = uid
        return com_to_link(_Client(p), cache[key], np.asarray(T))

    for root in args.roots:
        if (root / MARKER).exists():
            print(f"{root}: already in link frame, skipping")
            continue
        src, backup = root / "mesh_pose_list", root / "mesh_pose_list_com"
        if not backup.exists():
            shutil.copytree(src, backup)
        files = sorted(backup.glob("*.npz"))
        for f in files:
            rows = np.load(f, allow_pickle=True)["pc"]
            fixed = [(m, s, link_pose(m, s, T)) for m, s, T in rows]
            out = np.empty(len(fixed), dtype=object)
            out[:] = fixed
            np.savez_compressed(src / f.name, pc=out)
        (root / MARKER).touch()
        print(f"{root}: converted {len(files)} scenes (originals in {backup.name}/)")


class _Client:
    """Just enough of a bullet client for com_to_link on a raw connection."""

    def __init__(self, cid):
        self.cid = cid

    def getDynamicsInfo(self, uid, link):
        return pybullet.getDynamicsInfo(uid, link, physicsClientId=self.cid)

    def getMatrixFromQuaternion(self, q):
        return pybullet.getMatrixFromQuaternion(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+", help="raw data roots (with mesh_pose_list/)")
    main(parser.parse_args())
