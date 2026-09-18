import json
import uuid

import numpy as np
import pandas as pd

from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


def write_setup(root, size, intrinsic, max_opening_width, finger_depth):
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict(),
        "max_opening_width": max_opening_width,
        "finger_depth": finger_depth,
    }
    write_json(data, root / "setup.json")


def read_setup(root):
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    max_opening_width = data["max_opening_width"]
    finger_depth = data["finger_depth"]
    return size, intrinsic, max_opening_width, finger_depth


def write_sensor_data(root, depth_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path = root / "scenes" / (scene_id + ".npz")
    assert not path.exists()
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def write_full_sensor_data(root, depth_imgs, extrinsics, scene_id=None):
    if scene_id is None:
        scene_id = uuid.uuid4().hex
    path = root / "full_scenes" / (scene_id + ".npz")
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def read_sensor_data(root, scene_id):
    data = np.load(root / "scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]

def read_full_sensor_data(root, scene_id):
    data = np.load(root / "full_scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]


def write_grasp(root, scene_id, grasp, label, yaw_step_deg=0.0, yaw_ok=0,
                robust=float("nan"), robust_ok=0):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label",
             "yaw_step_deg", "yaw_ok", "robust", "robust_ok"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label, yaw_step_deg, yaw_ok,
               robust, robust_ok)


MAX_VALID_ROTATIONS = 48  # 2 (gripper symmetry) x up to 24 yaws


def valid_rotations(quat, yaw_step_deg=0.0, yaw_ok=0):
    """Every orientation counted as correct for one grasp row, as quaternions.

    The stored orientation and its 180-degree flip about the approach axis
    (the gripper is symmetric), plus, when the row carries yaw_ok, every other
    yaw that also succeeded. Padded to MAX_VALID_ROTATIONS by repeating the
    first entry, so batches have a fixed shape and a min over the set is
    unchanged. Rows without yaw_ok (older data) get just the pair.
    """
    ori = Rotation.from_quat(quat)
    flip = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
    has_mask = np.isfinite(yaw_step_deg) and yaw_step_deg > 0 and np.isfinite(yaw_ok)
    mask = (int(yaw_ok) if has_mask else 0) or 1  # bit 0 = the stored yaw
    rots = []
    for j in range(MAX_VALID_ROTATIONS // 2):
        if mask >> j & 1:
            r = ori * Rotation.from_euler("z", np.radians(j * yaw_step_deg))
            rots += [r.as_quat(), (r * flip).as_quat()]
    out = np.repeat(np.asarray(rots[:1], dtype=np.single), MAX_VALID_ROTATIONS, axis=0)
    out[:len(rots)] = rots[:MAX_VALID_ROTATIONS]
    return out


def read_grasp(df, i):
    scene_id = df.loc[i, "scene_id"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, grasp, label


def read_df(root):
    return pd.read_csv(root / "grasps.csv")


def write_df(df, root):
    df.to_csv(root / "grasps.csv", index=False)


def write_voxel_grid(root, scene_id, voxel_grid):
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, grid=voxel_grid)


def write_point_cloud(root, scene_id, point_cloud, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    try:
        data = np.asarray(point_cloud)
    except ValueError:
        # mesh_pose_list rows are (mesh_path, scale, 4x4 pose) -- ragged, which
        # numpy < 1.24 coerced to an object array on its own but newer numpy
        # rejects. Readers of this file already pass allow_pickle=True.
        data = np.array(point_cloud, dtype=object)
    np.savez_compressed(path, pc=data)

def read_voxel_grid(root, scene_id):
    path = root / "scenes" / (scene_id + ".npz")
    return np.load(path)["grid"]

def read_point_cloud(root, scene_id, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    return np.load(path)["pc"]

def read_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")


def apply_label_mode(df, mode="success", robust_th=0.75, fragile="negative"):
    """Return df with `label` (and rotation targets) set for a label mode.

    success   label as generated: the grasp worked once. Unchanged df.
    robust    positive only if the stored grasp also succeeds under pose
              noise at least robust_th of the time (column `robust`).
              Successes below that are "fragile": relabelled 0 with
              fragile="negative" (teaches the net to avoid them) or removed
              with fragile="drop" (neither rewarded nor punished). Rotation
              targets become the robust yaws (robust_ok) where recorded.
    """
    if mode == "success":
        return df
    if "robust" not in df.columns or not df.loc[df.label == 1, "robust"].notna().any():
        raise ValueError("robust labels requested but the data has no 'robust' column; "
                         "generate with --robust-trials or run scripts/relabel_robust.py")
    df = df.copy()
    fragile_rows = (df.label == 1) & ~(df.robust >= robust_th)
    if fragile == "drop":
        df = df[~fragile_rows].reset_index(drop=True)
    else:
        df.loc[fragile_rows, "label"] = 0
    if "robust_ok" in df.columns and "yaw_ok" in df.columns:
        use = (df.label == 1) & (df.robust_ok > 0)
        df.loc[use, "yaw_ok"] = df.loc[use, "robust_ok"]
    return df
