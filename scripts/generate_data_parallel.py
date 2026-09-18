# One thread per worker. numpy, Open3D and scikit-image each start an OpenMP
# pool sized to the whole machine, so with --num-proc N every worker tries to
# use all N cores and they spend their time fighting instead of working:
# measured on a 4-core box, a scene that takes 0.12s in a single process took
# 2.85s inside a 4-way pool, with the parent showing 27s of CPU against 4m20s
# of wall clock. Parallelism here comes from the pool, not from the libraries.
#
# This has to happen before numpy is imported -- the thread pools are sized at
# import time -- and it is inherited by spawned workers through os.environ.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim, grasp_robustness
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world


OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6


def main(args, rank):
    GRASPS_PER_SCENE = args.grasps_per_scene
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    grasps_per_worker = args.num_grasps // args.num_proc
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    if rank == 0:
        (args.root / "scenes").mkdir(parents=True, exist_ok=True)
        write_setup(
            args.root,
            sim.size,
            sim.camera.intrinsic,
            sim.gripper.max_opening_width,
            sim.gripper.finger_depth,
        )
        if args.save_scene:
            (args.root / "mesh_pose_list").mkdir(parents=True, exist_ok=True)

    for _ in range(grasps_per_worker // GRASPS_PER_SCENE):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()

        # render synthetic depth images
        n = MAX_VIEWPOINT_COUNT
        depth_imgs, extrinsics = render_images(sim, n)
        depth_imgs_side, extrinsics_side = render_side_images(sim, 1, args.random)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the raw data
        scene_id = write_sensor_data(args.root, depth_imgs_side, extrinsics_side)
        if args.save_scene:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
            write_point_cloud(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")

        for _ in range(GRASPS_PER_SCENE):
            # sample and evaluate a grasp point
            point, normal, depth = sample_grasp_point(pc, finger_depth)
            grasp, label, yaw_ok, robust, robust_ok = evaluate_grasp_point(
                sim, point, normal, depth, num_rotations=args.num_rotations,
                robust_trials=args.robust_trials, robust_th=args.robust_th)

            # store the sample
            write_grasp(args.root, scene_id, grasp, label,
                        yaw_step_deg=180.0 / args.num_rotations, yaw_ok=yaw_ok,
                        robust=robust, robust_ok=robust_ok)
            pbar.update()

    pbar.close()
    print('Process %d finished!' % rank)


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

def render_side_images(sim, n=1, random=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    """Pick a surface point, its normal, and how deep to grasp at it.

    This used to return `point + normal * grasp_depth` already applied, which
    silently tied the TCP offset to the surface normal. That is only correct
    when the approach IS the anti-normal: for a point on the side of a short
    object the TCP then stays at the object's side height, and a top-down
    approach from there drives the 103mm fingers straight through the table.
    The offset is now applied per approach axis, in evaluate_grasp_point.
    """
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    return point, normal, grasp_depth


def approach_axes(normal):
    """Approach directions to try at a sampled surface point.

    VGN only ever used the surface anti-normal, so a point on the side of a
    short object was only tested with a horizontal approach. With FRIDA's
    198mm-wide body that drives the palm into the table (measured: 62% of
    pregrasp poses collide palm-vs-table), the point is labelled a failure,
    and the network learns "do not grasp here" rather than "grasp here from
    above" -- even though the rotation head regresses a full orientation per
    voxel and could represent exactly that distinction.

    Returned axes point along the approach (into the object).
    """
    axes = [-normal / np.linalg.norm(normal)]
    top = np.r_[0.0, 0.0, -1.0]
    if np.linalg.norm(np.cross(axes[0], top)) > 1e-3:
        mid = axes[0] + top
        n = np.linalg.norm(mid)
        if n > 1e-6:
            axes.append(mid / n)
        axes.append(top)
    return axes


def frame_from_axis(z_axis):
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    return Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)


def widest_run_center(mask):
    """Index at the middle of the longest run of True in a circular mask.

    Yaw wraps around (the last yaw sits next to yaw 0), so the array is
    rotated to start at a False before looking for runs, which keeps a run
    that crosses the ends in one piece. An all-True mask returns its middle.
    """
    mask = np.asarray(mask, dtype=float)
    n = len(mask)
    shift = int(np.argmin(mask))
    peaks, properties = signal.find_peaks(x=np.r_[0, np.roll(mask, -shift), 0], height=1, width=1)
    return (peaks[np.argmax(properties["widths"])] - 1 + shift) % n


def relative_mask(flags, idx):
    """Bitmask of True entries, bit j meaning index (idx + j) mod n."""
    n = len(flags)
    return sum(1 << j for j in range(n) if flags[(idx + j) % n])


def evaluate_grasp_point(sim, surface_point, normal, grasp_depth, num_rotations=12,
                         robust_trials=0, robust_th=0.75, rng=np.random):
    """Try several approach directions, each over a yaw sweep.

    The label is the best outcome over every (approach, yaw) tried, and the
    returned Grasp carries the orientation that achieved it -- so the rotation
    target the network regresses is the approach that actually worked at this
    point, not whichever one the surface normal happened to dictate.

    Returns (grasp, label, yaw_ok, robust, robust_ok):

    yaw_ok     bitmask of every yaw that worked on the chosen approach,
               relative to the stored one: bit j set means
               orientation * Rz(j * step) also succeeded (step = pi /
               num_rotations). 0 for failures.

    With robust_trials > 0 every yaw that succeeded is re-executed
    robust_trials times under small pose noise (vgn.simulation
    .grasp_robustness):

    robust     success fraction of the stored yaw under that noise (NaN when
               robust_trials == 0, 0 for failures).
    robust_ok  like yaw_ok, but only yaws whose fraction reaches robust_th.

    Robustness never removes a success: `label` is still "worked once". It
    also does not stop at the first approach that works. If that one is
    fragile the next approaches are still swept, and the most robust yaw
    across all of them is stored, so a fragile early success cannot hide a
    robust grasp at the same point.
    """
    yaws = np.linspace(0.0, np.pi, num_rotations, endpoint=False)
    fallback = None  # (ori, pos) of the first approach, stored for failures
    best = None      # dict for the chosen successful yaw

    for z_axis in approach_axes(normal):
        R = frame_from_axis(z_axis)
        # Back off from the surface ALONG THIS APPROACH, so the fingers come in
        # from the direction actually being tested rather than from wherever
        # the surface normal happened to point.
        pos = surface_point - z_axis * grasp_depth
        oris = [R * Rotation.from_euler("z", yaw) for yaw in yaws]
        outcomes, widths = [], []
        for ori in oris:
            sim.restore_state()
            candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
            outcome, width = sim.execute_grasp(candidate, remove=False)
            outcomes.append(outcome)
            widths.append(width)
        if fallback is None:
            fallback = (oris[0], pos)

        successes = np.asarray(outcomes) == Label.SUCCESS
        if not successes.any():
            continue

        if robust_trials == 0:
            # plain labels: centre of the widest run of successful yaws, and
            # an approach that works is enough (keeps generation cheap)
            idx = widest_run_center(successes)
            best = dict(ori=oris[idx], pos=pos, width=widths[idx],
                        yaw_ok=relative_mask(successes, idx), robust=np.nan, robust_ok=0)
            break

        rob = np.zeros(num_rotations)
        for j in np.flatnonzero(successes):
            rob[j] = grasp_robustness(sim, oris[j], pos, robust_trials, rng)
        robust_yaws = rob >= robust_th
        # prefer the middle of the widest run of robust yaws; if none is
        # robust, the most robust yaw (ties go to the widest success run)
        if robust_yaws.any():
            idx = widest_run_center(robust_yaws)
        else:
            top = rob == rob.max()
            idx = widest_run_center(top & successes)
        cand = dict(ori=oris[idx], pos=pos, width=widths[idx],
                    yaw_ok=relative_mask(successes, idx), robust=float(rob[idx]),
                    robust_ok=relative_mask(robust_yaws, idx))
        if best is None or cand["robust"] > best["robust"]:
            best = cand
        if best["robust"] >= robust_th:
            break  # robust enough; further approaches cannot change the label

    if best is None:
        ori, pos = fallback
        robust = 0.0 if robust_trials else np.nan
        return (Grasp(Transform(ori, pos), sim.gripper.max_opening_width),
                int(Label.FAILURE), 0, robust, 0)
    return (Grasp(Transform(best["ori"], best["pos"]), best["width"]), int(Label.SUCCESS),
            best["yaw_ok"], best["robust"], best["robust_ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-rotations", type=int, default=12,
                        help="yaws tried per approach, spread over [0, pi)")
    parser.add_argument("--robust-trials", type=int, default=4,
                        help="noisy re-executions per successful yaw (0 = plain labels)")
    parser.add_argument("--robust-th", type=float, default=0.75,
                        help="success fraction under noise for a yaw to count as robust")
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--random", action="store_true", help="Add distrubation to camera pose")
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    args.save_scene = True
    if args.num_proc > 1:
        # PyBullet and Open3D both spin up OpenMP thread pools at import
        # time, and forking a process that already holds them deadlocks the
        # children: they sit at ~1% CPU forever and never write a file.
        # spawn hands each worker a clean interpreter instead.
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=args.num_proc)
        results = [
            pool.apply_async(func=main, args=(args, i))
            for i in range(args.num_proc)
        ]
        pool.close()
        # apply_async throws a worker's exception away unless the result is
        # read back. That is what made a crashed run look like a successful
        # one that happened to produce no data -- surface it instead.
        for r in results:
            r.get()
        pool.join()
    else:
        main(args, 0)
