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
from vgn.simulation import ClutterRemovalSim
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
            grasp, label, yaw_ok = evaluate_grasp_point(sim, point, normal, depth,
                                                        num_rotations=args.num_rotations)

            # store the sample
            write_grasp(args.root, scene_id, grasp, label,
                        yaw_step_deg=180.0 / args.num_rotations, yaw_ok=yaw_ok)
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


def evaluate_grasp_point(sim, surface_point, normal, grasp_depth, num_rotations=12):
    """Try several approach directions, each over a yaw sweep.

    The label is the best outcome over every (approach, yaw) tried, and the
    returned Grasp carries the orientation that achieved it -- so the rotation
    target the network regresses is the approach that actually worked at this
    point, not whichever one the surface normal happened to dictate.

    Also returns yaw_ok, a bitmask of every yaw that worked on that approach,
    relative to the stored one: bit j set means orientation * Rz(j * step)
    also succeeded (step = pi / num_rotations). Several yaws often work; the
    loss and the evaluation treat all of them as correct instead of only the
    one written to qx..qw. 0 for failures.
    """
    yaw_ok = 0
    best_ori = None
    best_pos = surface_point
    best_width = sim.gripper.max_opening_width
    best_outcome = Label.FAILURE

    for z_axis in approach_axes(normal):
        R = frame_from_axis(z_axis)
        # Back off from the surface ALONG THIS APPROACH, so the fingers come in
        # from the direction actually being tested rather than from wherever
        # the surface normal happened to point.
        pos = surface_point - z_axis * grasp_depth
        # The gripper is symmetric, so yaw 0 and yaw pi are the same grasp:
        # leave pi out or one of the tries is wasted on a repeat.
        yaws = np.linspace(0.0, np.pi, num_rotations, endpoint=False)
        outcomes, widths = [], []
        for yaw in yaws:
            ori = R * Rotation.from_euler("z", yaw)
            sim.restore_state()
            candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
            outcome, width = sim.execute_grasp(candidate, remove=False)
            outcomes.append(outcome)
            widths.append(width)
        if best_ori is None:
            best_ori = R * Rotation.from_euler("z", yaws[0])
            best_pos = pos

        # mid-point of the widest run of successful yaws. Yaw wraps around
        # (the last one sits next to yaw 0), so start the array at a failure
        # so no run is split across the ends.
        successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
        if np.sum(successes) and best_outcome != Label.SUCCESS:
            shift = int(np.argmin(successes))
            peaks, properties = signal.find_peaks(
                x=np.r_[0, np.roll(successes, -shift), 0], height=1, width=1
            )
            idx = (peaks[np.argmax(properties["widths"])] - 1 + shift) % num_rotations
            best_ori = R * Rotation.from_euler("z", yaws[idx])
            best_pos = pos
            best_width = widths[idx]
            best_outcome = Label.SUCCESS
            for j in range(num_rotations):
                if successes[(idx + j) % num_rotations]:
                    yaw_ok |= 1 << j
            break  # an approach that works is enough; keep generation cheap

    return Grasp(Transform(best_ori, best_pos), best_width), int(best_outcome), yaw_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-rotations", type=int, default=12,
                        help="yaws tried per approach, spread over [0, pi)")
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
