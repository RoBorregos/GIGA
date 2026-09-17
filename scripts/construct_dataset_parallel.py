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
import functools
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm
import multiprocessing as mp

from vgn.io import *
from vgn.perception import *
from vgn.utils.misc import apply_noise


RESOLUTION = 40

def process_one_scene(args, f, size, intrinsic):
    """Build one scene's TSDF grid and cropped point cloud.

    size/intrinsic are passed in rather than read from module globals: the
    pool uses the spawn start method, so a worker gets a fresh interpreter
    where whatever main() assigned to a global never happened.
    """
    if f.suffix != ".npz":
        return f.stem
    depth_imgs, extrinsics = read_sensor_data(args.raw, f.stem)
    # add noise
    depth_imgs = np.array([apply_noise(x, args.add_noise) for x in depth_imgs])
    if args.single_view:
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs[[0]], intrinsic, extrinsics[[0]])
    else:
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs, intrinsic, extrinsics)
    grid = tsdf.get_grid()
    write_voxel_grid(args.dataset, f.stem, grid)

    pc = tsdf.get_cloud()
    # crop surface and borders from point cloud
    # Was the literal [0.02,0.28] box of a 0.30m workspace. Scale with size,
    # and keep the floor just above the table (which sits at finger_depth).
    lower = np.array([0.02, 0.02, args.table_height + 0.005])
    upper = np.array([size - 0.02, size - 0.02, size])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    pc = pc.crop(bounding_box)
    pc = np.asarray(pc.points)
    write_point_cloud(args.dataset, f.stem, pc)
    return str(f.stem)

def process_one_scene_star(args, size, intrinsic, f):
    """Argument order flipped so functools.partial can bind everything but `f`.

    imap_unordered passes one item per call, and a spawned worker has to be
    able to import the target by name -- a lambda or closure cannot cross
    the process boundary.
    """
    return process_one_scene(args, f, size, intrinsic)


def log_result(result):
    g_num_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_num_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_num_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    if args.single_view:
        print('Loading first view only!')
    # create directory of new dataset
    (args.dataset / "scenes").mkdir(parents=True)
    (args.dataset / "point_clouds").mkdir(parents=True)

    global g_num_completed_jobs
    global g_num_total_jobs
    global g_starting_time

    # load setup information
    size, intrinsic, _, finger_depth = read_setup(args.raw)
    # size used to be asserted equal to 6 * finger_depth. Those are now
    # independent: the workspace is its own parameter and finger_depth is
    # the gripper's real fingertip offset.
    voxel_size = size / RESOLUTION
    args.table_height = finger_depth

    # create df
    df = read_df(args.raw)
    df["x"] /= voxel_size
    df["y"] /= voxel_size
    df["z"] /= voxel_size
    df["width"] /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    write_df(df, args.dataset)

    g_num_completed_jobs = []
    file_list = list((args.raw / "scenes").iterdir())
    g_num_total_jobs = len(file_list)
    g_starting_time = time.time()

    # create tsdfs and pcs

    if args.num_proc > 1:
        # PyBullet and Open3D both spin up OpenMP thread pools at import
        # time, and forking a process that already holds them deadlocks the
        # children: they sit at ~1% CPU forever and never write a file.
        # spawn hands each worker a clean interpreter instead.
        mp.set_start_method("spawn", force=True)

        # imap_unordered rather than apply_async with a callback: results are
        # consumed as they arrive on the calling thread, so a worker's
        # exception is raised here instead of being dropped, and the run does
        # not depend on the pool's callback thread staying alive to make
        # progress. apply_async left this step stalled with idle workers and
        # results that never resolved.
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        worker = functools.partial(process_one_scene_star, args, size, intrinsic)
        with mp.Pool(processes=args.num_proc) as pool:
            for _ in tqdm(pool.imap_unordered(worker, file_list, chunksize=1),
                          total=g_num_total_jobs):
                pass
    else:
        for f in tqdm(file_list, total=len(file_list)):
            process_one_scene(args, f, size, intrinsic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--single-view", action='store_true')
    parser.add_argument("--add-noise", type=str, default='')
    args = parser.parse_args()
    main(args)
