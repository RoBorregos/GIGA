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

import os
import glob
import time
import argparse
import numpy as np
import multiprocessing as mp

from vgn.utils.implicit import get_scene_from_mesh_pose_list, sample_iou_points

def sample_occ(mesh_pose_list_path, num_point, uniform):
    mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
    scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True)
    points, occ = sample_iou_points(mesh_list, scene.bounds, num_point, uniform=uniform)
    return points, occ

def save_occ(mesh_pose_list_path, args):
    points, occ = sample_occ(mesh_pose_list_path, args.num_point_per_file * args.num_file, args.uniform)
    points = points.astype(np.float16).reshape(args.num_file, args.num_point_per_file, 3)
    occ = occ.reshape(args.num_file, args.num_point_per_file)
    name = os.path.basename(mesh_pose_list_path)[:-4]
    save_root = os.path.join(args.raw, 'occ', name)
    os.makedirs(save_root)
    for i in range(args.num_file):
        np.savez(os.path.join(save_root, '%04d.npz' % i), points=points[i], occ=occ[i])

def log_result(result):
    g_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    mesh_list_files = glob.glob(os.path.join(args.raw, 'mesh_pose_list', '*.npz'))
    

    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time

    g_num_total_jobs = len(mesh_list_files)
    g_completed_jobs = []

    g_starting_time = time.time()

    if args.num_proc > 1:
        # PyBullet and Open3D both spin up OpenMP thread pools at import
        # time, and forking a process that already holds them deadlocks the
        # children: they sit at ~1% CPU forever and never write a file.
        # spawn hands each worker a clean interpreter instead.
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=args.num_proc)
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        results = [
            pool.apply_async(func=save_occ, args=(f, args), callback=log_result)
            for f in mesh_list_files
        ]
        pool.close()
        # apply_async throws a worker's exception away unless the result is
        # read back. That is what made a crashed run look like a successful
        # one that happened to produce no data -- surface it instead.
        for r in results:
            r.get()
        pool.join()
    else:
        for f in mesh_list_files:
            save_occ(f, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("raw", type=str)
    parser.add_argument("num_point_per_file", type=int)
    parser.add_argument("num_file", type=int)
    parser.add_argument("--uniform", action='store_true', help='sample uniformly in the bbox, else sample in the tight bbox')
    args = parser.parse_args()
    main(args)