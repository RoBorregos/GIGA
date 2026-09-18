"""Add robust-label columns to an existing raw dataset, without regenerating it.

For every successful grasp in grasps.csv this rebuilds its scene from
mesh_pose_list, re-executes the grasp --trials times under small pose noise
(vgn.simulation.grasp_robustness) and writes:

  robust     best success fraction under noise among the yaws that worked
             (only the stored one for data without yaw_ok). 0 for failures.
  robust_ok  bitmask of the yaws reaching --robust-th, relative to the
             stored yaw, like yaw_ok.

Labels themselves are not touched: label stays "worked once", and training
or evaluation choose robust labels with --label robust. The original csv is
kept as grasps_plain.csv. Unlike generating with --robust-trials, this cannot
try other approach directions (they were never stored), so a success whose
stored approach is fragile stays fragile here.

    python3 scripts/relabel_robust.py /workspace/giga_data/v6/raw_test --num-proc 4
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np

from diagnose_top1 import load_scene_objects, rebuild
from vgn.io import read_df, write_df
from vgn.simulation import ClutterRemovalSim, grasp_robustness
from vgn.utils.transform import Rotation


def relabel_scene(job):
    raw, scene_id, rows, trials, robust_th, seed, com_frame = job
    objects = load_scene_objects(raw, scene_id)
    object_set = str(Path(objects[0][0]).parent.relative_to("data/urdfs"))
    sim = ClutterRemovalSim("packed", object_set, gui=False)
    rebuild(sim, objects, com_frame)
    rng = np.random.default_rng(seed)
    out = []
    for i, quat, pos, step, mask in rows:
        ori = Rotation.from_quat(quat)
        mask = int(mask) if step > 0 and np.isfinite(mask) and mask > 0 else 1
        n = int(round(180.0 / step)) if step > 0 else 1
        best, robust_ok = 0.0, 0
        for j in range(n):
            if mask >> j & 1:
                r = grasp_robustness(sim, ori * Rotation.from_euler("z", np.radians(j * step)),
                                     pos, trials, rng)
                best = max(best, r)
                if r >= robust_th:
                    robust_ok |= 1 << j
        out.append((i, best, robust_ok))
    return out


def main(args):
    os.chdir(Path(__file__).resolve().parents[1])  # mesh paths are relative to the giga root
    raw = args.raw.resolve()
    df = read_df(raw)
    if "robust" in df.columns and df.loc[df.label == 1, "robust"].notna().all() and not args.force:
        print(f"{raw}: already has robust labels (use --force to redo)")
        return
    if not (raw / "grasps_plain.csv").exists():
        shutil.copy(raw / "grasps.csv", raw / "grasps_plain.csv")
    for col, default in (("yaw_step_deg", 0.0), ("yaw_ok", 0)):
        if col not in df.columns:
            df[col] = default
    com_frame = not (raw / ".mesh_pose_link_frame").exists()

    jobs = []
    for k, (scene_id, part) in enumerate(df[df.label == 1].groupby("scene_id")):
        rows = [(i, r[["qx", "qy", "qz", "qw"]].to_numpy(float), r[["x", "y", "z"]].to_numpy(float),
                 float(r.yaw_step_deg), float(r.yaw_ok)) for i, r in part.iterrows()]
        jobs.append((raw, scene_id, rows, args.trials, args.robust_th, args.seed + k, com_frame))

    df["robust"] = 0.0
    df["robust_ok"] = 0
    with mp.get_context("spawn").Pool(args.num_proc) as pool:
        for n, res in enumerate(pool.imap_unordered(relabel_scene, jobs), 1):
            for i, best, ok in res:
                df.at[i, "robust"] = best
                df.at[i, "robust_ok"] = ok
            print(f"\r{n}/{len(jobs)} scenes", end="", flush=True)
    print()
    write_df(df, raw)

    pos = df[df.label == 1]
    frac = (pos.robust >= args.robust_th).mean()
    print(f"{raw}: {len(pos)} successes, {frac:.1%} robust (>= {args.robust_th} of {args.trials} noisy tries), "
          f"mean robustness {pos.robust.mean():.2f}; positive rate {df.label.mean():.3f} -> "
          f"{(df.label.astype(bool) & (df.robust >= args.robust_th)).mean():.3f} with --label robust")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, help="raw root with grasps.csv and mesh_pose_list/")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--robust-th", type=float, default=0.75)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
