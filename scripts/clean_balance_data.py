from pathlib import Path
import argparse

import numpy as np

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform

def main(args):
    root = args.root

    # print
    df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    print("Before clean and balance:")
    print("Number of samples:", len(df.index))
    print("Number of positives:", len(positives.index))
    print("Number of negatives:", len(negatives.index))

    # clean -- the bounds used to be the literal 0.02/0.28 of a 0.30m
    # workspace, which silently dropped every grasp beyond 0.28 once the
    # workspace grew. Read the real size from the run's own setup.json.
    size, _, _, _ = read_setup(root)
    lo, hi = 0.02, size - 0.02
    df = read_df(root)
    for axis in ("x", "y", "z"):
        df.drop(df[df[axis] < lo].index, inplace=True)
        df.drop(df[df[axis] > hi].index, inplace=True)
    # write_df(df, root)

    print("After crop to [%.2f, %.2f]:" % (lo, hi), len(df.index), "samples")

    # balance -- skip it for evaluation data (keep the real positive rate) or
    # when training with a positive-class weight instead of dropping negatives.
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    if args.no_balance:
        print("Not balancing (--no-balance)")
    elif len(negatives.index) >= len(positives.index):
        rng = np.random.default_rng(args.seed)
        i = rng.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
        df = df.drop(i)
    else:
        print("More positives than negatives, not balancing")
    write_df(df, root)

    # remove unreferenced scenes.
    # df = read_df(root)
    scenes = df["scene_id"].values
    for f in (root / "scenes").iterdir():
        if f.suffix == ".npz" and f.stem not in scenes:
            print("Removed", f)
            f.unlink()

    # print
    df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    print("After clean and balance:")
    print("Number of samples:", len(df.index))
    print("Number of positives:", len(positives.index))
    print("Number of negatives:", len(negatives.index))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--no-balance", action="store_true",
                        help="only crop; keep every negative (use for test data)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for choosing which negatives to drop")
    args = parser.parse_args()
    main(args)