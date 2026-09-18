"""Why does the model's best grasp fail in some scenes?

For every test scene this takes the attempt the model scores highest (what
`top1_success` in eval_offline.py counts) and reports what it is: score,
label, approach type, height, and the object it lands on. Then it groups the
failures by object and approach, so you can see whether one object or one
kind of grasp is behind them.

With --resim N it rebuilds each scene in PyBullet from mesh_pose_list and
re-executes that top-1 grasp:

  replay_exact      same pose as the label, no noise. Should reproduce the
                    label; if it often does not, the reconstruction (or the
                    physics) is not deterministic and the labels are noisy.
  label_rot_rate    N tries with small pose noise, labelled orientation.
                    A success label with a low rate here was a lucky grasp.
  pred_rot_rate     N tries with small pose noise, the orientation the
                    network predicts. This is what the robot would execute;
                    far below label_rot_rate means the rotation head is the
                    bottleneck, not the quality head.

    python3 scripts/diagnose_top1.py --model best_vgn_giga_*.pt \\
        --dataset /workspace/giga_data/v6/processed_test \\
        --dataset_raw /workspace/giga_data/v6/raw_test --resim 5 --out top1.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_offline import approach_type, predict
from vgn.grasp import Grasp, Label
from vgn.io import read_df, read_setup
from vgn.networks import load_network
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform


def load_scene_objects(raw_root, scene_id):
    data = np.load(raw_root / "mesh_pose_list" / (scene_id + ".npz"), allow_pickle=True)["pc"]
    return [(str(p), float(s), np.asarray(T)) for p, s, T in data]


def target_object(objects, pos):
    """Name of the object whose origin is closest to the grasp point."""
    d = [np.linalg.norm(T[:3, 3] - pos) for _, _, T in objects]
    return Path(objects[int(np.argmin(d))][0]).stem


def rebuild(sim, objects, com_frame):
    sim.world.reset()
    sim.world.set_gravity([0.0, 0.0, -9.81])
    sim.place_table(sim.gripper.finger_depth)
    for mesh_path, scale, T in objects:
        pose = Transform.from_matrix(T)
        body = sim.world.load_urdf(Path(mesh_path).with_suffix(".urdf"), pose, scale=scale)
        if com_frame:
            # data from before fix_mesh_pose_frames.py stores the COM pose,
            # which is what resetBasePositionAndOrientation expects
            body.set_pose(pose)
    sim.wait_for_objects_to_rest()
    sim.save_state()


def try_grasp(sim, rot, pos, rng, noise_pos, noise_deg):
    if noise_pos or noise_deg:
        pos = pos + rng.normal(0.0, noise_pos, 3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rot = rot * Rotation.from_rotvec(axis * np.radians(rng.normal(0.0, noise_deg)))
    sim.restore_state()
    outcome, _ = sim.execute_grasp(Grasp(Transform(rot, pos), sim.gripper.max_opening_width), remove=False)
    return int(outcome == Label.SUCCESS)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_df(args.dataset_raw).reset_index(drop=True)
    size, _, _, _ = read_setup(args.dataset_raw)
    net = load_network(args.model, device, args.type)
    net.eval()
    qual, rot, _ = predict(net, df, args.dataset, size, device)
    labels = df["label"].values.astype(bool)
    quats = df[["qx", "qy", "qz", "qw"]].to_numpy()
    kind, angle = approach_type(quats)

    sim, rng = None, np.random.default_rng(args.seed)
    com_frame = not (args.dataset_raw / ".mesh_pose_link_frame").exists()
    rows = []
    for scene_id, idx in df.groupby("scene_id").groups.items():
        idx = np.asarray(list(idx))
        i = idx[np.argmax(qual[idx])]
        pos = df.loc[i, ["x", "y", "z"]].to_numpy(float)
        objects = load_scene_objects(args.dataset_raw, scene_id)
        row = {
            "scene_id": scene_id,
            "score": float(qual[i]),
            "label": int(labels[i]),
            "approach": str(kind[i]),
            "approach_deg": float(angle[i]),
            "z": float(pos[2]),
            "target": target_object(objects, pos),
            "objects": [Path(p).stem for p, _, _ in objects],
            "scene_positives": int(labels[idx].sum()),
            "scene_attempts": int(len(idx)),
        }
        if args.resim:
            if sim is None:
                object_set = str(Path(objects[0][0]).parent.relative_to("data/urdfs"))
                sim = ClutterRemovalSim("packed", object_set, gui=False)
            rebuild(sim, objects, com_frame)
            r_lab = Rotation.from_quat(quats[i])
            q = rot[i] / np.linalg.norm(rot[i])
            r_pred = Rotation.from_quat(q)
            row["replay_exact"] = try_grasp(sim, r_lab, pos, rng, 0.0, 0.0)
            row["label_rot_rate"] = float(np.mean([try_grasp(sim, r_lab, pos, rng, args.noise_pos, args.noise_deg)
                                                   for _ in range(args.resim)]))
            row["pred_rot_rate"] = float(np.mean([try_grasp(sim, r_pred, pos, rng, args.noise_pos, args.noise_deg)
                                                  for _ in range(args.resim)]))
        rows.append(row)

    fails = [r for r in rows if not r["label"]]
    summary = {"model": str(args.model), "scenes": len(rows), "top1_failures": len(fails)}
    for key in ("target", "approach"):
        agg = defaultdict(lambda: [0, 0])
        for r in rows:
            agg[r[key]][0] += 1
            agg[r[key]][1] += 1 - r["label"]
        summary[f"by_{key}"] = {k: {"picked": n, "failed": f, "fail_rate": round(f / n, 2)}
                                for k, (n, f) in sorted(agg.items(), key=lambda kv: -kv[1][1])}
    if args.resim:
        summary["replay_matches_label"] = float(np.mean([r["replay_exact"] == r["label"] for r in rows]))
        for name, sel in (("label_success", [r for r in rows if r["label"]]), ("label_failure", fails)):
            if sel:
                summary[f"{name}_mean_rates"] = {
                    "label_rot": float(np.mean([r["label_rot_rate"] for r in sel])),
                    "pred_rot": float(np.mean([r["pred_rot_rate"] for r in sel])),
                }
        summary["robot_like_success"] = float(np.mean([r["pred_rot_rate"] for r in rows]))

    print(json.dumps(summary, indent=2))
    print("\nfailed top-1 grasps:")
    for r in sorted(fails, key=lambda r: -r["score"]):
        extra = (f" replay={r['replay_exact']} lab={r['label_rot_rate']:.2f} pred={r['pred_rot_rate']:.2f}"
                 if args.resim else "")
        print(f"  {r['scene_id'][:8]} score={r['score']:.3f} {r['approach']:8s} z={r['z']:.3f} "
              f"target={r['target']} scene_pos={r['scene_positives']}/{r['scene_attempts']}{extra}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type", default="giga")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset_raw", type=Path, required=True)
    parser.add_argument("--resim", type=int, default=0, help="noisy re-executions per top-1 grasp (0 = skip)")
    parser.add_argument("--noise-pos", type=float, default=0.005, help="std of position noise, m")
    parser.add_argument("--noise-deg", type=float, default=5.0, help="std of orientation noise, degrees")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    main(parser.parse_args())
