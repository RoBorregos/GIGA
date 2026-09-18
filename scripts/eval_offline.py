"""Score a trained GIGA model on held-out grasp attempts, without running the robot sim.

The validation numbers train_giga.py prints come from grasps in scenes (and
objects) the network trained on, on 50/50-balanced data. This script uses a
dataset the model never saw -- normally raw_test after
`clean_balance_data.py --no-balance` -- so the positive rate is the real one,
and it reports metrics that do not depend on the 0.5 threshold:

  auc / ap        ranking quality of the quality head over all attempts
  top1            per scene, is the highest-scored attempt a success?
                  (what `--best` does at run time, restricted to sampled points)
  rot / width     errors on positives only, the rows those heads train on

and, for picking an operating threshold where false positives (a grasp that
drops the object) matter more than missed grasps:

  recall_at_precision   lowest threshold whose precision still reaches
                        --target-precision, and the recall left there
  scene_top1_sweep      per threshold: share of scenes whose best grasp
                        clears it (coverage) and how often that best grasp
                        succeeds; `recommended_qual_th` is the lowest one
                        whose top-1 success reaches --target-precision
  precision_at_top_k    per scene, success rate of the k highest-scored
  calibration           Brier score and score-vs-actual per bin; a model
                        trained on balanced data will over-predict here

It also checks for a shortcut: in generate_data_parallel.py a positive keeps
the TCP position of the approach that worked (often straight above the
object) while a negative always keeps the anti-normal one. So position alone
can carry the label. `baseline_auc` is how well TCP height alone, or the
approach direction, separates the classes, and `by_approach` shows whether
the model still ranks well *within* one approach type. If the model's AUC is
close to the height-only AUC, it has mostly learned height.

    python3 scripts/eval_offline.py --model best_vgn_giga_*.pt \\
        --dataset /workspace/giga_data/v6/processed_test \\
        --dataset_raw /workspace/giga_data/v6/raw_test --out results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vgn.dataset_voxel import yaw_mask
from vgn.io import apply_label_mode, read_df, read_setup, read_voxel_grid, valid_rotations
from vgn.networks import load_network
from vgn.utils.transform import Rotation


def auc(scores, labels):
    """ROC AUC via the rank-sum (Mann-Whitney) formula, ties averaged."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = np.asarray(scores)[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores, labels):
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def at_threshold(scores, labels, th):
    pred = scores >= th
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    return {
        "threshold": th,
        "accuracy": float((pred == labels).mean()),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "predicted_positive_rate": float(pred.mean()),
    }


def recall_at_precision(scores, labels, target):
    """Lowest threshold whose precision is still >= target."""
    order = np.argsort(-scores, kind="mergesort")
    s, hits = scores[order], labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    # only cut between distinct scores, so ties are kept or dropped together
    cut = np.r_[s[1:] != s[:-1], True]
    ok = np.flatnonzero(cut & (precision >= target))
    if len(ok) == 0 or labels.sum() == 0:
        return {"target_precision": target, "threshold": None, "precision": None,
                "recall": 0.0, "predicted_positive_rate": 0.0}
    k = ok[-1]
    return {
        "target_precision": target,
        "threshold": float(s[k]),
        "precision": float(precision[k]),
        "recall": float(tp[k] / labels.sum()),
        "predicted_positive_rate": float((k + 1) / len(s)),
    }


def scene_top1_sweep(qual, labels, scene_rows, target):
    """What the robot sees: it executes a scene's best grasp only if it clears th."""
    best_score = np.array([qual[r].max() for r in scene_rows])
    best_ok = np.array([labels[r[np.argmax(qual[r])]] for r in scene_rows])
    sweep, recommended = [], None
    for th in np.round(np.arange(0.5, 1.0, 0.05), 2).tolist() + [0.97, 0.99, 0.995, 0.999]:
        m = best_score >= th
        prec = float(best_ok[m].mean()) if m.any() else float("nan")
        sweep.append({"threshold": th, "coverage": float(m.mean()),
                      "top1_success": prec, "scenes": int(m.sum())})
        if recommended is None and m.any() and prec >= target:
            recommended = th
    return sweep, recommended


def precision_at_top_k(qual, labels, scene_rows, ks=(1, 3, 5)):
    out = {}
    for k in ks:
        vals = [labels[r[np.argsort(-qual[r])[:k]]].mean() for r in scene_rows if len(r) >= k]
        out[str(k)] = float(np.mean(vals)) if vals else float("nan")
    return out


def calibration(qual, labels, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)):
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (qual >= lo) & ((qual < hi) if hi < 1.0 else (qual <= hi))
        if not m.any():
            continue
        mean_score, rate = float(qual[m].mean()), float(labels[m].mean())
        ece += m.mean() * abs(mean_score - rate)
        bins.append({"score_range": [lo, hi], "rows": int(m.sum()),
                     "mean_score": mean_score, "actual_success": rate})
    return {"brier": float(np.mean((qual - labels) ** 2)), "ece": float(ece), "bins": bins}


def approach_type(quats):
    """Angle of the gripper's approach (+z) axis from straight down, bucketed."""
    z = Rotation.from_quat(quats).as_matrix()[:, :, 2]
    angle = np.degrees(np.arccos(np.clip(-z[:, 2], -1.0, 1.0)))
    return np.where(angle < 22.5, "top", np.where(angle < 67.5, "diagonal", "side")), angle


@torch.no_grad()
def predict(net, df, dataset_root, size, device):
    qual = np.zeros(len(df))
    rot = np.zeros((len(df), 4))
    width = np.zeros(len(df))
    for scene_id, rows in df.groupby("scene_id").groups.items():
        rows = np.asarray(list(rows))
        grid = read_voxel_grid(dataset_root, scene_id)
        x = torch.from_numpy(grid).float().to(device)  # 1 x 40 x 40 x 40
        pos = df.loc[rows, ["x", "y", "z"]].to_numpy(np.float32) / size - 0.5
        pos = torch.from_numpy(pos)[None].to(device)
        q, r, w = net(x, pos)
        qual[rows] = q.reshape(-1).cpu().numpy()
        rot[rows] = r.reshape(-1, 4).cpu().numpy()
        width[rows] = w.reshape(-1).cpu().numpy()
    return qual, rot, width


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # evaluation always counts fragile successes as failures in robust mode:
    # the question is whether the robot's grasp holds, not how it trained
    df = apply_label_mode(read_df(args.dataset_raw).reset_index(drop=True), args.label, args.robust_th)
    size, _, _, _ = read_setup(args.dataset_raw)
    net = load_network(args.model, device, args.type)
    net.eval()

    qual, rot, width = predict(net, df, args.dataset, size, device)
    labels = df["label"].values.astype(bool)
    quats = df[["qx", "qy", "qz", "qw"]].to_numpy()
    kind, angle = approach_type(quats)
    z = df["z"].values

    res = {
        "model": str(args.model),
        "label": args.label if args.label == "success" else f"robust>={args.robust_th}",
        "rows": int(len(df)),
        "scenes": int(df["scene_id"].nunique()),
        "positive_rate": float(labels.mean()),
        "auc": auc(qual, labels),
        "ap": average_precision(qual, labels),
        "ap_of_random_ranking": float(labels.mean()),
        "at_0.5": at_threshold(qual, labels, 0.5),
        f"at_{args.qual_th}": at_threshold(qual, labels, args.qual_th),
    }

    # per scene: does the top-scored attempt succeed?
    scene_rows = [np.asarray(list(r)) for r in df.groupby("scene_id").groups.values()]
    res["top1_success"] = float(np.mean([labels[r[np.argmax(qual[r])]] for r in scene_rows]))
    res["scenes_with_any_success"] = float(np.mean([labels[r].any() for r in scene_rows]))

    # operating point: false positives cost more than missed grasps
    res["recall_at_precision"] = recall_at_precision(qual, labels, args.target_precision)
    res["scene_top1_sweep"], res["recommended_qual_th"] = scene_top1_sweep(
        qual, labels, scene_rows, args.target_precision)
    res["precision_at_top_k"] = precision_at_top_k(qual, labels, scene_rows)
    res["calibration"] = calibration(qual, labels)

    # heads trained on positives only, so scored on positives only
    if labels.any():
        # distance to the closest orientation that worked: the stored one, its
        # 180-degree flip, and any other successful yaw the data recorded
        pos_rows = np.flatnonzero(labels)
        q_valid = np.stack([valid_rotations(quats[i], *yaw_mask(df, i)) for i in pos_rows])
        q_pred = rot[labels] / np.linalg.norm(rot[labels], axis=1, keepdims=True)
        dot = np.abs(np.einsum("nd,nkd->nk", q_pred, q_valid)).max(axis=1)
        rot_err = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
        width_err_mm = np.abs(width[labels] - df["width"].values[labels] / size) * size * 1000
        res["positives_rot_error_deg"] = {"median": float(np.median(rot_err)), "p90": float(np.percentile(rot_err, 90))}
        res["positives_width_error_mm"] = {"median": float(np.median(width_err_mm)), "p90": float(np.percentile(width_err_mm, 90))}

    # shortcut check
    res["baseline_auc"] = {
        "tcp_height_only": auc(z, labels),
        "approach_angle_only": auc(-angle, labels),
    }
    res["by_approach"] = {}
    for k in ("top", "diagonal", "side"):
        m = kind == k
        if m.any():
            res["by_approach"][k] = {
                "rows": int(m.sum()),
                "positive_rate": float(labels[m].mean()),
                "model_mean_score": float(qual[m].mean()),
                "model_auc_within": auc(qual[m], labels[m]),
            }
    edges = np.quantile(z, [0, 0.25, 0.5, 0.75, 1.0])
    res["by_tcp_height_quartile"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z <= hi)
        res["by_tcp_height_quartile"].append({
            "z_range_m": [float(lo), float(hi)],
            "rows": int(m.sum()),
            "positive_rate": float(labels[m].mean()),
            "model_mean_score": float(qual[m].mean()),
            "model_auc_within": auc(qual[m], labels[m]),
        })

    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type", default="giga")
    parser.add_argument("--dataset", type=Path, required=True, help="processed root (TSDF scenes)")
    parser.add_argument("--dataset_raw", type=Path, required=True, help="raw root (grasps.csv, setup.json)")
    parser.add_argument("--qual-th", type=float, default=0.9, help="the threshold sim_grasp_multiple.py uses")
    parser.add_argument("--target-precision", type=float, default=0.9,
                        help="precision the operating threshold has to reach")
    parser.add_argument("--label", choices=["success", "robust"], default="success",
                        help="ground truth: worked once, or still works under pose noise")
    parser.add_argument("--robust-th", type=float, default=0.75)
    parser.add_argument("--out", type=Path, default=None)
    main(parser.parse_args())
