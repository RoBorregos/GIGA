import argparse
from pathlib import Path
from datetime import datetime
import functools

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Average, Accuracy, Precision, Recall
import numpy as np
import torch
from torch.utils import tensorboard
import torch.nn.functional as F

#from vgn.dataset_pc import DatasetPCOcc
from vgn.dataset_voxel import DatasetVoxelOccFile
from vgn.networks import get_network, load_network
from vgn.utils.misc import set_random_seed

# loss_rot / loss_width are only trained on positives, so they are logged over
# positives only -- averaging them over negatives (whose targets are arbitrary)
# made the numbers meaningless.
LOSS_KEYS = ['loss_all', 'loss_qual', 'loss_rot', 'loss_width', 'loss_occ']

def main(args):
    if args.seed is not None:
        set_random_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 16, "pin_memory": True} if use_cuda else {}

    # create log directory
    if args.savedir == '':
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "{}_dataset={},augment={},net={},batch_size={},lr={:.0e},{}".format(
            time_stamp,
            args.dataset.name,
            args.augment,
            args.net,
            args.batch_size,
            args.lr,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.dataset_raw, args.batch_size, args.val_split, args.augment, kwargs,
        split=args.split, seed=args.seed)

    # Weight positives in the quality loss. "auto" = negatives/positives of the
    # training rows, for data that was cropped but not balanced.
    labels = train_loader.dataset.dataset.df["label"].values[train_loader.dataset.indices]
    if args.pos_weight == "auto":
        pos_weight = float((labels == 0).sum()) / max(1, int((labels == 1).sum()))
    else:
        pos_weight = float(args.pos_weight)
    print(f"train rows: {len(labels)}, positive rate: {labels.mean():.3f}, pos_weight: {pos_weight:.2f}")
    loss_fn = functools.partial(weighted_loss_fn, pos_weight=pos_weight)

    # build the network or load
    if args.load_path == '':
        net = get_network(args.net).to(device)
    else:
        net = load_network(args.load_path, device, args.net)

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    metrics = {
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
        "precision": Precision(lambda out: (torch.round(out[1][0]), out[2][0])),
        "recall": Recall(lambda out: (torch.round(out[1][0]), out[2][0])),
    }
    for k in LOSS_KEYS:
        metrics[k] = Average(lambda out, sk=k: out[3][sk])

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True, dynamic_ncols=True, disable=args.silence).attach(trainer)

    train_writer, val_writer = create_summary_writers(net, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        for k, v in metrics.items():
            train_writer.add_scalar(k, v, epoch)

        msg = 'Train'
        for k, v in metrics.items():
            msg += f' {k}: {v:.4f}'
        print(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        for k, v in metrics.items():
            val_writer.add_scalar(k, v, epoch)
            
        msg = 'Val'
        for k, v in metrics.items():
            msg += f' {k}: {v:.4f}'
        print(msg)

    def default_score_fn(engine):
        # Accuracy moves in steps of 1/len(val); on a few hundred rows the
        # "best" epoch is mostly noise. Quality loss is smoother.
        if args.select_by == "accuracy":
            return engine.state.metrics['accuracy']
        return -engine.state.metrics['loss_qual']

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "vgn",
        n_saved=1,
        require_empty=True,
    )
    best_checkpoint_handler = ModelCheckpoint(
        logdir,
        "best_vgn",
        n_saved=1,
        score_name="val_acc" if args.select_by == "accuracy" else "neg_val_loss_qual",
        score_function=default_score_fn,
        require_empty=True,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, best_checkpoint_handler, {args.net: net}
    )

    # v6 hit its best val loss at epoch 10 of 30 and only memorised after
    # that. Halve the LR as soon as validation stalls for an epoch, and stop
    # once it has not improved for `patience` epochs; best_vgn_* already
    # holds the best epoch.
    if args.lr_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_plateau)

        @evaluator.on(Events.EPOCH_COMPLETED)
        def step_lr(engine):
            scheduler.step(default_score_fn(engine))
            print(f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    if args.patience:
        evaluator.add_event_handler(
            Events.COMPLETED,
            EarlyStopping(patience=args.patience, score_function=default_score_fn, trainer=trainer),
        )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(root, root_raw, batch_size, val_split, augment, kwargs,
                             split="scene", seed=None):
    # augment is deliberately NOT forwarded: apply_transform works in voxel
    # units (z offset 6..34, centre 20) but this dataset stores positions in
    # meters, so enabling it would corrupt the grasp targets.
    dataset = DatasetVoxelOccFile(root, root_raw)
    rng = np.random.default_rng(seed)
    if split == "scene":
        # Every scene holds ~120 grasps on the same TSDF. Splitting rows lets
        # validation grasps share a scene with training grasps, which measures
        # interpolation inside a scene the network has already seen.
        scene_ids = dataset.df["scene_id"].values
        scenes = rng.permutation(np.unique(scene_ids))
        n_val = max(1, int(round(val_split * len(scenes))))
        is_val = np.isin(scene_ids, scenes[:n_val])
        val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
        print(f"scene split: {len(scenes) - n_val} train / {n_val} val scenes, "
              f"{len(train_idx)} / {len(val_idx)} rows")
    else:
        perm = rng.permutation(len(dataset))
        val_size = int(val_split * len(dataset))
        val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_set = torch.utils.data.Subset(dataset, val_idx.tolist())
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    pc, (label, rotations, width), pos, pos_occ, occ_value = batch
    pc = pc.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    width = width.float().to(device)
    pos.unsqueeze_(1) # B, 1, 3
    pos = pos.float().to(device)
    pos_occ = pos_occ.float().to(device)
    occ_value = occ_value.float().to(device)
    return pc, (label, rotations, width, occ_value), pos, pos_occ


def select(out):
    qual_out, rot_out, width_out, occ = out
    rot_out = rot_out.squeeze(1)
    occ = torch.sigmoid(occ) # to probability
    return qual_out.squeeze(-1), rot_out, width_out.squeeze(-1), occ


def weighted_loss_fn(y_pred, y, pos_weight=1.0):
    label_pred, rotation_pred, width_pred, occ_pred = y_pred
    label, rotations, width, occ = y
    loss_qual = _qual_loss_fn(label_pred, label)
    loss_rot = _rot_loss_fn(rotation_pred, rotations)
    loss_width = _width_loss_fn(width_pred, width)
    loss_occ = _occ_loss_fn(occ_pred, occ)
    qual_weight = 1.0 + (pos_weight - 1.0) * label
    loss = qual_weight * loss_qual + label * (loss_rot + 0.01 * loss_width) + loss_occ
    n_pos = label.sum().clamp(min=1.0)
    loss_dict = {'loss_qual': loss_qual.mean(),
                 'loss_rot': (label * loss_rot).sum() / n_pos,
                 'loss_width': (label * loss_width).sum() / n_pos,
                 'loss_occ': loss_occ.mean(),
                 'loss_all': loss.mean()}
    return loss.mean(), loss_dict


loss_fn = weighted_loss_fn


def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none")


def _rot_loss_fn(pred, target):
    # target is (B, K, 4): every orientation that worked for the row (see
    # vgn.io.valid_rotations). Only the closest one is penalised, so the net
    # is not pulled towards the average of several valid yaws.
    return (1.0 - torch.abs(torch.einsum("bd,bkd->bk", pred, target))).min(dim=1).values


def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def _width_loss_fn(pred, target):
    return F.mse_loss(40 * pred, 40 * target, reduction="none")

def _occ_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)


def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()
        # forward
        x, y, pos, pos_occ = prepare_batch(batch, device)
        y_pred = select(net(x, pos, p_tsdf=pos_occ))
        loss, loss_dict = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss_dict

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, pos, pos_occ = prepare_batch(batch, device)
            y_pred = select(net(x, pos, p_tsdf=pos_occ))
            loss, loss_dict = loss_fn(y_pred, y)
        return x, y_pred, y, loss_dict

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="giga")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset_raw", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--load-path", type=str, default='')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3,
                        help="stop after this many epochs without val improvement (0 = off)")
    parser.add_argument("--lr-plateau", type=int, default=1,
                        help="halve LR after this many epochs without improvement (0 = off)")
    parser.add_argument("--split", choices=["scene", "grasp"], default="scene",
                        help="hold out whole scenes (default) or individual grasps (old behaviour)")
    parser.add_argument("--pos-weight", default="1",
                        help="weight on positives in the quality loss; 'auto' = neg/pos of train rows")
    parser.add_argument("--select-by", choices=["loss_qual", "accuracy"], default="loss_qual",
                        help="validation metric that picks the best_vgn checkpoint")
    args = parser.parse_args()
    print(args)
    main(args)
