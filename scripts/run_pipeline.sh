#!/usr/bin/env bash
# Full GIGA training pipeline:
#   generate data -> clean/balance -> build dataset -> occupancy data -> train
# The grasp-trial eval on frida/test only runs with --run-test.
#
# Run inside the giga-train container (any folder). Logs and the run config
# are saved in $GIGA_DATA_ROOT/logs/<run-name>/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIGA_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$GIGA_ROOT"

# .bashrc (which runs setup_giga.sh) is skipped by non-interactive
# `docker exec`, so run it here if GIGA is not installed yet.
if [[ -f /root/setup_giga.sh ]] && ! python3 -c "import vgn" >/dev/null 2>&1; then
  echo "vgn not importable yet -- running setup_giga.sh first"
  . /root/setup_giga.sh
fi

DATA_ROOT="${GIGA_DATA_ROOT:-/workspace/giga_data}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
NUM_PROC="${NUM_PROC:-$(nproc)}"
NUM_GRASPS_TRAIN="${NUM_GRASPS_TRAIN:-10000}"
NUM_GRASPS_TEST="${NUM_GRASPS_TEST:-2000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-4}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
GRASPS_PER_SCENE="${GRASPS_PER_SCENE:-120}"
NUM_ROTATIONS="${NUM_ROTATIONS:-12}"   # yaws tried per approach when labelling
# 0 = keep every negative (scores stay calibrated to the real success rate);
# 1 = drop negatives down to 50/50 like the original GIGA. Test data is
# never balanced, so its metrics use the real positive rate.
BALANCE_TRAIN="${BALANCE_TRAIN:-0}"
SEED="${SEED:-0}"
SPLIT="${SPLIT:-scene}"               # scene | grasp (validation split)
POS_WEIGHT="${POS_WEIGHT:-1}"
SELECT_BY="${SELECT_BY:-loss_qual}"   # loss_qual | accuracy
TARGET_PRECISION="${TARGET_PRECISION:-0.9}"
PATIENCE="${PATIENCE:-3}"             # early stopping, epochs without val improvement (0 = off)
RUN_TEST=0
FORCE_REDO=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--run-test] [--force] [--run-name NAME]

  --run-test       after training, also run the held-out grasp-trial
                    evaluation (sim_grasp_multiple.py) on frida/test.
                    Without this flag, the script stops after training
                    and just prints the eval command for later.
  --force          re-run generate/construct steps even if their output
                    already exists (by default they're skipped -- data
                    generation is the expensive part of this pipeline).
  --run-name NAME  label for this run's logs/checkpoints (default: timestamp).

Tune via env vars: NUM_PROC, NUM_GRASPS_TRAIN, NUM_GRASPS_TEST, EPOCHS,
BATCH_SIZE, LR, VAL_SPLIT, GRASPS_PER_SCENE, NUM_ROTATIONS, BALANCE_TRAIN,
SEED, SPLIT, POS_WEIGHT, SELECT_BY, TARGET_PRECISION, PATIENCE, GIGA_DATA_ROOT.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-test) RUN_TEST=1; shift ;;
    --force) FORCE_REDO=1; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

LOG_DIR="$DATA_ROOT/logs/$RUN_NAME"
RAW_TRAIN="$DATA_ROOT/raw_train"
RAW_TEST="$DATA_ROOT/raw_test"
PROC_TRAIN="$DATA_ROOT/processed_train"
PROC_TEST="$DATA_ROOT/processed_test"
LOGDIR_TRAIN="$DATA_ROOT/runs/$RUN_NAME"
EVAL_DIR="$DATA_ROOT/eval/$RUN_NAME"
# The container runs as root. On exit (success or failure), give the files
# back to whoever owns the closest existing data folder, so you can delete
# them from the host. Checked before mkdir, which would create it as root.
owner_dir="$DATA_ROOT"
while [[ ! -d "$owner_dir" ]]; do owner_dir="$(dirname "$owner_dir")"; done
DATA_OWNER="$(stat -c '%u:%g' "$owner_dir")"
give_data_back() {
  if [[ "$(id -u)" -eq 0 && "$DATA_OWNER" != "0:0" ]]; then
    chown -R "$DATA_OWNER" "$DATA_ROOT" || true
  fi
}
trap give_data_back EXIT

mkdir -p "$LOG_DIR" "$LOGDIR_TRAIN" "$EVAL_DIR"
[[ "$FORCE_REDO" -eq 1 ]] && rm -f "$DATA_ROOT"/.done_*

CONFIG_LOG="$LOG_DIR/config.txt"
{
  echo "run_name=$RUN_NAME"
  echo "started=$(date -Iseconds)"
  echo "num_proc=$NUM_PROC num_grasps_train=$NUM_GRASPS_TRAIN num_grasps_test=$NUM_GRASPS_TEST"
  echo "epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR val_split=$VAL_SPLIT"
  echo "grasps_per_scene=$GRASPS_PER_SCENE num_rotations=$NUM_ROTATIONS balance_train=$BALANCE_TRAIN"
  echo "seed=$SEED split=$SPLIT pos_weight=$POS_WEIGHT select_by=$SELECT_BY target_precision=$TARGET_PRECISION patience=$PATIENCE"
  echo "run_test=$RUN_TEST force=$FORCE_REDO"
} | tee "$CONFIG_LOG"

# Runs "$@", saves its output to $LOG_DIR/<name>.log, and stops on failure.
# A step is skipped next time only if it left a .done_<name> marker, which is
# written only on success (a half-finished output folder does not count).
step() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  local marker="$DATA_ROOT/.done_${name}"
  if [[ "$FORCE_REDO" -eq 0 ]] && [[ -f "$marker" ]]; then
    echo "[$name] completed on an earlier run, skipping (use --force to redo)" | tee "$log"
    return 0
  fi
  echo "[$name] $(date -Iseconds) start: $*" | tee "$log"
  local t0=$SECONDS
  # `|| rc=$?` stops `set -e` from exiting early; pipefail makes rc the
  # command's exit code instead of tee's.
  local rc=0
  "$@" 2>&1 | tee -a "$log" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    touch "$marker"
    echo "[$name] $(date -Iseconds) done in $((SECONDS - t0))s" | tee -a "$log"
  else
    echo "[$name] $(date -Iseconds) FAILED after $((SECONDS - t0))s -- see $log" | tee -a "$log"
    exit 1
  fi
}

# Each worker needs at least one scene worth of grasps, or it silently
# generates 0 scenes.
for pair in "TRAIN:$NUM_GRASPS_TRAIN" "TEST:$NUM_GRASPS_TEST"; do
  split="${pair%%:*}"; n="${pair##*:}"
  if (( n / NUM_PROC < GRASPS_PER_SCENE )); then
    echo "NUM_GRASPS_$split=$n with NUM_PROC=$NUM_PROC gives $((n / NUM_PROC)) grasps/worker," >&2
    echo "under the $GRASPS_PER_SCENE grasps-per-scene floor -- that generates 0 scenes. Raise it to at least $((GRASPS_PER_SCENE * NUM_PROC))." >&2
    exit 1
  fi
done

step "01_generate_train" \
  python3 scripts/generate_data_parallel.py "$RAW_TRAIN" \
    --scene packed --object-set frida/train \
    --num-grasps "$NUM_GRASPS_TRAIN" --num-proc "$NUM_PROC" --save-scene \
    --grasps-per-scene "$GRASPS_PER_SCENE" --num-rotations "$NUM_ROTATIONS"

step "02_generate_test" \
  python3 scripts/generate_data_parallel.py "$RAW_TEST" \
    --scene packed --object-set frida/test \
    --num-grasps "$NUM_GRASPS_TEST" --num-proc "$NUM_PROC" --save-scene \
    --grasps-per-scene "$GRASPS_PER_SCENE" --num-rotations "$NUM_ROTATIONS"

BALANCE_FLAG=()
[[ "$BALANCE_TRAIN" -eq 1 ]] || BALANCE_FLAG=(--no-balance)
step "03_clean_balance_train" \
  python3 scripts/clean_balance_data.py "$RAW_TRAIN" "${BALANCE_FLAG[@]}" --seed "$SEED"

step "03_clean_balance_test" \
  python3 scripts/clean_balance_data.py "$RAW_TEST" --no-balance

step "04_construct_train" \
  python3 scripts/construct_dataset_parallel.py --num-proc "$NUM_PROC" --single-view --add-noise dex \
    "$RAW_TRAIN" "$PROC_TRAIN"

step "04_construct_test" \
  python3 scripts/construct_dataset_parallel.py --num-proc "$NUM_PROC" --single-view --add-noise dex \
    "$RAW_TEST" "$PROC_TEST"

step "05_occ_train" \
  python3 scripts/save_occ_data_parallel.py "$RAW_TRAIN" 100000 4 --num-proc "$NUM_PROC"

step "05_occ_test" \
  python3 scripts/save_occ_data_parallel.py "$RAW_TEST" 100000 4 --num-proc "$NUM_PROC"

step "06_train" \
  python3 scripts/train_giga.py --net giga \
    --dataset "$PROC_TRAIN" --dataset_raw "$RAW_TRAIN" \
    --logdir "$LOGDIR_TRAIN" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" --val-split "$VAL_SPLIT" \
    --seed "$SEED" --split "$SPLIT" --pos-weight "$POS_WEIGHT" --select-by "$SELECT_BY" --patience "$PATIENCE"

# train_giga.py saves into a timestamped subfolder of --logdir, so search
# for the newest best_vgn_*.pt instead of globbing the folder directly.
BEST_CKPT="$(find "$LOGDIR_TRAIN" -name 'best_vgn_*.pt' -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -1 | cut -d' ' -f2- || true)"
LAST_CKPT="$(find "$LOGDIR_TRAIN" -name 'vgn_*.pt' ! -name 'best_*' -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -1 | cut -d' ' -f2- || true)"
echo "best_checkpoint=$BEST_CKPT" | tee -a "$CONFIG_LOG"
echo "last_checkpoint=$LAST_CKPT" | tee -a "$CONFIG_LOG"

# Offline scoring on the unbalanced test split: seconds, no physics. Both
# checkpoints, since the validation pick is not always the better one.
for pair in "best:$BEST_CKPT" "last:$LAST_CKPT"; do
  tag="${pair%%:*}"; ckpt="${pair#*:}"
  [[ -n "$ckpt" ]] || continue
  rm -f "$DATA_ROOT/.done_06_eval_offline_${RUN_NAME}_$tag"
  step "06_eval_offline_${RUN_NAME}_$tag" \
    python3 scripts/eval_offline.py --model "$ckpt" \
      --dataset "$PROC_TEST" --dataset_raw "$RAW_TEST" \
      --target-precision "$TARGET_PRECISION" --out "$EVAL_DIR/offline_$tag.json"
done

if [[ "$RUN_TEST" -eq 1 ]]; then
  if [[ -z "$BEST_CKPT" ]]; then
    echo "no checkpoint found under $LOGDIR_TRAIN, cannot run test eval" | tee "$LOG_DIR/07_test_eval.log"
    exit 1
  fi
  step "07_test_eval" \
    python3 scripts/sim_grasp_multiple.py \
      --model "$BEST_CKPT" --type giga --scene packed --object-set frida/test \
      --num-view 1 --num-rounds 100 --sideview --add-noise dex \
      --force --best --result-path "$EVAL_DIR"
else
  cat <<EOF | tee -a "$CONFIG_LOG"

Training done. Best checkpoint: $BEST_CKPT
To run the held-out grasp-trial evaluation later:
  python3 scripts/sim_grasp_multiple.py --model "$BEST_CKPT" --type giga \\
    --scene packed --object-set frida/test --num-view 1 --num-rounds 100 \\
    --sideview --add-noise dex --force --best --result-path "$EVAL_DIR"
EOF
fi

echo "finished=$(date -Iseconds)" | tee -a "$CONFIG_LOG"
echo "All logs saved under $LOG_DIR"
