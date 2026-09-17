#!/usr/bin/env bash
# Full GIGA pipeline, chained: generate raw data (train+test) -> clean/balance
# -> construct dataset (TSDF+views, single-view, dex noise) -> occupancy data
# -> train. Training only ever uses the train split; the held-out grasp-trial
# eval on frida/test (sim_grasp_multiple.py) is opt-in via --run-test, since
# it's a separate decision from training and can be run later on its own.
#
# Run from inside the giga-train-cpu container (any cwd -- the script cd's
# into the giga package itself). Every step's stdout/stderr and a run config
# are saved under $GIGA_DATA_ROOT/logs/<run-name>/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIGA_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$GIGA_ROOT"

# setup_giga.sh (installs vgn + builds ConvONets extensions) normally only
# runs via .bashrc on an interactive shell -- if this script is invoked any
# other way (non-interactive docker exec, cron, etc.) that never happens,
# so make sure it's run here regardless of how we were launched.
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
BATCH_SIZE, LR, VAL_SPLIT, GIGA_DATA_ROOT.
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
mkdir -p "$LOG_DIR" "$LOGDIR_TRAIN" "$EVAL_DIR"
[[ "$FORCE_REDO" -eq 1 ]] && rm -f "$DATA_ROOT"/.done_*

CONFIG_LOG="$LOG_DIR/config.txt"
{
  echo "run_name=$RUN_NAME"
  echo "started=$(date -Iseconds)"
  echo "num_proc=$NUM_PROC num_grasps_train=$NUM_GRASPS_TRAIN num_grasps_test=$NUM_GRASPS_TEST"
  echo "epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR val_split=$VAL_SPLIT"
  echo "run_test=$RUN_TEST force=$FORCE_REDO"
} | tee "$CONFIG_LOG"

# Runs "$@", tees output to $LOG_DIR/<name>.log, and stops the pipeline on
# failure.
#
# Resuming is driven by a marker written only after a step exits 0, not by
# whether its output directory has anything in it. A step that dies partway
# still leaves that directory populated -- construct_dataset_parallel.py
# creates scenes/ and point_clouds/ before processing a single scene -- so
# an "is the output there?" check silently accepts a half-written dataset
# and every later step builds on it.
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
  # `|| rc=$?` keeps `set -e` from killing the script before the marker
  # logic runs; `pipefail` (set at the top) is what makes rc the command's
  # own status rather than tee's, which is always 0.
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

# generate_data_parallel.py loops `range(num_grasps // num_proc // 120)`, so
# anything under 120 grasps per worker silently produces zero scenes and the
# whole pipeline then "succeeds" on an empty dataset.
for pair in "TRAIN:$NUM_GRASPS_TRAIN" "TEST:$NUM_GRASPS_TEST"; do
  split="${pair%%:*}"; n="${pair##*:}"
  if (( n / NUM_PROC < 120 )); then
    echo "NUM_GRASPS_$split=$n with NUM_PROC=$NUM_PROC gives $((n / NUM_PROC)) grasps/worker," >&2
    echo "under the 120 grasps-per-scene floor -- that generates 0 scenes. Raise it to at least $((120 * NUM_PROC))." >&2
    exit 1
  fi
done

step "01_generate_train" \
  python3 scripts/generate_data_parallel.py "$RAW_TRAIN" \
    --scene packed --object-set frida/train \
    --num-grasps "$NUM_GRASPS_TRAIN" --num-proc "$NUM_PROC" --save-scene

step "02_generate_test" \
  python3 scripts/generate_data_parallel.py "$RAW_TEST" \
    --scene packed --object-set frida/test \
    --num-grasps "$NUM_GRASPS_TEST" --num-proc "$NUM_PROC" --save-scene

step "03_clean_balance_train" \
  python3 scripts/clean_balance_data.py "$RAW_TRAIN"

step "03_clean_balance_test" \
  python3 scripts/clean_balance_data.py "$RAW_TEST"

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
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" --val-split "$VAL_SPLIT"

# train_giga.py does not write into --logdir directly: it creates a
# timestamped, hyperparameter-named subdirectory under it and saves there, so
# a plain "$LOGDIR_TRAIN"/best_vgn_*.pt glob never matches and the pipeline
# reported an empty checkpoint after a training run that had worked.
BEST_CKPT="$(find "$LOGDIR_TRAIN" -name 'best_vgn_*.pt' -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -1 | cut -d' ' -f2- || true)"
echo "best_checkpoint=$BEST_CKPT" | tee -a "$CONFIG_LOG"

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

# The container runs as root (setup_giga.sh needs to write to dist-packages),
# so everything it just wrote into the bind-mounted data root is root-owned
# and the host user cannot delete their own datasets. Hand it back to
# whoever owns the mount point.
if [[ "$(id -u)" -eq 0 ]]; then
  owner="$(stat -c '%u:%g' "$DATA_ROOT")"
  [[ "$owner" != "0:0" ]] && chown -R "$owner" "$DATA_ROOT" || true
fi

# The container runs as root (setup_giga.sh needs to write to dist-packages),
# so everything just written into the bind-mounted data root is root-owned and
# the host user cannot delete their own datasets. Hand it back.
if [[ "$(id -u)" -eq 0 ]]; then
  owner="$(stat -c '%u:%g' "$DATA_ROOT")"
  [[ "$owner" != "0:0" ]] && chown -R "$owner" "$DATA_ROOT" || true
fi

echo "finished=$(date -Iseconds)" | tee -a "$CONFIG_LOG"
echo "All logs saved under $LOG_DIR"
