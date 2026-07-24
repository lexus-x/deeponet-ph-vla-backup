#!/usr/bin/env bash
# ============================================================================
# Train + eval DeepONet (m3) and DeepONet+PH (m4) on LIBERO Spatial / Object / Long
# with the SAME recipe the flow baseline used (30K steps, batch 48).
#
#   m3 = --head deeponet --variant baseline
#   m4 = --head deeponet --variant ph --lambda_ph 0.02 --ph_k 8
#
# 6 training runs total (2 models x 3 suites). Each suite then evaluates both
# models in-distribution (20 episodes/task), makes per-task plots, and saves
# results BOTH ways: full all-task average AND average excluding the lowest task.
# Resilient (one failure doesn't stop the rest), crash-safe (intermediate ckpts).
#
# RUN (detached) from inside this folder, with the python env already active:
#   export MUJOCO_GL=egl
#   setsid nohup bash run_deeponet_m3m4.sh > run.out 2>&1 < /dev/null &
#
# Results -> ./deeponet_results/{Spatial,Object,Long}/...  Progress -> PROGRESS.log
# ============================================================================
set -o pipefail
cd "$(dirname "$0")"

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
# DeepONet head hyperparameters (used by evaluate.py); match the v2 design + training defaults
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet

# sanity: env must have lerobot + libero
python -c "import lerobot, libero, torch; print('env OK, cuda=', torch.cuda.is_available())" \
  || { echo "ENV NOT READY: activate the venv with lerobot+libero installed first"; exit 1; }

BASE=./deeponet_results
mkdir -p "$BASE"
PROG="$BASE/PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }

# ---- SAME recipe as the flow runs (30K = stage1 1650 + stage2 28350, batch 48) ----
COMMON="--seed 0 --stage1_steps 1650 --stage2_steps 28350 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 --ckpt_every 10000 \
  --epoch_steps 200 --num_workers 8 \
  --deeponet_p 256 --deeponet_blocks 3 --deeponet_queries 8 --deeponet_fourier 16"

prune(){ local pd=$1; local l; l=$(cat "$pd/checkpoints/LATEST.txt" 2>/dev/null) || return
  for c in "$pd"/checkpoints/*/; do [ "$(basename "$c")" != "$l" ] && rm -rf "$c" || true; done; }
guard(){ local kb; kb=$(df --output=avail . | tail -1)
  if [ "$kb" -lt 10485760 ]; then note "ABORT: <10GB free"; exit 1; fi; }

NAMES=(Spatial Object Long)
DATA=(lerobot/libero_spatial_image lerobot/libero_object_image lerobot/libero_10_image)
FLAG=(libero_spatial libero_object libero_10)
# model | head | variant | extra-args
MODELS=("m3|deeponet|baseline|" "m4|deeponet|ph|--lambda_ph 0.02 --ph_k 8")

note "==== DeepONet m3 + m4 campaign START (30K, batch 48; Spatial/Object/Long) ===="
for k in 0 1 2; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}
  dir="$BASE/$nm"; mkdir -p "$dir/runs" "$dir/logs"

  # ---- train both models ----
  for mdef in "${MODELS[@]}"; do
    IFS='|' read -r mn mh mv mex <<< "$mdef"
    guard
    note "TRAIN start  $nm/$mn  ($mh/$mv) 30K"
    if python train.py --head "$mh" --variant "$mv" $mex $COMMON \
         --dataset "$ds" --out "$dir/runs/${mn}_s0" 2>&1 | tee "$dir/logs/train_${mn}.log"; then
      prune "$dir/runs/${mn}_s0"; note "TRAIN done   $nm/$mn"
    else
      note "TRAIN FAILED $nm/$mn — skipping its eval"
    fi
  done

  # ---- eval each model (in-distribution, 20 ep/task) + plots + both-way summaries ----
  for mn in m3 m4; do
    ck="$dir/runs/${mn}_s0/checkpoints/LATEST"
    [ -f "$dir/runs/${mn}_s0/checkpoints/LATEST.txt" ] || { note "no ckpt for $nm/$mn — skip eval"; continue; }
    guard
    note "EVAL start   $nm/$mn -> eval_${mn}"
    if python evaluate.py --suite "$fl" --dataset "$ds" \
         --model "${mn}=deeponet=${ck}" --only indist --indist_episodes 20 --replan 5 \
         --out "$dir/runs/eval_${mn}" 2>&1 | tee "$dir/logs/eval_${mn}.log"; then
      note "EVAL done    $nm/$mn"
      python summarize_suite.py --suite_dir "$dir" --suite "$fl" --model "$mn" \
        --eval_subdir "eval_${mn}" --drop auto 2>&1 | tee "$dir/logs/summary_${mn}.log" || note "summary failed $nm/$mn"
      python plot_10_and_9.py --suite_dir "$dir" --suite "$fl" --model "$mn" \
        --eval_subdir "eval_${mn}" --drop auto --label "$nm $mn 30K" 2>&1 | tee -a "$dir/logs/summary_${mn}.log" || note "plot failed $nm/$mn"
      note "######## RESULT READY: $nm/$mn -> $dir/runs/eval_${mn} (10-task + 9-task plots & summaries) ########"
    else
      note "EVAL FAILED  $nm/$mn"
    fi
  done
done
note "==== DeepONet m3 + m4 campaign DONE ===="
touch "$BASE/ALL_DONE"
