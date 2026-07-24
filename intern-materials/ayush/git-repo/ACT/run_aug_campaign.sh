#!/usr/bin/env bash
# Option B: full augmentation campaign. Retrain ALL 4 suites x 3 variants WITH image
# augmentation, then re-eval in-dist (3-seed) + LIBERO-Plus. Object is run FIRST so the
# key result (does aug lift Object-Plus off 0?) lands early. Writes to a SEPARATE dir
# (act_results_aug/); the original non-augmented results are untouched. Augmentation is
# applied identically to ACT and both DeepONet variants (training only; eval never augments).
set -u
cd "/home/user/Desktop/Ayush PH test/ACT"
source ../venv/bin/activate
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=./act_results
AUG=./act_results_aug
PROG="$AUG/AUG_PROGRESS.log"
mkdir -p "$AUG"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
guard(){ local kb; kb=$(df --output=avail . | tail -1); [ "$kb" -lt 10485760 ] && { note "ABORT <10GB free"; exit 1; } || true; }

NAMES=(Object Spatial Long Goal)
DATA=(lerobot/libero_object_image lerobot/libero_spatial_image lerobot/libero_10_image lerobot/libero_goal_image)
FLAG=(libero_object libero_spatial libero_10 libero_goal)
VARIANTS=(act act_deeponet act_deeponet_ph)
COMMON="--steps 30000 --batch 64 --ckpt_every 30000 --lr 1e-4 --lr_backbone 1e-5 \
  --ema 0.999 --epoch_steps 200 --num_workers 32 --ph_warmup 5000 --ph_trigger 0.15 \
  --lambda_ph 0.02 --ph_k 8 --seed 0 --augment"
INDIST="--indist_episodes 10 --test_seeds 3 --replan 5 --only indist"
INDIST_MS=(520 520 700 520)   # Long needs the longer horizon
PLUS_MS=(300 300 600 300)

note "waiting for clean Plus re-run to finish (CLEAN_PLUS_DONE)…"
while [ ! -f "$BASE/CLEAN_PLUS_DONE" ]; do sleep 60; done
note "==== START augmentation campaign (Option B: Object first, then Spatial/Long/Goal) ===="

for k in 0 1 2 3; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}; dir="$AUG/$nm"
  mkdir -p "$dir/runs" "$dir/logs"
  for v in "${VARIANTS[@]}"; do
    guard
    note "TRAIN+aug start $nm/$v"
    if python train_act.py --variant "$v" --dataset "$ds" --out "$dir/runs/${v}" $COMMON \
         2>&1 | tee "$dir/logs/train_${v}.log"; then note "TRAIN+aug done $nm/$v"
    else note "TRAIN+aug FAIL $nm/$v"; fi
  done
  MS=""; for v in "${VARIANTS[@]}"; do MS="$MS --model ${v}=$dir/runs/${v}/checkpoints/LATEST"; done
  guard; note "INDIST start $nm (max_steps=${INDIST_MS[$k]})"
  python evaluate_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_indist" $INDIST --max_steps ${INDIST_MS[$k]} \
    2>&1 | tee "$dir/logs/eval_indist.log" && note "INDIST done $nm"
  guard; note "PLUS start $nm (max_steps=${PLUS_MS[$k]})"
  python evaluate_plus_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_plus" --n_per_cat 12 --replan 5 --max_steps ${PLUS_MS[$k]} \
    2>&1 | tee "$dir/logs/eval_plus.log" && note "PLUS done $nm"
  note "######## AUG RESULT READY: $nm -> $dir ########"
done
note "==== AUG CAMPAIGN DONE ===="; touch "$AUG/AUG_DONE"
