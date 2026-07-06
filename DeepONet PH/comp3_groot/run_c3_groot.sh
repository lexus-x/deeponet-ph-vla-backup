#!/usr/bin/env bash
# =============================================================================
# COMP-3: GR00T N1.6-3B  —  baseline (diffusion) vs +DeepONet vs +DeepONet+PH
# Head-only (frozen backbone) finetune. Protocol mirrors comp-1:
#   per variant: 40-task combined PRETRAIN (15K) -> per-suite FINETUNE (15K)
#                -> in-dist + LIBERO-Plus eval on Spatial/Object/Long/Goal.
# AUTO-START: guards until the comp-1 replan=5 campaign (+ any pi05 job) releases
#   the GPU, so it never contends. Launch this detached now; it fires by itself.
# CORRECTNESS SAFETY NETS:
#   * GR00T_ACTION_HEAD env var is set at PRETRAIN, FINETUNE, AND EVAL (the head is
#     rebuilt from this var at every model load; wrong/unset -> trained head silently
#     discarded). Eval server log is asserted to show 'action head = deeponet'.
#   * SELF-SMOKE GATE: tiny end-to-end (real batch, 10-step train -> server -> 1-ep
#     eval) BEFORE the multi-day run; aborts on any failure so no days are wasted.
#   * DISK-SAFE: save_total_limit 1; finetune/pretrain ckpts pruned after each eval;
#     all heavy downloads happen post-guard when the drive has freed.
# Client (sim) runs in the PROJECT venv; server (3B model) in the gr00t conda env.
# Status -> $ROOT/PROGRESS_c3.log
# =============================================================================
set -uo pipefail
ROOT="/media/user/C2FE578FFE577A9D/sota_campaign"
DRIVE="/media/user/C2FE578FFE577A9D"
REPO="/home/user/Desktop/Ayush PH test"
GR00T="/home/user/Isaac-GR00T"
CONDA="/home/user/anaconda3/bin/conda"
VENV="$REPO/venv/bin/python"          # project venv = the eval CLIENT interpreter
PIP="$REPO/venv/bin/pip"
PROG="$ROOT/PROGRESS_c3.log"
RESULTS="$ROOT/results_c3"
OUTDIR="$ROOT/outputs_c3"
DATA="$DRIVE/gr00t_libero"
BASE="$DRIVE/hf_cache/GR00T-N1.6-3B"
EVAL="$ROOT/comp3_groot/eval_groot.py"
PORT=5757
STEPS=15000; SEED=0; GBATCH=16; GACC=4
INDIST_EP=12; PLUS_EP=8; MAXSTEPS=520; HORIZON=8

export HF_HOME="$DRIVE/hf_cache"
export PYTHONNOUSERSITE=1   # ignore ~/.local (transformers 4.57.6) so gr00t env's pinned 4.51.3 loads
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NO_ALBUMENTATIONS_UPDATE=1
export TMPDIR="$ROOT/tmp"; mkdir -p "$ROOT/tmp" "$RESULTS" "$OUTDIR" "$DATA"

SUITES=(Spatial Object Long Goal)
declare -A DSMAP=( [Spatial]=libero_spatial_no_noops_1.0.0_lerobot
                   [Object]=libero_object_no_noops_1.0.0_lerobot
                   [Long]=libero_10_no_noops_1.0.0_lerobot
                   [Goal]=libero_goal_no_noops_1.0.0_lerobot )
# tag | GR00T_ACTION_HEAD | GR00T_PH | GR00T_LAMBDA_PH
VARIANTS=( "c3_groot|diffusion|0|0.0"
           "c3_groot_deeponet|deeponet|0|0.0"
           "c3_groot_deeponet_ph|deeponet_ph|1|0.02" )
ALL4="$DATA/${DSMAP[Spatial]},$DATA/${DSMAP[Object]},$DATA/${DSMAP[Long]},$DATA/${DSMAP[Goal]}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
free_gb(){ df -BG "$DRIVE" | awk 'NR==2{gsub("G","",$4);print $4}'; }
check_disk(){ local g; g="$(free_gb)"; if [ "${g:-0}" -lt "${1:-20}" ]; then log "LOW DISK: ${g}G free (< ${1:-20}G)"; return 1; fi; return 0; }
find_ckpt(){ ls -d "$1"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1; }

# ---- train (head-only, single-GPU). env vars select the head variant. ---------
gtrain(){ # $1 head  $2 ph  $3 lambda  $4 base_ckpt  $5 dataset(s)  $6 out  $7 steps
  ( cd "$GR00T" && env GR00T_ACTION_HEAD="$1" GR00T_PH="$2" GR00T_LAMBDA_PH="$3" \
      CUDA_VISIBLE_DEVICES=0 "$CONDA" run --no-capture-output -n gr00t python \
      "$GR00T/gr00t/experiment/launch_finetune.py" \
      --base_model_path "$4" --dataset_path "$5" --embodiment_tag LIBERO_PANDA \
      --num_gpus 1 --global_batch_size "$GBATCH" --gradient_accumulation_steps "$GACC" \
      --output_dir "$6" --max_steps "$7" --save_steps "$7" --save_total_limit 1 \
      --learning_rate 1e-4 --warmup_ratio 0.05 --weight_decay 1e-5 \
      --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
      --dataloader_num_workers 4 )
}

# ---- server (gr00t env) + readiness poll --------------------------------------
SRV_PID=""
start_server(){ # $1 head  $2 ckpt
  : > "$ROOT/log_c3_server.out"
  # setsid -> server runs in its OWN process group, so stop_server kills the whole tree
  # (conda-run wrapper + the 3B python), never orphaning it on the GPU / port.
  # PYTHONUNBUFFERED=1 + python -u: the "[GR00T] action head = deeponet" marker is a
  # print() to stdout, which is BLOCK-buffered when redirected to a file -> it stays
  # trapped in the buffer while the server blocks in recv(), so the head-assert grep
  # below never sees it. Unbuffered stdout flushes the marker as soon as it prints.
  setsid bash -c "env GR00T_ACTION_HEAD='$1' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 '$CONDA' run --no-capture-output -n gr00t python -u '$GR00T/gr00t/eval/run_gr00t_server.py' --model-path '$2' --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --port '$PORT'" \
      >>"$ROOT/log_c3_server.out" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 180); do   # up to ~6 min for the 3B model to load + bind
    # `timeout 12` bounds the probe: PolicyClient.ping() has NO socket timeout, so a server
    # that dies before binding would otherwise hang the poll forever (dead-coding the
    # death-check + budget below).
    if timeout 12 "$VENV" -c "import sys;sys.path.insert(0,'$GR00T');from gr00t.policy.server_client import PolicyClient;sys.exit(0 if PolicyClient(host='127.0.0.1',port=$PORT).ping() else 1)" 2>/dev/null; then
      return 0
    fi
    kill -0 "$SRV_PID" 2>/dev/null || { log "server died during load (see log_c3_server.out)"; stop_server; return 1; }
    sleep 2
  done
  log "server did not become ready in time"; stop_server; return 1
}
stop_server(){
  [ -n "$SRV_PID" ] && kill -TERM -"$SRV_PID" 2>/dev/null   # negative = whole process group
  pkill -f "run_gr00t_server.py --model-path" 2>/dev/null
  [ -n "$SRV_PID" ] && wait "$SRV_PID" 2>/dev/null
  SRV_PID=""
  sleep 3                                                    # let GPU mem + port :$PORT free
}

eval_suite(){ # $1 head  $2 ckpt  $3 tag  $4 suite
  start_server "$1" "$2" || { log "$3/$4 server FAILED"; return 1; }
  if [ "$1" != "diffusion" ]; then   # HARD assert: wrong/unset head => stock diffusion head under a deeponet label
    grep -q "action head = deeponet" "$ROOT/log_c3_server.out" \
      || { log "!!!! $3/$4 ABORT: server did NOT load the deeponet head (GR00T_ACTION_HEAD not applied) !!!!"; stop_server; return 1; }
  fi
  "$VENV" "$EVAL" --variant "$3" --suite "$4" --results "$RESULTS" \
      --host 127.0.0.1 --port "$PORT" --indist_episodes "$INDIST_EP" --plus_episodes "$PLUS_EP" \
      --max_steps "$MAXSTEPS" --horizon "$HORIZON" --only both \
      >"$ROOT/log_c3_${3}_${4}_eval.out" 2>&1
  local rc=$?
  stop_server
  { [ $rc -eq 0 ] && [ -f "$RESULTS/${3}__${4}.json" ]; } || return 1   # real result required
  return 0
}

# ---- prep: client deps + datasets --------------------------------------------
prep(){
  log "#### PREP: client deps + datasets  [free $(free_gb)G] ####"
  check_disk 30 || { log "PREP ABORT: need >=30G free for dataset downloads"; return 1; }
  "$PIP" install -q "pyzmq==27.0.1" "msgpack==1.1.0" >>"$ROOT/log_c3_prep.out" 2>&1 \
    || { log "pip pyzmq/msgpack FAILED"; return 1; }
  for name in "${DSMAP[@]}"; do
    local d="$DATA/$name"
    if [ ! -f "$d/meta/episodes.jsonl" ]; then
      log "download $name"
      "$CONDA" run --no-capture-output -n gr00t python -c \
        "from huggingface_hub import snapshot_download; snapshot_download(repo_id='IPEC-COMMUNITY/$name', repo_type='dataset', local_dir='$d')" \
        >>"$ROOT/log_c3_prep.out" 2>&1 || { log "download $name FAILED"; return 1; }
    fi
    cp -n "$GR00T/examples/LIBERO/modality.json" "$d/meta/" 2>/dev/null || true
  done
  cp -n "$GR00T/examples/LIBERO/patches/episode_000082.mp4" \
     "$DATA/${DSMAP[Goal]}/videos/chunk-000/observation.images.wrist_image/" 2>/dev/null || true
  log "PREP done  [free $(free_gb)G]"
}

# ---- self-smoke gate ----------------------------------------------------------
smoke(){
  log "#### SELF-SMOKE: tiny end-to-end (deeponet, real batch) ####"
  local sd="$OUTDIR/_smoke"; rm -rf "$sd"
  gtrain deeponet 0 0.0 "$BASE" "$DATA/${DSMAP[Spatial]}" "$sd/ft" 10 \
      >"$ROOT/log_c3_smoke_train.out" 2>&1 || { log "SMOKE train FAILED (see log_c3_smoke_train.out)"; return 1; }
  local fck; fck="$(find_ckpt "$sd/ft")"; [ -n "$fck" ] || { log "SMOKE: no ckpt produced"; return 1; }
  start_server deeponet "$fck" || { log "SMOKE server FAILED"; return 1; }
  grep -q "action head = deeponet" "$ROOT/log_c3_server.out" \
      || { log "SMOKE: server did NOT load the deeponet head!"; stop_server; return 1; }
  "$VENV" "$EVAL" --variant c3_smoke --suite Spatial --results "$RESULTS" \
      --host 127.0.0.1 --port "$PORT" --indist_episodes 1 --plus_episodes 1 \
      --max_steps 60 --horizon 8 --only both >"$ROOT/log_c3_smoke_eval.out" 2>&1
  local rc=$?; stop_server
  { [ $rc -eq 0 ] && [ -f "$RESULTS/c3_smoke__Spatial.json" ]; } \
      || { log "SMOKE eval FAILED (rc=$rc, see log_c3_smoke_eval.out)"; return 1; }
  rm -rf "$sd" "$RESULTS/c3_smoke__Spatial.json" "$RESULTS/_c3_smoke__Spatial_raw"
  log "#### SELF-SMOKE PASSED -> proceeding to full comp-3 ####"
}

run_variant(){ # $1 tag  $2 head  $3 ph  $4 lambda
  local tag="$1" head="$2" ph="$3" lam="$4" base="$OUTDIR/$1"
  check_disk || { log "$tag ABORT: low disk before pretrain"; return 1; }
  log "#### $tag : PRETRAIN(40-task) head=$head  [free $(free_gb)G] ####"
  gtrain "$head" "$ph" "$lam" "$BASE" "$ALL4" "$base/pretrain" "$STEPS" \
      >"$ROOT/log_c3_${tag}_pretrain.out" 2>&1 && log "$tag pretrain DONE" \
      || { log "$tag pretrain FAILED -> skip variant"; return 1; }
  local pck; pck="$(find_ckpt "$base/pretrain")"; [ -n "$pck" ] || { log "$tag no pretrain ckpt -> skip"; return 1; }
  for s in "${SUITES[@]}"; do
    if [ -f "$RESULTS/${tag}__${s}.json" ]; then log "$tag/$s already done -> skip"; continue; fi
    check_disk || { log "$tag skip $s: low disk"; continue; }
    log "#### $tag : FINETUNE $s  [free $(free_gb)G] ####"
    gtrain "$head" "$ph" "$lam" "$pck" "$DATA/${DSMAP[$s]}" "$base/$s" "$STEPS" \
        >"$ROOT/log_c3_${tag}_${s}.out" 2>&1 && log "$tag finetune $s DONE" \
        || { log "$tag finetune $s FAILED"; continue; }
    local fck; fck="$(find_ckpt "$base/$s")"; [ -n "$fck" ] || { log "$tag no $s ckpt"; continue; }
    log "#### $tag : EVAL $s (in-dist + LIBERO-Plus) ####"
    if eval_suite "$head" "$fck" "$tag" "$s"; then
      log "######## $tag / $s RESULT READY ########"
      rm -rf "$base/$s"                 # prune ONLY after a confirmed non-null result
    else
      log "$tag eval $s FAILED -> KEEPING ckpt $base/$s for retry (see log_c3_${tag}_${s}_eval.out)"
    fi
  done
  rm -rf "$base/pretrain"
  log "==== $tag COMPLETE  [free $(free_gb)G] ===="
}

# ============================ GUARD then RUN =================================
log "==== COMP-3 (GR00T) armed; waiting for comp-1 replan=5 / pi05 to release the GPU ===="
while pgrep -f "run_c1_replan5.sh" >/dev/null || pgrep -f "train_pi05.py" >/dev/null \
      || pgrep -f "eval_pi05.py" >/dev/null; do sleep 120; done
log "==== GPU free. Starting comp-3 (GR00T N1.6, ${STEPS} steps, seed=$SEED)  [free $(free_gb)G] ===="

prep  || { log "==== COMP-3 ABORTED at PREP ===="; exit 1; }
smoke || { log "==== COMP-3 ABORTED at self-smoke gate (pipeline not validated) ===="; exit 1; }

for v in "${VARIANTS[@]}"; do
  IFS='|' read -r tag head ph lam <<< "$v"
  run_variant "$tag" "$head" "$ph" "$lam"
done
log "==== COMP-3 ALL VARIANTS DONE ===="
"$VENV" "$ROOT/aggregate_sota.py" >>"$ROOT/aggregate.out" 2>&1 && log "comp-3 tables regenerated" || true
