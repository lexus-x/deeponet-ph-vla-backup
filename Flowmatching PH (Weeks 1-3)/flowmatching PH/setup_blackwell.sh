#!/usr/bin/env bash
# =============================================================================
# setup_blackwell.sh
# -----------------------------------------------------------------------------
# One-shot environment setup for the SmolVLA + Persistent Homology project on
# an NVIDIA Blackwell (B100/B200, compute capability sm_100) GPU.
#
# Goals:
#   * pip-only, NO sudo (creates a local venv in ./venv)
#   * Install a PyTorch build that actually has Blackwell (sm_100) kernels
#   * Install LeRobot (+ SmolVLA), gudhi, wandb, imageio-ffmpeg, and helpers
#   * Download LIBERO + the LIBERO-10 (libero_10) HDF5 demos
#   * Verify: (1) torch sees the GPU with sm_100, (2) SmolVLA loads,
#             (3) EGL headless MuJoCo rendering works
#   * Attempt official LIBERO-V; fall back to our wrapper (file 3) if absent
#
# This script is intentionally verbose and FAILS LOUDLY on the things that
# actually break on Blackwell (torch wheels, EGL). It is safe to re-run.
#
# Usage:
#   bash setup_blackwell.sh            # full setup
#   bash setup_blackwell.sh --verify   # only re-run the verification checks
# =============================================================================

set -uo pipefail   # NOTE: not -e; we want to catch & report failures ourselves

# -----------------------------------------------------------------------------
# 0. Config (override via env vars before calling, e.g. CUDA_TAG=cu126 bash ...)
# -----------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
LIBERO_DIR="${LIBERO_DIR:-$PROJECT_DIR/third_party/LIBERO}"
PY="${PY:-python3}"

# Blackwell needs CUDA 12.8 wheels (sm_100). cu128 is the correct default.
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
# If the stable cu128 channel ever lacks sm_100, set USE_NIGHTLY=1.
USE_NIGHTLY="${USE_NIGHTLY:-0}"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR" "$DATA_DIR" "$PROJECT_DIR/third_party"
SETUP_LOG="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
c_green='\033[0;32m'; c_red='\033[0;31m'; c_yellow='\033[1;33m'; c_blue='\033[0;34m'; c_off='\033[0m'
log()   { echo -e "${c_blue}[setup]${c_off} $*" | tee -a "$SETUP_LOG"; }
ok()    { echo -e "${c_green}[ ok ]${c_off} $*" | tee -a "$SETUP_LOG"; }
warn()  { echo -e "${c_yellow}[warn]${c_off} $*" | tee -a "$SETUP_LOG"; }
err()   { echo -e "${c_red}[FAIL]${c_off} $*" | tee -a "$SETUP_LOG"; }
section(){ echo -e "\n${c_blue}========== $* ==========${c_off}" | tee -a "$SETUP_LOG"; }

VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

# -----------------------------------------------------------------------------
# 1. Sanity: GPU + driver present
# -----------------------------------------------------------------------------
section "1. Host / GPU sanity"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$SETUP_LOG"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    ok "GPU detected: ${GPU_NAME}"
    case "$GPU_NAME" in
        *B100*|*B200*|*GB200*|*Blackwell*) ok "Blackwell-class GPU confirmed." ;;
        *) warn "GPU does not look Blackwell ('$GPU_NAME'). cu128/sm_100 settings still applied; adjust CUDA_TAG if needed." ;;
    esac
else
    err "nvidia-smi not found. No NVIDIA driver visible. Cannot continue meaningfully."
    err "If you are on a login node without a GPU, run this on the GPU node instead."
    # We continue so the venv/deps can still be built, but verification will fail.
fi

# -----------------------------------------------------------------------------
# 2. Python venv (pip-only, no sudo)
# -----------------------------------------------------------------------------
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    section "2. Python virtual environment"
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating venv at $VENV_DIR"
        "$PY" -m venv "$VENV_DIR" || { err "venv creation failed"; exit 1; }
    else
        ok "venv already exists at $VENV_DIR (reusing)"
    fi
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate" || { err "could not activate venv"; exit 1; }
python -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "$SETUP_LOG" >/dev/null
ok "Using python: $(which python)  ($(python --version 2>&1))"

# -----------------------------------------------------------------------------
# 3. PyTorch for Blackwell  (the #1 thing that breaks)
# -----------------------------------------------------------------------------
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    section "3. PyTorch (Blackwell / sm_100)"
    if [[ "$USE_NIGHTLY" -eq 1 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/nightly/${CUDA_TAG}"
        log "Installing NIGHTLY torch from $TORCH_INDEX"
        python -m pip install --pre torch torchvision \
            --index-url "$TORCH_INDEX" 2>&1 | tee -a "$SETUP_LOG"
    else
        log "Installing stable torch (>=2.7) from $TORCH_INDEX"
        python -m pip install "torch>=2.7" torchvision \
            --index-url "$TORCH_INDEX" 2>&1 | tee -a "$SETUP_LOG"
    fi
fi

# -----------------------------------------------------------------------------
# 4. Core ML / project dependencies
# -----------------------------------------------------------------------------
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    section "4. Project dependencies"
    # LeRobot with SmolVLA extra. If the extra name drifts, fall back to base.
    log "Installing lerobot[smolvla]"
    python -m pip install "lerobot[smolvla]" 2>&1 | tee -a "$SETUP_LOG" || {
        warn "lerobot[smolvla] failed; trying plain lerobot + transformers"
        python -m pip install lerobot transformers 2>&1 | tee -a "$SETUP_LOG"
    }

    log "Installing topology / logging / video / plotting deps"
    python -m pip install \
        gudhi \
        wandb \
        imageio imageio-ffmpeg \
        "mujoco>=3.1" \
        h5py \
        matplotlib \
        "numpy>=2.2,<2.3" \
        opencv-python-headless \
        tqdm \
        reportlab \
        numba scipy \
        2>&1 | tee -a "$SETUP_LOG"
    ok "Core deps installed (see log for any individual failures)."

    # LIBERO/robosuite runtime deps. robosuite 1.4.0's own pins (numpy 1.22 etc.)
    # would clobber the torch stack, so install --no-deps and add only the light
    # runtime imports it actually needs. numpy is held at 2.2.x for numba.
    log "Installing LIBERO sim backend (robosuite 1.4.0, bddl) without deps"
    python -m pip install --no-deps "robosuite==1.4.0" "bddl==1.0.1" 2>&1 | tee -a "$SETUP_LOG"
    python -m pip install --no-deps easydict thop cloudpickle hydra-core omegaconf \
        antlr4-python3-runtime future "gym==0.25.2" robomimic 2>&1 | tee -a "$SETUP_LOG"
    ok "LIBERO sim deps installed."
fi

# -----------------------------------------------------------------------------
# 5. LIBERO + LIBERO-10 demos
# -----------------------------------------------------------------------------
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    section "5. LIBERO benchmark + libero_10 demos"
    if [[ ! -d "$LIBERO_DIR/.git" ]]; then
        log "Cloning LIBERO into $LIBERO_DIR"
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR" \
            2>&1 | tee -a "$SETUP_LOG" || warn "LIBERO clone failed (check network/proxy)."
    else
        ok "LIBERO repo already present."
    fi
    if [[ -d "$LIBERO_DIR" ]]; then
        log "Installing LIBERO (editable, pip-only)"
        python -m pip install -e "$LIBERO_DIR" 2>&1 | tee -a "$SETUP_LOG" || \
            warn "LIBERO pip install reported errors; some sim features may be limited."

        # LIBERO's top-level package dir has no __init__.py, so the editable
        # finder maps nothing -> add the repo root to the venv path explicitly.
        SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
        echo "$LIBERO_DIR" > "$SP/libero_root.pth"
        ok "Added LIBERO root to venv path via $SP/libero_root.pth"

        # Point LIBERO's asset/bddl/init paths at THIS install (avoids stale
        # ~/.libero/config.yaml from any previous checkout).
        mkdir -p "$HOME/.libero"
        LB="$LIBERO_DIR/libero/libero"
        cat > "$HOME/.libero/config.yaml" <<EOF
benchmark_root: $LB
bddl_files: $LB/bddl_files
init_states: $LB/init_files
datasets: $DATA_DIR/libero_datasets
assets: $LB/assets
EOF
        ok "Wrote ~/.libero/config.yaml pointing at this LIBERO install"

        log "Downloading libero_10 HDF5 demos (this can be several GB)"
        # LIBERO ships a dataset downloader; libero_10 == the long-horizon suite.
        python - <<'PYDL' 2>&1 | tee -a "$SETUP_LOG"
try:
    from libero.libero import benchmark, get_libero_path
    import subprocess, sys
    # Preferred: official downloader script
    try:
        from libero.libero import download_datasets  # noqa: F401
        download_datasets.main(datasets="libero_10")  # type: ignore
        print("[libero] libero_10 downloaded via download_datasets")
    except Exception as e1:
        print(f"[libero] module downloader unavailable ({e1}); trying CLI script")
        subprocess.run([sys.executable, "-m", "libero.libero.benchmark.download_datasets",
                        "--datasets", "libero_10"], check=False)
except Exception as e:
    print(f"[libero] dataset download could not be initiated automatically: {e}")
    print("[libero] You may need to run LIBERO's benchmark/download_datasets.py manually.")
PYDL
    fi
fi

# -----------------------------------------------------------------------------
# 6. Attempt official LIBERO-V; record fallback decision
# -----------------------------------------------------------------------------
section "6. LIBERO-V (visual perturbations)"
LIBERO_V_STATUS="wrapper"
if python -c "import libero_v" >/dev/null 2>&1; then
    LIBERO_V_STATUS="official"
    ok "Official 'libero_v' package importable -> will use it."
else
    warn "No official 'libero_v' package found -> evaluation will use our libero_v_wrapper.py (file 3)."
fi
echo "LIBERO_V_BACKEND=${LIBERO_V_STATUS}" > "$PROJECT_DIR/.libero_v_backend"
log "Recorded backend choice in .libero_v_backend"

# =============================================================================
# 7. VERIFICATION (always runs)
# =============================================================================
section "7. Verification"
VERIFY_FAILS=0

# --- 7a. torch + Blackwell kernels -------------------------------------------
log "7a. torch / CUDA / sm_100 check"
python - <<'PYV' 2>&1 | tee -a "$SETUP_LOG" || VERIFY_FAILS=$((VERIFY_FAILS+1))
import sys, torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available to torch"
name = torch.cuda.get_device_name(0)
cap  = torch.cuda.get_device_capability(0)
print("device:", name, "| capability: sm_%d%d" % cap)
archs = torch.cuda.get_arch_list()
print("compiled arch list:", archs)
# Blackwell is sm_100. Warn (not hard-fail) if this wheel lacks it.
if cap[0] >= 10 and not any("sm_100" in a or "sm_90" in a for a in archs):
    print("WARNING: wheel arch list has no sm_100/sm_90 entry; kernels may fall back.")
# Actually exercise a kernel so a 'no kernel image' error surfaces NOW, not in training.
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print("bf16 matmul on GPU ok, checksum finite:", (y == y))
print("TORCH_OK")
PYV
grep -q "TORCH_OK" "$SETUP_LOG" && ok "torch sees Blackwell GPU and ran a bf16 kernel." || err "torch GPU verification failed."

# --- 7b. SmolVLA loads -------------------------------------------------------
log "7b. SmolVLA pretrained load check (lerobot/smolvla_base)"
python - <<'PYS' 2>&1 | tee -a "$SETUP_LOG" || VERIFY_FAILS=$((VERIFY_FAILS+1))
try:
    # lerobot >= 0.5 layout
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except Exception:
    # Older lerobot used the lerobot.common.* layout.
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import torch
if SmolVLAPolicy is not None:
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    n = sum(p.numel() for p in policy.parameters())
    print(f"SmolVLA loaded. total params: {n/1e6:.1f}M")
    print("has forward:", hasattr(policy, "forward"),
          "| has select_action:", hasattr(policy, "select_action"))
print("SMOLVLA_OK")
PYS
grep -q "SMOLVLA_OK" "$SETUP_LOG" && ok "SmolVLA loaded from pretrained." || err "SmolVLA load failed (check HF auth/network/lerobot version)."

# --- 7c. EGL headless MuJoCo render -----------------------------------------
log "7c. EGL headless rendering check (MUJOCO_GL=egl)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python - <<'PYE' 2>&1 | tee -a "$SETUP_LOG" || VERIFY_FAILS=$((VERIFY_FAILS+1))
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
XML = """
<mujoco>
  <visual><global offwidth='256' offheight='256'/></visual>
  <worldbody>
    <light pos='0 0 2'/>
    <geom type='box' size='.2 .2 .2' rgba='.8 .3 .3 1'/>
  </worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 256, 256)   # this allocates the EGL context
mujoco.mj_forward(m, d)
r.update_scene(d)
img = r.render()
assert img.shape == (256, 256, 3) and img.dtype == np.uint8, img.shape
assert img.sum() > 0, "rendered frame is all black (EGL likely not really working)"
print("rendered frame shape:", img.shape, "mean px:", float(img.mean()))
print("EGL_OK")
PYE
grep -q "EGL_OK" "$SETUP_LOG" && ok "EGL headless rendering works." || err "EGL render failed. Try: export MUJOCO_GL=egl ; ensure libEGL present (no-sudo: conda libglvnd or module load)."

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
section "8. Summary"
log "LIBERO-V backend : ${LIBERO_V_STATUS}"
log "venv             : ${VENV_DIR}"
log "data dir         : ${DATA_DIR}"
log "full log         : ${SETUP_LOG}"
if [[ "$VERIFY_FAILS" -eq 0 ]]; then
    ok  "ALL VERIFICATION CHECKS PASSED. Activate with:  source \"$VENV_DIR/bin/activate\""
    ok  "Next: report back and I'll send file 2 (ph_loss.py)."
    exit 0
else
    err "$VERIFY_FAILS verification check(s) FAILED. Scroll up / read $SETUP_LOG."
    err "Do NOT proceed to training until 7a/7b/7c all pass. Paste the failing block and we'll debug."
    exit 1
fi
