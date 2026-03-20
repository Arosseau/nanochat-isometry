#!/bin/bash
#SBATCH --job-name=nanochat-d12-all
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH -M hydra
#SBATCH -p hopper_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/dev/null

# All d12 isometry experiments in a single script (1×H200, single GPU).
# Runs 7 experiments sequentially:
#   1. baseline_muon   — Karpathy's default Muon+AdamW with weight decay (control)
#   2. muono           — Muon + decoupled ortho reg, no weight decay
#   3. muono_relu      — MuonO with ReLU activation scale (2.0)
#   4. muono_coupled   — MuonO coupled (auxiliary loss through Muon moments)
#   5. adamo           — AdamW everywhere + decoupled ortho reg, no weight decay
#   6. adamo_relu      — AdamO with ReLU activation scale (2.0)
#   7. adamw_baseline  — AdamW everywhere with standard weight decay (control)
#
# Usage:
#   sbatch runs/h200_d12_all.sh
#   bash runs/h200_d12_all.sh
#   SERIES_NAME=myexp sbatch runs/h200_d12_all.sh

export OMP_NUM_THREADS=1
SCRATCH_BASE="${VSC_SCRATCH}/nanochat-isometry"
export NANOCHAT_BASE_DIR="${SCRATCH_BASE}/nanochat_cache"
mkdir -p "$SCRATCH_BASE" "$NANOCHAT_BASE_DIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0

# Load secrets (WANDB_API_KEY, GITHUB_TOKEN)
source "${SCRATCH_BASE}/secrets.sh"
export WANDB_API_KEY

GITHUB_TOKEN="${GITHUB_TOKEN:?Must set GITHUB_TOKEN in secrets.sh}"
REPO_OWNER="Arosseau"
REPO_NAME="nanochat-isometry"
REPO_URL="https://oauth2:${GITHUB_TOKEN}@github.com/${REPO_OWNER}/${REPO_NAME}.git"
REPO_DIR="${SCRATCH_BASE}"

# --- Clone if missing, otherwise pull latest ---
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning ${REPO_NAME}..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repo exists — pulling latest changes..."
    cd "$REPO_DIR"
    git reset --hard HEAD
    git clean -fd
    git pull --rebase origin master
fi

cd "$REPO_DIR"

# --- Setup (uv, venv, deps, dataset, tokenizer) ---
export UV_INSTALL_DIR="${SCRATCH_BASE}/bin"
export UV_CACHE_DIR="${SCRATCH_BASE}/.uv_cache"
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR"
export PATH="${UV_INSTALL_DIR}:${HOME}/.local/bin:${PATH}"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Karpathy uses 1000 for his d12-d26 miniseries but says it "can probably be reduced, TODO".
# ~100 shards (~10GB) is our estimate for d12 alone.
python -m nanochat.dataset -n 100
TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.json"
if [ "${SKIP_TOKENIZER:-0}" = "1" ] && [ -f "$TOKENIZER_FILE" ]; then
    echo "Tokenizer already exists, skipping (SKIP_TOKENIZER=1)."
else
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
fi

# -----------------------------------------------------------------------------
# Configuration
SERIES_NAME="${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
DEPTH=12
RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_isometry_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "name,val_bpb,train_time_sec" > "$RESULTS_FILE"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

run_exp() {
    local NAME="$1"
    shift
    local TAG="${SERIES_NAME}_d12_${NAME}"
    local LOG="$RESULTS_DIR/${TAG}.log"

    log "Running: $NAME"
    START=$(date +%s)

    python -m scripts.base_train \
        --depth=$DEPTH \
        --run="${SERIES_NAME}_isometry" \
        --model-tag="${TAG}" \
        --core-metric-every=999999 \
        --sample-every=-1 \
        --save-every=-1 \
        # --ortho-init \
        "$@" \
        2>&1 | tee "$LOG"

    END=$(date +%s)
    ELAPSED=$((END - START))
    VAL_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')
    log "  $NAME: bpb=$VAL_BPB, time=${ELAPSED}s"
    echo "$NAME,$VAL_BPB,$ELAPSED" >> "$RESULTS_FILE"
}

log "============================================================"
log "${SERIES_NAME} ALL d${DEPTH} Isometry Experiments (1×H200)"
log "============================================================"

# ============================================================================
# BASELINES
# ============================================================================

# 1) Muon+AdamW baseline (Karpathy's default: weight decay, no ortho reg)
log "--- BASELINES ---"
run_exp "baseline_muon" \
    --weight-decay=0.28

# ============================================================================
# MUON + ORTHOGONAL REGULARIZATION (MuonO)
# ============================================================================
log "--- MuonO EXPERIMENTS ---"

# 2) MuonO: Muon + decoupled ortho reg, no weight decay
run_exp "muono" \
    --weight-decay=0.0 \
    --orth-reg-lambda=1e-3 \
    --orth-reg-decoupled

# 3) MuonO with ReLU activation scale (2.0) to compensate relu^2 signal loss
run_exp "muono_relu" \
    --weight-decay=0.0 \
    --orth-reg-lambda=1e-3 \
    --orth-reg-decoupled \
    --orth-reg-activation-scale=2.0

# 4) MuonO coupled (auxiliary loss, gradients flow through Muon moments)
run_exp "muono_coupled" \
    --weight-decay=0.0 \
    --orth-reg-lambda=1e-3

# ============================================================================
# ADAMW + ORTHOGONAL REGULARIZATION (AdamO)
# ============================================================================
log "--- AdamO EXPERIMENTS ---"

# 5) AdamO: AdamW everywhere + decoupled ortho reg, no weight decay
run_exp "adamo" \
    --optimizer=adamw \
    --matrix-lr=3e-4 \
    --weight-decay=0.0 \
    --orth-reg-lambda=1e-3 \
    --orth-reg-decoupled

# 6) AdamO with ReLU activation scale (2.0) to compensate relu^2 signal loss
run_exp "adamo_relu" \
    --optimizer=adamw \
    --matrix-lr=3e-4 \
    --weight-decay=0.0 \
    --orth-reg-lambda=1e-3 \
    --orth-reg-decoupled \
    --orth-reg-activation-scale=2.0

# 7) AdamW baseline (standard weight decay, no ortho reg) as control
run_exp "adamw_baseline" \
    --optimizer=adamw \
    --matrix-lr=3e-4 \
    --weight-decay=0.01

# ============================================================================
# SUMMARY
# ============================================================================
log "============================================================"
log "All d${DEPTH} experiments complete!"
log "============================================================"
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
