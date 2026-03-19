#!/bin/bash
#SBATCH --job-name=nanochat-d24-1gpu-adamo
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH -M hydra
#SBATCH -p hopper_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# d24 AdamO experiments on a single H200.
# Same as run_h200_d24_adamo.sh but 1 GPU — no torchrun, ~8x slower wall clock.
# FP8 still enabled (H200 Hopper SM9.0 supports it).
# Runs 3 variants sequentially: adamo, adamo+relu, adamw baseline.
#
# Usage:
#   sbatch runs/h200_d24_1gpu_adamo.sh
#   bash runs/h200_d24_1gpu_adamo.sh
#   SERIES_NAME=myexp sbatch runs/h200_d24_1gpu_adamo.sh

export OMP_NUM_THREADS=1
SCRATCH_BASE="${VSC_SCRATCH}/nanochat-isometry"
export NANOCHAT_BASE_DIR="${SCRATCH_BASE}/nanochat_cache"
mkdir -p "$SCRATCH_BASE" "$NANOCHAT_BASE_DIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0

source "${SCRATCH_BASE}/secrets.sh"
export WANDB_API_KEY

GITHUB_TOKEN="${GITHUB_TOKEN:?Must set GITHUB_TOKEN in secrets.sh}"
REPO_OWNER="Arosseau"
REPO_NAME="nanochat-isometry"
REPO_URL="https://oauth2:${GITHUB_TOKEN}@github.com/${REPO_OWNER}/${REPO_NAME}.git"
REPO_DIR="${SCRATCH_BASE}"

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    cd "$REPO_DIR"
    git reset --hard HEAD
    git clean -fd
    git pull --rebase origin main
fi

cd "$REPO_DIR"

export UV_INSTALL_DIR="${SCRATCH_BASE}/bin"
export UV_CACHE_DIR="${SCRATCH_BASE}/.uv_cache"
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR"
export PATH="${UV_INSTALL_DIR}:${HOME}/.local/bin:${PATH}"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.dataset -n 170
TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.json"
if [ "${SKIP_TOKENIZER:-0}" = "1" ] && [ -f "$TOKENIZER_FILE" ]; then
    echo "Tokenizer already exists, skipping (SKIP_TOKENIZER=1)."
else
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
fi

# -----------------------------------------------------------------------------
SERIES_NAME="${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
DEPTH=24
RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_isometry_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"
[ ! -f "$RESULTS_FILE" ] && echo "name,val_bpb,train_time_sec" > "$RESULTS_FILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

run_exp() {
    local NAME="$1"
    local WANDB_NAME="$2"
    shift 2
    local TAG="${SERIES_NAME}_d24_1gpu_adamo_${NAME}"
    local LOG="$RESULTS_DIR/${TAG}.log"

    log "Running: $NAME"
    START=$(date +%s)

    python -m scripts.base_train \
        --depth=$DEPTH \
        --target-param-data-ratio=8 \
        --device-batch-size=16 \
        --fp8 \
        --run="${SERIES_NAME} ${WANDB_NAME}" \
        --model-tag="${TAG}" \
        --core-metric-every=999999 \
        --sample-every=-1 \
        --save-every=-1 \
        "$@" \
        2>&1 | tee "$LOG"

    END=$(date +%s)
    ELAPSED=$((END - START))
    VAL_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')
    log "  $NAME: bpb=$VAL_BPB, time=${ELAPSED}s"
    echo "d24_1gpu_${NAME},$VAL_BPB,$ELAPSED" >> "$RESULTS_FILE"
}

log "=================================================="
log "${SERIES_NAME} d24 AdamO Experiments (1×H200, FP8)"
log "=================================================="

run_exp "adamo" "adamo 1e-3 decoupled no-wd" \
    --optimizer=adamw \
    --matrix-lr=3e-4 \
    --weight-decay=0.0 \
    --orth-reg-lambda=1e-3 \
    --orth-reg-decoupled

run_exp "adamo_relu" "adamo 1e-3 decoupled relu no-wd" \
    --optimizer=adamw \
    --matrix-lr=3e-4 \
    --weight-decay=0.0 \
    --orth-reg-lambda=1e-3 \
    --orth-reg-decoupled \
    --orth-reg-activation-scale=2.0

run_exp "adamw_baseline" "adamw wd=0.01" \
    --optimizer=adamw \
    --matrix-lr=3e-4 \
    --weight-decay=0.01

log "=================================================="
log "d24 1×H200 AdamO experiments complete!"
log "=================================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
