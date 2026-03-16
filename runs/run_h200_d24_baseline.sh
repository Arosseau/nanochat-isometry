#!/bin/bash
#SBATCH --job-name=nanochat-d24-base
#SBATCH --time=12:00:00
#SBATCH --gpus=8
#SBATCH -M hydra
#SBATCH -p hopper_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

# d24 Baseline experiment: Karpathy's default Muon+AdamW with weight decay.
# Uses 8×H200 GPUs with torchrun, FP8 training, and ~170 data shards.
# Following speedrun.sh best practices for d24.
#
# Usage:
#   sbatch runs/run_h200_d24_baseline.sh
#   bash runs/run_h200_d24_baseline.sh
#   SERIES_NAME=myexp sbatch runs/run_h200_d24_baseline.sh

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
REPO_DIR="${SCRATCH_BASE}/${REPO_NAME}"

# --- Clone if missing, otherwise pull latest ---
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning ${REPO_NAME}..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repo exists — pulling latest changes..."
    cd "$REPO_DIR"
    git reset --hard HEAD
    git clean -fd
    git pull --rebase origin main
fi

cd "$REPO_DIR"

# --- Setup (uv, venv, deps, dataset, tokenizer) ---
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# d24 needs more data shards than d12 (~170 shards, following speedrun.sh)
python -m nanochat.dataset -n 170
TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.json"
if [ "${SKIP_TOKENIZER:-0}" = "1" ] && [ -f "$TOKENIZER_FILE" ]; then
    echo "Tokenizer already exists, skipping (SKIP_TOKENIZER=1)."
else
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
fi

# -----------------------------------------------------------------------------
# Configuration
SERIES_NAME="${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
DEPTH=24
TAG="${SERIES_NAME}_d24_baseline_muon"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_isometry_results"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${TAG}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running d24 baseline Muon+AdamW (8×H200, FP8)"
START=$(date +%s)

# d24: 8 GPUs, FP8, slightly undertrained (ratio=9.5) following speedrun.sh
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=$DEPTH \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run="${SERIES_NAME}_isometry" \
    --model-tag="${TAG}" \
    --weight-decay=0.28 \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
    2>&1 | tee "$LOG"

END=$(date +%s)
ELAPSED=$((END - START))
VAL_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')

echo "[$(date '+%Y-%m-%d %H:%M:%S')] d24_baseline_muon: bpb=$VAL_BPB, time=${ELAPSED}s"

# Save results
RESULTS_FILE="$RESULTS_DIR/results.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "name,val_bpb,train_time_sec" > "$RESULTS_FILE"
fi
echo "d24_baseline_muon,$VAL_BPB,$ELAPSED" >> "$RESULTS_FILE"
