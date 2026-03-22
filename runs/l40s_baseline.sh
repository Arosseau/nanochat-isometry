#!/bin/bash
#SBATCH --job-name=nanochat-l40s-base
#SBATCH --time=12:00:00
#SBATCH --gres=shard:4
#SBATCH -M anansi
#SBATCH -p ada_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dev/null

# d12 Baseline experiment on 1×L40S (Ada Lovelace, full GPU via 4 shards).
# No FP8 (requires Hopper SM9.0+). BF16 auto-detected (L40S SM8.9 supports it).
#
# Usage:
#   sbatch runs/l40s_baseline.sh
#   bash runs/l40s_baseline.sh
#   SERIES_NAME=myexp sbatch runs/l40s_baseline.sh

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
    git pull --rebase origin master
fi

cd "$REPO_DIR"

export UV_INSTALL_DIR="${SCRATCH_BASE}/bin"
export UV_CACHE_DIR="${SCRATCH_BASE}/.uv_cache"
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR"
export PATH="${UV_INSTALL_DIR}:${HOME}/.local/bin:${PATH}"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
# Install FA2 for sliding window support on Ada/L40S (compiled against local CUDA)

source .venv/bin/activate
# Install FA2 for sliding window support on Ada/L40S (must be after venv activation)
pip install flash-attn --no-build-isolation

python -m nanochat.dataset -n 100
TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.json"
if [ "${SKIP_TOKENIZER:-0}" = "1" ] && [ -f "$TOKENIZER_FILE" ]; then
    echo "Tokenizer already exists, skipping (SKIP_TOKENIZER=1)."
else
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
fi

# -----------------------------------------------------------------------------
SERIES_NAME="${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
DEPTH=12
TAG="${SERIES_NAME}_l40s_baseline_muon"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_isometry_results"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${TAG}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running d${DEPTH} baseline Muon+AdamW (1×L40S, BF16)"
START=$(date +%s)

python -m scripts.base_train \
    --depth=$DEPTH \
    --run="${SERIES_NAME} baseline wd=0.2" \
    --model-tag="${TAG}" \
    --weight-decay=0.2 \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
    2>&1 | tee "$LOG"

END=$(date +%s)
ELAPSED=$((END - START))
VAL_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')
echo "[$(date '+%Y-%m-%d %H:%M:%S')] l40s_baseline_muon: bpb=$VAL_BPB, time=${ELAPSED}s"

RESULTS_FILE="$RESULTS_DIR/results.csv"
[ ! -f "$RESULTS_FILE" ] && echo "name,val_bpb,train_time_sec" > "$RESULTS_FILE"
echo "l40s_baseline_muon,$VAL_BPB,$ELAPSED" >> "$RESULTS_FILE"
