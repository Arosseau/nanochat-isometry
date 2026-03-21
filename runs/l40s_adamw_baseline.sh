#!/bin/bash
#SBATCH --job-name=nanochat-l40s-adamw
#SBATCH --time=12:00:00
#SBATCH --gres=shard:4
#SBATCH -M anansi
#SBATCH -p ada_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dev/null

# d12 AdamW-only baseline on 1×L40S — LR sweep.
# Uses AdamW for ALL parameters (including matrix params that normally use Muon).
# Betas: (0.9, 0.95) — standard for transformer AdamW (not the speedrun (0.8, 0.95)).
# Runs three LR values: 1e-3 / 3e-3 / 1e-2. Each takes ~4h, ~12h total.
#
# Usage:
#   sbatch runs/l40s_adamw_baseline.sh
#   bash runs/l40s_adamw_baseline.sh
#   SERIES_NAME=myexp sbatch runs/l40s_adamw_baseline.sh

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
TAG="${SERIES_NAME}_l40s_adamw_baseline"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_isometry_results"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${TAG}.log"

run_exp() {
    local TAG_SUFFIX="$1"
    local WANDB_NAME="$2"
    local EXTRA_ARGS="${@:3}"
    local EXP_TAG="${SERIES_NAME}_l40s_adamw_${TAG_SUFFIX}"
    local LOG="$RESULTS_DIR/${EXP_TAG}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: ${WANDB_NAME}"
    local START=$(date +%s)

    python -m scripts.base_train \
        --depth=$DEPTH \
        --optimizer=adamw \
        --adam-beta1=0.9 \
        --adam-beta2=0.95 \
        --weight-decay=0.2 \
        --run="${SERIES_NAME} ${WANDB_NAME}" \
        --model-tag="${EXP_TAG}" \
        --core-metric-every=999999 \
        --sample-every=-1 \
        --save-every=-1 \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG"

    local END=$(date +%s)
    local ELAPSED=$((END - START))
    local VAL_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${EXP_TAG}: bpb=$VAL_BPB, time=${ELAPSED}s"

    RESULTS_FILE="$RESULTS_DIR/results.csv"
    [ ! -f "$RESULTS_FILE" ] && echo "name,val_bpb,train_time_sec" > "$RESULTS_FILE"
    echo "${EXP_TAG},$VAL_BPB,$ELAPSED" >> "$RESULTS_FILE"
}

# LR sweep: 1e-4 / 3e-4 / 1e-2 with standard AdamW betas (0.9, 0.95)
# Literature suggests 1e-4 is typical for ~300M models; 3e-4 may be marginal.
run_exp "lr1e-4" "adamw lr=1e-4 β=(0.9,0.95)" --matrix-lr=1e-4
run_exp "lr3e-4" "adamw lr=3e-4 β=(0.9,0.95)" --matrix-lr=3e-4
# run_exp "lr1e-2" "adamw lr=1e-2 β=(0.9,0.95)" --matrix-lr=1e-2
