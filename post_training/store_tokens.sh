#!/bin/bash
# SFT data prep: download datasets, classify time-aware vs timeless, tokenize timeless data.
# Usage: bash post_training/store_tokens.sh [--force]   (classification needs OPENAI_API_KEY)
# Steps whose outputs already exist are skipped unless --force is given.

set -euo pipefail

# Run from the repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Activated venv"
else
    echo "⚠️  No venv found at venv/bin/activate, using system Python"
fi

FORCE="${1:-}"
FORCE_FLAG=""
if [ "$FORCE" = "--force" ]; then
    FORCE_FLAG="--force"
    echo "🔄 Force mode: re-running all steps"
fi

# Datasets downloaded by store_raw_datasets.py and classified by classify_time_aware.py (must match their lists)
DATA_DIR="data/post_training_dataset"
DATASETS=(
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated"
    "ai2-adapt-dev/personahub_code_v2_34999"
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k"
    "ai2-adapt-dev/numinamath_tir_math_decontaminated"
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"
    "argilla/ifeval-like-data"
)

# True if $DATA_DIR/<dataset>/$1 exists and is non-empty for every dataset
all_exist() {
    local name
    for name in "${DATASETS[@]}"; do
        [ -s "$DATA_DIR/$name/$1" ] || return 1
    done
}

echo ""
echo "============================================================"
echo "  Step 1/3: Download raw datasets from HuggingFace"
echo "============================================================"
echo ""

if [ -z "$FORCE_FLAG" ] && all_exist "raw/data.jsonl"; then
    echo "⏭  Raw datasets already exist (${#DATASETS[@]} datasets found), skipping."
    echo "   Use --force to re-download."
else
    python post_training/store_raw_datasets.py $FORCE_FLAG
fi

echo ""
echo "============================================================"
echo "  Step 2/3: Classify time-aware vs timeless"
echo "============================================================"
echo ""

if [ -z "$FORCE_FLAG" ] && all_exist "classified/timeless/data.jsonl"; then
    echo "⏭  Classification already done (${#DATASETS[@]} datasets classified), skipping."
    echo "   Use --force to re-classify."
else
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "⚠️  OPENAI_API_KEY not set. Classification requires the OpenAI API."
        echo "   Set it with: export OPENAI_API_KEY=sk-..."
        echo "   Skipping classification step."
    else
        python post_training/classify_time_aware.py
    fi
fi

echo ""
echo "============================================================"
echo "  Step 3/3: Tokenize SFT data (code + math + IF)"
echo "============================================================"
echo ""

SFT_OUTPUT="data/sft_tokenized"
if [ -d "$SFT_OUTPUT" ] && [ -z "$FORCE_FLAG" ]; then
    echo "⏭  SFT tokens already exist at $SFT_OUTPUT, skipping."
else
    python post_training/sft_tokens.py \
        --output-dir "$SFT_OUTPUT" \
        --max-length 2048
fi



echo ""
echo "============================================================"
echo "  ✅ All token preparation complete!"
echo "============================================================"
echo ""
echo "  Outputs:"
echo "    SFT  → $SFT_OUTPUT"
echo ""
echo "  Next: run training with:"
echo "    CHECKPOINT_PATH=<CKPT.pt> torchrun --nproc_per_node=8 post_training/sft.py"
echo "    (or SFT + merge + IFEval per checkpoint: bash post_training/sft_all.sh <CKPT.pt> ...)"
echo ""
