#!/bin/bash
# ============================================================
#  store_tokens.sh — Full data preparation pipeline
#
#  Goes through:
#    1. Download raw datasets from HuggingFace  (store_raw_datasets.py)
#    2. Classify time-aware vs timeless          (classify_time_aware.py)
#    3. Tokenize SFT data                        (sft_tokens.py)
#
#  Usage:
#    bash post_training/store_tokens.sh
#
#  To skip already-completed steps, each step checks if output
#  exists before running. Use --force to re-run everything.
# ============================================================

set -euo pipefail

# ── Config ──
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

echo ""
echo "============================================================"
echo "  Step 1/3: Download raw datasets from HuggingFace"
echo "============================================================"
echo ""

if [ -d "data/post_training_dataset" ] && [ -z "$FORCE_FLAG" ]; then
    # Check if all raw files exist
    RAW_COUNT=$(find data/post_training_dataset -path "*/raw/data.jsonl" -type f 2>/dev/null | wc -l)
    if [ "$RAW_COUNT" -ge 9 ]; then
        echo "⏭  Raw datasets already exist ($RAW_COUNT datasets found), skipping."
        echo "   Use --force to re-download."
    else
        python post_training/store_raw_datasets.py $FORCE_FLAG
    fi
else
    python post_training/store_raw_datasets.py $FORCE_FLAG
fi

echo ""
echo "============================================================"
echo "  Step 2/3: Classify time-aware vs timeless"
echo "============================================================"
echo ""

TIMELESS_COUNT=$(find data/post_training_dataset -path "*/classified/timeless/data.jsonl" -type f 2>/dev/null | wc -l)
if [ "$TIMELESS_COUNT" -ge 9 ] && [ -z "$FORCE_FLAG" ]; then
    echo "⏭  Classification already done ($TIMELESS_COUNT datasets classified), skipping."
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
echo "    torchrun --nproc_per_node=8 post_training/sft.py"
echo ""
