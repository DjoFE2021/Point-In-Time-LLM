#!/bin/bash
# Per 4B checkpoint: IFEval (base), LoRA SFT, merge, IFEval (merged), then a summary table.
# Usage: bash post_training/sft_all.sh <checkpoint1.pt> [checkpoint2.pt ...]
#   (in background: nohup bash post_training/sft_all.sh ckpt1.pt ckpt2.pt > sft_all.log 2>&1 &)
# Outputs: model-sft/<name>/ (LoRA adapter), model-sft/<name>_merged.pt,
#   ifeval_outputs_new/<name>_summary.txt (base IFEval) and <name>_merged_summary.txt (SFT IFEval).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ── Activate venv ──
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Activated venv"
fi

# ── Validate args ──
if [ $# -eq 0 ]; then
    echo "Usage: bash post_training/sft_all.sh <checkpoint1.pt> [checkpoint2.pt] ..."
    echo "Example:"
    echo "  bash post_training/sft_all.sh <CHECKPOINT_DIR>/*_checkpoint.pt"
    exit 1
fi

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
SFT_DIR="model-sft"
IFEVAL_DIR="ifeval_outputs_new"  # passed to eval/ifeval_test.py --output-dir

# Checkpoint name as derived by sft.py and eval/ifeval_test.py (e.g. "2021-12_step=42325")
ckpt_name() {
    basename "$1" | sed -e 's/_checkpoint\.pt//g' -e 's/\.pt//g'
}

# Prompt-level strict accuracy from an ifeval_test.py summary file ("--" if the file or line is missing)
ifeval_score() {
    local score=""
    if [ -f "$1" ]; then
        score=$(awk '$1 == "prompt_level_strict_acc:" {print $2}' "$1")
    fi
    echo "${score:---}"
}

echo ""
echo "============================================================"
echo "  SFT + IFEval Pipeline"
echo "  $(date)"
echo "============================================================"
echo "  Checkpoints:  $#"
echo "  GPUs:         $NUM_GPUS"
echo "  Output:       $SFT_DIR/{checkpoint_name}/"
echo "============================================================"
echo ""

for CKPT in "$@"; do
    CKPT_NAME=$(ckpt_name "$CKPT")
    ADAPTER_DIR="$SFT_DIR/$CKPT_NAME"
    MERGED_PT="$SFT_DIR/${CKPT_NAME}_merged.pt"
    # Written by eval/ifeval_test.py, which names its files after the checkpoint file
    BASE_SUMMARY="$IFEVAL_DIR/${CKPT_NAME}_summary.txt"
    SFT_SUMMARY="$IFEVAL_DIR/${CKPT_NAME}_merged_summary.txt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $CKPT_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Step 0: IFEval on base model (before SFT) ──────────────
    if [ -f "$BASE_SUMMARY" ]; then
        echo "  ⏭  Base IFEval already evaluated: $BASE_SUMMARY"
    else
        echo "  📋 Evaluating IFEval on base model..."
        torchrun --nproc_per_node=1 eval/ifeval_test.py \
            --checkpoint "$CKPT" \
            --model 4B \
            --output-dir "$IFEVAL_DIR"
    fi

    # ── Step 1: SFT training ────────────────────────────────────
    if [ -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
        echo "  ⏭  SFT adapter already exists: $ADAPTER_DIR"
    else
        echo "  🔧 Training SFT..."
        CHECKPOINT_PATH="$CKPT" \
        OUTPUT_DIR="$SFT_DIR" \
            torchrun --nproc_per_node="$NUM_GPUS" post_training/sft.py
    fi

    # ── Step 2: Merge LoRA ──────────────────────────────────────
    if [ -f "$MERGED_PT" ]; then
        echo "  ⏭  Merged checkpoint already exists: $MERGED_PT"
    else
        echo "  🔗 Merging LoRA adapter..."
        python post_training/merge_lora.py \
            --adapter "$ADAPTER_DIR" \
            --base-checkpoint "$CKPT" \
            --model 4B \
            --output "$MERGED_PT"
    fi

    # ── Step 3: IFEval on merged SFT model ──────────────────────
    if [ -f "$SFT_SUMMARY" ]; then
        echo "  ⏭  SFT IFEval already evaluated: $SFT_SUMMARY"
    else
        echo "  📋 Evaluating IFEval on SFT model..."
        torchrun --nproc_per_node=1 eval/ifeval_test.py \
            --checkpoint "$MERGED_PT" \
            --model 4B \
            --output-dir "$IFEVAL_DIR"
    fi

    echo "  ✅ Done: $CKPT_NAME"
done

echo ""
echo "============================================================"
echo "  ✅ All SFT + IFEval evaluations complete! $(date)"
echo "============================================================"
echo ""
# Summary table: IFEval prompt-level strict accuracy, base vs SFT
printf "  %-30s %10s %10s\n" "Checkpoint" "Base" "SFT"
printf "  %-30s %10s %10s\n" "──────────────────────────────" "──────────" "──────────"
for CKPT in "$@"; do
    CKPT_NAME=$(ckpt_name "$CKPT")
    BASE_SCORE=$(ifeval_score "$IFEVAL_DIR/${CKPT_NAME}_summary.txt")
    SFT_SCORE=$(ifeval_score "$IFEVAL_DIR/${CKPT_NAME}_merged_summary.txt")
    printf "  %-30s %10s %10s\n" "$CKPT_NAME" "$BASE_SCORE" "$SFT_SCORE"
done
echo ""
