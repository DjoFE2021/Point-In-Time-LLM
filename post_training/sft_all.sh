#!/bin/bash
# ============================================================
#  sft_all.sh — SFT + IFEval for multiple point-in-time LLMs
#
#  Trains SFT on each checkpoint, merges LoRA, evaluates IFEval.
#
#  Usage:
#    bash post_training/sft_all.sh \
#        /mnt/llms/fineweb_pit/checkpoints/4B/2019-12_step=32559_checkpoint.pt \
#        /mnt/llms/fineweb_pit/checkpoints/4B/2020-12_step=37442_checkpoint.pt \
#        /mnt/llms/fineweb_pit/checkpoints/4B/2021-12_step=42325_checkpoint.pt
#
#    # Background:
#    nohup bash post_training/sft_all.sh ckpt1.pt ckpt2.pt > sft_all.log 2>&1 &
#
#  Output structure:
#    model-sft/{checkpoint_name}/adapter_model.safetensors
#    model-sft/{checkpoint_name}/adapter_config.json
#    model-sft/{checkpoint_name}_merged.pt
#    ifeval_outputs/{checkpoint_name}_sft_outputs.json
#    ifeval_outputs/{checkpoint_name}_sft_summary.txt
# ============================================================

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
    echo "  bash post_training/sft_all.sh /mnt/llms/fineweb_pit/checkpoints/4B/*_checkpoint.pt"
    exit 1
fi

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
SFT_DIR="model-sft"

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
    # Extract checkpoint name (e.g. "2021-12_step=42325" from path)
    CKPT_NAME=$(basename "$CKPT" | sed 's/_checkpoint\.pt$//' | sed 's/\.pt$//')
    ADAPTER_DIR="$SFT_DIR/$CKPT_NAME"
    MERGED_PT="$SFT_DIR/${CKPT_NAME}_merged.pt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $CKPT_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Step 0: IFEval on base model (before SFT) ──────────────
    BASE_SUMMARY="ifeval_outputs/${CKPT_NAME}_base_summary.txt"
    if [ -f "$BASE_SUMMARY" ]; then
        echo "  ⏭  Base IFEval already evaluated: $BASE_SUMMARY"
    else
        echo "  📋 Evaluating IFEval on base model..."
        torchrun --nproc_per_node=1 eval/ifeval_test.py \
            --checkpoint "$CKPT" \
            --model 4B
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

    # ── Step 3: IFEval on SFT model (after) ─────────────────────
    SFT_SUMMARY="ifeval_outputs/${CKPT_NAME}_sft_summary.txt"
    if [ -f "$SFT_SUMMARY" ]; then
        echo "  ⏭  SFT IFEval already evaluated: $SFT_SUMMARY"
    else
        echo "  📋 Evaluating IFEval on SFT model..."
        torchrun --nproc_per_node=1 eval/ifeval_test.py \
            --checkpoint "$MERGED_PT" \
            --model 4B
    fi

    echo "  ✅ Done: $CKPT_NAME"
done

echo ""
echo "============================================================"
echo "  ✅ All SFT + IFEval evaluations complete! $(date)"
echo "============================================================"
echo ""
printf "  %-30s %10s %10s\n" "Checkpoint" "Base" "SFT"
printf "  %-30s %10s %10s\n" "──────────────────────────────" "──────────" "──────────"
for CKPT in "$@"; do
    CKPT_NAME=$(basename "$CKPT" | sed 's/_checkpoint\.pt$//' | sed 's/\.pt$//')
    BASE_SUMMARY="ifeval_outputs/${CKPT_NAME}_base_summary.txt"
    SFT_SUMMARY="ifeval_outputs/${CKPT_NAME}_sft_summary.txt"
    BASE_SCORE="--"
    SFT_SCORE="--"
    if [ -f "$BASE_SUMMARY" ]; then
        BASE_SCORE=$(grep "Prompt-Level Strict" "$BASE_SUMMARY" | awk '{print $NF}')
    fi
    if [ -f "$SFT_SUMMARY" ]; then
        SFT_SCORE=$(grep "Prompt-Level Strict" "$SFT_SUMMARY" | awk '{print $NF}')
    fi
    printf "  %-30s %10s %10s\n" "$CKPT_NAME" "$BASE_SCORE" "$SFT_SCORE"
done
echo ""
