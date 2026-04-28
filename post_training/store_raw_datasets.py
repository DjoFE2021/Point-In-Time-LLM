"""
Download all HuggingFace datasets used in training.sh and save as JSONL.

Saves each dataset to:
    data/post_training_dataset/{hf_path}/raw/data.jsonl

Datasets:
  SFT subsets (from allenai/tulu-3-sft-mixture):
    - ai2-adapt-dev/evol_codealpaca_heval_decontaminated
    - ai2-adapt-dev/personahub_code_v2_34999
    - ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k
    - ai2-adapt-dev/numinamath_tir_math_decontaminated
    - ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980
  Standalone:
    - argilla/ifeval-like-data
    - allenai/llama-3.1-tulu-3-8b-preference-mixture
    - openai/gsm8k

Usage:
    # Download all datasets:
    python post_training/parse_timeless_prompt.py

    # Download specific dataset(s) only:
    python post_training/parse_timeless_prompt.py --only openai/gsm8k argilla/ifeval-like-data

    # Force re-download even if file exists:
    python post_training/parse_timeless_prompt.py --force
"""

import argparse
import json
import os
import re
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer


# ── Dataset registry ─────────────────────────────────────────────────
# Each entry: (output_path, parent_dataset, hf_config, source_filter)
#   - output_path:    key under data/post_training_dataset/
#   - parent_dataset: HuggingFace dataset ID to load
#   - hf_config:      config/subset name passed to load_dataset (or None)
#   - source_filter:  value of "source" column to filter on (or None = keep all)

TULU_SFT_MIXTURE = "allenai/tulu-3-sft-mixture"

SFT_SUBSETS = [
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "ai2-adapt-dev/personahub_code_v2_34999",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
]

STANDALONE_DATASETS = [
    # (output_path, hf_dataset_id, hf_config)
    ("argilla/ifeval-like-data", "argilla/ifeval-like-data", None),
    ("allenai/llama-3.1-tulu-3-8b-preference-mixture",
     "allenai/llama-3.1-tulu-3-8b-preference-mixture", None),
    ("openai/gsm8k", "openai/gsm8k", "main"),
]

BASE_DIR = "./data/post_training_dataset"


# ── Non-English detection ─────────────────────────────────────────────

_NON_LATIN_RE = re.compile(
    r'[\u4e00-\u9fff'          # CJK Unified Ideographs (Chinese)
    r'\u3400-\u4dbf'           # CJK Extension A
    r'\uf900-\ufaff'           # CJK Compatibility Ideographs
    r'\u3000-\u303f'           # CJK Symbols and Punctuation
    r'\u3040-\u309f'           # Hiragana (Japanese)
    r'\u30a0-\u30ff'           # Katakana (Japanese)
    r'\uac00-\ud7af'           # Hangul Syllables (Korean)
    r'\u0600-\u06ff'           # Arabic
    r'\u0750-\u077f'           # Arabic Supplement
    r'\u0590-\u05ff'           # Hebrew
    r'\u0e00-\u0e7f'           # Thai
    r'\u0900-\u097f'           # Devanagari (Hindi, Sanskrit, etc.)
    r'\u0980-\u09ff'           # Bengali
    r'\u0a80-\u0aff'           # Gujarati
    r'\u0b80-\u0bff'           # Tamil
    r'\u0c00-\u0c7f'           # Telugu
    r'\u0c80-\u0cff'           # Kannada
    r'\u0d00-\u0d7f'           # Malayalam
    r'\u10a0-\u10ff'           # Georgian
    r'\u0530-\u058f'           # Armenian
    r'\u1200-\u137f'           # Ethiopic
    r'\u1780-\u17ff'           # Khmer
    r'\u1000-\u109f'           # Myanmar
    r']'
)

_CYRILLIC_RE = re.compile(r'[\u0400-\u04ff]')
CYRILLIC_THRESHOLD = 0.05  # >5% of alpha chars = non-English


def is_non_english(text: str) -> bool:
    """Return True if text contains significant non-English script."""
    if _NON_LATIN_RE.search(text):
        return True
    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    if cyrillic_count > 0:
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count > 0 and (cyrillic_count / alpha_count) > CYRILLIC_THRESHOLD:
            return True
    return False


def extract_text(row: dict) -> str:
    """Extract all text from a row (handles all dataset formats)."""
    parts = []
    # SFT: messages list
    if "messages" in row and isinstance(row["messages"], list):
        for msg in row["messages"]:
            if isinstance(msg, dict):
                parts.append(msg.get("content", ""))
    # DPO: chosen/rejected
    for key in ("chosen", "rejected"):
        if key in row and isinstance(row[key], list):
            for msg in row[key]:
                if isinstance(msg, dict):
                    parts.append(msg.get("content", ""))
    # argilla: instruction + response
    for key in ("instruction", "response", "question", "answer", "prompt", "completion"):
        if key in row:
            parts.append(str(row[key]))
    return " ".join(parts)


# ── Token counting ────────────────────────────────────────────────────

def count_tokens(row: dict, tokenizer) -> int:
    """Count total tokens for a row using the GPT-2 tokenizer.

    Format-aware:
      - SFT (messages):    sum of all message contents
      - DPO (chosen/rejected): max(chosen, rejected) since training uses
                               the longer one
      - argilla / GSM8K:   instruction + response / question + answer
    """
    parts = []
    if "messages" in row and isinstance(row["messages"], list):
        for msg in row["messages"]:
            if isinstance(msg, dict):
                parts.append(msg.get("content", ""))
    elif "chosen" in row and isinstance(row["chosen"], list):
        # DPO: count the longer side
        chosen_parts = [m.get("content", "") for m in row["chosen"] if isinstance(m, dict)]
        rejected_parts = [m.get("content", "") for m in row.get("rejected", []) if isinstance(m, dict)]
        chosen_len = len(tokenizer.encode(" ".join(chosen_parts)))
        rejected_len = len(tokenizer.encode(" ".join(rejected_parts)))
        return max(chosen_len, rejected_len)
    else:
        for key in ("instruction", "response", "question", "answer", "prompt", "completion"):
            if key in row:
                parts.append(str(row[key]))
    return len(tokenizer.encode(" ".join(parts)))


# ── Helpers ───────────────────────────────────────────────────────────

def save_jsonl(dataset, path, tokenizer=None, max_length=None):
    """Save a HuggingFace Dataset to a JSONL file, filtering non-English
    and over-length rows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = 0
    kept = 0
    removed_lang = 0
    removed_len = 0
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset:
            total += 1
            text = extract_text(example)
            if is_non_english(text):
                removed_lang += 1
                continue
            if tokenizer:
                n_tokens = count_tokens(example, tokenizer)
                if max_length and n_tokens > max_length:
                    removed_len += 1
                    continue
                example["number_of_tokens"] = n_tokens
            kept += 1
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    if removed_lang > 0:
        pct = removed_lang / total * 100
        print(f"   🌐 Removed {removed_lang:,d} non-English rows ({pct:.1f}%)")
    if removed_len > 0:
        pct = removed_len / total * 100
        print(f"   📏 Removed {removed_len:,d} over-length rows ({pct:.1f}%) [>{max_length} tokens]")
    return kept


def output_path(name):
    """Return the JSONL output path for a dataset name."""
    return os.path.join(BASE_DIR, name, "raw", "data.jsonl")


def should_skip(path, force):
    """Return True if the file exists and is non-empty (and not --force)."""
    if force:
        return False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    return False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download all training datasets to JSONL"
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Download only these dataset(s), by name "
             "(e.g. openai/gsm8k ai2-adapt-dev/personahub_code_v2_34999)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the JSONL file already exists",
    )
    parser.add_argument(
        "--num-proc", type=int, default=8,
        help="Number of processes for filtering (default: 8)",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Remove rows with more than N tokens (GPT-2 tokenizer). "
             "E.g. --max-length 2048",
    )
    args = parser.parse_args()

    # Load tokenizer once if max-length filtering is requested
    tokenizer = None
    if args.max_length:
        print(f"📏 Token length filter: max {args.max_length} tokens (gpt2 tokenizer)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # Build the set of dataset names we actually want to process
    all_names = set(SFT_SUBSETS) | {s[0] for s in STANDALONE_DATASETS}
    if args.only:
        requested = set(args.only)
        unknown = requested - all_names
        if unknown:
            print(f"⚠️  Unknown dataset(s): {unknown}")
            print(f"   Known: {sorted(all_names)}")
            return
        targets = requested
    else:
        targets = all_names

    # ── SFT subsets (load parent once, filter into each subset) ───────
    sft_targets = [s for s in SFT_SUBSETS if s in targets]
    if sft_targets:
        # Check which ones actually need downloading
        sft_todo = [s for s in sft_targets
                    if not should_skip(output_path(s), args.force)]
        sft_skip = [s for s in sft_targets if s not in sft_todo]

        for s in sft_skip:
            print(f"⏭  {s}  (already exists, skipping)")

        if sft_todo:
            print(f"\n{'═'*60}")
            print(f"  Loading {TULU_SFT_MIXTURE} ...")
            print(f"{'═'*60}")
            sft_ds = load_dataset(TULU_SFT_MIXTURE, split="train")
            print(f"  Raw examples: {len(sft_ds):,d}")

            # Show all source counts for reference
            counts = Counter(sft_ds["source"])

            for subset_name in sft_todo:
                out = output_path(subset_name)
                print(f"\n── Extracting: {subset_name}")

                filtered = sft_ds.filter(
                    lambda ex, name=subset_name: ex["source"] == name,
                    num_proc=args.num_proc,
                )
                print(f"   {len(filtered):,d} examples")

                if len(filtered) == 0:
                    print(f"   ⚠️  No examples matched! Skipping.")
                    # Show closest matches
                    for src in sorted(counts.keys()):
                        if subset_name.split("/")[-1][:10] in src:
                            print(f"      Did you mean: {src} ({counts[src]:,d})?")
                    continue

                save_jsonl(filtered, out, tokenizer=tokenizer, max_length=args.max_length)
                print(f"   ✅ Saved → {out}")

            del sft_ds

    # ── Standalone datasets ───────────────────────────────────────────
    for name, hf_id, hf_config in STANDALONE_DATASETS:
        if name not in targets:
            continue

        out = output_path(name)
        if should_skip(out, args.force):
            print(f"⏭  {name}  (already exists, skipping)")
            continue

        print(f"\n{'═'*60}")
        print(f"  Loading {hf_id} ...")
        print(f"{'═'*60}")

        if hf_config:
            ds = load_dataset(hf_id, hf_config, split="train")
        else:
            ds = load_dataset(hf_id, split="train")

        print(f"  {len(ds):,d} examples")
        save_jsonl(ds, out, tokenizer=tokenizer, max_length=args.max_length)
        print(f"  ✅ Saved → {out}")
        del ds

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Summary")
    print(f"{'═'*60}")
    for name in sorted(all_names):
        out = output_path(name)
        if os.path.exists(out):
            size_mb = os.path.getsize(out) / (1024 * 1024)
            # Count lines
            with open(out, "r") as f:
                n_lines = sum(1 for _ in f)
            print(f"  ✅ {name:<58s}  {n_lines:>8,d} rows  {size_mb:>7.1f} MB")
        else:
            print(f"  ❌ {name:<58s}  (missing)")


if __name__ == "__main__":
    main()
