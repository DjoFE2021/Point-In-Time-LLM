"""
Tokenize the SFT mixture from local classified/timeless JSONL files.

Reads data from:  data/post_training_dataset/*/*/classified/timeless/data.jsonl

Usage:
    # Tokenize the default mixture (code+math+IF) into a single dataset:
    python post_training/sft_tokens.py

    # Custom output dir and max length:
    python post_training/sft_tokens.py --output-dir ./data/sft_tok --max-length 4096

    # Cap a specific source:
    python post_training/sft_tokens.py --cap "argilla/ifeval-like-data:300000"

The saved dataset can then be loaded by tulu3_lora.py or similar for training.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, OrderedDict
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


# ───────────── Source definitions ─────────────

# Code & Math sources
CODE_MATH_SOURCES = [
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "ai2-adapt-dev/personahub_code_v2_34999",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
]

# IF / eval-like sources
IF_SOURCES = [
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    "argilla/ifeval-like-data",
]

# Default caps  (source_name → max_examples)
DEFAULT_CAPS = {
    "argilla/ifeval-like-data": 300_000,
}

ALL_SOURCES = CODE_MATH_SOURCES + IF_SOURCES


# ───────────── Chat formatting (matches tulu3_tokens.py) ─────────────

def format_messages(messages):
    """Format a list of {'role': ..., 'content': ...} dicts into plain text."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    return "\n".join(parts)


def record_to_messages(record):
    """Normalise any JSONL schema into a list of message dicts."""
    # Schema 1: messages list (ai2-adapt-dev/*)
    if "messages" in record and isinstance(record["messages"], list):
        return record["messages"]

    # Schema 2: instruction / response (argilla/ifeval-like-data)
    if "instruction" in record:
        return [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record.get("response", "")},
        ]

    # Schema 3: question / answer (openai/gsm8k)
    if "question" in record:
        return [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record.get("answer", "")},
        ]

    raise ValueError(f"Unknown schema, keys: {list(record.keys())}")


def to_chat_text(record):
    """Convert a JSONL record to full-conversation plain text."""
    return format_messages(record_to_messages(record))


# ───────────── CJK filter (same as tulu3_tokens.py) ─────────────

_CJK_RE = re.compile(
    r'[\u4e00-\u9fff'
    r'\u3400-\u4dbf'
    r'\uf900-\ufaff'
    r'\u3000-\u303f'
    r'\u3040-\u309f'
    r'\u30a0-\u30ff'
    r'\uac00-\ud7af]'
)

def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


# ───────────── I/O helpers ─────────────

def load_jsonl(path: str):
    """Stream-read a JSONL file, return list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def discover_timeless(base_dir: str, source_name: str) -> str:
    """Return the path to classified/timeless/data.jsonl for a given source."""
    path = os.path.join(base_dir, source_name, "classified", "timeless", "data.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing: {path}")
    return path


# ───────────── Tokenization ─────────────

def tokenize_example(text, tokenizer, max_length):
    """Tokenize a single chat text into input_ids, attention_mask, labels."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    # Causal LM SFT: labels[i] = input_ids[i+1]
    encoding["labels"] = encoding["input_ids"][1:] + [-100]
    return encoding


# ───────────── Main ─────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize SFT mixture from local classified/timeless JSONL files"
    )
    parser.add_argument(
        "--base-dir", type=str, default="./data/post_training_dataset",
        help="Root directory containing org/dataset/classified/timeless structure",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/sft_tokenized",
        help="Where to save the tokenized dataset",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--num-proc", type=int, default=8,
        help="Number of processes for parallel tokenization",
    )
    parser.add_argument(
        "--cap", nargs="+", default=None, metavar="SOURCE:COUNT",
        help="Cap specific sources, e.g.  --cap 'argilla/ifeval-like-data:300000'",
    )
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Override default sources (space-separated full names)",
    )
    args = parser.parse_args()

    # ── Parse caps ──
    caps = dict(DEFAULT_CAPS)
    if args.cap:
        for entry in args.cap:
            src, count = entry.rsplit(":", 1)
            caps[src] = int(count)

    sources = args.sources if args.sources else ALL_SOURCES

    # ── Load all sources ──
    all_texts = []       # list of chat-formatted strings
    all_sources = []     # parallel list of source names
    source_counts = OrderedDict()
    code_math_total = 0
    if_total = 0

    print("=" * 70)
    print("  SFT Mixture — Loading from local timeless JSONL files")
    print("=" * 70)

    for src in sources:
        path = discover_timeless(args.base_dir, src)
        records = load_jsonl(path)
        n_raw = len(records)

        # ── Filter CJK ──
        filtered = []
        for rec in records:
            try:
                msgs = record_to_messages(rec)
                if any(contains_cjk(m.get("content", "")) for m in msgs):
                    continue
                filtered.append(rec)
            except ValueError:
                continue
        n_english = len(filtered)
        cjk_removed = n_raw - n_english

        # ── Apply cap ──
        if src in caps and len(filtered) > caps[src]:
            random.seed(42)
            random.shuffle(filtered)
            filtered = filtered[:caps[src]]
            cap_msg = f" (capped from {n_english:,d} → {caps[src]:,d})"
        else:
            cap_msg = ""

        n_final = len(filtered)

        # ── Convert to chat text ──
        for rec in filtered:
            all_texts.append(to_chat_text(rec))
            all_sources.append(src)

        # ── Categorise ──
        if src in CODE_MATH_SOURCES:
            code_math_total += n_final
        elif src in IF_SOURCES:
            if_total += n_final

        source_counts[src] = n_final

        # ── Per-source log ──
        short_name = src.split("/")[-1] if "/" in src else src
        cjk_note = f"  [CJK removed: {cjk_removed:,d}]" if cjk_removed else ""
        print(f"  {short_name:<55s}  {n_final:>8,d}{cap_msg}{cjk_note}")

    # ── Summary table (paper-ready) ──
    print()
    print("─" * 70)
    print("  Dataset breakdown (for paper):")
    print("─" * 70)
    for src, cnt in source_counts.items():
        print(f"    {src:<55s}  {cnt:>8,d}")
    print("─" * 70)
    print(f"    {'Code & Math total':<55s}  {code_math_total:>8,d}")
    print(f"    {'IF / eval-like total':<55s}  {if_total:>8,d}")
    print(f"    {'Grand total':<55s}  {code_math_total + if_total:>8,d}")
    print("─" * 70)

    # ── Build HuggingFace Dataset ──
    print(f"\nBuilding HuggingFace Dataset from {len(all_texts):,d} examples ...")
    ds = Dataset.from_dict({"text": all_texts, "source": all_sources})

    # ── Tokenize ──
    print(f"Loading tokenizer (gpt2) ...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizing with max_length={args.max_length} ...")
    def tok_fn(batch):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for text in batch["text"]:
            enc = tokenize_example(text, tokenizer, args.max_length)
            results["input_ids"].append(enc["input_ids"])
            results["attention_mask"].append(enc["attention_mask"])
            results["labels"].append(enc["labels"])
        return results

    ds = ds.map(
        tok_fn,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        desc="Tokenizing",
    )

    # ── Keep only trainer columns ──
    keep_cols = {"input_ids", "attention_mask", "labels"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    # ── Drop over-length (safety) ──
    before_len = len(ds)
    ds = ds.filter(
        lambda ex: len(ex["input_ids"]) <= args.max_length,
        num_proc=args.num_proc,
        desc="Filtering over-length",
    )
    dropped = before_len - len(ds)
    if dropped > 0:
        print(f"  ✂ Dropped {dropped:,d} / {before_len:,d} examples exceeding {args.max_length} tokens")

    # ── Final stats ──
    total_tokens = sum(len(ids) for ids in ds["input_ids"])
    print(f"\n  ✅ {len(ds):,d} examples remaining (≤ {args.max_length} tokens)")
    print(f"  Total tokens: {total_tokens:,d}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Example input_ids length: {len(ds[0]['input_ids'])}")

    # ── Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(f"\n  💾 Saved tokenized dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
