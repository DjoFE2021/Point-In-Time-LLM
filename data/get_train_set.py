#!/usr/bin/env python3
"""
Parallel FineWeb monthly shard builder.

Usage:
    # Process specific months
    python cached_monthly_test_async.py --months 2013-05 2013-06 2013-12
    
    # Process a range by index (0-based)
    python cached_monthly_test_async.py --start-idx 0 --end-idx 10
    
    # Process all months
    python cached_monthly_test_async.py --all
"""
import os
import argparse
import shutil
import numpy as np
import pyarrow.dataset as pds
from tqdm import tqdm
from datetime import datetime
import time, requests
from huggingface_hub import snapshot_download, list_repo_files
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer

from dump_coverage import MONTH_MULTIPLIER
# --- Configuration ---
HF_DATASET_REPO = "HuggingFaceFW/fineweb"
HF_DATASET_REPO_TYPE = "dataset"

# --- Helper: month difference ---
def month_diff(m1: str, m2: str) -> int:
    y1, mo1 = map(int, m1.split("-"))
    y2, mo2 = map(int, m2.split("-"))
    return (y2 - y1) * 12 + (mo2 - mo1)

def robust_snapshot_download(**kwargs):
    for attempt in range(4):
        try:
            return snapshot_download(**kwargs)
        except requests.exceptions.ReadTimeout as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
# --- Streaming Version: Writes directly to disk, O(batch) memory ---
def process_snapshot_month_batched_streaming(
    config: str,
    year_month: str,
    tokenizer: PreTrainedTokenizer,
    target_tokens: int,
    output_path: str,
    local_root: str,
    dtype=np.uint16,
    batch_size_files: int = 50
):
    """Same as process_snapshot_month_batched but streams tokens to disk."""
    # Compute date range
    year, month = map(int, year_month.split("-"))
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1
    lower, upper = f"{year:04d}-{month:02d}-01", f"{next_year:04d}-{next_month:02d}-01"

    eos = tokenizer.eos_token_id
    count = 0

    # 1. List all .parquet files for the dump
    all_files = list_repo_files(HF_DATASET_REPO, repo_type=HF_DATASET_REPO_TYPE)
    parquet_files = sorted(f for f in all_files if f.startswith(f"data/{config}/") and f.endswith(".parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found for {config} on HuggingFace")
        return

    print(f"🔍 Found {len(parquet_files)} parquet files for {config}. Processing {year_month} (streaming)...")

    with open(output_path, "wb") as f:
        # Write placeholder header (we'll update token count at the end)
        header = np.zeros(256, dtype=np.int64)
        header[0], header[1] = 20240520, 1  # magic, version (count will be updated)
        f.write(header.tobytes())

        for i in range(0, len(parquet_files), batch_size_files):
            batch_files = parquet_files[i:i+batch_size_files]
            if not batch_files:
                break

            # 2. Download only this batch of files
            robust_snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_DATASET_REPO_TYPE,
                local_dir=local_root,
                allow_patterns=batch_files,
                etag_timeout=300, 
            )

            parquet_dir = os.path.join(local_root, os.path.dirname(batch_files[0]))
            dataset = pds.dataset(parquet_dir, format="parquet")

            # 3. Filter by date
            date_col = "date"
            date_field = pds.field(date_col)
            filter_expr = (date_field >= lower) & (date_field < upper)

            scanner = dataset.scanner(filter=filter_expr, columns=[date_col, "text"], batch_size=4096)

            for batch in scanner.to_batches():
                if count >= target_tokens:
                    break
                texts = batch.column("text").to_pylist()
                for text in texts:
                    toks = tokenizer.encode(text or "", add_special_tokens=False)
                    if not toks:
                        continue
                    rem = target_tokens - count
                    if len(toks) >= rem:
                        toks = toks[: max(0, rem - 1)]
                    toks.append(eos)
                    
                    # Write tokens directly to file (streaming!)
                    f.write(np.array(toks, dtype=dtype).tobytes())
                    count += len(toks)
                    
                    if count >= target_tokens:
                        break
                print(f"📊 Collected {count:,}/{target_tokens:,} tokens for {config} {year_month}")

            # 4. Delete the downloaded Parquet files
            shutil.rmtree(parquet_dir, ignore_errors=True)
            if count >= target_tokens:
                break

        # Update header with actual token count
        f.seek(2 * 8)  # position of header[2]
        f.write(np.array([count], dtype=np.int64).tobytes())

    if count == 0:
        print(f"⚠️ No tokens collected for {config} {year_month}.")
        os.remove(output_path)
        return

    print(f"✅ Wrote {count:,} tokens to {output_path}")


def process_months(
    months: list[str],
    output_dir: str,
    snapshot_parquet_root: str,
    tokens_per_month: int = 8_000_000_000,
):
    """Process a list of specific months."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.model_max_length = 10**12

    os.makedirs(output_dir, exist_ok=True)

    for year_month in months:
        if year_month not in MONTH_MULTIPLIER:
            print(f"❌ Unknown month: {year_month}")
            continue

        output_path = os.path.join(output_dir, f"{year_month}.bin")
        if os.path.exists(output_path):
            print(f"⏭️  Skipping {year_month} (already exists)")
            continue

        info = MONTH_MULTIPLIER[year_month]
        config = info["config"]
        multiplier = info["multiplier"]
        target_tokens = tokens_per_month * multiplier

        print(f"\n⏳ Processing {year_month} | config={config} | multiplier={multiplier} | target={target_tokens:,}")

        process_snapshot_month_batched_streaming(
            config=config,
            year_month=year_month,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            output_path=output_path,
            local_root=snapshot_parquet_root,
            batch_size_files=2,
        )

    print("\n🎉 Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FineWeb monthly shards (parallel-friendly)")
    
    # Month selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--months", nargs="+", help="Specific months to process (e.g., 2013-05 2013-06)")
    group.add_argument("--start-idx", type=int, help="Start index (0-based) for month range")
    group.add_argument("--all", action="store_true", help="Process all months")
    
    parser.add_argument("--end-idx", type=int, help="End index (exclusive) for month range")
    parser.add_argument("--output-dir", default="/home/tengandrea/fineweb_pit/8B", help="Output directory")
    parser.add_argument("--tokens-per-month", type=int, default=8_000_000_000, help="Base tokens per month (default: 8B)")
    
    default_cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    parser.add_argument("--snapshot-parquet-root", default=default_cache, help="HF cache directory")

    args = parser.parse_args()

    # Get sorted list of all months
    all_months = sorted(MONTH_MULTIPLIER.keys())
    
    # Determine which months to process
    if args.months:
        months_to_process = args.months
    elif args.start_idx is not None:
        end_idx = args.end_idx if args.end_idx else len(all_months)
        months_to_process = all_months[args.start_idx:end_idx]
    else:  # --all
        months_to_process = all_months

    print(f"📅 Will process {len(months_to_process)} months: {months_to_process[:5]}{'...' if len(months_to_process) > 5 else ''}")
    print(f"📂 Output: {args.output_dir}")
    print(f"🎯 Tokens per month: {args.tokens_per_month:,}")

    process_months(
        months=months_to_process,
        output_dir=args.output_dir,
        snapshot_parquet_root=args.snapshot_parquet_root,
        tokens_per_month=args.tokens_per_month,
    )
