#!/usr/bin/env python3
"""
Verify token counts in the 8B dataset files.
Usage: python verify_8B_tokens.py [--dir /path/to/8B]
"""
import os
import argparse
import numpy as np
from dump_coverage import MONTH_MULTIPLIER
def peek_header(filepath):
    """Read the header and return token count."""
    with open(filepath, "rb") as f:
        header = np.frombuffer(f.read(256 * 8), dtype=np.int64)
    magic, version, ntok = header[0], header[1], header[2]
    return {"magic": magic, "version": version, "tokens": ntok}

def verify_files(output_dir: str, tokens_per_month: int = 8_000_000_000):
    """Verify all .bin files in the directory."""
    files = sorted([f for f in os.listdir(output_dir) if f.endswith(".bin")])
    
    complete = []
    incomplete = []
    
    print(f"{'Month':<10} {'Actual':>12} {'Expected':>12} {'%':>6} {'Size':>10} {'Status':<10}")
    print("-" * 70)
    
    for fname in files:
        month = fname.replace(".bin", "")
        fpath = os.path.join(output_dir, fname)
        
        # Expected tokens
        info = MONTH_MULTIPLIER.get(month, {"multiplier": 1})
        expected_tokens = tokens_per_month * info["multiplier"]
        
        # Read header
        header = peek_header(fpath)
        actual_tokens = header["tokens"]
        file_size = os.path.getsize(fpath)
        
        pct = (actual_tokens / expected_tokens) * 100 if expected_tokens > 0 else 0
        status = "✅ OK" if pct >= 99 else "❌ INCOMPLETE"
        
        print(f"{month:<10} {actual_tokens/1e9:>10.2f}B {expected_tokens/1e9:>10.2f}B {pct:>5.1f}% {file_size/1e9:>8.1f}GB {status}")
        if pct >= 99:
            complete.append(month)
        else:
            incomplete.append((month, actual_tokens, expected_tokens))
    
    print("-" * 70)
    print(f"Complete: {len(complete)} / {len(files)} files")
    print(f"Total months needed: {len(MONTH_MULTIPLIER)}")
    print(f"Remaining: {len(MONTH_MULTIPLIER) - len(complete)} months")
    
    if incomplete:
        print(f"\n❌ Incomplete files ({len(incomplete)}):")
        for month, actual, expected in incomplete:
            print(f"  {month}: {actual/1e9:.2f}B / {expected/1e9:.2f}B")
    
    return complete, incomplete

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/mnt/seagate_8T/datasets/fineweb_pit/8B", help="Directory with .bin files")
    parser.add_argument("--tokens-per-month", type=int, default=8_000_000_000, help="Base tokens per month")
    args = parser.parse_args()
    
    verify_files(args.dir, args.tokens_per_month)
