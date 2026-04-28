"""
Classify dataset rows as time-aware (references real-world events) or not.

Reads from:   data/post_training_dataset/{name}/raw/data.jsonl
Writes to:
    data/post_training_dataset/{name}/classified/timeless/data.jsonl    (time_aware=0)
    data/post_training_dataset/{name}/classified/time_aware/data.jsonl  (time_aware=1)

Each row gets a "time_aware" key:
    1 = references real-world events, people, or time-sensitive facts
    0 = generic, hypothetical, mathematical, or coding content

Uses GPT-5 Mini via OpenAI API for classification.

Usage:
    # Classify all datasets:
    OPENAI_API_KEY=sk-... python post_training/classify_time_aware.py

    # Classify specific dataset(s):
    python post_training/classify_time_aware.py --only openai/gsm8k

    # Adjust concurrency:
    python post_training/classify_time_aware.py --workers 20

    # Resume from where you left off (automatic, reads progress file):
    python post_training/classify_time_aware.py
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Dataset names (must match store_raw_datasets.py) ──────────────────

ALL_DATASETS = [
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "ai2-adapt-dev/personahub_code_v2_34999",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    "argilla/ifeval-like-data",
    "allenai/llama-3.1-tulu-3-8b-preference-mixture",
    "openai/gsm8k",
]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(_PROJECT_ROOT, "data", "post_training_dataset")
MODEL = "gpt-5-nano"

# ── Classification prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a binary classifier. Determine if text contains TIME-SENSITIVE FACTS that could become outdated or incorrect over time.

Output ONLY "0" (timeless) or "1" (time-aware).

Output "1" ONLY if the text states facts that could BECOME OUTDATED, such as:
- Who currently holds a political office ("Biden is the president")
- Recent or dated events ("The 2024 Olympics were held in Paris")
- Current statistics, prices, or rankings ("Tesla stock is at $X")
- A real person doing something specific at a specific time ("Elon Musk announced X in 2024")
- Laws, policies, or regulations tied to a specific time frame

Output "0" if the text is:
- Math problems or solutions (even with character names like "Alice" or "John")
- Programming/coding tasks or tutorials (even mentioning real tools: Python, Java, NetBeans, Eclipse, AWS, Docker, etc.)
- General educational content about real software, frameworks, or technologies
- Generic advice or best practices (even about real products)
- Hypothetical scenarios (even with human names)
- Instruction-following tasks
- Scientific facts that don't change ("water boils at 100C")
- Abstract reasoning or logic puzzles

KEY DISTINCTION: Mentioning a real tool, company, or product by name does NOT make text time-aware. Only FACTUAL CLAIMS THAT COULD BECOME OUTDATED do.

Examples:
- "Use NetBeans profiler for CPU analysis" -> 0 (generic advice about a tool)
- "Write a REST API using Flask and AWS Lambda" -> 0 (coding tutorial)
- "Google announced Gemini 2.0 in December 2024" -> 1 (dated event)
- "The president of the United States is Joe Biden" -> 1 (will become outdated)
- "Tesla's market cap exceeded $1 trillion in 2024" -> 1 (time-sensitive fact)
- "Solve: 2x + 3 = 7" -> 0 (math)
- "Anna bought 5 apples at $2 each" -> 0 (hypothetical)
- "Python's GIL prevents true multithreading" -> 0 (technical fact, stable)
- "React 18 introduced concurrent rendering" -> 0 (historical tech fact, won't change)
- "As of 2024, React is the most popular framework" -> 1 (ranking changes over time)"""


# ── Text extraction ───────────────────────────────────────────────────

def extract_text(row: dict) -> str:
    """Extract all text from a row (handles all dataset formats)."""
    parts = []
    if "messages" in row and isinstance(row["messages"], list):
        for msg in row["messages"]:
            if isinstance(msg, dict):
                parts.append(msg.get("content", ""))
    for key in ("chosen", "rejected"):
        if key in row and isinstance(row[key], list):
            for msg in row[key]:
                if isinstance(msg, dict):
                    parts.append(msg.get("content", ""))
    for key in ("instruction", "response", "question", "answer", "prompt", "completion"):
        if key in row:
            parts.append(str(row[key]))
    return " ".join(parts)


# ── API call ──────────────────────────────────────────────────────────

def classify_row(client, text, model, max_chars=4000):
    """Call the model to classify a single row.

    Returns (label, raw_answer):
        label:      0, 1, or -1 (error)
        raw_answer:  the model's raw response string, or the error message
    """
    # Truncate to save tokens — first + last portion captures most signals
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n...\n" + text[-half:]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_completion_tokens=1500,
            # NOTE: gpt-5-nano does NOT support temperature=0
            #       (API returns 400: "Only the default (1) value is supported")
        )
        answer = (resp.choices[0].message.content or "").strip()
        if answer in ("0", "1"):
            return int(answer), answer
        # Model returned garbage — treat as error, not silent classification
        print(f"    ⚠️  Unexpected model output: {answer!r}")
        return -1, answer
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "403" in err_str or "invalid_api_key" in err_str:
            raise SystemExit(f"\n❌ API key error: {e}")
        print(f"    ⚠️  API error: {e}")
        return -1, f"ERROR: {e}"


# ── Processing ────────────────────────────────────────────────────────

def raw_path(name):
    return os.path.join(BASE_DIR, name, "raw", "data.jsonl")


def timeless_path(name):
    return os.path.join(BASE_DIR, name, "classified", "timeless", "data.jsonl")


def time_aware_path(name):
    return os.path.join(BASE_DIR, name, "classified", "time_aware", "data.jsonl")


def progress_path(name):
    return os.path.join(BASE_DIR, name, "classified", ".classify_progress")


def errors_path(name):
    return os.path.join(BASE_DIR, name, "classified", "errors.jsonl")


def retry_errors(client, name, model, workers=10, max_passes=3):
    """Retry rows that previously failed classification."""
    out_errors = errors_path(name)
    out_timeless = timeless_path(name)
    out_time_aware = time_aware_path(name)

    for pass_num in range(1, max_passes + 1):
        if not os.path.exists(out_errors) or os.path.getsize(out_errors) == 0:
            return

        with open(out_errors, "r", encoding="utf-8") as f:
            error_entries = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    error_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip corrupted lines

        if not error_entries:
            return

        print(f"  🔄 Retrying {len(error_entries):,d} errors (pass {pass_num}/{max_passes})...")

        resolved_timeless = []
        resolved_time_aware = []
        still_failed = []

        # Parallel retry
        texts = [extract_text(entry["row"]) for entry in error_entries]
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, text in enumerate(texts):
                future = pool.submit(classify_row, client, text, model)
                futures[future] = idx

            results = [None] * len(texts)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        for entry, (label, raw_answer) in zip(error_entries, results):
            row = entry["row"]
            if label == -1:
                entry["last_answer"] = raw_answer
                still_failed.append(entry)
            else:
                row["classifier_answer"] = raw_answer
                if label == 1:
                    resolved_time_aware.append(row)
                else:
                    resolved_timeless.append(row)

        # Append resolved rows to output files
        if resolved_timeless:
            with open(out_timeless, "a", encoding="utf-8") as f:
                for row in resolved_timeless:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if resolved_time_aware:
            with open(out_time_aware, "a", encoding="utf-8") as f:
                for row in resolved_time_aware:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Rewrite errors file with only still-failing rows
        with open(out_errors, "w", encoding="utf-8") as f:
            for entry in still_failed:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        resolved = len(resolved_timeless) + len(resolved_time_aware)
        print(f"    ✅ Resolved {resolved:,d} "
              f"(timeless={len(resolved_timeless):,d}, time_aware={len(resolved_time_aware):,d}), "
              f"{len(still_failed):,d} still failing")

        if not still_failed:
            open(out_errors, "w").close()  # truncate, don't delete
            return

    if still_failed:
        print(f"    ⚠️  {len(still_failed):,d} rows still failing after {max_passes} retry passes")


def process_dataset(client, name, model, workers=10, force=False):
    """Classify all rows in a dataset into timeless / time_aware."""
    inp = raw_path(name)
    out_timeless = timeless_path(name)
    out_time_aware = time_aware_path(name)
    out_errors = errors_path(name)
    prog = progress_path(name)

    if not os.path.exists(inp):
        print(f"  ❌ Raw file not found: {inp}")
        return

    # Count total lines
    with open(inp, "r") as f:
        total_lines = sum(1 for _ in f)

    # Check resume point
    start_line = 0
    if not force and os.path.exists(prog):
        with open(prog, "r") as f:
            start_line = int(f.read().strip())
        if start_line >= total_lines:
            print(f"  ⏭  Already fully classified ({total_lines:,d} rows)")
            # Still retry any leftover errors
            retry_errors(client, name, model, workers)
            return
        print(f"  ▶ Resuming from line {start_line:,d} / {total_lines:,d}")
    elif not force:
        # No progress file — check if output files exist
        has_outputs = any(os.path.exists(p) for p in (out_timeless, out_time_aware))
        if has_outputs:
            existing = 0
            for p in (out_timeless, out_time_aware, out_errors):
                if os.path.exists(p):
                    with open(p, "r") as f:
                        existing += sum(1 for _ in f)
            if existing >= total_lines:
                print(f"  ⏭  Already fully classified ({total_lines:,d} rows)")
                retry_errors(client, name, model, workers)
                return
            # Partial results with no progress file = ambiguous state
            print(f"  ❌ Found {existing:,d} output rows but no progress file.")
            print(f"     Cannot safely resume. Use --force to re-classify from scratch.")
            return

    # Open output files (append if resuming, write if starting fresh)
    mode = "a" if start_line > 0 else "w"
    os.makedirs(os.path.dirname(out_timeless), exist_ok=True)
    os.makedirs(os.path.dirname(out_time_aware), exist_ok=True)
    f_timeless = open(out_timeless, mode, encoding="utf-8")
    f_time_aware = open(out_time_aware, mode, encoding="utf-8")
    f_errors = open(out_errors, mode, encoding="utf-8")

    # Read all lines to classify
    with open(inp, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    classified_0 = 0
    classified_1 = 0
    error_count = 0
    batch_size = workers  # one API call per thread per batch
    t0 = time.time()

    # Process in batches with thread pool
    i = start_line
    while i < total_lines:
        batch_end = min(i + batch_size, total_lines)
        batch_lines = lines[i:batch_end]
        batch_rows = [json.loads(line) for line in batch_lines]
        batch_texts = [extract_text(row) for row in batch_rows]

        # Parallel API calls
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, text in enumerate(batch_texts):
                future = pool.submit(classify_row, client, text, model)
                futures[future] = idx

            results = [None] * len(batch_texts)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        # Write results to the appropriate file
        for idx, (row, (label, raw_answer)) in enumerate(zip(batch_rows, results)):
            if label == -1:
                # Log error with row index for later retry
                err_entry = {"line_index": i + idx, "row": row, "last_answer": raw_answer}
                f_errors.write(json.dumps(err_entry, ensure_ascii=False) + "\n")
                error_count += 1
            else:
                row["classifier_answer"] = raw_answer
                line_out = json.dumps(row, ensure_ascii=False) + "\n"
                if label == 1:
                    f_time_aware.write(line_out)
                    classified_1 += 1
                else:
                    f_timeless.write(line_out)
                    classified_0 += 1

        # Flush output buffers, then save progress
        f_timeless.flush()
        f_time_aware.flush()
        f_errors.flush()
        i = batch_end
        with open(prog, "w") as f:
            f.write(str(i))

        # Progress log
        elapsed = time.time() - t0
        rate = (i - start_line) / elapsed if elapsed > 0 else 0
        remaining = (total_lines - i) / rate if rate > 0 else 0
        err_str = f"  err={error_count}" if error_count else ""
        print(f"\r  {i:>9,d} / {total_lines:,d}  "
              f"({i/total_lines*100:.1f}%)  "
              f"{rate:.0f} rows/s  "
              f"ETA {remaining/60:.0f}m  "
              f"[time_aware=1: {classified_1:,d}{err_str}]", end="", flush=True)

    f_timeless.close()
    f_time_aware.close()
    f_errors.close()
    print()  # newline after progress

    total_done = classified_0 + classified_1
    pct = (classified_1 / total_done * 100) if total_done > 0 else 0
    print(f"  ✅ Done: {classified_1:,d} time-aware ({pct:.1f}%), "
          f"{classified_0:,d} timeless")
    if error_count > 0:
        print(f"  ⚠️  {error_count:,d} errors logged → {out_errors}")

    # ── Retry errors ──────────────────────────────────────────────
    retry_errors(client, name, model, workers)

    # Write final progress (keep file so re-runs detect completion)
    with open(prog, "w") as f:
        f.write(str(total_lines))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classify dataset rows as time-aware using GPT-5 Mini"
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Process only these dataset(s)",
    )
    parser.add_argument(
        "--workers", type=int, default=100,
        help="Number of concurrent API calls (default: 100)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-classify from scratch (ignore progress)",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL,
        help=f"OpenAI model to use (default: {MODEL})",
    )
    args = parser.parse_args()

    client = OpenAI()  # uses OPENAI_API_KEY env var

    if args.only:
        datasets = [d for d in args.only if d in ALL_DATASETS]
        unknown = set(args.only) - set(ALL_DATASETS)
        if unknown:
            print(f"⚠️  Unknown: {unknown}")
    else:
        datasets = ALL_DATASETS

    # Sort by raw file size (smallest first) so small datasets finish quickly
    datasets = sorted(datasets, key=lambda d: os.path.getsize(raw_path(d)) if os.path.exists(raw_path(d)) else float("inf"))

    print(f"{'═'*60}")
    print(f"  Classifying time-aware rows (model: {MODEL})")
    print(f"  Workers: {args.workers}")
    print(f"{'═'*60}")

    for name in datasets:
        print(f"\n── {name}")
        process_dataset(client, name, model=args.model, workers=args.workers, force=args.force)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Summary")
    print(f"{'═'*60}")
    def count_lines(p):
        if not os.path.exists(p):
            return 0
        with open(p, "r") as f:
            return sum(1 for _ in f)

    for name in ALL_DATASETS:
        tl = timeless_path(name)
        ta = time_aware_path(name)
        er = errors_path(name)
        if os.path.exists(tl) or os.path.exists(ta):
            n_timeless = count_lines(tl)
            n_time_aware = count_lines(ta)
            n_errors = count_lines(er)
            total = n_timeless + n_time_aware
            pct = n_time_aware / total * 100 if total > 0 else 0
            err_str = f"  errors={n_errors:,d}" if n_errors else ""
            print(f"  ✅ {name:<55s} {total:>8,d} rows  "
                  f"timeless={n_timeless:,d}  time_aware={n_time_aware:,d} ({pct:.1f}%){err_str}")
        else:
            print(f"  ❌ {name:<55s} (not classified)")


if __name__ == "__main__":
    main()
