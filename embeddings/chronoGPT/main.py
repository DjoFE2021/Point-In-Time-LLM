import argparse
import os
import glob
import pickle
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

from utils import load_model

DATASET_DIR = "/scratch/jschwab/dataset/jkp_matched"
OUTPUT_BASE = "/scratch/jschwab/embeddings"
FIRST_MODEL_YEAR = 2012
LAST_MODEL_YEAR = 2024
MAX_TOKENS = 1792
BATCH_SIZE = 8


def get_model_year(file_year: int) -> int:
    """Map a data year to the ChronoGPT snapshot used to embed it.

    Articles are embedded fully out-of-sample: year Y is embedded with the
    (Y-1)-12-31 model.  Everything up to and including 2012 uses the 2012
    model (the only in-sample period).  Years beyond LAST_MODEL_YEAR are
    not supported.
    """
    if file_year > LAST_MODEL_YEAR:
        raise ValueError(f"No ChronoGPT snapshot available for year {file_year}.")
    # OOS shift: use prior year's model, floored at FIRST_MODEL_YEAR
    return max(file_year - 1, FIRST_MODEL_YEAR)


def embed_articles(model, tokenizer, articles: list[str], device: torch.device) -> np.ndarray:
    """Return last-layer RMS-normed last-token embeddings for each article.

    ChronoGPT's forward() returns (logits, layer_outputs) where layer_outputs[i]
    is norm(x) after block i — already RMS-normalised, in float32-cast bfloat16.
    We use layer_outputs[-1] directly instead of a hook on blocks[-1], which
    would capture the raw (unnormalised) residual stream and produce large,
    coarsely-quantised values.

    Args:
        model: ChronoGPT instance.
        tokenizer: tiktoken encoding (gpt2).
        articles: List of raw article strings.
        device: Torch device.

    Returns:
        numpy array of shape (len(articles), model_dim).
    """
    # Capture lm_head input via pre-hook: this is norm(x) in both the base
    # and instruct variants, regardless of what forward() returns.
    captured: dict = {}

    def _pre_hook(_, args):
        captured["hidden"] = args[0]

    handle = model.lm_head.register_forward_pre_hook(_pre_hook)

    embeddings = []
    try:
        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i : i + BATCH_SIZE]

            # Tokenize and truncate to first MAX_TOKENS tokens
            token_ids = [tokenizer.encode(text)[:MAX_TOKENS] for text in batch]

            # Left-pad so every article's last real token sits at position -1
            max_len = max(len(t) for t in token_ids)
            padded = [[0] * (max_len - len(t)) + t for t in token_ids]

            input_ids = torch.tensor(padded, dtype=torch.long).to(device)

            model(input_ids)

            # captured["hidden"]: (batch, seq_len, model_dim) — RMS-normalised
            # With left-padding every article's last real token is at position -1
            batch_emb = captured["hidden"][:, -1, :]
            embeddings.append(batch_emb.float().cpu().numpy())
    finally:
        handle.remove()

    return np.vstack(embeddings)


def build_work_list(groups: dict, out_dir: Path) -> list[tuple[int, str]]:
    """Return a sorted list of (model_year, fpath) for files not yet embedded.

    Sorted by (model_year, fpath) so that contiguous chunks assigned to each
    GPU minimise the number of model reloads per GPU.
    """
    pairs = []
    for model_year, files in groups.items():
        for fpath in sorted(files):
            out_path = out_dir / f"{Path(fpath).stem}_embeddings.pkl"
            if not out_path.exists():
                pairs.append((model_year, fpath))
    return sorted(pairs)


def worker(rank: int, num_gpus: int, work_list: list, model_type: str, out_dir: Path) -> None:
    """Embed a contiguous chunk of the work list on GPU `rank`.

    The work list is pre-sorted by model_year so each GPU loads at most a
    handful of models (usually just one for the large 2012 group).
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    prefix = f"[GPU {rank}]" if torch.cuda.is_available() else "[CPU]"

    # Contiguous chunk — better load balance than round-robin by model year
    chunk_size = (len(work_list) + num_gpus - 1) // num_gpus
    start = rank * chunk_size
    chunk = work_list[start : start + chunk_size]

    if not chunk:
        print(f"  {prefix} No files to process.", flush=True)
        return

    print(f"  {prefix} {len(chunk)} files to embed.", flush=True)

    current_model_year = None
    model = None
    tokenizer = None

    try:
        for model_year, fpath in chunk:
            # Load a new model only when the model year changes
            if model_year != current_model_year:
                if model is not None:
                    del model
                    torch.cuda.empty_cache()
                print(f"\n=== {prefix} Loading model year {model_year} ===", flush=True)
                tokenizer, model = load_model(date(model_year, 12, 31), model_type=model_type)
                model.to(device)
                model.eval()
                current_model_year = model_year

            stem = Path(fpath).stem
            out_path = out_dir / f"{stem}_embeddings.pkl"
            if out_path.exists():
                print(f"  {prefix} [skip] {stem}", flush=True)
                continue

            with open(fpath, "rb") as f:
                df = pickle.load(f)

            articles = df["Article"].fillna("").tolist()
            print(f"  {prefix} Embedding {len(articles):>5} articles from {stem} ...", flush=True)

            embs = embed_articles(model, tokenizer, articles, device)
            df["embedding"] = list(embs)

            with open(out_path, "wb") as f:
                pickle.dump(df, f)

            print(f"  {prefix} Saved → {out_path}", flush=True)
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()


def aggregate_embeddings(out_dir: Path) -> None:
    """Average embeddings: article → daily → monthly, grouped by (permno, month).

    Two-step aggregation:
      1. Mean over all articles for the same stock on the same day.
      2. Mean over all days for the same stock in the same month.

    Saves a single DataFrame indexed by (permno, year_month) with one
    embedding vector per cell to out_dir/embeddings_monthly.pkl.
    """
    emb_files = sorted(out_dir.glob("*_embeddings.pkl"))
    if not emb_files:
        print("No embedding files found — skipping aggregation.")
        return

    print(f"\n=== Aggregating {len(emb_files)} files ===")
    chunks = []
    for fpath in emb_files:
        with open(fpath, "rb") as f:
            df = pickle.load(f)
        # Keep only what we need
        chunks.append(df[["permno", "Date", "embedding"]].copy())

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.dropna(subset=["permno", "Date", "embedding"])

    # Date is stored as "YYYYMMDD" string → derive year_month "YYYY-MM"
    combined["year_month"] = combined["Date"].astype(str).str[:7].str.replace(r"(\d{4})(\d{2})", r"\1-\2", regex=True)

    def mean_embeddings(series):
        return np.stack(series.values).mean(axis=0)

    # Step 1: average per (permno, date)
    daily = (
        combined
        .groupby(["permno", "Date"])["embedding"]
        .agg(mean_embeddings)
        .reset_index()
    )

    # Step 2: average per (permno, year_month)
    daily["year_month"] = daily["Date"].astype(str).str[:7].str.replace(r"(\d{4})(\d{2})", r"\1-\2", regex=True)
    monthly = (
        daily
        .groupby(["permno", "year_month"])["embedding"]
        .agg(mean_embeddings)
    )

    out_path = out_dir / "embeddings_monthly.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(monthly, f)
    print(f"Saved monthly embeddings ({len(monthly)} rows) → {out_path}")


def test_mode(model_type: str) -> None:
    """Embed a single file with 2 articles max and print sanity checks."""
    import torch.nn.functional as F

    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))
    if not all_files:
        print("No dataset files found — check DATASET_DIR.")
        return

    fpath = all_files[0]
    fname = Path(fpath).name
    file_year = int(fname.split("_")[1].split("-")[0])
    model_year = get_model_year(file_year)

    print(f"=== TEST MODE ===")
    print(f"File       : {fname}")
    print(f"Model year : {model_year}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    tokenizer, model = load_model(date(model_year, 12, 31), model_type=model_type)
    model.to(device)
    model.eval()

    with open(fpath, "rb") as f:
        df = pickle.load(f)

    articles = df["Article"].fillna("").tolist()[:3]
    print(f"Articles   : {len(articles)}")

    embs = embed_articles(model, tokenizer, articles, device)

    print(f"\nEmbedding shape : {embs.shape}")
    print(f"dtype           : {embs.dtype}")
    print(f"min / max       : {embs.min():.4f} / {embs.max():.4f}")
    print(f"mean / std      : {embs.mean():.4f} / {embs.std():.4f}")
    print(f"any NaN         : {np.isnan(embs).any()}")
    print(f"any Inf         : {np.isinf(embs).any()}")
    print(f"Sample values   : {embs[0, :8]}")

    # Cosine similarity between first two articles (should be < 1 if distinct)
    if len(embs) >= 2:
        cos = float(
            (embs[0] * embs[1]).sum()
            / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9)
        )
        print(f"Cosine sim [0,1]: {cos:.4f}")

    print("\nTest passed.")


def main():
    parser = argparse.ArgumentParser(description="Embed financial articles with ChronoGPT.")
    parser.add_argument(
        "--model-type", choices=["base", "instruct"], default="instruct",
        help="ChronoGPT variant to use (default: instruct).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a quick sanity check on one file and exit.",
    )
    args = parser.parse_args()

    if args.test:
        test_mode(args.model_type)
        return

    out_dir = Path(OUTPUT_BASE) / f"chronogpt_{args.model_type}"

    num_gpus = torch.cuda.device_count()
    print(f"Model type : {args.model_type}")
    print(f"Output dir : {out_dir}")
    print(f"Found {num_gpus} GPU(s).")

    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))

    # Group files by the model year that should embed them
    groups: dict[int, list[str]] = {}
    for fpath in all_files:
        fname = Path(fpath).name          # DJN_YYYY-MM_retmatched.pkl
        file_year = int(fname.split("_")[1].split("-")[0])
        try:
            model_year = get_model_year(file_year)
        except ValueError as e:
            print(f"  [warn] Skipping {fname}: {e}")
            continue
        groups.setdefault(model_year, []).append(fpath)

    out_dir.mkdir(parents=True, exist_ok=True)
    work_list = build_work_list(groups, out_dir)
    print(f"Files remaining: {len(work_list)} across {len(groups)} model years.")

    if not work_list:
        print("Nothing to do.")
    elif num_gpus > 1:
        mp.spawn(
            worker,
            args=(num_gpus, work_list, args.model_type, out_dir),
            nprocs=num_gpus,
            join=True,
        )
    else:
        worker(0, 1, work_list, args.model_type, out_dir)

    aggregate_embeddings(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
