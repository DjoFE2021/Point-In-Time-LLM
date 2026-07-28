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
FIRST_MODEL_YEAR = 2013
LAST_MODEL_YEAR = 2024
MAX_TOKENS = 1792
BATCH_SIZE = 128


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


def embed_articles(model, tokenizer, articles: list[str], device: torch.device, padding: str = "right") -> np.ndarray:
    """Return last-layer RMS-normed last-token embeddings for each article.

    Captures the input to lm_head via a pre-hook — norm(x) in both base and
    instruct variants, regardless of what forward() returns.

    Args:
        model: ChronoGPT instance.
        tokenizer: tiktoken encoding (gpt2).
        articles: List of raw article strings.
        device: Torch device.
        padding: "right" (default) pads on the right and uses each article's
                 true token length to index the last real token.  "left" pads
                 on the left so the last real token is always at position -1.
                 ChronoGPT's forward() applies lm_head over the full sequence,
                 so no full-sequence flag is needed for the right-padded path.

    Returns:
        numpy array of shape (len(articles), model_dim).
    """
    captured: dict = {}

    class _EarlyExit(Exception):
        pass

    def _pre_hook(_, args):
        captured["hidden"] = args[0].detach()
        raise _EarlyExit()

    handle = model.lm_head.register_forward_pre_hook(_pre_hook)

    # Tokenize all articles upfront and sort by length to minimise padding waste.
    # Articles that tokenize to nothing have no last real token, so they are held
    # out of the batches and left as NaN for aggregation to drop — batching them
    # would either index a pad position or produce a zero-width batch.
    all_token_ids = [tokenizer.encode(text)[:MAX_TOKENS] for text in articles]
    model_dim = model.lm_head.weight.shape[1]

    nonempty = [i for i, t in enumerate(all_token_ids) if len(t) > 0]
    n_empty = len(articles) - len(nonempty)
    if n_empty:
        print(f"  {n_empty} empty article(s) → NaN embeddings", flush=True)
    order = sorted(nonempty, key=lambda i: len(all_token_ids[i]))

    result = np.full((len(articles), model_dim), np.nan, dtype=np.float32)
    try:
        for i in range(0, len(order), BATCH_SIZE):
            batch_idx = order[i : i + BATCH_SIZE]
            batch_ids = [all_token_ids[k] for k in batch_idx]

            lengths = [len(t) for t in batch_ids]
            max_len = max(lengths)

            if padding == "right":
                padded = [t + [0] * (max_len - len(t)) for t in batch_ids]
            else:
                padded = [[0] * (max_len - len(t)) + t for t in batch_ids]

            input_ids = torch.tensor(padded, dtype=torch.long).to(device)

            with torch.no_grad():
                try:
                    model(input_ids)
                except _EarlyExit:
                    pass

            # captured["hidden"]: (batch, seq_len, model_dim) — RMS-normalised
            hidden = captured["hidden"].float().cpu().numpy()
            if padding == "right":
                batch_emb = np.stack([hidden[j, lengths[j] - 1, :] for j in range(len(batch_ids))])
            else:
                batch_emb = hidden[:, -1, :]

            # Scatter back into original article order
            for j, k in enumerate(batch_idx):
                result[k] = batch_emb[j]
    finally:
        handle.remove()

    return result


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


def worker(rank: int, num_gpus: int, work_list: list, model_type: str, out_dir: Path, padding: str = "right") -> None:
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

            embs = embed_articles(model, tokenizer, articles, device, padding=padding)
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
    # Empty articles carry NaN vectors — drop them so they cannot poison a mean
    finite = combined["embedding"].map(lambda e: e is not None and np.all(np.isfinite(e)))
    if not finite.all():
        print(f"  Dropping {(~finite).sum()} rows with non-finite embeddings.")
    combined = combined[finite]

    # Date is stored as "YYYYMMDD" string → derive year_month "YYYY-MM"
    combined["year_month"] = combined["Date"].astype(str).str[:4] + "-" + combined["Date"].astype(str).str[4:6]

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
    daily["year_month"] = daily["Date"].astype(str).str[:4] + "-" + daily["Date"].astype(str).str[4:6]
    monthly = (
        daily
        .groupby(["permno", "year_month"])["embedding"]
        .agg(mean_embeddings)
    )

    out_path = out_dir / "embeddings_monthly.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(monthly, f)
    print(f"Saved monthly embeddings ({len(monthly)} rows) → {out_path}")


def test_mode(model_type: str, padding: str = "right", seed: int = None) -> None:
    """Embed 3 randomly sampled articles and print sanity checks."""
    rng = np.random.default_rng(seed)

    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))
    if not all_files:
        print("No dataset files found — check DATASET_DIR.")
        return

    fpath = all_files[rng.integers(len(all_files))]
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

    all_articles = df["Article"].fillna("").tolist()
    indices = rng.choice(len(all_articles), size=min(3, len(all_articles)), replace=False)
    articles = [all_articles[i] for i in sorted(indices)]
    token_lens = [len(tokenizer.encode(a)) for a in articles]
    print(f"Articles   : {len(articles)} (indices {list(sorted(indices))},"
          f" token lengths {token_lens}, truncated to {MAX_TOKENS})")
    print(f"Padding    : {padding}")

    embs = embed_articles(model, tokenizer, articles, device, padding=padding)

    print(f"\nEmbedding shape : {embs.shape}")
    print(f"dtype           : {embs.dtype}")
    print(f"min / max       : {embs.min():.4f} / {embs.max():.4f}")
    print(f"mean / std      : {embs.mean():.4f} / {embs.std():.4f}")
    print(f"any NaN         : {np.isnan(embs).any()}")
    print(f"any Inf         : {np.isinf(embs).any()}")
    for idx in range(len(embs)):
        print(f"Sample[{idx}]      : {embs[idx, :8]}")

    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    normed = embs / norms
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            cos = float((normed[i] * normed[j]).sum())
            print(f"Cosine sim [{i},{j}]: {cos:.4f}")

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
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for article selection in test mode (for reproducibility).",
    )
    parser.add_argument(
        "--padding", choices=["left", "right"], default="right",
        help="Padding side: 'left' keeps last real token at position -1; "
             "'right' right-pads and uses each article's true length to index last token.",
    )
    args = parser.parse_args()

    if args.test:
        test_mode(args.model_type, padding=args.padding, seed=args.seed)
        return

    out_dir = Path(OUTPUT_BASE) / f"chronogpt_{args.model_type}-v2"

    num_gpus = torch.cuda.device_count()
    print(f"Model type : {args.model_type}")
    print(f"Output dir : {out_dir}")
    print(f"Padding    : {args.padding}")
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
            args=(num_gpus, work_list, args.model_type, out_dir, args.padding),
            nprocs=num_gpus,
            join=True,
        )
    else:
        worker(0, 1, work_list, args.model_type, out_dir, padding=args.padding)

    aggregate_embeddings(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
