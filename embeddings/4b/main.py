import argparse
import os
import glob
import pickle
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import tiktoken
from collections import OrderedDict
from pathlib import Path

# Make models/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.GPT import GPT
from models.GPTConfig import GPT2_4B

DATASET_DIR  = "/scratch/jschwab/dataset/jkp_matched"
CKPT_DIR     = "/scratch/jschwab/checkpoints/4B"
OUTPUT_DIR   = "/scratch/jschwab/embeddings/4b"
MAX_TOKENS   = 2048
BATCH_SIZE   = 8

# Available checkpoints sorted chronologically as (year, month) tuples.
# Add or remove entries here if checkpoints are added later.
CHECKPOINTS: list[tuple[int, int]] = [
    (2013, 12),
    (2014, 12),
    (2015, 11),
    (2016, 12),
    (2017, 12),
    (2018, 12),
    (2019, 12),
    (2020, 12),
]


def ckpt_stem(year: int, month: int) -> str:
    return f"{year}-{month:02d}_checkpoint.pt"


def get_checkpoint_ym(file_year: int, file_month: int) -> tuple[int, int]:
    """Return the (year, month) of the checkpoint to use for a given file date.

    Uses the most recent checkpoint strictly before (file_year, file_month),
    floored at the earliest available checkpoint (2013-12).

    Examples
    --------
    - 2014-01 .. 2014-12  →  2013-12  (only checkpoint before 2014)
    - 2015-01 .. 2015-11  →  2014-12
    - 2015-12 .. 2016-12  →  2015-11  (no 2015-12 checkpoint exists)
    - 2017-01 .. 2017-12  →  2016-12
    """
    file_ym = (file_year, file_month)
    best = CHECKPOINTS[0]           # floor: use earliest if nothing is strictly before
    for ckpt_ym in CHECKPOINTS:
        if ckpt_ym < file_ym:
            best = ckpt_ym
    return best


def load_4b_model(ckpt_year: int, ckpt_month: int, device: torch.device) -> GPT:
    path = os.path.join(CKPT_DIR, ckpt_stem(ckpt_year, ckpt_month))
    print(f"  Loading checkpoint: {path}", flush=True)

    model = GPT(GPT2_4B())

    ckpt = torch.load(path, map_location="cpu")
    raw_sd = ckpt["model"]
    new_sd = OrderedDict()
    for k, v in raw_sd.items():
        name = k.replace("module._orig_mod.", "").replace("_orig_mod.", "")
        new_sd[name] = v
    model.load_state_dict(new_sd)
    del ckpt, raw_sd, new_sd
    torch.cuda.empty_cache()

    model.to(device)
    model.eval()
    return model


def embed_articles(model, tokenizer, articles: list[str], device: torch.device) -> np.ndarray:
    """Return RMS-normed last-token embeddings for each article.

    Captures the input to lm_head via a pre-hook — this is F.rms_norm(x)
    after the last transformer block, identical to what lm_head projects.
    """
    captured: dict = {}

    def _pre_hook(_, args):
        captured["hidden"] = args[0]

    handle = model.lm_head.register_forward_pre_hook(_pre_hook)

    embeddings = []
    try:
        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i : i + BATCH_SIZE]

            token_ids = [tokenizer.encode(text)[:MAX_TOKENS] for text in batch]
            max_len = max(len(t) for t in token_ids)
            padded = [[0] * (max_len - len(t)) + t for t in token_ids]

            input_ids = torch.tensor(padded, dtype=torch.long).to(device)

            with torch.no_grad():
                model(input_ids)

            # (batch, seq_len, n_embd) — left-padded so last token is at -1
            batch_emb = captured["hidden"][:, -1, :]
            embeddings.append(batch_emb.float().cpu().numpy())
    finally:
        handle.remove()

    return np.vstack(embeddings)


def build_work_list(out_dir: Path) -> list[tuple[tuple[int, int], str]]:
    """Return sorted list of ((ckpt_year, ckpt_month), fpath) for unprocessed files."""
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))
    pairs = []
    for fpath in all_files:
        fname = Path(fpath).name      # DJN_YYYY-MM_retmatched.pkl
        date_part = fname.split("_")[1]
        file_year, file_month = int(date_part.split("-")[0]), int(date_part.split("-")[1])
        ckpt_ym = get_checkpoint_ym(file_year, file_month)
        out_path = out_dir / f"{Path(fpath).stem}_embeddings.pkl"
        if not out_path.exists():
            pairs.append((ckpt_ym, fpath))
    # Sort by (ckpt_ym, fpath) so each GPU loads each checkpoint only once
    return sorted(pairs)


def aggregate_embeddings(out_dir: Path) -> None:
    """Average embeddings article → daily → monthly, grouped by (permno, month)."""
    emb_files = sorted(out_dir.glob("*_embeddings.pkl"))
    if not emb_files:
        print("No embedding files found — skipping aggregation.")
        return

    print(f"\n=== Aggregating {len(emb_files)} files ===")
    chunks = []
    for fpath in emb_files:
        with open(fpath, "rb") as f:
            df = pickle.load(f)
        chunks.append(df[["permno", "Date", "embedding"]].copy())

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.dropna(subset=["permno", "Date", "embedding"])
    combined["year_month"] = (
        combined["Date"].astype(str).str[:7]
        .str.replace(r"(\d{4})(\d{2})", r"\1-\2", regex=True)
    )

    def mean_embeddings(series):
        return np.stack(series.values).mean(axis=0)

    daily = (
        combined.groupby(["permno", "Date"])["embedding"]
        .agg(mean_embeddings).reset_index()
    )
    daily["year_month"] = (
        daily["Date"].astype(str).str[:7]
        .str.replace(r"(\d{4})(\d{2})", r"\1-\2", regex=True)
    )
    monthly = (
        daily.groupby(["permno", "year_month"])["embedding"]
        .agg(mean_embeddings)
    )

    out_path = out_dir / "embeddings_monthly.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(monthly, f)
    print(f"Saved monthly embeddings ({len(monthly)} rows) → {out_path}")


def worker(rank: int, num_gpus: int, work_list: list, out_dir: Path) -> None:
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    prefix = f"[GPU {rank}]" if torch.cuda.is_available() else "[CPU]"

    chunk_size = (len(work_list) + num_gpus - 1) // num_gpus
    start = rank * chunk_size
    chunk = work_list[start : start + chunk_size]

    if not chunk:
        print(f"  {prefix} No files to process.", flush=True)
        return

    print(f"  {prefix} {len(chunk)} files to embed.", flush=True)

    current_ckpt_ym = None
    model = None
    tokenizer = tiktoken.get_encoding("gpt2")

    try:
        for ckpt_ym, fpath in chunk:
            if ckpt_ym != current_ckpt_ym:
                if model is not None:
                    del model
                    torch.cuda.empty_cache()
                print(f"\n=== {prefix} Checkpoint {ckpt_ym[0]}-{ckpt_ym[1]:02d} ===", flush=True)
                model = load_4b_model(*ckpt_ym, device)
                current_ckpt_ym = ckpt_ym

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


def test_mode() -> None:
    """Embed 3 articles from the first file and print sanity checks."""
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))
    if not all_files:
        print("No dataset files found — check DATASET_DIR.")
        return

    fpath = all_files[0]
    fname = Path(fpath).name
    date_part = fname.split("_")[1]
    file_year, file_month = int(date_part.split("-")[0]), int(date_part.split("-")[1])
    ckpt_ym = get_checkpoint_ym(file_year, file_month)

    print(f"=== TEST MODE ===")
    print(f"File       : {fname}")
    print(f"Checkpoint : {ckpt_ym[0]}-{ckpt_ym[1]:02d}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_4b_model(*ckpt_ym, device)

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

    if len(embs) >= 2:
        cos = float(
            (embs[0] * embs[1]).sum()
            / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9)
        )
        print(f"Cosine sim [0,1]: {cos:.4f}")

    print("\nTest passed.")


def main():
    parser = argparse.ArgumentParser(description="Embed financial articles with the 4B model.")
    parser.add_argument("--test", action="store_true", help="Run sanity check on one file and exit.")
    args = parser.parse_args()

    if args.test:
        test_mode()
        return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    print(f"Output dir : {out_dir}")
    print(f"Found {num_gpus} GPU(s).")

    work_list = build_work_list(out_dir)
    print(f"Files remaining: {len(work_list)}")

    if not work_list:
        print("Nothing to do.")
    elif num_gpus > 1:
        mp.spawn(
            worker,
            args=(num_gpus, work_list, out_dir),
            nprocs=num_gpus,
            join=True,
        )
    else:
        worker(0, 1, work_list, out_dir)

    aggregate_embeddings(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
