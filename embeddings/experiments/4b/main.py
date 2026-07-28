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
from typing import List, Tuple, Optional

# Make models/ importable — the repo root is the nearest ancestor holding models/,
# which keeps this working under both the repo layout (embeddings/experiments/4b)
# and the flat cluster layout (embeddings/4b).
sys.path.insert(0, str(next(
    p for p in Path(__file__).resolve().parents if (p / "models").is_dir()
)))
from models.GPT import GPT
from models.GPTConfig import GPT2_4B

DATASET_DIR       = "/scratch/jschwab/dataset/jkp_matched"
CKPT_DIR          = "/scratch/jschwab/checkpoints/4B"
CKPT_DIR_FULL     = "/scratch/jschwab/checkpoints/4B-full"
OUTPUT_DIR        = "/scratch/jschwab/embeddings/4b-v2"
OUTPUT_DIR_FULL   = "/scratch/jschwab/embeddings/4b-full-v2"
MAX_TOKENS        = 2048
BATCH_SIZE        = 64

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


def load_4b_model(ckpt_year: int, ckpt_month: int, device: torch.device,
                  ckpt_dir: str = CKPT_DIR, last_ckpt: bool = False) -> GPT:
    if last_ckpt:
        pts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if len(pts) != 1:
            raise RuntimeError(f"Expected exactly one .pt file in {ckpt_dir}, found: {pts}")
        path = pts[0]
    else:
        path = os.path.join(ckpt_dir, ckpt_stem(ckpt_year, ckpt_month))
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


def embed_articles(model, tokenizer, articles: list[str], device: torch.device, padding: str = "right") -> np.ndarray:
    """Return RMS-normed last-token embeddings for each article.

    Captures the input to lm_head via a pre-hook — this is F.rms_norm(x)
    after the last transformer block, identical to what lm_head projects.

    Args:
        padding: "right" (default) pads on the right and uses each article's
                 true token length to index the last real token.  "left" pads
                 on the left so the last real token is always at position -1.
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
    n_embd = model.lm_head.weight.shape[1]

    nonempty = [i for i, t in enumerate(all_token_ids) if len(t) > 0]
    n_empty = len(articles) - len(nonempty)
    if n_empty:
        print(f"  {n_empty} empty article(s) → NaN embeddings", flush=True)
    order = sorted(nonempty, key=lambda i: len(all_token_ids[i]))

    result = np.full((len(articles), n_embd), np.nan, dtype=np.float32)
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

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                try:
                    model(input_ids, full_sequence=(padding == "right"))
                except _EarlyExit:
                    pass

            hidden = captured["hidden"].float().cpu().numpy()  # (batch, seq_len, n_embd)
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


def build_work_list(out_dir: Path, last_ckpt: bool = False) -> List[Tuple[Optional[Tuple[int, int]], str]]:
    """Return sorted list of ((ckpt_year, ckpt_month), fpath) for unprocessed files.

    When last_ckpt=True, ckpt_ym is None for every entry (single fixed checkpoint).
    """
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))
    pairs = []
    for fpath in all_files:
        fname = Path(fpath).name      # DJN_YYYY-MM_retmatched.pkl
        date_part = fname.split("_")[1]
        file_year, file_month = int(date_part.split("-")[0]), int(date_part.split("-")[1])
        ckpt_ym = None if last_ckpt else get_checkpoint_ym(file_year, file_month)
        out_path = out_dir / f"{Path(fpath).stem}_embeddings.pkl"
        if not out_path.exists():
            pairs.append((ckpt_ym, fpath))
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
    # Empty articles carry NaN vectors — drop them so they cannot poison a mean
    finite = combined["embedding"].map(lambda e: e is not None and np.all(np.isfinite(e)))
    if not finite.all():
        print(f"  Dropping {(~finite).sum()} rows with non-finite embeddings.")
    combined = combined[finite]
    combined["year_month"] = (
        combined["Date"].astype(str).str[:4] + "-" + combined["Date"].astype(str).str[4:6]
    )

    def mean_embeddings(series):
        return np.stack(series.values).mean(axis=0)

    daily = (
        combined.groupby(["permno", "Date"])["embedding"]
        .agg(mean_embeddings).reset_index()
    )
    daily["year_month"] = (
        daily["Date"].astype(str).str[:4] + "-" + daily["Date"].astype(str).str[4:6]
    )
    monthly = (
        daily.groupby(["permno", "year_month"])["embedding"]
        .agg(mean_embeddings)
    )

    out_path = out_dir / "embeddings_monthly.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(monthly, f)
    print(f"Saved monthly embeddings ({len(monthly)} rows) → {out_path}")


def worker(rank: int, num_gpus: int, work_list: list, out_dir: Path,
           padding: str = "right", ckpt_dir: str = CKPT_DIR, last_ckpt: bool = False) -> None:
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    prefix = f"[GPU {rank}]" if torch.cuda.is_available() else "[CPU]"

    chunk_size = (len(work_list) + num_gpus - 1) // num_gpus
    start = rank * chunk_size
    chunk = work_list[start : start + chunk_size]

    if not chunk:
        print(f"  {prefix} No files to process.", flush=True)
        return

    print(f"  {prefix} {len(chunk)} files to embed.", flush=True)

    current_ckpt_ym = "unloaded"  # sentinel distinct from None (used by last_ckpt)
    model = None
    tokenizer = tiktoken.get_encoding("gpt2")

    try:
        for ckpt_ym, fpath in chunk:
            if ckpt_ym != current_ckpt_ym:
                if model is not None:
                    del model
                    torch.cuda.empty_cache()
                if last_ckpt:
                    print(f"\n=== {prefix} Full checkpoint (last_ckpt) ===", flush=True)
                else:
                    print(f"\n=== {prefix} Checkpoint {ckpt_ym[0]}-{ckpt_ym[1]:02d} ===", flush=True)
                model = load_4b_model(
                    *(ckpt_ym if ckpt_ym else (None, None)), device,
                    ckpt_dir=ckpt_dir, last_ckpt=last_ckpt,
                )
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

            embs = embed_articles(model, tokenizer, articles, device, padding=padding)
            df["embedding"] = list(embs)

            with open(out_path, "wb") as f:
                pickle.dump(df, f)

            print(f"  {prefix} Saved → {out_path}", flush=True)
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()


def test_mode(padding: str = "right", seed: int = None,
              ckpt_dir: str = CKPT_DIR, last_ckpt: bool = False) -> None:
    """Embed 3 randomly sampled articles and print sanity checks."""
    rng = np.random.default_rng(seed)

    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))
    if not all_files:
        print("No dataset files found — check DATASET_DIR.")
        return

    fpath = all_files[rng.integers(len(all_files))]
    fname = Path(fpath).name
    date_part = fname.split("_")[1]
    file_year, file_month = int(date_part.split("-")[0]), int(date_part.split("-")[1])
    ckpt_ym = None if last_ckpt else get_checkpoint_ym(file_year, file_month)

    print(f"=== TEST MODE ===")
    print(f"File       : {fname}")
    if last_ckpt:
        print(f"Checkpoint : (last_ckpt) {ckpt_dir}")
    else:
        print(f"Checkpoint : {ckpt_ym[0]}-{ckpt_ym[1]:02d}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_4b_model(
        *(ckpt_ym if ckpt_ym else (None, None)), device,
        ckpt_dir=ckpt_dir, last_ckpt=last_ckpt,
    )

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
    parser = argparse.ArgumentParser(description="Embed financial articles with the 4B model.")
    parser.add_argument("--test", action="store_true", help="Run sanity check on one file and exit.")
    parser.add_argument("--padding", choices=["left", "right"], default="right",
                        help="Padding side: 'left' keeps last real token at position -1; "
                             "'right' right-pads and uses each article's true length to index last token.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for article selection in test mode.")
    parser.add_argument("--last_ckpt", action="store_true",
                        help="Use the single final checkpoint from 4B-full instead of time-varying checkpoints. "
                             "Saves results to the 4b-full output directory.")
    args = parser.parse_args()

    ckpt_dir = CKPT_DIR_FULL if args.last_ckpt else CKPT_DIR
    out_dir_str = OUTPUT_DIR_FULL if args.last_ckpt else OUTPUT_DIR

    if args.test:
        test_mode(padding=args.padding, seed=args.seed, ckpt_dir=ckpt_dir, last_ckpt=args.last_ckpt)
        return

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    print(f"Output dir : {out_dir}")
    print(f"Padding    : {args.padding}")
    print(f"Last ckpt  : {args.last_ckpt}")
    print(f"Found {num_gpus} GPU(s).")

    work_list = build_work_list(out_dir, last_ckpt=args.last_ckpt)
    print(f"Files remaining: {len(work_list)}")

    if not work_list:
        print("Nothing to do.")
    elif num_gpus > 1:
        mp.spawn(
            worker,
            args=(num_gpus, work_list, out_dir, args.padding, ckpt_dir, args.last_ckpt),
            nprocs=num_gpus,
            join=True,
        )
    else:
        worker(0, 1, work_list, out_dir, padding=args.padding, ckpt_dir=ckpt_dir, last_ckpt=args.last_ckpt)

    aggregate_embeddings(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
