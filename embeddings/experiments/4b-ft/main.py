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

# Make models/ importable (repo root — this file lives at
# <root>/embeddings/experiments/4b-ft/main.py, so the root is parents[3])
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.GPT import GPT
from models.GPTConfig import GPT2_4B

DATASET_DIR       = "/scratch/jschwab/dataset/jkp_matched"
CKPT_DIR          = "/scratch/jschwab/checkpoints/4B-FT"
CKPT_DIR_FULL     = "/scratch/jschwab/checkpoints/4B-FT-full"
OUTPUT_DIR        = "/scratch/jschwab/embeddings/4b-ft-v2"
OUTPUT_DIR_FULL   = "/scratch/jschwab/embeddings/4b-ft-full"
MAX_TOKENS        = 2048
BATCH_SIZE        = 64

# Available checkpoints sorted chronologically as (year, month) tuples.
CHECKPOINTS: list[tuple[int, int]] = [
    (2013, 12),
    (2014, 12),
    (2015, 11),
    (2016, 12),
    (2017, 12),
    (2018, 12),
    (2019, 12),
]


def ckpt_stem(year: int, month: int) -> str:
    return f"{year}-{month:02d}_merged.pt"


def get_checkpoint_ym(file_year: int, file_month: int) -> tuple[int, int]:
    """Return the (year, month) of the checkpoint to use for a given file date.

    Uses the most recent checkpoint strictly before (file_year, file_month),
    floored at the earliest available checkpoint (2013-12).
    """
    file_ym = (file_year, file_month)
    best = CHECKPOINTS[0]           # floor: use earliest if nothing is strictly before
    for ckpt_ym in CHECKPOINTS:
        if ckpt_ym < file_ym:
            best = ckpt_ym
    return best


def load_4b_ft_model(ckpt_year: int, ckpt_month: int, device: torch.device,
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
        padding: "left" pads on the left so the last real token is always at
                 position -1.  "right" pads on the right and uses each
                 article's true token length to index the last real token.
    """
    captured: dict = {}

    class _EarlyExit(Exception):
        pass

    def _pre_hook(_, args):
        captured["hidden"] = args[0].detach()
        raise _EarlyExit()

    handle = model.lm_head.register_forward_pre_hook(_pre_hook)

    # Tokenize all articles upfront and sort by length to minimise padding waste
    all_token_ids = [tokenizer.encode(text)[:MAX_TOKENS] for text in articles]
    sorted_indices = np.argsort([len(t) for t in all_token_ids])
    sorted_token_ids = [all_token_ids[i] for i in sorted_indices]

    sorted_embs = np.empty((len(articles), ), dtype=object)
    try:
        for i in range(0, len(sorted_token_ids), BATCH_SIZE):
            batch_ids = sorted_token_ids[i : i + BATCH_SIZE]

            max_len = max(len(t) for t in batch_ids)
            lengths = [len(t) for t in batch_ids]

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

            for j, emb in enumerate(batch_emb):
                sorted_embs[i + j] = emb
    finally:
        handle.remove()

    # Restore original article order
    result = np.empty((len(articles), ), dtype=object)
    for sorted_pos, orig_pos in enumerate(sorted_indices):
        result[orig_pos] = sorted_embs[sorted_pos]
    return np.vstack(result)


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
                model = load_4b_ft_model(
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
    """Embed one article from each of 3 different date-spread files and print sanity checks."""
    rng = np.random.default_rng(seed)

    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "DJN_*_retmatched.pkl")))
    if not all_files:
        print("No dataset files found — check DATASET_DIR.")
        return

    # Pick 3 files spread evenly across the date range
    n = len(all_files)
    bucket_size = n // 3
    chosen_files = [
        all_files[rng.integers(i * bucket_size, (i + 1) * bucket_size)]
        for i in range(3)
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"=== TEST MODE ===")
    print(f"Device     : {device}")
    print(f"Padding    : {padding}")
    print(f"Last ckpt  : {last_ckpt}")
    print()

    articles = []
    labels = []
    current_model = None
    current_ckpt = "unloaded"

    for fpath in chosen_files:
        fname = Path(fpath).name
        date_part = fname.split("_")[1]
        file_year, file_month = int(date_part.split("-")[0]), int(date_part.split("-")[1])
        ckpt_ym = None if last_ckpt else get_checkpoint_ym(file_year, file_month)

        # Load model only if checkpoint changed
        if ckpt_ym != current_ckpt:
            if last_ckpt:
                print(f"  Loading full checkpoint from {ckpt_dir} ...")
            else:
                print(f"  Loading checkpoint {ckpt_ym[0]}-{ckpt_ym[1]:02d} ...")
            current_model = load_4b_ft_model(
                *(ckpt_ym if ckpt_ym else (None, None)), device,
                ckpt_dir=ckpt_dir, last_ckpt=last_ckpt,
            )
            current_ckpt = ckpt_ym

        with open(fpath, "rb") as f:
            df = pickle.load(f)

        all_arts = df["Article"].fillna("").tolist()
        # Prefer longer articles (top-25% by token length), pick one randomly
        tok_lens = [len(tokenizer.encode(a)[:MAX_TOKENS]) for a in all_arts]
        threshold = np.percentile(tok_lens, 75)
        candidates = [i for i, l in enumerate(tok_lens) if l >= threshold]
        idx = int(rng.choice(candidates))
        article = all_arts[idx]
        tok_len = tok_lens[idx]

        emb = embed_articles(current_model, tokenizer, [article], device, padding=padding)
        articles.append(emb[0])
        labels.append((fname, idx, tok_len, ckpt_ym))
        print(f"  [{fname}]  article {idx}, {tok_len} tokens, ckpt {ckpt_ym[0]}-{ckpt_ym[1]:02d}")

    embs = np.vstack(articles)
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
    parser = argparse.ArgumentParser(description="Embed financial articles with the 4B-FT model.")
    parser.add_argument("--test", action="store_true", help="Run sanity check on one file and exit.")
    parser.add_argument("--padding", choices=["left", "right"], default="right",
                        help="Padding side: 'left' keeps last real token at position -1; "
                             "'right' right-pads and uses each article's true length to index last token.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for article selection in test mode.")
    parser.add_argument("--last_ckpt", action="store_true",
                        help="Use the single final checkpoint from 4B-FT-full instead of time-varying checkpoints. "
                             "Saves results to the 4b-ft-full output directory.")
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


if __name__ == "__main__":
    main()
