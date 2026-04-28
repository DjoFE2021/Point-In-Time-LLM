import os
import sys

# ensure the script's directory is on the path so utils imports work
# regardless of where the script is invoked from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import load_dotenv
load_dotenv.load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from tqdm import tqdm

from utils.portfolios import produce_random_feature_managed_returns_chunked
from utils.load_data import load_matched_ret_emb
from utils.path_manager import get_embeddings_path
from utils.constants import DEFAULT_SHRINKAGE_GRID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS          = ["chronogpt_base-right", "chronogpt_instruct-right", "PIT-4B-right", "PIT-4B-FT-right"]
MODEL_LABELS    = ["ChronoGPT-base", "ChronoGPT-instruct", "PIT-4B", "PIT-4B-FT"]
MODEL_COLORS    = ["#10b981", "#f59e0b", "#2563eb", "#9ca3af"]   # green, amber, blue, light grey, purple, pink
SIZE_GROUPS     = ["all"]
MIN_TRAIN       = 12   # minimum months before first OOS prediction
ROLLING_WINDOW  = None# set to an int (e.g. 120) for rolling window, None for expanding
PORTFOLIOS      = ["linear"]
PORTFOLIO_TITLES = {"linear" : "Linear", "random_feature": "Random Features"}
N_RANDOM_FEAT   = 50
FILTER_SMALL=False

_BASE    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(_BASE, "results", "raw",   "msrr")
PLOT_DIR = os.path.join(_BASE, "results", "plots", "msrr")
for _d in (RAW_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

RESULTS_PATHS = {
    "linear":         os.path.join(RAW_DIR, "results_msrr_linear.pkl"),
    "random_feature": os.path.join(RAW_DIR, "results_msrr_random_feature.pkl"),
}
SPLIT_DATE = pd.Timestamp("2013-12-31")

PERIODS = {
    "Full sample":   (None, None),
    "In-sample":     (None, SPLIT_DATE),
    "Out-of-sample": (SPLIT_DATE, None),
}


def compute_sharpe(rets):
    return np.mean(rets, axis=0) / np.std(rets, axis=0) * np.sqrt(12)


def build_portfolio(df, portfolio, n_random_feat=N_RANDOM_FEAT):
    if portfolio == "linear":
        signals = df.drop(columns=["r_1", "size_grp"])
        return (signals * df["r_1"].values.reshape(-1, 1)).groupby(
            df.index.get_level_values("date")
        ).mean()
    elif portfolio == "random_feature":
        return produce_random_feature_managed_returns_chunked(
            P=n_random_feat,
            r1=df["r_1"],
            signals=df.drop(columns=["r_1", "size_grp"]),
            num_seeds=100,
            scale=1.0,
            activation="relu",
            base_seed=0,
        )
    else:
        raise ValueError(f"Unknown portfolio type: {portfolio}")


def _ridge_fit_predict_gpu(
    X_train: torch.Tensor,   # (t, p)
    X_test:  torch.Tensor,   # (1, p)
    shrinkage: torch.Tensor, # (n_z,)
    normalize_by_trace: bool = True,
) -> torch.Tensor:           # (n_z,)
    """Single-step ridge fit + predict, fully on GPU."""
    t_, p_ = X_train.shape

    if normalize_by_trace:
        trace = (X_train ** 2).mean()
        z = (p_ / t_) * shrinkage * trace
    else:
        z = shrinkage

    if p_ < t_:
        # standard: decompose p×p  S'S/t
        M = X_train.T @ X_train / t_
        eig_vals, eig_vecs = torch.linalg.eigh(M)          # (p,), (p,p)
        means = X_train.T @ torch.ones(t_, device=DEVICE, dtype=X_train.dtype) / t_  # (p,)
        proj  = eig_vecs.T @ means                          # (p,)
        denom = eig_vals.unsqueeze(1) + z.unsqueeze(0)      # (p, n_z)
        intermed = proj.unsqueeze(1) / denom                # (p, n_z)
        betas = eig_vecs @ intermed                         # (p, n_z)
    else:
        # wide: decompose t×t  SS'/t
        M = X_train @ X_train.T / t_
        eig_vals, eig_vecs = torch.linalg.eigh(M)          # (t,), (t,t)
        means = torch.ones(t_, device=DEVICE, dtype=X_train.dtype) / t_  # (t,)
        proj  = eig_vecs.T @ means                          # (t,)
        denom = eig_vals.unsqueeze(1) + z.unsqueeze(0)      # (t, n_z)
        intermed = proj.unsqueeze(1) / denom                # (t, n_z)
        betas = (X_train.T @ eig_vecs) @ intermed           # (p, n_z)

    return (X_test @ betas).squeeze(0)                      # (n_z,)


def run_experiment(df_port, rolling_window: int = None):
    """
    OOS for every ridge shrinkage value on the grid, computed on GPU.

    Returns {period_label: np.ndarray of shape (n_z,)} — annualised Sharpe.
    """
    shrinkage = torch.tensor(DEFAULT_SHRINKAGE_GRID, dtype=torch.float32, device=DEVICE)

    # move entire portfolio matrix to GPU once
    X_gpu = torch.tensor(df_port.values, dtype=torch.float32, device=DEVICE)
    all_dates = np.array(df_port.index)

    oos_ret, pred_dates = [], []
    for step in tqdm(range(MIN_TRAIN, len(df_port)), leave=False):
        train_start = max(0, step - rolling_window) if rolling_window is not None else 0
        X_train = X_gpu[train_start:step]        # (t, p) — GPU slice, zero-copy
        X_test  = X_gpu[step: step + 1]          # (1, p)

        with torch.no_grad():
            pred = _ridge_fit_predict_gpu(X_train, X_test, shrinkage)  # (n_z,)
        oos_ret.append(pred.cpu())
        pred_dates.append(all_dates[step])

    oos_ret    = torch.stack(oos_ret).numpy()     # (T, n_z)
    pred_dates = np.array(pred_dates)

    TARGET_STD = 0.10 / np.sqrt(12)   # 10% annualised → monthly

    # Rescale each ridge column to target std, then average
    col_std = oos_ret.std(axis=0, ddof=1)
    col_std = np.where(col_std < 1e-12, np.nan, col_std)
    oos_ret_scaled = oos_ret * (TARGET_STD / col_std)
    avg_ret = np.nanmean(oos_ret_scaled, axis=1, keepdims=True)  # (T, 1)

    ridge_cols = [str(z) for z in DEFAULT_SHRINKAGE_GRID]
    all_cols = ridge_cols + ["avg"]
    ret_df = pd.DataFrame(
        np.hstack([oos_ret, avg_ret]),
        index=pd.Index(pred_dates, name="date"),
        columns=all_cols,
    )

    sharpes = {}
    for label, (start, end) in PERIODS.items():
        mask = np.ones(len(pred_dates), dtype=bool)
        if start is not None:
            mask &= pred_dates > start
        if end is not None:
            mask &= pred_dates <= end
        r       = oos_ret[mask]
        r_avg   = avg_ret[mask]
        per_ridge = compute_sharpe(r) if len(r) > 1 else np.full(oos_ret.shape[1], np.nan)
        avg_sharpe = compute_sharpe(r_avg).item() if len(r_avg) > 1 else np.nan
        sharpes[label] = np.append(per_ridge, avg_sharpe)   # (n_z + 1,)

    return sharpes, ret_df  # ({period: ndarray (n_z+1,)}, DataFrame (T, n_z+1))


def _sharpes_from_csv(csv_path: str) -> dict:
    """Recompute per-ridge + avg Sharpes from a saved returns CSV."""
    TARGET_STD = 0.10 / np.sqrt(12)
    ret_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # keep only columns matching the current shrinkage grid, drop avg (recomputed below)
    ridge_cols = [str(z) for z in DEFAULT_SHRINKAGE_GRID]
    ret_df = ret_df[[c for c in ridge_cols if c in ret_df.columns]]

    oos_ret    = ret_df.values.astype(np.float32)          # (T, n_z)
    pred_dates = ret_df.index.to_numpy()

    col_std        = oos_ret.std(axis=0, ddof=1)
    col_std        = np.where(col_std < 1e-12, np.nan, col_std)
    oos_ret_scaled = oos_ret * (TARGET_STD / col_std)
    avg_ret        = np.nanmean(oos_ret_scaled, axis=1, keepdims=True)

    # overwrite CSV with avg column included
    ret_df["avg"] = avg_ret[:, 0]
    ret_df.to_csv(csv_path)

    sharpes = {}
    for label, (start, end) in PERIODS.items():
        mask = np.ones(len(pred_dates), dtype=bool)
        if start is not None:
            mask &= pred_dates > np.datetime64(start)
        if end is not None:
            mask &= pred_dates <= np.datetime64(end)
        r     = oos_ret[mask]
        r_avg = avg_ret[mask]
        per_ridge  = compute_sharpe(r) if len(r) > 1 else np.full(oos_ret.shape[1], np.nan)
        avg_sharpe = compute_sharpe(r_avg).item() if len(r_avg) > 1 else np.nan
        sharpes[label] = np.append(per_ridge, avg_sharpe)
    return sharpes


def run_all():
    # results[portfolio][model][size_grp] = {period: scalar}
    results = {p: {m: {sg: None for sg in SIZE_GROUPS} for m in MODELS} for p in PORTFOLIOS}

    for portfolio in PORTFOLIOS:
        path = RESULTS_PATHS[portfolio]
        if os.path.exists(path):
            with open(path, "rb") as f:
                saved = pickle.load(f)
            for m in MODELS:
                for sg in SIZE_GROUPS:
                    if m in saved and sg in saved[m]:
                        results[portfolio][m][sg] = saved[m][sg]
            print(f"Loaded existing results from {path}")

    # Always recompute avg from the returns CSV if it exists
    for portfolio in PORTFOLIOS:
        for model in MODELS:
            for sg in SIZE_GROUPS:
                csv_path = os.path.join(RAW_DIR, f"returns_{portfolio}_{model}_{sg}.csv")
                if os.path.exists(csv_path):
                    print(f"  Computing avg from CSV: {model} / {sg}")
                    results[portfolio][model][sg] = _sharpes_from_csv(csv_path)
                    with open(RESULTS_PATHS[portfolio], "wb") as f:
                        pickle.dump(results[portfolio], f)

    for portfolio in PORTFOLIOS:
        print(f"\n{'='*50}\nPortfolio: {portfolio}")

        for model in MODELS:
            all_done = all(
                results[portfolio][model][sg] is not None for sg in SIZE_GROUPS
            )
            if all_done:
                print(f"  Skipping {model} (already complete)")
                continue

            print(f"\n  === Model: {model} ===")
            df_full = load_matched_ret_emb(get_embeddings_path(model), residualize=True, filter_small = FILTER_SMALL)
            df_full = df_full.sort_index()

            for size_grp in SIZE_GROUPS:
                if results[portfolio][model][size_grp] is not None:
                    print(f"    Skipping size_grp={size_grp} (already computed)")
                    continue

                df = df_full[df_full["size_grp"] == size_grp] if size_grp != "all" else df_full

                print(f"    Building {portfolio} portfolios for size_grp={size_grp} ...")
                df_port = build_portfolio(df, portfolio)

                sharpes, ret_df = run_experiment(df_port, rolling_window=ROLLING_WINDOW)
                results[portfolio][model][size_grp] = sharpes

                csv_path = os.path.join(
                    RAW_DIR, f"returns_{portfolio}_{model}_{size_grp}.csv"
                )
                ret_df.to_csv(csv_path)
                print(f"    Saved returns to {csv_path}")

                print_table(results[portfolio][model], portfolio)
                with open(RESULTS_PATHS[portfolio], "wb") as f:
                    pickle.dump(results[portfolio], f)
                    
    breakpoint()

    save_oos_table(results)
    return results


def save_oos_table(results, period="Out-of-sample"):
    ridge_labels = [str(z) for z in DEFAULT_SHRINKAGE_GRID]
    all_cols = ridge_labels + ["avg"]
    for portfolio in PORTFOLIOS:
        records = []
        for model, sg in [(m, sg) for m in MODELS for sg in SIZE_GROUPS]:
            entry = results[portfolio][model][sg]
            if entry is None or period not in entry:
                continue
            vals = entry[period]
            has_avg = len(vals) == len(ridge_labels) + 1
            cols = ridge_labels + ["avg"] if has_avg else ridge_labels
            label = f"{model}" if len(SIZE_GROUPS) == 1 else f"{model}|{sg}"
            records.append({"model": label, **dict(zip(cols, vals))})
        if not records:
            continue
        table = pd.DataFrame(records).set_index("model").reindex(columns=all_cols)
        table.index.name = "model"
        table.columns.name = "ridge"
        csv_path = os.path.join(RAW_DIR, f"sharpe_oos_{portfolio}.csv")
        table.to_csv(csv_path, float_format="%.4f")
        print(f"\nOOS Sharpe table saved to {csv_path}")
        print(table.to_string(float_format="{:.3f}".format))


def _get_sharpe(results, portfolio, model, sg, period):
    """Return Sharpe array (n_z,) for (portfolio, model, sg, period)."""
    entry = results[portfolio][model][sg]
    if entry is None:
        return None
    return entry[period]


def print_table(model_results, portfolio):
    col_labels = [str(z) for z in DEFAULT_SHRINKAGE_GRID] + ["avg"]
    for period in PERIODS:
        rows = {}
        for sg in SIZE_GROUPS:
            entry = model_results[sg]
            if entry is not None and period in entry:
                rows[sg] = entry[period]
        if not rows:
            continue
        table = pd.DataFrame(rows, index=col_labels).T
        table.index.name = "size_grp"
        table.columns.name = "ridge"
        print(f"\n    Sharpe by ridge — {period} [{portfolio}]:")
        print(table.to_string(float_format="{:.3f}".format))
    print()


def plot_results(results):
    period   = "Out-of-sample"
    n_models = len(MODELS)
    n_groups = len(SIZE_GROUPS)
    x        = np.arange(n_groups)
    total_w  = 0.65
    bar_w    = total_w / n_models

    fig, axes = plt.subplots(
        len(PORTFOLIOS), 1,
        figsize=(9, 4 * len(PORTFOLIOS)),
        facecolor="white",
    )
    if len(PORTFOLIOS) == 1:
        axes = [axes]

    for ax, portfolio in zip(axes, PORTFOLIOS):
        ax.set_facecolor("white")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

        n_ridge = len(DEFAULT_SHRINKAGE_GRID)
        for m_idx, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLORS)):
            sharpe_vals = []
            for sg in SIZE_GROUPS:
                arr = _get_sharpe(results, portfolio, model, sg, period)
                # index n_ridge = avg column
                val = float(arr[n_ridge]) if arr is not None and len(arr) > n_ridge else np.nan
                sharpe_vals.append(val)
            offset = (m_idx - (n_models - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, sharpe_vals, bar_w,
                label=label, color=color, zorder=2, edgecolor="none",
            )
            for bar, val in zip(bars, sharpe_vals):
                va  = "bottom" if val >= 0 else "top"
                pad = 0.01  if val >= 0 else -0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + pad,
                    f"{val:.2f}",
                    ha="center", va=va,
                    fontsize=7, color="#333333",
                )

        ax.set_title(PORTFOLIO_TITLES[portfolio], fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Sharpe ratio", fontsize=9)
        ax.set_xlabel("Size group", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(SIZE_GROUPS, fontsize=9)

        ymin, ymax = ax.get_ylim()
        if portfolio == "linear":
            ymin = -0.5
        tick_min = np.ceil(ymin / 0.2) * 0.2
        tick_max = np.floor(ymax / 0.2) * 0.2
        ax.set_yticks(np.arange(tick_min, tick_max + 1e-9, 0.2))
        ax.set_ylim(ymin, ymax)

        ax.legend(
            fontsize=8, frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            framealpha=1.0,
            edgecolor="#cccccc",
            fancybox=False,
        )

    plt.tight_layout(pad=2.0)
    fname = os.path.join(PLOT_DIR, "msrr_oos_avg_z.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_stock_counts(emb_path=None):
    """Plot number of stocks per date for each size group."""
    from utils.path_manager import get_embeddings_path as _gep
    path = emb_path or _gep(MODELS[0])
    df = load_matched_ret_emb(path, filter_small=False)
    dates = df.index.get_level_values("date")
    

    groups = {sg: df[df["size_grp"] == sg] if sg != "all" else df for sg in SIZE_GROUPS}
    colors = {"large": "#2563eb", "mega": "#f59e0b", "all": "#6b7280"}

    fig, ax = plt.subplots(figsize=(11, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    for sg, sub in groups.items():
        counts = sub.groupby(sub.index.get_level_values("date")).size()
        ax.plot(counts.index, counts.values, label=sg, color=colors.get(sg), linewidth=1.2)

    ax.set_title("Number of stocks per date by size group", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("# stocks", fontsize=9)
    ax.legend(fontsize=8, frameon=True, framealpha=1.0, edgecolor="#cccccc", fancybox=False)

    plt.tight_layout(pad=2.0)
    fname = os.path.join(PLOT_DIR, "stock_counts.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_avg_returns():
    """Cumulative avg-strategy returns per size group, one line per model."""
    n_groups = len(SIZE_GROUPS)
    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(11, 4 * n_groups),
        facecolor="white",
        squeeze=False,
    )

    for row, sg in enumerate(SIZE_GROUPS):
        ax = axes[row, 0]
        ax.set_facecolor("white")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

        for portfolio in PORTFOLIOS:
            for model, color in zip(MODELS, MODEL_COLORS):
                csv_path = os.path.join(RAW_DIR, f"returns_{portfolio}_{model}_{sg}.csv")
                if not os.path.exists(csv_path):
                    continue
                ret_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if "avg" not in ret_df.columns:
                    continue
                cum = (1 + ret_df["avg"]).cumprod()
                label = f"{model}" + (f" [{portfolio}]" if len(PORTFOLIOS) > 1 else "")
                ax.plot(cum.index, cum.values, label=label, color=color, linewidth=1.2)

        ax.set_title(
            f"Cumulative avg-strategy return — size group: {sg}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_ylabel("Cumulative return", fontsize=9)
        ax.legend(
            fontsize=8, frameon=True, framealpha=1.0,
            edgecolor="#cccccc", fancybox=False,
            loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        )

    plt.tight_layout(pad=2.0)
    fname = os.path.join(PLOT_DIR, "avg_cumret_by_size.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    results = run_all()
    plot_results(results)
    plot_avg_returns()
