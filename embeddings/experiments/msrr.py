import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from utils.portfolios import produce_random_feature_managed_returns_chunked
from utils.load_data import load_matched_ret_emb
from utils.path_manager import get_embeddings_path
from utils.ridge import Ridge
from utils.constants import DEFAULT_SHRINKAGE_GRID

MODELS          = ["chronogpt_base-right", "chronogpt_instruct-right", "PIT-4B-right", "PIT-4B-FT-right"]
MODEL_LABELS    = ["ChronoGPT-base", "ChronoGPT-instruct", "PIT-4B", "PIT-4B-FT"]
MODEL_COLORS    = ["#10b981", "#f59e0b", "#2563eb", "#9ca3af"]   # green, amber, blue, light grey
SIZE_GROUPS     = ["micro", "small", "large", "mega", "all"]
ROLLING_WINDOWS = [360]
PORTFOLIOS      = ["linear", "random_feature"]
PORTFOLIO_TITLES = {"linear": "Linear", "random_feature": "Random Features"}
N_RANDOM_FEAT   = 70

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


def run_experiment(df_port, rolling_window):
    """Returns {period_label: sharpe array of shape (n_z,)}."""
    ridge_regressor = Ridge()
    oos_ret, pred_dates = [], []

    for step in tqdm(range(rolling_window, len(df_port)), leave=False):
        train      = df_port.iloc[step - rolling_window: step, :]
        test       = df_port.iloc[step: step + 1, :]
        pred_date  = df_port.index[step]

        ridge_regressor.fit(train.values, np.ones(train.shape[0]))
        oos_ret.append(ridge_regressor.predict(test.values))   # (1, n_z)
        pred_dates.append(pred_date)

    oos_ret    = np.concatenate(oos_ret)          # (T, n_z)
    pred_dates = np.array(pred_dates)

    results = {}
    for label, (start, end) in PERIODS.items():
        mask = np.ones(len(pred_dates), dtype=bool)
        if start is not None:
            mask &= pred_dates > start
        if end is not None:
            mask &= pred_dates <= end
        results[label] = compute_sharpe(oos_ret[mask])

    return results  # {period_label: array(n_z)}


def run_all():
    # results[portfolio][model][size_grp][T] = {period: array(n_z)}
    results = {p: {m: {sg: {} for sg in SIZE_GROUPS} for m in MODELS} for p in PORTFOLIOS}

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

    for portfolio in PORTFOLIOS:
        print(f"\n{'='*50}\nPortfolio: {portfolio}")

        for model in MODELS:
            all_done = all(
                T in results[portfolio][model][sg]
                for sg in SIZE_GROUPS for T in ROLLING_WINDOWS
            )
            if all_done:
                print(f"  Skipping {model} (already complete)")
                continue

            print(f"\n  === Model: {model} ===")
            df_full = load_matched_ret_emb(get_embeddings_path(model))
            df_full = df_full.sort_index()

            for size_grp in SIZE_GROUPS:
                df = df_full[df_full["size_grp"] == size_grp] if size_grp != "all" else df_full

                print(f"    Building {portfolio} portfolios for size_grp={size_grp} ...")
                df_port = build_portfolio(df, portfolio)

                for T in ROLLING_WINDOWS:
                    if T in results[portfolio][model][size_grp]:
                        print(f"      Skipping T={T} (already computed)")
                        continue
                    print(f"      T={T}")
                    results[portfolio][model][size_grp][T] = run_experiment(df_port, T)
                    print_table(results[portfolio][model], portfolio)
                    with open(RESULTS_PATHS[portfolio], "wb") as f:
                        pickle.dump(results[portfolio], f)

    return results


def _avg_sharpe(results, portfolio, model, sg, T, period):
    """Average Sharpe across all z values on the shrinkage grid."""
    arr = results[portfolio][model][sg][T][period]   # shape (n_z,)
    return float(np.mean(arr))


def print_table(model_results, portfolio):
    T = ROLLING_WINDOWS[0]
    table = pd.DataFrame(index=SIZE_GROUPS, columns=list(PERIODS.keys()), dtype=float)
    for sg in SIZE_GROUPS:
        if T in model_results[sg]:
            for label in PERIODS:
                table.loc[sg, label] = np.mean(model_results[sg][T][label])
    print(f"\n    Avg Sharpe (mean over z grid) [{portfolio}] T={T}:")
    print(table.to_string(float_format="{:.3f}".format))
    print()


def plot_results(results):
    T        = ROLLING_WINDOWS[0]
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

        for m_idx, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLORS)):
            sharpe_vals = [
                _avg_sharpe(results, portfolio, model, sg, T, period)
                for sg in SIZE_GROUPS
            ]
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

        legend_loc = "lower right" if portfolio == "linear" else "upper right"
        ax.legend(
            fontsize=8, frameon=True,
            loc=legend_loc,
            framealpha=1.0,
            edgecolor="#cccccc",
            fancybox=False,
        )

    plt.tight_layout(pad=2.0)
    fname = os.path.join(PLOT_DIR, "msrr_oos_avg_z.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    results = run_all()
    plot_results(results)
