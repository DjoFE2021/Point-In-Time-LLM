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

MODELS          = ["chronogpt_base", "chronogpt_instruct"]
SIZE_GROUPS     = ["large", "small", "mega", "micro", "all"]
ROLLING_WINDOWS = [360]
PORTFOLIOS      = ["linear", "random_feature"]
N_RANDOM_FEAT   = 100

_BASE    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(_BASE, "results", "raw",   "msrr")
PLOT_DIR = os.path.join(_BASE, "results", "plots", "msrr")
for _d in (RAW_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

RESULTS_PATHS = {
    "linear":         os.path.join(RAW_DIR, "results_msrr_linear.pkl"),
    "random_feature": os.path.join(RAW_DIR, "results_msrr_random_feature.pkl"),
}
SPLIT_DATE      = pd.Timestamp("2013-12-31")

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


def print_table(model_results, portfolio):
    T = ROLLING_WINDOWS[0]
    table = pd.DataFrame(index=SIZE_GROUPS, columns=list(PERIODS.keys()), dtype=float)
    for sg in SIZE_GROUPS:
        if T in model_results[sg]:
            for label in PERIODS:
                table.loc[sg, label] = model_results[sg][T][label].max()
    print(f"\n    Best OOS Sharpe [{portfolio}] T={T} (max over z):")
    print(table.to_string(float_format="{:.3f}".format))
    print()


def plot_results(results):
    x = np.arange(len(SIZE_GROUPS))
    w = 0.35
    T = ROLLING_WINDOWS[0]

    for portfolio in PORTFOLIOS:
        for z_idx, z_val in enumerate(DEFAULT_SHRINKAGE_GRID):
            fig, axes = plt.subplots(
                len(PERIODS), 1,
                figsize=(8, 4 * len(PERIODS)),
                sharey=False,
            )
            fig.suptitle(f"OOS Sharpe [{portfolio}]  T={T}  —  z = {z_val:.0e}", fontsize=13)

            for row_idx, period_label in enumerate(PERIODS):
                ax = axes[row_idx]

                for m_idx, model in enumerate(MODELS):
                    sharpe_vals = [
                        results[portfolio][model][sg][T][period_label][z_idx]
                        for sg in SIZE_GROUPS
                    ]
                    offset = (m_idx - (len(MODELS) - 1) / 2) * w
                    ax.bar(x + offset, sharpe_vals, w, label=model)

                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_ylabel(f"{period_label}\nSharpe", fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels(SIZE_GROUPS, fontsize=9)
                ax.legend(fontsize=7)
                if row_idx == len(PERIODS) - 1:
                    ax.set_xlabel("Size group", fontsize=9)

            plt.tight_layout()
            fname = os.path.join(PLOT_DIR, f"msrr_{portfolio}_z{z_idx:02d}_1e{int(np.log10(z_val))}.png")
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"Saved {fname}")


if __name__ == "__main__":
    results = run_all()
    plot_results(results)
