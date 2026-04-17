import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from utils.load_data import load_matched_ret_emb
from utils.path_manager import get_embeddings_path
from utils.ridge import Ridge
from utils.constants import DEFAULT_SHRINKAGE_GRID

MODELS          = ["chronogpt_base", "chronogpt_instruct"]
SIZE_GROUPS     = ["large", "small", "mega", "micro", "all"]
ROLLING_WINDOWS = [12]
METHODS         = ["linear"]#, "random_feature"]
N_RANDOM_FEAT   = 2000          # P=10k risks OOM for large groups; 2k is safe
SPLIT_DATE      = pd.Timestamp("2013-12-31")

_BASE    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(_BASE, "results", "raw",   "mse")
PLOT_DIR = os.path.join(_BASE, "results", "plots", "mse")
for _d in (RAW_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

RESULTS_PATHS = {
    "linear":         os.path.join(RAW_DIR, "results_mse_linear.pkl"),
    "random_feature": os.path.join(RAW_DIR, "results_mse_random_feature.pkl"),
}

PERIODS = {
    "Full sample":   (None, None),
    "In-sample":     (None, SPLIT_DATE),
    "Out-of-sample": (SPLIT_DATE, None),
}

METRICS = {
    "mse": {"label": "MSE", "best": "min"},
    "r2":  {"label": "R²",  "best": "max"},
}


def compute_mse(predictions, true):
    return np.mean((predictions - true) ** 2, axis=0)   # (n_z,)


def compute_r2(predictions, true):
    ss_res = np.sum((predictions - true) ** 2, axis=0)  # (n_z,)
    ss_tot = np.sum(true**2, axis=0)
    return 1 - ss_res / ss_tot                           # (n_z,)


def make_omega(d, P, seed=0):
    """Random feature projection matrix (P × d), fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((P, d)) * (2.0 / d) ** 0.5).astype(np.float32)


def apply_rf(X, omega):
    """ReLU random features: (N, d) → (N, P)."""
    return np.maximum(0, X.astype(np.float32) @ omega.T)


def run_experiment(df, rolling_window, omega=None):
    """
    omega=None  → linear (raw embeddings)
    omega=(P,d) → random feature ridge

    Returns {period: {"mse": array(n_z), "r2": array(n_z)}}.
    """
    ridge_regressor = Ridge()
    hat_ret, true_ret, pred_dates = [], [], []

    df = df.sort_index()
    dates = df.index.get_level_values("date").unique().sort_values()

    for step in tqdm(range(rolling_window, len(dates)), leave=False):
        date_start     = dates[step - rolling_window]
        date_end_train = dates[step - 1]
        date_predict   = dates[step]

        train     = df.loc[(slice(None), slice(date_start, date_end_train)), :]
        X_train_r = train.drop(columns=["r_1", "size_grp"]).values
        y_train   = train["r_1"].values

        X_test_r  = df.loc[(slice(None), date_predict), :].drop(columns=["r_1", "size_grp"]).values
        n_stocks  = X_test_r.shape[0]

        if omega is not None:
            X_train = apply_rf(X_train_r, omega)
            X_test  = apply_rf(X_test_r,  omega)
        else:
            X_train = X_train_r
            X_test  = X_test_r

        ridge_regressor.fit(X_train, y_train)
        hat_ret.append(ridge_regressor.predict(X_test))
        true_ret.append(df.loc[(slice(None), date_predict), "r_1"].values)
        pred_dates.extend([date_predict] * n_stocks)

    hat_ret    = np.concatenate(hat_ret)                # (N_total, n_z)
    true_ret   = np.concatenate(true_ret).reshape(-1, 1)
    pred_dates = np.array(pred_dates)

    results = {}
    for label, (start, end) in PERIODS.items():
        mask = np.ones(len(pred_dates), dtype=bool)
        if start is not None:
            mask &= pred_dates > start
        if end is not None:
            mask &= pred_dates <= end
        results[label] = {
            "mse": compute_mse(hat_ret[mask], true_ret[mask]),
            "r2":  compute_r2(hat_ret[mask],  true_ret[mask]),
        }

    return results


def run_all():
    # results[method][model][size_grp][T] = {period: {"mse": arr, "r2": arr}}
    results = {mth: {m: {sg: {} for sg in SIZE_GROUPS} for m in MODELS} for mth in METHODS}

    for mth in METHODS:
        path = RESULTS_PATHS[mth]
        if os.path.exists(path):
            with open(path, "rb") as f:
                saved = pickle.load(f)
            for m in MODELS:
                for sg in SIZE_GROUPS:
                    if m in saved and sg in saved[m]:
                        results[mth][m][sg] = saved[m][sg]
            print(f"Loaded existing results from {path}")

    for method in METHODS:
        print(f"\n{'='*50}\nMethod: {method}")

        for model in MODELS:
            all_done = all(
                T in results[method][model][sg]
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

                # Generate omega once per (model, size_grp) — reused across all T
                if method == "random_feature":
                    d     = df.drop(columns=["r_1", "size_grp"]).shape[1]
                    omega = make_omega(d, N_RANDOM_FEAT)
                else:
                    omega = None

                for T in ROLLING_WINDOWS:
                    if T in results[method][model][size_grp]:
                        print(f"    Skipping size_grp={size_grp}  T={T} (already computed)")
                        continue
                    print(f"    size_grp={size_grp}  T={T}")
                    results[method][model][size_grp][T] = run_experiment(df, T, omega)
                    print_table(results[method][model], method)
                    with open(RESULTS_PATHS[method], "wb") as f:
                        pickle.dump(results[method], f)

    return results


def print_table(model_results, method):
    table = pd.DataFrame(index=SIZE_GROUPS, columns=ROLLING_WINDOWS, dtype=float)
    for sg in SIZE_GROUPS:
        for T in ROLLING_WINDOWS:
            if T in model_results[sg]:
                table.loc[sg, T] = model_results[sg][T]["Full sample"]["mse"].min()
    print(f"\n    Best MSE [{method}] — Full sample (min over z):")
    print(table.to_string(float_format="{:.5f}".format))
    print()


def _plot_metric(results, metric, method):
    meta = METRICS[metric]
    x    = np.arange(len(ROLLING_WINDOWS))
    w    = 0.35

    for z_idx, z_val in enumerate(DEFAULT_SHRINKAGE_GRID):
        fig, axes = plt.subplots(
            len(PERIODS), len(SIZE_GROUPS),
            figsize=(4 * len(SIZE_GROUPS), 4 * len(PERIODS)),
            sharey="row",
        )
        fig.suptitle(f"OOS {meta['label']} [{method}]  —  z = {z_val:.0e}", fontsize=13)

        for row_idx, period_label in enumerate(PERIODS):
            for col_idx, size_grp in enumerate(SIZE_GROUPS):
                ax = axes[row_idx, col_idx]

                for m_idx, model in enumerate(MODELS):
                    vals = [
                        results[method][model][size_grp][T][period_label][metric][z_idx]
                        for T in ROLLING_WINDOWS
                    ]
                    offset = (m_idx - (len(MODELS) - 1) / 2) * w
                    ax.bar(x + offset, vals, w, label=model)

                if metric == "r2":
                    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                if row_idx == 0:
                    ax.set_title(size_grp, fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(f"{period_label}\n{meta['label']}", fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels([str(T) for T in ROLLING_WINDOWS], fontsize=8)
                if row_idx == len(PERIODS) - 1:
                    ax.set_xlabel("Rolling window (months)", fontsize=9)
                if row_idx == 0 and col_idx == len(SIZE_GROUPS) - 1:
                    ax.legend(fontsize=7)

        plt.tight_layout()
        fname = os.path.join(PLOT_DIR, f"{metric}_{method}_z{z_idx:02d}_1e{int(np.log10(z_val))}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")


def plot_results(results):
    for method in METHODS:
        for metric in METRICS:
            _plot_metric(results, metric, method)


if __name__ == "__main__":
    results = run_all()
    plot_results(results)
