"""Load the JKP stock panel and monthly text embeddings, align them on (permno, date).

Optionally residualizes embeddings per date on GPU; needs JKP_PANEL_PATH (env or .env).
"""
import gc
import os
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_FILE)  # embeddings/experiments/.env; already-set variables take precedence

_JKP_PATH = os.environ.get("JKP_PANEL_PATH")
if not _JKP_PATH:
    raise RuntimeError(f"JKP_PANEL_PATH is not set: add it to {_ENV_FILE} (see .env.example)")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resid_I_minus_S_ridge_torch(
    E: torch.Tensor,
    S: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project out the column space of S from E via ridge-stabilised OLS."""
    St = S.T
    G = St @ S + eps * torch.eye(S.shape[1], device=S.device, dtype=S.dtype)
    beta = torch.linalg.solve(G, St @ E)
    return E - S @ beta


def _resid_by_day_cuda(
    E_np: np.ndarray,    # (N, n_e) float32
    S_np: np.ndarray,    # (N, n_s) float32
    starts: np.ndarray,
    ends: np.ndarray,
    eps: float = 1e-8,
    dtype=torch.float32,
) -> np.ndarray:
    """Cross-sectionally residualise E against S within each date block, on GPU."""
    out = np.empty_like(E_np)
    with torch.no_grad():
        for a, b in zip(starts, ends):
            E_t = torch.nan_to_num(
                torch.as_tensor(E_np[a:b], device=_DEVICE, dtype=dtype), nan=0.0
            )
            S_t = torch.nan_to_num(
                torch.as_tensor(S_np[a:b], device=_DEVICE, dtype=dtype), nan=0.0
            )
            out[a:b] = _resid_I_minus_S_ridge_torch(E_t, S_t, eps=eps).cpu().numpy()
    return out


def load_jkp(file_path: str = _JKP_PATH, filter_small: bool = True) -> pd.DataFrame:
    """Load the JKP panel indexed by (permno, date); filter_small keeps only large/mega caps."""
    df = pd.read_pickle(file_path)
    if filter_small:
        df = df[df.size_grp.isin(["large", "mega"])]
    df.rename(columns={"id": "permno"}, inplace=True)
    df.set_index(["permno", "date"], inplace=True)
    return df


def load_embeddings(file_path: str) -> pd.DataFrame:
    """Load monthly embeddings as a (permno, month-end date) × emb_i DataFrame."""
    df = pd.DataFrame(pd.read_pickle(file_path))

    # Expand a column of embedding vectors into one column per dimension
    emb_col = df.columns[0] if isinstance(df, pd.DataFrame) else None
    if emb_col is not None and df[emb_col].dtype == object:
        arr = np.stack(df[emb_col].values)
        df = pd.DataFrame(arr, index=df.index,
                          columns=[f"emb_{i}" for i in range(arr.shape[1])])

    df.index.names = ["permno", "date"]

    # Month labels (e.g. "YYYY-MM") → month-end timestamps
    date_level = df.index.get_level_values("date")
    if not pd.api.types.is_datetime64_any_dtype(date_level):
        eom_dates = pd.PeriodIndex(date_level, freq="M").to_timestamp("M")
    else:
        eom_dates = date_level

    # Always normalise to [permno, date] order
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values("permno"), eom_dates],
        names=["permno", "date"],
    )

    return df


def load_matched_ret_emb(
    emb_path: str,
    jkp_path: str = _JKP_PATH,
    emb_dim: int = 15,
    demean: bool = False,
    standardize: bool = False,
    residualize: bool | str = False,
    filter_small: bool = True,
) -> pd.DataFrame:
    """
    Return a (permno, date) DataFrame with r_1, size_grp and the embedding columns.

    residualize options:
      False / "none" — no residualization
      True  / "JKP+1" — JKP factors + intercept  (legacy default)
      "1"            — intercept only
      "JKP"          — JKP factors only, no intercept
    """
    meta_cols = ["r_1", "size_grp"]

    jkp_df = load_jkp(jkp_path, filter_small=filter_small)
    emb_df = load_embeddings(emb_path)

    emb_cols = list(emb_df.columns)

    # Align indices without building a fat merged DataFrame
    common_idx = jkp_df.index.intersection(emb_df.index)
    emb_sub = emb_df.loc[common_idx].copy()
    del emb_df
    gc.collect()

    # Normalise residualize to a string token
    if residualize is True:
        residualize = "JKP+1"
    elif residualize is False:
        residualize = "none"

    if residualize != "none":
        jkp_factor_cols = [c for c in jkp_df.columns if c not in meta_cols]

        # Sort rows by date so each date's rows form a contiguous block
        date_level = common_idx.names.index("date") if "date" in common_idx.names else 1
        date_vals = common_idx.get_level_values(date_level).to_numpy()
        sort_order = np.argsort(date_vals, kind="stable")
        sorted_dates = date_vals[sort_order]
        change = np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1
        starts = np.r_[0, change]
        ends   = np.r_[change, len(common_idx)]

        # Build regressors as plain float32 numpy arrays (not DataFrames) to save memory
        jkp_sub = jkp_df.loc[common_idx]
        parts = []
        if residualize in ("JKP", "JKP+1"):
            parts.append(
                jkp_sub.iloc[sort_order][jkp_factor_cols].to_numpy(dtype=np.float32, copy=True)
            )
        if residualize in ("1", "JKP+1"):
            parts.append(np.ones((len(common_idx), 1), dtype=np.float32))
        S_np = np.hstack(parts)
        E_np = emb_sub.iloc[sort_order].to_numpy(dtype=np.float32, copy=True)

        # Keep only meta cols from JKP, then free the full JKP block
        meta_sub = jkp_df.loc[common_idx, meta_cols].copy()
        del jkp_df, jkp_sub
        gc.collect()

        E_resid = _resid_by_day_cuda(E_np, S_np, starts, ends)
        del E_np, S_np
        gc.collect()

        # Undo the date sort to restore common_idx order
        inv_order = np.argsort(sort_order)
        emb_sub = pd.DataFrame(E_resid[inv_order], index=common_idx, columns=emb_cols)
        del E_resid
        gc.collect()
    else:  # "none"
        meta_sub = jkp_df.loc[common_idx, meta_cols].copy()
        del jkp_df
        gc.collect()

    result = pd.concat([meta_sub, emb_sub], axis=1)

    # Optional cross-sectional demeaning / standardization per date
    if demean or standardize:
        dates = result.index.get_level_values("date")
        group_keys = [dates]
        if standardize:
            result[emb_cols] = result.groupby(group_keys)[emb_cols].transform(
                lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
            )
        else:
            result[emb_cols] = result.groupby(group_keys)[emb_cols].transform(
                lambda x: x - x.mean()
            )

    return result
