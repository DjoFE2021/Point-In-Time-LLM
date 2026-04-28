import gc
import os
import numpy as np
import pandas as pd
import torch

import load_dotenv

load_dotenv.load_dotenv()  # Load environment variables from .env file

_JKP_PATH = os.environ["JKP_PANEL_PATH"]
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
    df = pd.read_pickle(file_path)
    if filter_small:
        df = df[df.size_grp.isin(["large", "mega"])]
    df.rename(columns={"id": "permno"}, inplace=True)
    df.set_index(["permno", "date"], inplace=True)
    return df


def load_embeddings(file_path: str) -> pd.DataFrame:
    if "dummy" in file_path:
        return None

    df = pd.DataFrame(pd.read_pickle(file_path))

    emb_col = df.columns[0] if isinstance(df, pd.DataFrame) else None
    if emb_col is not None and df[emb_col].dtype == object:
        arr = np.stack(df[emb_col].values)
        df = pd.DataFrame(arr, index=df.index,
                          columns=[f"emb_{i}" for i in range(arr.shape[1])])

    if "mistral" in file_path:
        df.index.names = ["date", "permno"]
    else:
        df.index.names = ["permno", "date"]

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
    residualize: bool = False,
    filter_small: bool = True,
) -> pd.DataFrame:
    meta_cols = ["r_1", "size_grp"]

    jkp_df = load_jkp(jkp_path, filter_small=filter_small)
    emb_df = load_embeddings(emb_path)

    if emb_df is None:
        rng = np.random.default_rng(seed=0)
        emb_df = pd.DataFrame(
            rng.standard_normal((len(jkp_df), emb_dim)),
            index=jkp_df.index,
            columns=[f"emb_{i}" for i in range(emb_dim)],
        )

    emb_cols = list(emb_df.columns)

    # Align indices without building a fat merged DataFrame
    common_idx = jkp_df.index.intersection(emb_df.index)
    emb_sub = emb_df.loc[common_idx].copy()
    del emb_df
    gc.collect()

    if residualize:
        jkp_factor_cols = [c for c in jkp_df.columns if c not in meta_cols]

        # Sort rows by date so day blocks are contiguous
        date_level = common_idx.names.index("date") if "date" in common_idx.names else 1
        date_vals = common_idx.get_level_values(date_level).to_numpy()
        sort_order = np.argsort(date_vals, kind="stable")
        sorted_dates = date_vals[sort_order]
        change = np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1
        starts = np.r_[0, change]
        ends   = np.r_[change, len(common_idx)]

        # Build regressors as plain float32 numpy — never keep as a DF
        jkp_sub = jkp_df.loc[common_idx]
        S_np = np.hstack([
            jkp_sub.iloc[sort_order][jkp_factor_cols].to_numpy(dtype=np.float32, copy=True),
            np.ones((len(common_idx), 1), dtype=np.float32),   # intercept
        ])
        E_np = emb_sub.iloc[sort_order].to_numpy(dtype=np.float32, copy=True)

        # Keep only meta cols from JKP, then free the full JKP block
        meta_sub = jkp_df.loc[common_idx, meta_cols].copy()
        del jkp_df, jkp_sub
        gc.collect()

        E_resid = _resid_by_day_cuda(E_np, S_np, starts, ends)
        del E_np, S_np
        gc.collect()

        inv_order = np.argsort(sort_order)
        emb_sub = pd.DataFrame(E_resid[inv_order], index=common_idx, columns=emb_cols)
        del E_resid
        gc.collect()
    else:
        meta_sub = jkp_df.loc[common_idx, meta_cols].copy()
        del jkp_df
        gc.collect()

    result = pd.concat([meta_sub, emb_sub], axis=1)

    if demean or standardize:
        dates = result.index.get_level_values("date")
        group_keys = [dates, result["size_grp"]]
        if standardize:
            result[emb_cols] = result.groupby(group_keys)[emb_cols].transform(
                lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
            )
        else:
            result[emb_cols] = result.groupby(group_keys)[emb_cols].transform(
                lambda x: x - x.mean()
            )

    return result
