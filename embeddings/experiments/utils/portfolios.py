import pandas as pd
import numpy as np


def produce_random_feature_managed_returns_chunked(
    P,
    r1,
    signals,
    num_seeds=2,
    scale=1.0,
    activation="relu",
    base_seed=0,
    device=None,  # kept for API compatibility, ignored (CPU-only)
    dtype=None,   # kept for API compatibility, ignored
):
    """
    RAM-safe random-feature managed returns (NumPy implementation).

    RFs are generated in chunks and immediately collapsed to T×P.
    No NT×(P·num_seeds) object is ever stored.

    Returns
    -------
    managed_returns : DataFrame (T × (P·num_seeds))
    """
    X = np.array(signals.values if isinstance(signals, pd.DataFrame) else signals, dtype=np.float32)
    r = np.array(r1.values, dtype=np.float32).reshape(-1, 1)

    dates = r1.index.get_level_values("date")

    NT, d = X.shape
    sqrt2_over_d = (2.0 / d) ** 0.5

    managed_blocks = []

    for seed in range(num_seeds):
        rng = np.random.default_rng(base_seed + seed)

        # --- RF weights (P × d)
        omega = scale * rng.standard_normal((P, d)).astype(np.float32) * sqrt2_over_d

        # --- RF activations (NT × P)
        Z = X @ omega.T

        if activation == "relu":
            Z = np.maximum(0, Z)
        elif activation == "sincos":
            Z = np.concatenate([np.sqrt(2.0) * np.sin(Z), np.sqrt(2.0) * np.cos(Z)], axis=1)
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        # --- managed returns: collapse N → T immediately
        weighted = Z * r
        df = pd.DataFrame(weighted, index=r1.index)

        managed = df.groupby(dates).mean()
        managed_blocks.append(managed)

        del omega, Z, weighted, df
        
    result = pd.concat(managed_blocks, axis=1)
    result.columns = [f"RF_{i}" for i in range(result.shape[1])]

    return result
