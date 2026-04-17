import numpy as np
import pandas as pd


def load_jkp(file_path : str = "/Users/jschwab/Desktop/Projects/data/usa_153_per_size_ranks_False.pkl") -> pd.DataFrame: 
    
    df = pd.read_pickle(file_path)
    df = df[["id", "date", "size_grp", "r_1"]].copy()
    
    df.rename(columns = {"id": "permno"}, inplace = True)
    
    df.set_index(["permno", "date"], inplace = True)
    
    return df

def load_embeddings(file_path: str) -> pd.DataFrame:

    if "dummy" in file_path:
        return None

    df = pd.DataFrame(pd.read_pickle(file_path))

    # Expand embedding lists into columns if needed
    emb_col = df.columns[0] if isinstance(df, pd.DataFrame) else None
    if emb_col is not None and df[emb_col].dtype == object:
        arr = np.stack(df[emb_col].values)
        df = pd.DataFrame(arr, index=df.index,
                          columns=[f"emb_{i}" for i in range(arr.shape[1])])

    # Ensure index levels are named permno / date
    df.index.names = ["permno", "date"]

    # Convert year-month dates (e.g. "1994-02") to end-of-month timestamps
    date_level = df.index.get_level_values("date")
    if not pd.api.types.is_datetime64_any_dtype(date_level):
        eom_dates = pd.PeriodIndex(date_level, freq="M").to_timestamp("M")
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values("permno"), eom_dates],
            names=["permno", "date"],
        )

    return df

def load_matched_ret_emb(
    emb_path: str,
    jkp_path: str = "/Users/jschwab/Desktop/Projects/data/usa_153_per_size_ranks_False.pkl",
    emb_dim: int = 15,
) -> pd.DataFrame:

    jkp_df = load_jkp(jkp_path)
    emb_df = load_embeddings(emb_path)

    if emb_df is None:
        rng = np.random.default_rng(seed=0)
        emb_df = pd.DataFrame(
            rng.standard_normal((len(jkp_df), emb_dim)),
            index=jkp_df.index,
            columns=[f"emb_{i}" for i in range(emb_dim)],
        )
        
    merged_df = jkp_df.join(emb_df, how="inner")
    
    return merged_df
    
    
if __name__ == "__main__":
    
    load_jkp()