from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_raw_csv(path: str) -> pd.DataFrame:
    """
    Load a raw daily OHLCV CSV file.

    Expected columns (at minimum):
    - Date
    - Open
    - High
    - Low
    - Close
    - Volume
    """
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Input CSV must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    if price_col not in df.columns:
        raise ValueError(f"{price_col} column not found in DataFrame.")
    df["log_ret"] = np.log(df[price_col]).diff()
    return df


def compute_realized_volatility(
    df: pd.DataFrame,
    ret_col: str = "log_ret",
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute a simple realized volatility proxy:
    rvol_t = sqrt( sum_{i=t-window+1}^{t} r_i^2 ).

    This is a rolling measure over 'window' days.
    """
    df = df.copy()
    if ret_col not in df.columns:
        raise ValueError(f"{ret_col} column not found in DataFrame.")
    rv = (df[ret_col] ** 2).rolling(window=window).sum()
    df["rvol"] = np.sqrt(rv)
    return df


def prepare_vol_data(
    raw_csv_path: str,
    processed_path: str,
    price_col: str = "Close",
    window: int = 21,
) -> pd.DataFrame:
    """
    Full pipeline for Week 1:

    1. Load raw prices
    2. Compute log returns
    3. Compute realized volatility
    4. Save to parquet at `processed_path`

    Returns the processed DataFrame.
    """
    df = load_raw_csv(raw_csv_path)
    df = compute_log_returns(df, price_col=price_col)
    df = compute_realized_volatility(df, ret_col="log_ret", window=window)

    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)

    return df


def train_val_test_split_indices(
    n: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[int, int]:
    """
    Return integer indices (train_end, val_end), where:
    - train indices: [0, train_end)
    - val indices:   [train_end, val_end)
    - test indices:  [val_end, n)
    """
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end
