import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data.load_prices import prepare_vol_data, train_val_test_split_indices
from src.training.metrics import regression_metrics


def build_dataset_for_baseline(df: pd.DataFrame, window: int = 21):
    """
    Very simple tabular baseline dataset:

    - Features: past `window` days of realized vol (rvol).
    - Target: today's realized vol.
    """
    df = df.copy()
    if "rvol" not in df.columns:
        raise ValueError("DataFrame must contain 'rvol' column for baselines.")

    # drop early rows where rvol is NaN
    df = df.dropna(subset=["rvol"]).reset_index(drop=True)

    X_list = []
    y_list = []

    for i in range(window, len(df)):
        past_rvol = df["rvol"].iloc[i - window : i].values  # shape (window,)
        X_list.append(past_rvol)
        y_list.append(df["rvol"].iloc[i])

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def naive_baseline(y: np.ndarray) -> np.ndarray:
    """
    Naive baseline: y_hat_t = y_{t-1}. We'll implement this by shifting.
    For the first point, we just copy y_0.
    """
    y_hat = np.empty_like(y)
    y_hat[0] = y[0]
    y_hat[1:] = y[:-1]
    return y_hat


def main():
    parser = argparse.ArgumentParser(description="Week 1 baselines for vol forecasting")
    parser.add_argument("--raw_csv", type=str, required=True,
                        help="Path to raw daily OHLCV CSV file.")
    parser.add_argument("--processed_path", type=str, default="data/processed/spy_daily.parquet",
                        help="Path to save processed parquet.")
    parser.add_argument("--window", type=int, default=21,
                        help="Rolling window for realized volatility and features.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    processed_path = Path(args.processed_path)
    if not processed_path.exists():
        print(f"[INFO] Processed file not found at {processed_path}. Running preprocessing...")
        df = prepare_vol_data(
            raw_csv_path=args.raw_csv,
            processed_path=str(processed_path),
            window=args.window,
        )
    else:
        print(f"[INFO] Loading existing processed file from {processed_path}.")
        df = pd.read_parquet(processed_path)

    print(f"[INFO] Data shape after preprocessing: {df.shape}")

    # Build tabular dataset: (X, y)
    X, y = build_dataset_for_baseline(df, window=args.window)
    n = len(y)
    print(f"[INFO] Baseline dataset size: {n} samples.")

    train_end, val_end = train_val_test_split_indices(
        n, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 1) Naive baseline (shifted y)
    print("\n=== Naive baseline (shifted previous rvol) ===")
    y_hat_naive = naive_baseline(y_test)
    metrics_naive = regression_metrics(y_test, y_hat_naive)
    print("Test metrics:", metrics_naive)

    # 2) Linear regression on past rvol window
    print("\n=== Linear Regression on past rvol window ===")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_hat_lr = lr.predict(X_test)
    metrics_lr = regression_metrics(y_test, y_hat_lr)
    print("Test metrics:", metrics_lr)

    # Optional: print validation metrics too
    y_hat_lr_val = lr.predict(X_val)
    metrics_lr_val = regression_metrics(y_val, y_hat_lr_val)
    print("Val metrics (Linear Regression):", metrics_lr_val)


if __name__ == "__main__":
    main()
