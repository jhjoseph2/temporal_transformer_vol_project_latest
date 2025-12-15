import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

# Re-use your project's logic to ensure exact data matching
from src.config import ExperimentConfig
from src.data.vol_dataset import TimeSeriesVolDataset
from src.data.splitter import ExpandingWindowSplitter
from src.utils.logging import get_logger

logger = get_logger("baseline_wfv")

def get_numpy_data(dataset):
    """
    Extracts X and y from the PyTorch dataset into Numpy arrays 
    compatible with Scikit-Learn.
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # Get all data in one big batch
    for x, y in loader:
        # x shape: [Batch, Seq, Feat] -> Flatten for Linear Reg: [Batch, Seq*Feat]
        X_flat = x.numpy().reshape(x.shape[0], -1)
        y_flat = y.numpy()
        
        # Also extract the "Last Rvol" for the Naive baseline
        # Assuming Feature 1 is 'rvol' (check your config/dataset columns!)
        # x is [Batch, Seq, Features]. If rvol is index 1:
        # naive_pred = x[:, -1, 1] 
        # But to be safe, let's assume Naive = "Value at t-1" which is usually the last target.
        # Actually, standard Naive Vol is: Vol_t = Vol_{t-1}. 
        # In our dataset, X includes rvol. Let's assume the last column is rvol.
        last_val = x[:, -1, -1].numpy() 
        
        return X_flat, y_flat, last_val
    return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/spy_daily.parquet")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    
    # 1. Load Data
    if not Path(args.data_path).exists():
        logger.error("Data not found.")
        return
        
    df = pd.read_parquet(args.data_path)
    feature_cols = ["log_ret", "rvol"]
    target_col = "rvol"
    
    # 2. Setup Same Splitter as Transformer
    # MUST MATCH run_walk_forward.py EXACTLY
    splitter = ExpandingWindowSplitter(
        total_samples=len(df),
        initial_train_size=500, 
        test_size=120  # Use the Fixed size from your Day 7 update
    )

    # Store results
    naive_preds = []
    linear_preds = []
    actuals = []

    logger.info("Starting Walk-Forward Validation for Baselines...")

    for fold, (train_idx, val_idx, test_idx) in enumerate(splitter.split()):
        # We only care about Train and Test for Baselines (No Validation needed for Linear Regression usually)
        
        # Build Datasets (Re-using your existing class guarantees alignment)
        train_ds = TimeSeriesVolDataset(df, feature_cols, target_col, lookback=cfg.lookback, indices=train_idx)
        test_ds = TimeSeriesVolDataset(df, feature_cols, target_col, lookback=cfg.lookback, indices=test_idx)

        # Convert to Numpy for Sklearn
        X_train, y_train, _ = get_numpy_data(train_ds)
        X_test, y_test, naive_test = get_numpy_data(test_ds)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # --- MODEL 1: LINEAR REGRESSION (Ridge) ---
        # We use Ridge to prevent overfitting on the flattened high-dim features
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        pred_lin = model.predict(X_test)
        
        # --- MODEL 2: NAIVE (Random Walk) ---
        # Predict y_t = y_{t-1} (The last known rvol in the input window)
        pred_naive = naive_test

        # Accumulate
        linear_preds.append(pred_lin)
        naive_preds.append(pred_naive)
        actuals.append(y_test)
        
        logger.info(f"Fold {fold}: Processed {len(y_test)} samples.")

    # 3. Save Combined Results
    # Flatten list of arrays
    final_linear = np.concatenate(linear_preds)
    final_naive = np.concatenate(naive_preds)
    final_actual = np.concatenate(actuals)

    # Save Linear
    df_lin = pd.DataFrame({'actual': final_actual, 'predicted': final_linear})
    df_lin.to_csv("results/wfv_linear.csv", index=False)
    
    # Save Naive
    df_naive = pd.DataFrame({'actual': final_actual, 'predicted': final_naive})
    df_naive.to_csv("results/wfv_naive.csv", index=False)
    
    logger.info(f"Saved wfv_linear.csv and wfv_naive.csv with {len(final_actual)} rows.")

if __name__ == "__main__":
    main()