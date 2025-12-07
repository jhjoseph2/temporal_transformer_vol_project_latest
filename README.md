# Transformer Temporal Embeddings for Volatility Forecasting (Skeleton)

This is a lightweight project skeleton for experimenting with temporal embeddings
in Transformer-based models for realized volatility forecasting.

## Structure

- `data/raw/` — put your raw price CSV here (e.g. `spy_daily.csv`).
- `data/processed/` — processed parquet with log returns and realized volatility.
- `src/`
  - `config.py` — global configuration dataclass.
  - `utils/seed.py` — seed setting utility.
  - `utils/logging.py` — basic logger helper.
  - `data/load_prices.py` — data loading and realized volatility computation.
  - `data/vol_dataset.py` — PyTorch Dataset for sequence-to-one vol forecasting.
  - `models/time_embeddings.py` — (stubs) different temporal embedding modules.
  - `models/transformer_vol.py` — (stub) Transformer model for vol forecasting.
  - `training/metrics.py` — basic regression metrics (MSE/MAE).
  - `training/loop.py` — (stub) generic training loop for Transformer.
- `scripts/`
  - `run_baselines.py` — Week 1 baselines (naive and simple regression).
  - `train_transformer.py` — (stub) training script for Transformer experiments.
- `notebooks/`
  - empty placeholder for your own EDA / experiments.

## Quick start (Week 1)

1. Put a daily price CSV (with at least columns `Date, Open, High, Low, Close, Volume`)
   into `data/raw/spy_daily.csv`.
2. Run the preprocessing and baselines:

   ```bash
   python scripts/run_baselines.py \
       --raw_csv data/raw/spy_daily.csv \
       --processed_path data/processed/spy_daily.parquet
   ```

3. Inspect the printed baseline metrics and then iterate.
