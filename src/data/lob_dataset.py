import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class LOBDataset(Dataset):
    def __init__(self, txt_path, lookback=50, horizon=10):
        """
        Args:
            txt_path: Path to a specific 'DecPre' .txt file (e.g. Train_Dst_NoAuction_DecPre_CF_7.txt)
            lookback: Sequence length (e.g. 50 ticks)
            horizon: Prediction horizon (e.g. 10 ticks ahead)
        """
        self.lookback = lookback
        self.horizon = horizon
        
        print(f"Loading FI-2010 Fold: {txt_path}...")
        # 1. Load & Transpose
        # Original: [149, Ticks] -> Transpose to [Ticks, 149]
        raw_data = np.loadtxt(txt_path).T
        
        # 2. Extract Features (Rows 1-144 in original, now Cols 0-143)
        # FI-2010 DecPre Standard Layout for first 4 columns:
        # 0: Ask Price 1
        # 1: Ask Volume 1
        # 2: Bid Price 1
        # 3: Bid Volume 1
        self.ask_price = raw_data[:, 0]
        self.ask_vol   = raw_data[:, 1]
        self.bid_price = raw_data[:, 2]
        self.bid_vol   = raw_data[:, 3]
        
        # Calculate Mid Price (Crucial for Regression Targets)
        # Note: DecPre prices are usually scaled (e.g. by 10000), but 
        # Log Returns are scale-invariant, so this is safe.
        self.mid_price = (self.ask_price + self.bid_price) / 2.0
        
        # 3. Construct Input Features
        # We Normalize inputs relative to the Mid-Price to ensure stationarity
        # (Even though data is pre-normalized, relative scaling is safer for DL)
        self.features = np.stack([
            (self.bid_price / self.mid_price) - 1, # Normalized Bid
            (self.ask_price / self.mid_price) - 1, # Normalized Ask
            np.log1p(self.bid_vol),                # Log Bid Size
            np.log1p(self.ask_vol)                 # Log Ask Size
        ], axis=1).astype(np.float32)

        # 4. Create "Event Time"
        # FI-2010 is an Event-Based dataset. The "Time" is just the tick count.
        # Time2Vec/CTLPE will learn to embed the integer sequence index.
        self.timestamps = np.arange(len(self.features), dtype=np.float32)

        # 5. Create Regression Targets (Log Returns)
        # We IGNORE the classification labels (Cols 144-148)
        # Target = log( Price[t+h] / Price[t] )
        mid_series = pd.Series(self.mid_price)
        returns = np.log(mid_series.shift(-horizon) / mid_series)
        self.targets = returns.fillna(0).values.astype(np.float32)

        print(f"  > Loaded {len(self.features)} events.")
        print(f"  > Input Shape: {self.features.shape} (Used Level 1 Only)")

    def __len__(self):
        return len(self.features) - self.lookback - self.horizon

    def __getitem__(self, idx):
        # 1. Feature Window
        x = self.features[idx : idx + self.lookback]
        
        # 2. Time Window (Event Time)
        # Normalize to start at 0 for this window
        t_raw = self.timestamps[idx : idx + self.lookback]
        t_norm = (t_raw - t_raw[0]) # 0, 1, 2...
        
        # Scale down so values aren't huge (helps neural net stability)
        # For event time, dividing by lookback puts it in [0, 1] range
        t_norm = t_norm / self.lookback
        
        # 3. Target
        y = self.targets[idx + self.lookback - 1]
        
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            't': torch.tensor(t_norm, dtype=torch.float32).unsqueeze(-1),
            'y': torch.tensor([y], dtype=torch.float32)
        }