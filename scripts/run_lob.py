import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from src.models.transformer_vol import VolTransformer
from src.data.lob_dataset import LOBDataset
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger("lob_experiment")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=str, required=True, choices=["sinusoidal", "time2vec", "alibi", "ctlpe"])
    parser.add_argument("--train_file", type=str, required=True, help="Path to Training Fold .txt")
    parser.add_argument("--test_file", type=str, required=True, help="Path to Testing Fold .txt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # 1. Load Data (Explicit Train/Test Files)
    # Using Fold 9 is recommended for the most data
    logger.info("Initializing Datasets...")
    train_ds = LOBDataset(args.train_file, lookback=50, horizon=10)
    test_ds = LOBDataset(args.test_file, lookback=50, horizon=10)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True) # Larger batch for HFT
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # 2. Initialize Model
    # n_features=4 because we extracted Bid/Ask Price/Vol (Level 1)
    model = VolTransformer(
        n_features=4, 
        d_model=64, 
        n_layers=2, 
        n_heads=4, 
        embedding_type=args.embedding,
        dropout=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    logger.info(f"Starting {args.embedding} experiment on {device}...")

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            pred = model(x, t)
            loss = loss_fn(pred, y.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                t = batch['t'].to(device)
                y = batch['y'].to(device)
                pred = model(x, t)
                val_loss += loss_fn(pred, y.squeeze(-1)).item()
                
        logger.info(f"Epoch {epoch+1} | Train MSE: {train_loss/len(train_loader):.7f} | Test MSE: {val_loss/len(test_loader):.7f}")

    # 4. Save Results (Optional but good)
    save_path = Path(f"results/lob_{args.embedding}_results.csv")
    save_path.parent.mkdir(exist_ok=True)
    # Logic to save preds if needed...

if __name__ == "__main__":
    main()