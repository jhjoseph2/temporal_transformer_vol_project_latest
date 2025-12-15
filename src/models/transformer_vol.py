import torch
import torch.nn as nn
from typing import Literal, Optional

# Import the ingredients we just prepared
from .time_embeddings import (
    SinusoidalPositionalEncoding, 
    LearnedPositionalEncoding, 
    Time2Vec, 
    CTLPE,
    get_alibi_bias
)

class VolTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
        embedding_type: str = 'sinusoidal'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding_type = embedding_type.lower()

        # 1. Input Projection Layer
        self.input_linear = nn.Linear(n_features, d_model)
        
        # 2. Initialize the specific embedding module (if needed)
        self.time_embedding = None
        if self.embedding_type == 'sinusoidal':
            self.time_embedding = SinusoidalPositionalEncoding(d_model)
        elif self.embedding_type == 'learned':
            self.time_embedding = LearnedPositionalEncoding(d_model)
        elif self.embedding_type == 'time2vec':
            # Time2Vec maps 1 time feature -> d_model dims
            self.time_embedding = Time2Vec(d_model)
        elif self.embedding_type == 'ctlpe':
            self.time_embedding = CTLPE(d_model)
        # Note: ALiBi requires no registered embedding layer, it uses a dynamic mask

        # 3. The Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Better stability for time series
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Output Head
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch, seq_len, n_features]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # --- 1. Standard Transformer Forward Pass ---
        # A. Projection
        x_emb = self.input_linear(x)

        # B. Temporal Embeddings
        if self.embedding_type in ['time2vec', 'ctlpe']:
            if t is None:
                t = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0).unsqueeze(-1)
                t = t.expand(batch_size, -1, -1)
            x_emb = x_emb + self.time_embedding(t)
        elif self.embedding_type == 'sinusoidal':
            x_emb = self.time_embedding(x_emb)
        elif self.embedding_type == 'learned':
            x_emb = self.time_embedding(x_emb)
            
        # C. ALiBi Mask
        src_mask = None
        if self.embedding_type == 'alibi':
            alibi_bias = get_alibi_bias(seq_len, self.n_heads, device)
            src_mask = alibi_bias.repeat(batch_size, 1, 1)

        # D. Encoder
        x_enc = self.encoder(x_emb, mask=src_mask)

        # E. Prediction Head
        last_token = x_enc[:, -1, :]
        model_out = self.out(last_token).squeeze(-1) # [batch]
        
        # --- 2. THE RESIDUAL FIX (ResNet for Time Series) ---
        # We assume the target variable (rvol) is the LAST feature column.
        # Check your config: feature_cols = ["log_ret", "rvol"] -> rvol is at index -1.
        naive_forecast = x[:, -1, -1] # The most recent known volatility
        
        # The model now only learns the DIFFERENCE (Innovation)
        # Prediction = Naive + Learned_Change
        return naive_forecast + model_out