"""
tft_model.py

Defines a simplified Temporal Fusion Transformer-style model for predicting
next-day BTC up/down movements.

The architecture is inspired by the TFT paper but kept intentionally simpler
for a bachelor thesis project:

1. Linear projection from raw input features to a shared hidden size.
2. LSTM encoder over the past time steps.
3. Multi-head self-attention over the encoded sequence.
4. Position-wise feed-forward network with residual connections.
5. Aggregation over time (we use the last time step) and a final linear layer
   to output a single logit for binary classification.

Other modules (train_tft.py, evaluate_tft.py, etc.) should treat this as a
standard PyTorch model:

    from tft_model import TemporalFusionTransformer
"""

from __future__ import annotations

import torch
from torch import nn

import config


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer-style model.

    Expected input:
        x: Tensor of shape (batch_size, seq_length, input_size)

    Output:
        logits: Tensor of shape (batch_size, output_size)
                For this project, output_size = 1 (binary up/down logit).

    If return_attention=True, the forward pass also returns the attention
    weights from the multi-head attention layer, which can be used later
    for interpretability plots in the thesis.
    """

    def __init__(self, model_config: config.ModelConfig | None = None) -> None:
        super().__init__()

        # Use the global MODEL_CONFIG by default, but allow overriding.
        if model_config is None:
            model_config = config.MODEL_CONFIG

        self.config = model_config

        # -------- 1. Input projection --------
        # Project raw features (input_size) into a shared hidden_size.
        self.input_projection = nn.Linear(
            self.config.input_size, self.config.hidden_size
        )

        # -------- 2. LSTM encoder --------
        # Processes the sequence over time. We use batch_first=True so that
        # input/output shapes are (batch, seq, hidden_size).
        self.lstm_encoder = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            bidirectional=False,  # keep it simple: single direction
        )

        # LayerNorm after the LSTM + residual connection.
        self.lstm_layer_norm = nn.LayerNorm(self.config.hidden_size)

        # -------- 3. Multi-head self-attention --------
        # Allows the model to focus on important time steps in the encoded
        # sequence (e.g., recent volatility spikes).
        # batch_first=True => input/output: (batch, seq, hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )

        # LayerNorm after attention + residual connection.
        self.attn_layer_norm = nn.LayerNorm(self.config.hidden_size)

        # -------- 4. Position-wise feed-forward network --------
        # Applied to each time step independently (like a small MLP).
        self.ffn = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.ff_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.ff_hidden_size, self.config.hidden_size),
        )

        self.ffn_layer_norm = nn.LayerNorm(self.config.hidden_size)

        # -------- 5. Output layer --------
        # We will aggregate over time (take last time step) and map the
        # hidden_size vector to a single logit for up/down.
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.output_size)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TFT model.

        Args:
            x:
                Input tensor of shape (batch_size, seq_length, input_size).
                This should match the sequences created in data_pipeline.py.
            return_attention:
                If True, also return the attention weights from the
                multi-head attention layer. This is useful for later
                interpretability / visualization.

        Returns:
            logits:
                Tensor of shape (batch_size, output_size), where output_size=1.
                These are raw logits; you should apply torch.sigmoid(logits)
                outside the model when computing probabilities.
            attn_weights (optional):
                Tensor of shape (batch_size, seq_length, seq_length) with
                attention weights over time. Only returned if
                return_attention=True.
        """
        # x: (B, T, input_size)
        batch_size, seq_length, input_size = x.shape

        if input_size != self.config.input_size:
            raise ValueError(
                f"Expected input_size={self.config.input_size}, "
                f"but got {input_size}. Check FEATURE_COLS and ModelConfig."
            )

        # ---- Step 1: Input projection ----
        # Map raw features into hidden_size dimension.
        # x_proj: (B, T, H)
        x_proj = self.input_projection(x)

        # ---- Step 2: LSTM encoder ----
        # lstm_out: (B, T, H)
        lstm_out, _ = self.lstm_encoder(x_proj)

        # Residual connection: add the projected input to the LSTM output.
        # Then apply LayerNorm for stability.
        # enc: (B, T, H)
        enc = self.lstm_layer_norm(lstm_out + x_proj)

        # ---- Step 3: Multi-head self-attention ----
        # Self-attention uses the same tensor for query, key, and value.
        # attn_out: (B, T, H)
        # attn_weights: (B, T, T) - how much each time step attends to others.
        attn_out, attn_weights = self.attention(enc, enc, enc)

        # Another residual connection + LayerNorm.
        # attn_out: (B, T, H)
        attn_out = self.attn_layer_norm(attn_out + enc)

        # ---- Step 4: Position-wise feed-forward network ----
        # Apply the small MLP to each time step independently.
        # ff_out: (B, T, H)
        ff_out = self.ffn(attn_out)

        # Residual connection + LayerNorm again.
        ff_out = self.ffn_layer_norm(ff_out + attn_out)

        # ---- Step 5: Aggregate over time ----
        # We take the representation of the LAST time step as summary of the
        # whole history window. This fits the idea: "use last 30 days to
        # decide if tomorrow is up or down".
        # last_timestep: (B, H)
        last_timestep = ff_out[:, -1, :]

        # Map to final logits: (B, 1)
        logits = self.output_layer(last_timestep)

        if return_attention:
            return logits, attn_weights

        return logits


if __name__ == "__main__":
    # Small sanity check that the model runs end-to-end on dummy data.
    torch.manual_seed(0)

    # Use settings from config.py
    seq_length = config.SEQ_LENGTH
    input_size = config.MODEL_CONFIG.input_size

    batch_size = 4
    dummy_x = torch.randn(batch_size, seq_length, input_size)

    model = TemporalFusionTransformer()
    logits, attn_weights = model(dummy_x, return_attention=True)

    print("Input shape:       ", dummy_x.shape)       # (B, T, input_size)
    print("Logits shape:      ", logits.shape)        # (B, 1)
    print("Attention shape:   ", attn_weights.shape)  # (B, T, T)