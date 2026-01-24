"""
tft_model.py

Defines a simplified Temporal Fusion Transformer-style model for BTC.

The model supports one task mode controlled by config / ModelConfig:
  - "quantile_forecast":
      outputs quantile forecasts for the H-step return path:
        raw head: (B, H*Q) where output_size = H*Q
        reshaped: (B, H, Q)

Architecture (simplified):
1) Optional Variable Selection Network (VSN) or linear projection of past inputs
2) LSTM encoder over past sequence
3) Multi-head self-attention over encoded sequence
4) Temporal feed-forward block (GRN-gated or residual MLP)
5) Optional future-covariate encoder for t+H known features (calendar/halving)
6) Output head determined by output_size (see above)

Usage:
- train_tft.py / evaluate_tft.py build tensors shaped like:
    x_past:   (B, SEQ_LENGTH, len(FEATURE_COLS))
    x_future: (B, len(FUTURE_COVARIATE_COLS))  [only if enabled]
- The model returns y_hat shaped (B, FORECAST_HORIZON, len(QUANTILES)).
"""

from __future__ import annotations
import torch
from torch import nn
from final_model import config

# Public classes:
# - TemporalFusionTransformer: main model used by train_tft.py / evaluate_tft.py
# - GatedResidualNetwork: building block (used internally)
# - VariableSelectionNetwork: building block (used internally)

# ============================
# 1. Gated Residual Network
# ============================

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) with optional context input.

    This is a simplified version of the GRN used in the TFT paper:
        - small MLP (two linear layers with ELU)
        - gating mechanism (GLU-style: elementwise gate in [0, 1])
        - residual connection + LayerNorm

    Shapes:
        x:       (batch, ..., input_size)
        context: broadcastable to x (batch, ..., input_size) or (batch, input_size) or None

    Output:
        same shape as x, but with last dimension = output_size
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # First linear layer maps input (and optional context) to hidden size
        self.linear1 = nn.Linear(input_size, hidden_size)

        # Second linear layer maps hidden to output dimension
        self.linear2 = nn.Linear(hidden_size, output_size)

        # Optional projection for the residual path if dimensions differ
        if input_size != output_size:
            self.skip_projection = nn.Linear(input_size, output_size)
        else:
            self.skip_projection = None

        # Gating layer: produces a gate in [0, 1] for each element
        self.gate_layer = nn.Linear(output_size, output_size)

        # Non-linear activation and dropout
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Layer normalization applied after residual + gated output
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the GRN.

        Args:
            x:
                Input tensor of shape (batch, ..., input_size).
            context:
                Optional context tensor. For this simplified version we expect
                it to be broadcastable to the same shape as x before the
                first linear layer.

        Returns:
            Tensor of shape (batch, ..., output_size).
        """
        # Save residual for later
        residual = x

        # If context is provided and has fewer dimensions, try to unsqueeze it
        if context is not None:
            # Example: x: (B, T, D), context: (B, D)
            # -> unsqueeze context to (B, 1, D) so broadcasting works.
            if context.dim() == x.dim() - 1:
                context = context.unsqueeze(1)
            # Add context to x before non-linearity (simple conditioning)
            x = x + context

        # MLP part
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)

        # If dimensions do not match, project residual to output_size
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)

        # GLU-style gating: gate in [0, 1], then elementwise multiply
        gate = torch.sigmoid(self.gate_layer(x))
        x = gate * x

        # Residual connection + LayerNorm
        out = self.layer_norm(residual + x)

        return out


# ============================
# 2. Variable Selection Network
# ============================

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) over past covariates.

    This module learns:
      - a value projection that embeds each feature into a hidden dimension
      - a weight network that outputs importance scores per feature
      - a softmax over features to produce a weighted combination

    Given:
        x_past: (batch, seq_length, num_features)

    It returns:
        x_selected: (batch, seq_length, hidden_size)
            The feature-mixed representation per time step.
        selection_weights: (batch, seq_length, num_features)
            Softmax-normalised importance weights for each feature at each time.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        vsn_hidden_size: int,
        dropout: float = 0.1,
        use_gating: bool = True,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.use_gating = use_gating

        # Value projection: maps each (B, T, F) vector to (B, T, F * H),
        # then we reshape to (B, T, F, H).
        self.value_projection = nn.Linear(num_features, num_features * hidden_size)

        # Weight network: produces logits over features.
        # If gating is enabled, we reuse a GRN; otherwise, a small MLP.
        if use_gating:
            self.weight_network = GatedResidualNetwork(
                input_size=num_features,
                hidden_size=vsn_hidden_size,
                output_size=num_features,
                dropout=dropout,
            )
        else:
            self.weight_network = nn.Sequential(
                nn.Linear(num_features, vsn_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(vsn_hidden_size, num_features),
            )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                Tensor of shape (batch, seq_length, num_features)

        Returns:
            x_selected:
                Tensor of shape (batch, seq_length, hidden_size)
            selection_weights:
                Tensor of shape (batch, seq_length, num_features)
        """
        # x: (B, T, F)
        batch_size, seq_length, num_features = x.shape

        if num_features != self.num_features:
            raise ValueError(
                f"VSN expected num_features={self.num_features}, "
                f"but got {num_features}. Check FEATURE_COLS / PAST_COVARIATE_COLS."
            )

        # ---- Value projection ----
        # (B, T, F) -> (B, T, F * H) -> (B, T, F, H)
        value_raw = self.value_projection(x)  # (B, T, F * H)
        value = value_raw.view(
            batch_size, seq_length, self.num_features, self.hidden_size
        )  # (B, T, F, H)

        # ---- Weight network ----
        # Produce logits over features, then softmax.
        # weight_logits: (B, T, F)
        weight_logits = self.weight_network(x)
        selection_weights = self.softmax(weight_logits)  # (B, T, F)

        # ---- Weighted combination ----
        # Expand weights to match value embeddings: (B, T, F, 1)
        # Multiply and sum over F → (B, T, H)
        weights_expanded = selection_weights.unsqueeze(-1)  # (B, T, F, 1)
        x_selected = (weights_expanded * value).sum(dim=-2)  # sum over feature dim

        return x_selected, selection_weights


# ============================
# 3. Temporal Fusion Transformer (simplified)
# ============================

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer-style model for BTC.

    Expected input:
        x_past:
            Tensor of shape (batch_size, seq_length, input_size)
            Past covariates (OHLCV + technical indicators + optional external features)
            as built in data_pipeline.py using FEATURE_COLS (the exact set depends on
            your enabled data sources / toggles).
        x_future:
            Optional tensor of shape (batch_size, future_input_size)
            Known future covariates for t+H (calendar + halving), where
            H = config.FORECAST_HORIZON. If
            config.MODEL_CONFIG.use_future_covariates is True, x_future
            must be provided.

    Output:
        - quantile_forecast:
            quantile predictions reshaped to (batch_size, H, Q),
            where H = config.FORECAST_HORIZON and Q = len(config.QUANTILES)
            (implemented via output_size = H * Q).

    Advanced options controlled via config.ModelConfig:
        - use_gating:
            If True, wrap temporal blocks in GRNs (with GLU gating).
            If False, use simpler residual + LayerNorm + MLP blocks.
        - use_variable_selection:
            If True, use a Variable Selection Network over past covariates
            instead of a single linear input projection.
        - use_future_covariates:
            If True, encode x_future and fuse it with the temporal summary
            before the final output layer.
    """

    def __init__(self, model_config: config.ModelConfig | None = None) -> None:
        super().__init__()

        # Use the global MODEL_CONFIG by default, but allow overriding.
        if model_config is None:
            model_config = config.MODEL_CONFIG

        self.config = model_config

        hidden_size = self.config.hidden_size
        dropout = self.config.dropout

        # -------- 1. Input handling: VSN or linear projection --------
        # Plain linear projection (used when variable selection is disabled).
        self.input_projection = nn.Linear(
            self.config.input_size, hidden_size
        )

        # Variable Selection Network over past covariates.
        # We instantiate it unconditionally so you can switch use_variable_selection
        # on/off from config without changing this file.
        self.vsn_past = VariableSelectionNetwork(
            num_features=self.config.input_size,
            hidden_size=hidden_size,
            vsn_hidden_size=self.config.variable_selection_hidden_size,
            dropout=dropout,
            use_gating=self.config.use_gating,
        )

        # -------- 2. LSTM encoder --------
        # Processes the sequence over time. We use batch_first=True so that
        # input/output shapes are (batch, seq, hidden_size).
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            bidirectional=False,  # single direction
        )

        # When gating is disabled, we keep simple residual + LayerNorm blocks.
        if not self.config.use_gating:
            self.lstm_layer_norm = nn.LayerNorm(hidden_size)

        # -------- 3. Multi-head self-attention --------
        # Allows the model to focus on important time steps in the encoded
        # sequence (e.g., recent volatility spikes).
        # batch_first=True => input/output: (batch, seq, hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.config.num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if not self.config.use_gating:
            # Simple residual + LayerNorm when gating is off
            self.attn_layer_norm = nn.LayerNorm(hidden_size)

        # -------- 4. Temporal feed-forward / gating blocks --------
        if self.config.use_gating:
            # Use GRNs around key temporal parts:
            # - After LSTM (post-encoding)
            # - After attention
            # - As the temporal feed-forward block
            self.post_lstm_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=self.config.variable_selection_hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )

            self.attn_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=self.config.variable_selection_hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )

            self.temporal_ffn_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=self.config.ff_hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
        else:
            # Simpler feed-forward network with residual + LayerNorm
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, self.config.ff_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.config.ff_hidden_size, hidden_size),
            )
            self.ffn_layer_norm = nn.LayerNorm(hidden_size)

        # -------- 5. Future covariate encoder (optional) --------
        # Encodes known-future covariates aligned to the forecast endpoint (t+H),
        # where H = config.FORECAST_HORIZON, into a hidden_size context vector.
        if self.config.use_future_covariates and self.config.future_input_size > 0:
            if self.config.use_gating:
                # GRN-based encoder for future covariates
                self.future_encoder = GatedResidualNetwork(
                    input_size=self.config.future_input_size,
                    hidden_size=self.config.variable_selection_hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
            else:
                # Simple linear projection when gating is disabled
                self.future_encoder = nn.Sequential(
                    nn.Linear(self.config.future_input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
        else:
            self.future_encoder = None

        # GRN (or linear) to fuse temporal summary with future context
        if self.config.use_future_covariates and self.config.use_gating:
            self.decision_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=self.config.ff_hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
        elif self.config.use_future_covariates and not self.config.use_gating:
            # When gating is off, we'll concatenate and use a linear layer
            self.decision_linear = nn.Linear(
                hidden_size + hidden_size, hidden_size
            )

        # -------- 6. Output layer --------
        # We aggregate over time (last time step), optionally fuse
        # with future covariates, and map hidden_size → output_size.
        #
        # Output head (quantile-only):
        # output_size = H * Q -> raw (B, H*Q) then reshape to (B, H, Q)
        self.output_layer = nn.Linear(hidden_size, self.config.output_size)

        # -------- Quantile-head shape contract (fail fast) --------
        # Canonical API: config.FORECAST_HORIZON (int) + config.QUANTILES (list[float])
        # NOTE: This contract intentionally uses the *global* config values so that
        # training/evaluation stay consistent even if a custom `model_config` is passed in.
        self.horizon: int = int(config.FORECAST_HORIZON)
        self.quantiles: tuple[float, ...] = tuple(config.QUANTILES)
        self.n_quantiles: int = len(self.quantiles)

        if self.horizon <= 0:
            raise ValueError(f"FORECAST_HORIZON must be > 0, got {self.horizon}.")
        if self.n_quantiles <= 0:
            raise ValueError("QUANTILES must be a non-empty list.")
        if any((q <= 0.0 or q >= 1.0) for q in self.quantiles):
            raise ValueError(f"All QUANTILES must be in (0,1), got {self.quantiles}.")

        self.expected_output_size: int = self.horizon * self.n_quantiles

        if self.config.output_size != self.expected_output_size:
            raise ValueError(
                f"[tft_model] MODEL_CONFIG.output_size mismatch: "
                f"got {self.config.output_size} but expected H*Q={self.expected_output_size} "
                f"(H={self.horizon}, Q={self.n_quantiles}). "
                f"Fix config.MODEL_CONFIG.output_size or (preferably) keep it derived from "
                f"FORECAST_HORIZON * len(QUANTILES)."
            )

        # Extra guard: output layer must agree with the contract.
        if self.output_layer.out_features != self.expected_output_size:
            raise ValueError(
                f"[tft_model] output_layer.out_features mismatch: "
                f"got {self.output_layer.out_features} but expected {self.expected_output_size}."
            )

    def forward(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass (quantile multi-horizon forecasting).

        Args:
            x_past:
                Past covariates of shape (B, T, F), where:
                  - B = batch size
                  - T = seq_length
                  - F = config.MODEL_CONFIG.input_size (len(FEATURE_COLS))
            x_future:
                Optional known-future covariates of shape (B, F_future),
                aligned to the forecast endpoint (t+H).
                Required when config.MODEL_CONFIG.use_future_covariates=True.

        Returns:
            Quantile forecasts of shape (B, H, Q), where:
              - H = config.FORECAST_HORIZON
              - Q = len(config.QUANTILES)
        """
        # x_past: (B, T, F)
        batch_size, seq_length, input_size = x_past.shape

        if input_size != self.config.input_size:
            raise ValueError(
                f"TemporalFusionTransformer expected input_size="
                f"{self.config.input_size} (len(FEATURE_COLS)), but got "
                f"{input_size}. This usually means a mismatch between "
                f"the features used to build x_past and config.FEATURE_COLS "
                f"(including ONCHAIN_COLS). "
                f"Check config.FEATURE_COLS / config.ONCHAIN_COLS and the "
                f"shape of the tensor you pass into the model."
            )

        # Check / handle future covariates
        if self.config.use_future_covariates:
            if self.future_encoder is None or self.config.future_input_size <= 0:
                raise RuntimeError(
                    "use_future_covariates=True but future_encoder is not set or "
                    "future_input_size <= 0. Check config.FUTURE_COVARIATE_COLS "
                    "and ModelConfig.future_input_size."
                )
            if x_future is None:
                raise ValueError(
                    "x_future must be provided when use_future_covariates=True."
                )
            if x_future.shape[0] != batch_size:
                raise ValueError(
                    f"x_future batch dimension {x_future.shape[0]} does not match "
                    f"x_past batch dimension {batch_size}."
                )
            if x_future.shape[1] != self.config.future_input_size:
                raise ValueError(
                    f"Expected x_future.shape[1]={self.config.future_input_size}, "
                    f"but got {x_future.shape[1]}. Check FUTURE_COVARIATE_COLS."
                )
        else:
            # Ignore x_future if not used
            x_future = None

        # ---- Step 1: Variable selection or simple projection ----
        if self.config.use_variable_selection:
            # x_proj: (B, T, H)
            # We ignore selection weights here, but they can be logged for interpretability.
            x_proj, _ = self.vsn_past(x_past)
        else:
            # Plain linear projection: (B, T, F) -> (B, T, H)
            x_proj = self.input_projection(x_past)

        # ---- Step 2: LSTM encoder ----
        # lstm_out: (B, T, H)
        lstm_out, _ = self.lstm_encoder(x_proj)

        if self.config.use_gating:
            # Combine projected input and LSTM output, then pass through GRN.
            # This GRN includes residual + gating + LayerNorm internally.
            enc_input = lstm_out + x_proj
            enc = self.post_lstm_grn(enc_input)
        else:
            # Residual connection: add the projected input to the LSTM output.
            # Then apply LayerNorm for stability.
            enc = self.lstm_layer_norm(lstm_out + x_proj)

        # ---- Step 3: Multi-head self-attention ----
        # Self-attention uses the same tensor for query, key, and value.
        # attn_raw: (B, T, H)
        # We don't return attention weights in the simplified final model.
        attn_raw, _ = self.attention(enc, enc, enc)

        if self.config.use_gating:
            # Apply a GRN to the attention output. GRN internally handles
            # residual + gating + LayerNorm.
            attn_out = self.attn_grn(attn_raw)
        else:
            # Simple residual + LayerNorm when gating is off.
            attn_out = self.attn_layer_norm(attn_raw + enc)

        # ---- Step 4: Temporal feed-forward / gating ----
        if self.config.use_gating:
            # GRN as the temporal feed-forward block.
            ff_out = self.temporal_ffn_grn(attn_out)
        else:
            # Classic MLP + residual + LayerNorm.
            ff_core = self.ffn(attn_out)
            ff_out = self.ffn_layer_norm(ff_core + attn_out)

        # ---- Step 5: Aggregate over time ----
        # Use the representation of the LAST time step as summary of the
        # whole history window.
        # last_timestep: (B, hidden_size)
        last_timestep = ff_out[:, -1, :]

        # ---- Step 6: Fuse with future covariates (if used) ----
        if self.config.use_future_covariates and x_future is not None:
            # Encode future covariates to a context vector: (B, H)
            if self.config.use_gating:
                future_context = self.future_encoder(x_future)  # GRN encoder
            else:
                future_context = self.future_encoder(x_future)  # linear encoder

            # Fuse temporal summary and future context.
            if self.config.use_gating:
                # Use a GRN with future_context as "context" input.
                fused = self.decision_grn(last_timestep, context=future_context)
            else:
                # Concatenate and project back to hidden_size.
                combined = torch.cat([last_timestep, future_context], dim=-1)
                fused = self.decision_linear(combined)

            fusion_output = fused
        else:
            # No future covariates used: keep last_timestep as final representation.
            fusion_output = last_timestep

        # ---- Step 7: Final output layer ----
        raw = self.output_layer(fusion_output)  # (B, output_size)

        # Quantile-only head: reshape (B, H*Q) -> (B, H, Q)
        expected = self.expected_output_size
        if raw.shape[-1] != expected:
            raise ValueError(
                f"[tft_model] Output size mismatch: got {raw.shape[-1]} but expected {expected} "
                f"(H={self.horizon}, Q={self.n_quantiles}). "
                f"Check config.FORECAST_HORIZON / config.QUANTILES and config.MODEL_CONFIG.output_size."
            )

        outputs = raw.view(raw.size(0), self.horizon, self.n_quantiles)  # (B, H, Q)

        return outputs