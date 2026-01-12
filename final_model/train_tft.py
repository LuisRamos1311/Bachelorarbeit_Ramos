"""
train_tft.py

Train a Temporal Fusion Transformer (TFT) for multi-horizon quantile forecasting of BTC
forward returns.

Inputs:
  - BTC OHLCV candles (daily or hourly, selected via config.FREQUENCY)
  - optional on-chain and sentiment features (enabled via config.USE_ONCHAIN / config.USE_SENTIMENT)
  - known-future covariates (calendar / halving style) shifted to t+H (see data_pipeline.py)

This codebase now supports ONLY:
  - TASK_TYPE="quantile_forecast"

Quantile training:
  - model outputs (B, H, Q)
  - loss: pinball loss
  - logging: MAE on median (q=0.5) at horizon FORECAST_HORIZON

Train/val/test date windows, FREQUENCY, and data-source toggles (USE_ONCHAIN,
USE_SENTIMENT) are configured in config.py.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from final_model import config as cfg
from final_model.data_pipeline import prepare_datasets
from final_model.tft_model import TemporalFusionTransformer
from final_model import utils


# ============================================================
# 1. DATALOADER CREATION
# ============================================================


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap PyTorch Datasets into DataLoaders.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# 2. SINGLE-EPOCH TRAINING / EVAL
# ============================================================


def run_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
) -> Dict[str, float]:
    """
    Run one epoch of quantile-forecast training or evaluation.

    Batch format (from BTCTFTDataset):
      - (x_past, y) when no future covariates are provided
      - (x_past, x_future, y) when known-future covariates are enabled

    Shapes:
      - x_past:   (B, SEQ_LENGTH, input_size)
      - x_future: (B, future_input_size)  (per-sample vector aligned to step t+H)
      - y:        (B, H)
      - y_hat:    (B, H, Q)

    Returns:
      {"loss": avg_pinball, "mae": avg_mae_at_H}
      where mae is computed on q=0.5 at horizon FORECAST_HORIZON.
    """
    if train:
        model.train()
    else:
        model.eval()

    if train and optimizer is None:
        raise ValueError("optimizer must not be None when train=True")

    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    for batch in data_loader:
        # Unpack batch: either (X_past, y) or (X_past, X_future, y)
        if len(batch) == 2:
            x_past, y = batch
            x_future = None
        elif len(batch) == 3:
            x_past, x_future, y = batch
        else:
            raise ValueError(
                f"Expected batch of length 2 or 3, got {len(batch)}. "
                "Check BTCTFTDataset.__getitem__."
            )

        x_past = x_past.to(device)
        y = y.to(device)
        if x_future is not None:
            x_future = x_future.to(device)

        with torch.set_grad_enabled(train):
            # Forward
            if x_future is not None:
                y_hat = model(x_past, x_future)  # (B, H, Q)
            else:
                y_hat = model(x_past)  # (B, H, Q)

            # Loss
            loss = utils.pinball_loss(y_true=y, y_pred=y_hat, quantiles=cfg.QUANTILES)

            # MAE on q=0.5 at horizon FORECAST_HORIZON
            batch_mae = utils.mae_on_median_at_horizon(
                y_true=y,
                y_pred=y_hat,
                quantiles=cfg.QUANTILES,
                horizon_step=cfg.FORECAST_HORIZON,
            )

            # Backward + optimize
            if train:

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                clip_value = cfg.TRAINING_CONFIG.grad_clip
                if clip_value is not None and clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_mae += float(batch_mae) * batch_size
        num_samples += batch_size

    if num_samples == 0:
        raise RuntimeError("DataLoader has zero samples in run_epoch().")

    return {
        "loss": total_loss / num_samples,
        "mae": total_mae / num_samples,
    }


# ============================================================
# 3. MAIN TRAINING LOOP
# ============================================================

def main() -> None:
    """
    Entry point for training.

    Creates a fresh run folder under standard/, prepares datasets, trains the TFT, and
    saves the best checkpoint (by validation pinball loss) plus training artifacts.
    """
    utils.set_seed(cfg.TRAINING_CONFIG.seed)
    device = utils.get_device()

    # Safety: refuse to start a new run if "standard" already has content
    utils.guard_dir_missing_or_empty(cfg.STANDARD_RUN_DIR, display_name=cfg.STANDARD_RUN_DIRNAME)

    # Normal directory creation under standard/
    utils.ensure_dir(cfg.MODELS_DIR)
    utils.ensure_dir(cfg.EXPERIMENTS_DIR)
    utils.ensure_dir(cfg.PLOTS_DIR)

    # Derive a human-readable horizon description (quantile-only)
    horizon_steps = int(cfg.FORECAST_HORIZON)

    if cfg.FREQUENCY.upper() == "D":
        horizon_desc = f"{horizon_steps}-day-ahead"
    elif cfg.FREQUENCY.lower() in {"1h", "h"}:
        horizon_desc = f"{horizon_steps}-hour-ahead"
    else:
        horizon_desc = f"{horizon_steps}-step-ahead"

    # Print a short run summary so logs are self-documenting
    print(
        f"[train_tft] Run setup: {horizon_desc} quantile multi-horizon "
        f"H={cfg.FORECAST_HORIZON} | FREQUENCY='{cfg.FREQUENCY}' | "
        f"USE_OHLCV={cfg.USE_OHLCV} (active_cols={len(cfg.PRICE_VOLUME_COLS) if cfg.USE_OHLCV else 0}) | "
        f"USE_TALIB_INDICATORS={cfg.USE_TALIB_INDICATORS} (active_cols={len(cfg.INDICATOR_COLS) if cfg.USE_TALIB_INDICATORS else 0}) | "
        f"USE_ONCHAIN={cfg.USE_ONCHAIN} (active_cols={len(cfg.ONCHAIN_COLS) if cfg.USE_ONCHAIN else 0}) | "
        f"USE_SENTIMENT={cfg.USE_SENTIMENT} (active_cols={len(cfg.SENTIMENT_COLS) if cfg.USE_SENTIMENT else 0}) | "
        f"TOTAL_FEATURES={len(cfg.FEATURE_COLS)}"
    )

    # 1) Prepare datasets (explicitly pass SEQ_LENGTH to keep training/eval aligned)
    print("[train_tft] Preparing datasets.")
    train_dataset, val_dataset, _test_dataset, _scalers = prepare_datasets(
        seq_length=cfg.SEQ_LENGTH
    )

    # 2) Dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=cfg.TRAINING_CONFIG.batch_size,
    )

    # 3) Initialize model
    print("[train_tft] Initializing model.")
    model = TemporalFusionTransformer(cfg.MODEL_CONFIG).to(device)

    # 4) Loss function
    print(
        f"[train_tft] Using pinball loss for quantile forecasting "
        f"(H={cfg.FORECAST_HORIZON}, QUANTILES={list(cfg.QUANTILES)})."
    )

    # 5) Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAINING_CONFIG.learning_rate,
        weight_decay=cfg.TRAINING_CONFIG.weight_decay,
    )

    # 6) Training setup
    num_epochs = cfg.TRAINING_CONFIG.num_epochs

    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
    }

    # Path where the best model checkpoint will be saved
    best_model_path = cfg.BEST_MODEL_PATH

    best_val_loss = float("inf")
    best_epoch: Optional[int] = None

    # Run ID for saving history/plots
    run_id = f"tft_run_{utils.get_timestamp()}"

    print(f"[train_tft] Starting training for {num_epochs} epochs.")

    for epoch in range(1, num_epochs + 1):
        print(f"[train_tft] Epoch {epoch}/{num_epochs}")

        # --- Train epoch ---
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            device=device,
            optimizer=optimizer,
            train=True,
        )

        # --- Validation epoch ---
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            optimizer=None,
            train=False,
        )

        # Record in history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])

        print(
            f"[train_tft][Epoch {epoch}] "
            f"Train pinball={train_metrics['loss']:.4f}, "
            f"Val pinball={val_metrics['loss']:.4f}, "
            f"Train MAE@{cfg.FORECAST_HORIZON}={train_metrics['mae']:.6f}, "
            f"Val MAE@{cfg.FORECAST_HORIZON}={val_metrics['mae']:.6f}"
        )

        # --- Model selection (quantile-only: minimize val pinball loss) ---
        current_val_loss = val_metrics["loss"]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(
                f"[train_tft] New best checkpoint saved -> {best_model_path} "
                f"(epoch={best_epoch}, val_loss={best_val_loss:.6f})"
            )

    if best_epoch is None:
        raise RuntimeError("[train_tft] No checkpoint was saved (best_epoch is None).")

    print(
        f"[train_tft] Training finished. Best checkpoint: {best_model_path} "
        f"(epoch={best_epoch}, val_loss={best_val_loss:.6f})"
    )

    # Save history as JSON
    history_path = os.path.join(cfg.EXPERIMENTS_DIR, f"{run_id}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    curves_path = os.path.join(cfg.PLOTS_DIR, f"{run_id}_training_curves.png")
    utils.plot_training_curves(history, curves_path)

    # Compact artifacts summary
    print(
        "\n[train_tft] Artifacts summary:\n"
        f"  Best checkpoint: {best_model_path}\n"
        f"  History JSON:    {history_path}\n"
        f"  Training curves: {curves_path}"
    )

if __name__ == "__main__":
    main()