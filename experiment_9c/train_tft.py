"""
train_tft.py

Training script for the Temporal Fusion Transformer (TFT) on BTC data.

Experiment configuration:
  - 1-hour BTC candles (config.FREQUENCY = "1h")
  - H-step-ahead 3-class direction classification, where
        0 = DOWN, 1 = FLAT, 2 = UP
    and H = config.FORECAST_HORIZONS[0] (e.g. 24 hours ahead).
  - Target column is config.TRIPLE_DIRECTION_COLUMN ("direction_3c"),
    created from the H-step forward return using config.DIRECTION_THRESHOLD.
  - Forward return can be:
        * log(close_{t+H} / close_t)      if config.USE_LOG_RETURNS = True
        * close_{t+H} / close_t - 1.0     otherwise
    (see data_pipeline.add_target_column).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from experiment_9c.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    PLOTS_DIR,
    TASK_TYPE,
    NUM_CLASSES,
    SEQ_LENGTH,
    USE_LOG_RETURNS,
    DIRECTION_THRESHOLD,
    DIRECTION_LABEL_COLUMN,
    FORECAST_HORIZONS,
    FREQUENCY,
    BEST_MODEL_PATH,
    FEATURE_COLS,
    ONCHAIN_COLS,
    USE_ONCHAIN,
    USE_SENTIMENT,
    SENTIMENT_COLS,
    FORECAST_HORIZON,
    QUANTILES,
)
from experiment_9c import config as cfg
from experiment_9c.data_pipeline import prepare_datasets
from experiment_9c.tft_model import TemporalFusionTransformer
from experiment_9c import utils


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
    criterion: Optional[torch.nn.Module],
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run one epoch over a DataLoader, for either training or validation.

    Returns:
        Dict of metrics, always including "loss".
        - classification: also includes accuracy/precision/recall/f1/auc
        - quantile_forecast: also includes mae (median @ FORECAST_HORIZON)
    """
    if train:
        if optimizer is None:
            raise ValueError("optimizer must be provided when train=True")
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0  # only used for quantile_forecast
    num_samples = 0

    all_true = []
    all_outputs = []

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
            if TASK_TYPE == "classification":
                if criterion is None:
                    raise ValueError("criterion must not be None for classification")

                if x_future is not None:
                    logits = model(x_past, x_future)
                else:
                    logits = model(x_past)

                loss = criterion(logits, y)
                out_for_metrics = logits
                y_for_metrics = y

            elif TASK_TYPE == "quantile_forecast":
                # y_hat shape: (B, H, Q)
                if x_future is not None:
                    y_hat = model(x_past, x_future)
                else:
                    y_hat = model(x_past)

                loss = utils.pinball_loss(
                    y_true=y,
                    y_pred=y_hat,
                    quantiles=QUANTILES,
                )

                # MAE on q=0.5 at horizon FORECAST_HORIZON (e.g. 24)
                batch_mae = utils.mae_on_median_at_horizon(
                    y_true=y,
                    y_pred=y_hat,
                    quantiles=QUANTILES,
                    horizon_step=FORECAST_HORIZON,
                )

                out_for_metrics = y_hat
                y_for_metrics = y

            else:
                raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in run_epoch().")

            # Backward + optimize
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                clip_value = TRAINING_CONFIG.grad_clip
                if clip_value is not None and clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

        # Accumulate loss and predictions
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        if TASK_TYPE == "quantile_forecast":
            total_mae += float(batch_mae) * batch_size
        else:
            all_true.append(y_for_metrics.detach().cpu())
            all_outputs.append(out_for_metrics.detach().cpu())

    if num_samples == 0:
        raise RuntimeError("DataLoader has zero samples in run_epoch().")

    avg_loss = total_loss / num_samples

    # ---- Metrics depending on task type ----
    if TASK_TYPE == "classification":
        y_true_all = torch.cat(all_true, dim=0)
        y_out_all = torch.cat(all_outputs, dim=0)
        y_prob = torch.softmax(y_out_all, dim=1)

        metrics = utils.compute_multiclass_metrics(
            y_true=y_true_all,
            y_prob=y_prob,
        )
        metrics["loss"] = float(avg_loss)
        return metrics

    if TASK_TYPE == "quantile_forecast":
        avg_mae = total_mae / num_samples
        return {
            "loss": float(avg_loss),  # pinball
            "mae": float(avg_mae),    # MAE on median @ H
        }

    raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in run_epoch().")


# ============================================================
# 3. MAIN TRAINING LOOP
# ============================================================


def main() -> None:
    utils.set_seed(TRAINING_CONFIG.seed)
    device = utils.get_device()
    print(f"[train_tft] Using device: {device}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Derive a human-readable horizon description
    if FORECAST_HORIZONS:
        horizon_steps = FORECAST_HORIZONS[0]
    else:
        horizon_steps = 1

    if FREQUENCY.upper() == "D":
        horizon_desc = f"{horizon_steps}-day-ahead"
    elif FREQUENCY.lower() in {"1h", "h"}:
        horizon_desc = f"{horizon_steps}-hour-ahead"
    else:
        horizon_desc = f"{horizon_steps}-step-ahead"

    # Print a short experiment summary so logs are self-documenting
    onchain_count = len(ONCHAIN_COLS) if USE_ONCHAIN else 0
    sentiment_count = len(SENTIMENT_COLS) if USE_SENTIMENT else 0

    out_size = FORECAST_HORIZON * len(QUANTILES)

    if TASK_TYPE == "classification":
        print(
            f"[train_tft] Experiment setup: {horizon_desc} "
            f"{NUM_CLASSES}-class direction "
            f"label='{DIRECTION_LABEL_COLUMN}', "
            f"threshold={DIRECTION_THRESHOLD:.4f}, "
            f"use_log_returns={USE_LOG_RETURNS}, "
            f"seq_length={SEQ_LENGTH}, "
            f"frequency='{FREQUENCY}', "
            f"features={len(FEATURE_COLS)} "
            f"(incl. {onchain_count} on-chain, {sentiment_count} sentiment)"
        )
    elif TASK_TYPE == "quantile_forecast":
        print(
            f"[train_tft] Experiment setup: {horizon_desc} quantile multi-horizon "
            f"H={FORECAST_HORIZON}, QUANTILES={list(QUANTILES)}, "
            f"use_log_returns={USE_LOG_RETURNS}, "
            f"seq_length={SEQ_LENGTH}, "
            f"frequency='{FREQUENCY}', "
            f"features={len(FEATURE_COLS)} "
            f"(incl. {onchain_count} on-chain, {sentiment_count} sentiment), "
            f"output_size={out_size}"
        )
    else:
        raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in train_tft.")

    # Experiment 9a: data integrity flags (self-documenting logs)
    print("[train_tft] Integrity flags (Experiment 9a):")
    print(f"  FORECAST_HORIZONS        = {FORECAST_HORIZONS} (=> {horizon_desc})")
    print(f"  DAILY_FEATURE_LAG_DAYS   = {getattr(cfg, 'DAILY_FEATURE_LAG_DAYS', None)}")
    print(f"  DEBUG_DATA_INTEGRITY     = {getattr(cfg, 'DEBUG_DATA_INTEGRITY', None)}")
    print(f"  DROP_LAST_H_IN_EACH_SPLIT= {getattr(cfg, 'DROP_LAST_H_IN_EACH_SPLIT', None)}")

    # 1) Prepare datasets (explicitly pass SEQ_LENGTH to keep training/eval aligned)
    print("[train_tft] Preparing datasets.")
    train_dataset, val_dataset, test_dataset, scalers = prepare_datasets(
        seq_length=SEQ_LENGTH
    )

    print(f"[train_tft] Train samples: {len(train_dataset)}")
    print(f"[train_tft] Val samples:   {len(val_dataset)}")
    print(f"[train_tft] Test samples:  {len(test_dataset)}")

    # 2) Dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
    )

    # 3) Initialize model
    print("[train_tft] Initializing model.")
    model = TemporalFusionTransformer(MODEL_CONFIG).to(device)

    # 4) Loss function
    if TASK_TYPE == "classification":
        criterion = torch.nn.CrossEntropyLoss()
        print(
            f"[train_tft] Using CrossEntropyLoss for "
            f"{NUM_CLASSES}-class classification."
        )
    elif TASK_TYPE == "quantile_forecast":
        criterion = None  # pinball loss computed inside run_epoch()
        print(
            f"[train_tft] Using pinball loss for quantile forecasting "
            f"(H={FORECAST_HORIZON}, QUANTILES={list(QUANTILES)})."
        )
    else:
        raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in train_tft.")

    # 5) Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG.learning_rate,
        weight_decay=TRAINING_CONFIG.weight_decay,
    )

    print(model)

    # 6) Training setup
    num_epochs = TRAINING_CONFIG.num_epochs
    threshold = TRAINING_CONFIG.threshold  # kept for future binary/UP-vs-REST uses

    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    # History fields + model selection metric depending on task
    if TASK_TYPE == "classification":
        history.update(
            {
                "train_accuracy": [],
                "val_accuracy": [],
                "train_precision": [],
                "val_precision": [],
                "train_recall": [],
                "val_recall": [],
                "train_f1": [],
                "val_f1": [],
                "train_auc": [],
                "val_auc": [],
            }
        )
        selection_metric_name = "f1"
        best_val_metric = -1.0
        selection_mode = "max"
        print("[train_tft] Model selection metric: Val macro-F1 (maximize).")

    elif TASK_TYPE == "quantile_forecast":
        history.update(
            {
                "train_mae": [],
                "val_mae": [],
            }
        )
        selection_metric_name = "loss"  # pinball loss
        best_val_metric = float("inf")
        selection_mode = "min"
        print("[train_tft] Model selection metric: Val pinball loss (minimize).")

    else:
        raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in train_tft.")

    # Path where the best model checkpoint will be saved
    best_model_path = BEST_MODEL_PATH

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
            criterion=criterion,
            optimizer=optimizer,
            train=True,
            threshold=threshold,
        )

        # --- Validation epoch ---
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            train=False,
            threshold=threshold,
        )

        # Record in history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        if TASK_TYPE == "classification":
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["train_precision"].append(train_metrics["precision"])
            history["val_precision"].append(val_metrics["precision"])
            history["train_recall"].append(train_metrics["recall"])
            history["val_recall"].append(val_metrics["recall"])
            history["train_f1"].append(train_metrics["f1"])
            history["val_f1"].append(val_metrics["f1"])
            history["train_auc"].append(train_metrics["auc"])
            history["val_auc"].append(val_metrics["auc"])

            print(
                f"[train_tft][Epoch {epoch}] "
                f"Train loss={train_metrics['loss']:.4f}, "
                f"Val loss={val_metrics['loss']:.4f}, "
                f"Train F1={train_metrics['f1']:.4f}, "
                f"Val F1={val_metrics['f1']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )

        elif TASK_TYPE == "quantile_forecast":
            history["train_mae"].append(train_metrics["mae"])
            history["val_mae"].append(val_metrics["mae"])

            print(
                f"[train_tft][Epoch {epoch}] "
                f"Train pinball={train_metrics['loss']:.4f}, "
                f"Val pinball={val_metrics['loss']:.4f}, "
                f"Train MAE@{FORECAST_HORIZON}={train_metrics['mae']:.6f}, "
                f"Val MAE@{FORECAST_HORIZON}={val_metrics['mae']:.6f}"
            )

        else:
            raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in train_tft.")

        # --- Model selection ---
        current_val_metric = val_metrics[selection_metric_name]
        if selection_mode == "max":
            is_better = current_val_metric > best_val_metric
        else:
            is_better = current_val_metric < best_val_metric

        if is_better:
            best_val_metric = current_val_metric
            torch.save(model.state_dict(), best_model_path)
            print(
                f"[train_tft] New best model saved with "
                f"Val {selection_metric_name.upper()}={best_val_metric:.4f} "
                f"to {best_model_path}"
            )

    print(
        f"[train_tft] Training finished. "
        f"Best Val {selection_metric_name.upper()}: {best_val_metric:.4f}"
    )

    # Save history as JSON
    history_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train_tft] History saved to {history_path}")

    # Plot training curves
    curves_path = os.path.join(PLOTS_DIR, f"{run_id}_training_curves.png")
    utils.plot_training_curves(history, curves_path)
    print(f"[train_tft] Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()