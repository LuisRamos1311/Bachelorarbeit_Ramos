"""
train_tft.py

Training script for the Temporal Fusion Transformer (TFT) on BTC data.

Experiment 3 / 5 baseline:
  - 1-day-ahead 3-class direction classification:
        0 = DOWN, 1 = FLAT, 2 = UP
  - Target column is config.TRIPLE_DIRECTION_COLUMN ("direction_3c"),
    created from the 1-day forward return using config.DIRECTION_THRESHOLD.
  - Forward return can be:
        * log(close_{t+1} / close_t)      if config.USE_LOG_RETURNS = True
        * close_{t+1} / close_t - 1.0     otherwise
    (see data_pipeline.add_target_column).

Currently, only the multi-class classification path is implemented:
  - TASK_TYPE = "classification"
  - Loss: CrossEntropyLoss on 3 classes (DOWN / FLAT / UP)
  - Metrics: utils.compute_multiclass_metrics (macro-averaged)
  - Model selection: best validation macro-F1 ("f1" from compute_multiclass_metrics)

The code is written so that later you can extend it to:
  - multi-horizon regression or classification (by adding branches to
    run_epoch() and main(), and using config.FORECAST_HORIZONS).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from experiment_5.config import (
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
)
from experiment_5.data_pipeline import prepare_datasets
from experiment_5.tft_model import TemporalFusionTransformer
from experiment_5 import utils


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
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run one epoch over a DataLoader, for either training or validation.

    Args:
        model:
            The Temporal Fusion Transformer model.
        data_loader:
            DataLoader yielding (X_past, y) or (X_past, X_future, y).
        device:
            Torch device.
        criterion:
            Loss function:
              - CrossEntropyLoss for multi-class classification.
        optimizer:
            Optimizer (only required if train=True).
        train:
            If True, run in training mode and update weights.
            If False, run in eval mode and do not update weights.
        threshold:
            Currently unused in this Experiment 3/5 implementation
            (metrics are computed from full class probabilities), but
            kept for future binary / UP-vs-REST extensions.

    Returns:
        A dictionary of metrics. Always includes:
            - "loss"
        Additionally for classification:
            - "accuracy", "precision", "recall", "f1", "auc"
    """
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be provided when train=True.")
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true: list[torch.Tensor] = []
    all_outputs: list[torch.Tensor] = []

    num_samples = len(data_loader.dataset)

    for batch in data_loader:
        # Unpack batch: supports both (x_past, y) and (x_past, x_future, y)
        if len(batch) == 2:
            x_past, y = batch
            x_future = None
        else:
            x_past, x_future, y = batch

        x_past = x_past.to(device)
        y = y.to(device)
        if x_future is not None:
            x_future = x_future.to(device)

        with torch.set_grad_enabled(train):
            # Forward pass
            outputs = model(x_past) if x_future is None else model(x_past, x_future)

            if TASK_TYPE == "classification":
                # Multi-class classification:
                # outputs: (B, NUM_CLASSES) logits
                # y:       (B,) integer labels in {0,1,2}
                logits = outputs
                y_flat = y.view(-1).long()

                loss = criterion(logits, y_flat)

                y_for_metrics = y_flat
                out_for_metrics = logits

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
        batch_size = y.size(0)  # number of samples in batch
        total_loss += loss.item() * batch_size

        all_true.append(y_for_metrics.detach().cpu())
        all_outputs.append(out_for_metrics.detach().cpu())

    if num_samples == 0:
        raise RuntimeError("DataLoader has zero samples in run_epoch().")

    # Concatenate along batch dimension:
    #   classification: (N,)
    y_true_all = torch.cat(all_true, dim=0)
    y_out_all = torch.cat(all_outputs, dim=0)

    avg_loss = total_loss / num_samples

    # ---- Metrics depending on task type ----
    if TASK_TYPE == "classification":
        # Convert logits -> class probabilities
        y_prob = torch.softmax(y_out_all, dim=1)
        metrics = utils.compute_multiclass_metrics(
            y_true=y_true_all,
            y_prob=y_prob,
        )
        metrics["loss"] = float(avg_loss)

    else:
        raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in run_epoch().")

    return metrics


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

    # Print a short experiment summary so logs are self-documenting
    print(
        "[train_tft] Experiment setup: 1-day-ahead 3-class direction "
        f"label='{DIRECTION_LABEL_COLUMN}', "
        f"threshold={DIRECTION_THRESHOLD:.4f}, "
        f"use_log_returns={USE_LOG_RETURNS}, "
        f"seq_length={SEQ_LENGTH}"
    )

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
        # Multi-class (0=DOWN, 1=FLAT, 2=UP)
        # You can later add class weights here if needed.
        criterion = torch.nn.CrossEntropyLoss()
        print(f"[train_tft] Using CrossEntropyLoss for {NUM_CLASSES}-class classification.")
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
        # "f1" here is macro-F1 from utils.compute_multiclass_metrics
        selection_metric_name = "f1"
        best_val_metric = -1.0
        selection_mode = "max"
        print("[train_tft] Model selection metric: Val macro-F1 (maximize).")
    else:
        raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in train_tft.")

    best_model_path = os.path.join(MODELS_DIR, "tft_btc_best.pth")

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

    print(f"[train_tft] Training finished. Best Val {selection_metric_name.upper()}: {best_val_metric:.4f}")

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