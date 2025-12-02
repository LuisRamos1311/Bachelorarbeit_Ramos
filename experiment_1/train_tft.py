"""
train_tft.py

Training script for the Temporal Fusion Transformer (TFT) on BTC data.

Supports both:
  - classification (up/down)        -> target_up (binary label)
  - regression (multi-horizon returns)
        -> continuous forward returns for each horizon in
           config.FORECAST_HORIZONS (e.g. 1, 3, 7 days)

The behaviour is controlled by config.TASK_TYPE:
  - "classification": BCEWithLogitsLoss + classification metrics,
                      model selection by Val F1.
  - "regression":    MSELoss on multi-horizon returns, aggregated
                     regression metrics (MSE, RMSE, MAE, R^2) and
                     directional metrics, model selection by Val RMSE.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from experiment_1.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    PLOTS_DIR,
    TASK_TYPE,
    UP_THRESHOLD,
)
from experiment_1.data_pipeline import prepare_datasets
from experiment_1.tft_model import TemporalFusionTransformer
from experiment_1 import utils


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
            DataLoader yielding (X_past, X_future, y) or (X_past, y).
        device:
            Torch device.
        criterion:
            Loss function:
              - BCEWithLogitsLoss for classification
              - MSELoss (or similar) for regression
        optimizer:
            Optimizer (only required if train=True).
        train:
            If True, run in training mode and update weights.
            If False, run in eval mode and do not update weights.
        threshold:
            Classification threshold on probabilities for computing
            up/down labels (only used when TASK_TYPE="classification").

    Returns:
        A dictionary of metrics. Always includes:
            - "loss"
        Additionally:
            - classification: accuracy, precision, recall, f1, auc
            - regression: mse, mae, rmse, r2,
                          direction_accuracy, direction_f1
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
                # Single-horizon classification: flatten to (B,)
                outputs = outputs.view(-1)
                y_flat = y.view(-1)
                loss = criterion(outputs, y_flat)

                y_for_metrics = y_flat
                out_for_metrics = outputs

            elif TASK_TYPE == "regression":
                # Multi-horizon regression:
                #   outputs: (B, H)
                #   y:       (B, H)
                # MSELoss will average over all horizons.
                loss = criterion(outputs, y)

                # For metrics we keep the full (B, H) tensors;
                # compute_regression_metrics will flatten later.
                y_for_metrics = y
                out_for_metrics = outputs

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
        batch_size = y.size(0)  # number of samples in batch (not horizons)
        total_loss += loss.item() * batch_size

        all_true.append(y_for_metrics.detach().cpu())
        all_outputs.append(out_for_metrics.detach().cpu())

    if num_samples == 0:
        raise RuntimeError("DataLoader has zero samples in run_epoch().")

    # Concatenate along batch dimension:
    #   classification: (N,)
    #   regression (multi-horizon): (N, H)
    y_true_all = torch.cat(all_true, dim=0)
    y_out_all = torch.cat(all_outputs, dim=0)

    avg_loss = total_loss / num_samples

    # ---- Metrics depending on task type ----
    if TASK_TYPE == "classification":
        # Convert raw logits -> probabilities
        y_prob = torch.sigmoid(y_out_all)
        metrics = utils.compute_classification_metrics(
            y_true=y_true_all,
            y_prob=y_prob,
            threshold=threshold,
        )
        metrics["loss"] = float(avg_loss)

    elif TASK_TYPE == "regression":
        # Use raw outputs as predicted returns.
        # compute_regression_metrics will internally flatten (N, H) -> (N*H,)
        # for aggregate MSE/RMSE/MAE/R^2 + directional metrics.
        metrics = utils.compute_regression_metrics(
            y_true=y_true_all,
            y_pred=y_out_all,
            direction_threshold=UP_THRESHOLD,
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

    # 1) Prepare datasets
    print("[train_tft] Preparing datasets...")
    train_dataset, val_dataset, test_dataset, scalers = prepare_datasets()

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
    print("[train_tft] Initializing model...")
    model = TemporalFusionTransformer(MODEL_CONFIG).to(device)

    # 4) Loss function
    if TASK_TYPE == "classification":
        if TRAINING_CONFIG.pos_weight != 1.0:
            pos_weight_tensor = torch.tensor(
                [TRAINING_CONFIG.pos_weight], dtype=torch.float32, device=device
            )
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(
                f"[train_tft] Using BCEWithLogitsLoss with "
                f"pos_weight={TRAINING_CONFIG.pos_weight:.4f} for classification."
            )
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
            print("[train_tft] Using BCEWithLogitsLoss without class weighting.")
    elif TASK_TYPE == "regression":
        criterion = torch.nn.MSELoss()
        print("[train_tft] Using MSELoss for regression on continuous returns.")
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
    threshold = TRAINING_CONFIG.threshold  # classification prob threshold

    history = {
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
        print("[train_tft] Model selection metric: Val F1 (maximize).")
    elif TASK_TYPE == "regression":
        history.update(
            {
                "train_mse": [],
                "val_mse": [],
                "train_mae": [],
                "val_mae": [],
                "train_rmse": [],
                "val_rmse": [],
                "train_r2": [],
                "val_r2": [],
                "train_direction_accuracy": [],
                "val_direction_accuracy": [],
                "train_direction_f1": [],
                "val_direction_f1": [],
            }
        )
        selection_metric_name = "rmse"
        best_val_metric = float("inf")
        selection_mode = "min"
        print("[train_tft] Model selection metric: Val RMSE (minimize).")
    else:
        raise ValueError(f"Unsupported TASK_TYPE='{TASK_TYPE}' in train_tft.")

    best_model_path = os.path.join(MODELS_DIR, "tft_btc_best.pth")

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

        elif TASK_TYPE == "regression":
            history["train_mse"].append(train_metrics["mse"])
            history["val_mse"].append(val_metrics["mse"])
            history["train_mae"].append(train_metrics["mae"])
            history["val_mae"].append(val_metrics["mae"])
            history["train_rmse"].append(train_metrics["rmse"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["train_r2"].append(train_metrics["r2"])
            history["val_r2"].append(val_metrics["r2"])
            history["train_direction_accuracy"].append(train_metrics["direction_accuracy"])
            history["val_direction_accuracy"].append(val_metrics["direction_accuracy"])
            history["train_direction_f1"].append(train_metrics["direction_f1"])
            history["val_direction_f1"].append(val_metrics["direction_f1"])

            print(
                f"[train_tft][Epoch {epoch}] "
                f"Train loss={train_metrics['loss']:.6f}, "
                f"Val loss={val_metrics['loss']:.6f}, "
                f"Train RMSE={train_metrics['rmse']:.6f}, "
                f"Val RMSE={val_metrics['rmse']:.6f}, "
                f"Val DirAcc={val_metrics['direction_accuracy']:.4f}, "
                f"Val DirF1={val_metrics['direction_f1']:.4f}"
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

    print(
        f"[train_tft] Training finished. "
        f"Best Val {selection_metric_name.upper()}: {best_val_metric:.4f}"
    )

    # Save training history
    run_id = time.strftime("tft_run_%Y%m%d_%H%M%S")
    history_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train_tft] History saved to {history_path}")

    # Plot curves
    curves_path = os.path.join(PLOTS_DIR, f"{run_id}_training_curves.png")
    utils.plot_training_curves(history, curves_path)
    print(f"[train_tft] Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()