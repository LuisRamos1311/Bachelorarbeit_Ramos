"""
train_tft.py

Training script for the Temporal Fusion Transformer (TFT)
on the BTC up/down dataset.

Phase 3 (complete):
- Set up device and seeds
- Build train/val datasets & DataLoaders
- Instantiate the TFT model, loss, optimizer
- Run a full training loop with:
    * per-epoch train & validation metrics
    * best-model saving
    * experiment history logging
    * training curves saved to plots/
"""

from __future__ import annotations

import json
import os
import time
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader

from config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    PLOTS_DIR,
)
from data_pipeline import prepare_datasets
from tft_model import TemporalFusionTransformer
import utils


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap the PyTorch Datasets from data_pipeline.py into DataLoaders.

    We shuffle the training set, but NOT the validation set.
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


def run_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    train: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Run a single epoch (train or validation).

    Args:
        model:
            The TFT model.
        data_loader:
            DataLoader for train or validation set.
        criterion:
            BCEWithLogitsLoss (possibly with pos_weight).
        device:
            torch.device("cuda") or torch.device("cpu").
        threshold:
            Classification threshold used for metrics.
        train:
            If True, run in training mode (with backprop).
            If False, run in evaluation mode (no gradient updates).
        optimizer:
            Optimizer to use when train=True. Can be None when train=False.
        max_grad_norm:
            Max norm for gradient clipping (only used when train=True).

    Returns:
        Dictionary with:
            - "loss"
            - "accuracy"
            - "precision"
            - "recall"
            - "f1"
            - "auc"
    """
    if train:
        if optimizer is None:
            raise ValueError("optimizer must be provided when train=True.")
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true = []
    all_prob = []

    num_samples = len(data_loader.dataset)

    for batch in data_loader:
        # Batch can be:
        #   - (x_past, y)
        #   - (x_past, x_future, y)
        if len(batch) == 2:
            x_past, y = batch
            x_future = None
        else:
            x_past, x_future, y = batch

        x_past = x_past.to(device)
        y = y.to(device)

        if x_future is not None:
            x_future = x_future.to(device)

        # Forward pass
        with torch.set_grad_enabled(train):
            logits = model(x_past, x_future)         # (B, 1)
            loss = criterion(logits.view(-1), y.view(-1))

            if train:
                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size

        # Collect probabilities and labels for metrics
        probs = torch.sigmoid(logits).detach().cpu().view(-1)
        all_prob.append(probs)
        all_true.append(y.detach().cpu().view(-1))

    if num_samples == 0:
        raise ValueError("DataLoader has zero samples. Check your dataset/split sizes.")

    all_true_tensor = torch.cat(all_true)
    all_prob_tensor = torch.cat(all_prob)

    avg_loss = total_loss / num_samples
    metrics = utils.compute_classification_metrics(
        y_true=all_true_tensor,
        y_prob=all_prob_tensor,
        threshold=threshold,
    )
    metrics["loss"] = float(avg_loss)

    return metrics


def main() -> None:
    # -------------------------
    # 1. Reproducibility & device
    # -------------------------

    utils.set_seed(TRAINING_CONFIG.seed)
    device = utils.get_device()
    print(f"[train_tft] Using device: {device}")

    # Make sure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


    # -------------------------
    # 2. Build datasets & loaders
    # -------------------------

    print("[train_tft] Preparing datasets...")
    train_ds, val_ds, test_ds, scalers = prepare_datasets()

    print(f"[train_tft] Train samples: {len(train_ds)}")
    print(f"[train_tft] Val samples:   {len(val_ds)}")
    print(f"[train_tft] Test samples:  {len(test_ds)}")

    # -------------------------
    # 2a. Class balance & pos_weight
    # -------------------------

    # train_ds.labels is a 1D torch tensor of 0.0 / 1.0 on CPU (from BTCTFTDataset)
    train_labels_np = train_ds.labels.numpy()
    num_pos = float((train_labels_np == 1.0).sum())
    num_neg = float((train_labels_np == 0.0).sum())
    total = num_pos + num_neg

    if total == 0:
        raise ValueError("[train_tft] Training set has zero samples; check your splits.")

    pos_frac = num_pos / total
    print(
        f"[train_tft] Class balance (train): "
        f"negatives={int(num_neg)}, positives={int(num_pos)} "
        f"(pos_frac={pos_frac:.4f})"
    )

    # Decide which pos_weight to use according to config semantics
    if TRAINING_CONFIG.pos_weight < 0:
        # Auto-compute from data
        if num_pos == 0:
            raise ValueError(
                "[train_tft] No positive samples in training set; cannot compute pos_weight."
            )
        pos_weight_value = num_neg / num_pos
        TRAINING_CONFIG.pos_weight = float(pos_weight_value)  # store for reference
        print(
            f"[train_tft] Auto-computed pos_weight={pos_weight_value:.4f} "
            f"(= #neg / #pos on training labels)."
        )
    elif TRAINING_CONFIG.pos_weight == 1.0:
        # Explicitly no re-weighting
        pos_weight_value = 1.0
        print("[train_tft] Using unweighted BCE (pos_weight=1.0 from config).")
    else:
        # User-specified fixed value
        pos_weight_value = float(TRAINING_CONFIG.pos_weight)
        print(
            f"[train_tft] Using manual pos_weight from config: "
            f"{pos_weight_value:.4f}"
        )

    # Now we can build DataLoaders as before
    batch_size = TRAINING_CONFIG.batch_size
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
    )


    # -------------------------
    # 3. Model, loss, optimizer
    # -------------------------

    print("[train_tft] Initializing model...")
    model = TemporalFusionTransformer(MODEL_CONFIG).to(device)

    # Binary classification with logits, using the pos_weight we decided above
    if pos_weight_value == 1.0:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("[train_tft] Using BCEWithLogitsLoss with pos_weight=1.0 (no reweighting).")
    else:
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"[train_tft] Using BCEWithLogitsLoss with pos_weight={pos_weight_value:.4f}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG.learning_rate,
        weight_decay=TRAINING_CONFIG.weight_decay,
    )

    print(model)  # optional: prints architecture


    # -------------------------
    # 4. Training loop
    # -------------------------

    num_epochs = TRAINING_CONFIG.num_epochs
    threshold = TRAINING_CONFIG.threshold

    # History dictionary for logging & plotting
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
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

    best_val_f1 = -1.0
    best_model_path = os.path.join(MODELS_DIR, "tft_btc_best.pth")

    print(f"[train_tft] Starting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        print(f"\n[train_tft] Epoch {epoch}/{num_epochs}")

        # ---- Train epoch ----
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            train=True,
            optimizer=optimizer,
            max_grad_norm=1.0,
        )

        # ---- Validation epoch ----
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            train=False,
            optimizer=None,  # not used when train=False
            max_grad_norm=1.0,
        )

        # ---- Log metrics ----
        history["epoch"].append(epoch)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

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

        # ---- Save best model (by validation F1) ----
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"[train_tft] New best model saved with Val F1={best_val_f1:.4f} "
                f"to {best_model_path}"
            )

    print("\n[train_tft] Training finished.")
    print(f"[train_tft] Best Val F1: {best_val_f1:.4f}")


    # -------------------------
    # 5. Save experiment history & training curves
    # -------------------------

    run_id = time.strftime("tft_run_%Y%m%d_%H%M%S")

    # Save history as JSON
    history_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train_tft] History saved to {history_path}")

    # Save training curves plot
    curves_path = os.path.join(PLOTS_DIR, f"{run_id}_training_curves.png")
    utils.plot_training_curves(history, curves_path)
    print(f"[train_tft] Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()

