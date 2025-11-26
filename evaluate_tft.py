"""
evaluate_tft.py

Phase 4: Evaluate the trained Temporal Fusion Transformer (TFT)
on the held-out test period and generate thesis-ready plots.

This script:
- Rebuilds datasets via data_pipeline.prepare_datasets()
- Loads the best saved TFT model from models/
- Runs inference on the test set
- Computes final metrics (accuracy, precision, recall, F1, AUC)
- Saves:
    * test metrics as JSON under experiments/
    * confusion matrix plot under plots/
    * ROC curve plot under plots/
"""

from __future__ import annotations

import json
import os
import time
from typing import Tuple

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


def create_test_dataloader(test_dataset, batch_size: int) -> DataLoader:
    """
    Wrap the test dataset into a DataLoader (no shuffling).
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return test_loader


def main() -> None:
    # -------------------------
    # 1. Reproducibility & device
    # -------------------------
    utils.set_seed(TRAINING_CONFIG.seed)
    device = utils.get_device()
    print(f"[evaluate_tft] Using device: {device}")

    # Ensure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -------------------------
    # 2. Build datasets & test loader
    # -------------------------
    print("[evaluate_tft] Preparing datasets...")
    train_ds, val_ds, test_ds, scalers = prepare_datasets()
    print(f"[evaluate_tft] Train samples: {len(train_ds)}")
    print(f"[evaluate_tft] Val samples:   {len(val_ds)}")
    print(f"[evaluate_tft] Test samples:  {len(test_ds)}")

    test_loader = create_test_dataloader(
        test_dataset=test_ds,
        batch_size=TRAINING_CONFIG.batch_size,
    )

    # -------------------------
    # 3. Load best model
    # -------------------------
    best_model_path = os.path.join(MODELS_DIR, "tft_btc_best.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model file not found at {best_model_path}. "
            f"Make sure you ran train_tft.py and it saved the model."
        )

    print(f"[evaluate_tft] Loading model from {best_model_path} ...")
    model = TemporalFusionTransformer(MODEL_CONFIG).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Use the same loss as in training (for reference)
    if TRAINING_CONFIG.pos_weight != 1.0:
        pos_weight = torch.tensor([TRAINING_CONFIG.pos_weight], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # -------------------------
    # 4. Run inference on test set
    # -------------------------
    print("[evaluate_tft] Running inference on test set...")

    all_true = []
    all_prob = []
    total_loss = 0.0
    num_samples = len(test_ds)

    with torch.no_grad():
        for batch in test_loader:
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

            logits = model(x_past, x_future)  # (B, 1)
            loss = criterion(logits.view(-1), y.view(-1))

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits).detach().cpu().view(-1)
            all_prob.append(probs)
            all_true.append(y.detach().cpu().view(-1))

    if num_samples == 0:
        raise ValueError("Test dataset is empty. Check your date ranges and pipeline.")

    all_true_tensor = torch.cat(all_true)
    all_prob_tensor = torch.cat(all_prob)

    avg_loss = total_loss / num_samples

    # -------------------------
    # 5. Compute metrics
    # -------------------------
    threshold = TRAINING_CONFIG.threshold
    metrics = utils.compute_classification_metrics(
        y_true=all_true_tensor,
        y_prob=all_prob_tensor,
        threshold=threshold,
    )
    metrics["loss"] = float(avg_loss)

    print("\n[evaluate_tft] Test metrics:")
    print(f"  Test loss:      {metrics['loss']:.4f}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1-score:       {metrics['f1']:.4f}")
    print(f"  ROC AUC:        {metrics['auc']:.4f}")
    print(f"  Threshold used: {threshold:.3f}")

    # -------------------------
    # 6. Save metrics & plots
    # -------------------------
    run_id = time.strftime("tft_eval_%Y%m%d_%H%M%S")

    # 6.1 Save metrics JSON
    metrics_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate_tft] Test metrics saved to {metrics_path}")

    # 6.2 Confusion matrix & ROC plots
    y_true_np = all_true_tensor.cpu().numpy()
    y_prob_np = all_prob_tensor.cpu().numpy()
    y_pred_np = (y_prob_np >= threshold).astype(int)

    cm_path = os.path.join(PLOTS_DIR, f"{run_id}_confusion_matrix.png")
    utils.plot_confusion_matrix(
        y_true=y_true_np,
        y_pred=y_pred_np,
        out_path=cm_path,
        labels=("DOWN", "UP"),
    )
    print(f"[evaluate_tft] Confusion matrix saved to {cm_path}")

    roc_path = os.path.join(PLOTS_DIR, f"{run_id}_roc_curve.png")
    utils.plot_roc_curve(
        y_true=y_true_np,
        y_prob=y_prob_np,
        out_path=roc_path,
    )
    print(f"[evaluate_tft] ROC curve saved to {roc_path}")


if __name__ == "__main__":
    main()