"""
evaluate_tft_posweight_experiment.py

Phase 4: Evaluate the trained Temporal Fusion Transformer (TFT)
on the held-out test period and generate thesis-ready plots.

This script now:
- Rebuilds datasets via data_pipeline.prepare_datasets()
- Computes class balance & pos_weight from training labels (if configured)
- Loads the best saved TFT model from models/
- Runs inference on the *validation* set to tune the decision threshold (grid search)
- Uses the best validation threshold to evaluate the *test* set
- Computes final metrics (accuracy, precision, recall, F1, AUC)
- Saves:
    * test metrics as JSON under experiments/
    * confusion matrix plot under plots/
    * ROC curve plot under plots/
    * probability histograms for val/test under plots/
"""

from __future__ import annotations

import json
import os
import time
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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


def create_eval_dataloader(dataset, batch_size: int) -> DataLoader:
    """
    Wrap a dataset into a DataLoader (no shuffling, no drop_last).
    Used for both validation and test sets.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return loader


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Run forward passes over an evaluation DataLoader and collect:
        - all true labels (1D tensor)
        - all predicted probabilities (1D tensor)
        - average BCE loss

    This works for both validation and test sets.
    """
    model.eval()

    all_true = []
    all_prob = []
    total_loss = 0.0
    num_samples = len(data_loader.dataset)

    with torch.no_grad():
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

            logits = model(x_past, x_future)  # (B, 1)
            loss = criterion(logits.view(-1), y.view(-1))

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits).detach().cpu().view(-1)
            all_prob.append(probs)
            all_true.append(y.detach().cpu().view(-1))

    if num_samples == 0:
        raise ValueError("Dataset is empty. Check your date ranges and pipeline.")

    all_true_tensor = torch.cat(all_true)
    all_prob_tensor = torch.cat(all_prob)
    avg_loss = total_loss / num_samples

    return all_true_tensor, all_prob_tensor, avg_loss


def plot_probability_histogram(
    y_prob: torch.Tensor,
    out_path: str,
    threshold: float | None = None,
    title: str = "Predicted probabilities",
    bins: int = 30,
) -> None:
    """
    Plot and save a histogram of predicted probabilities.

    Args:
        y_prob:
            1D tensor of probabilities for the positive class (after sigmoid).
        out_path:
            File path (PNG) where the plot will be saved.
        threshold:
            Optional vertical line showing the classification threshold.
        title:
            Plot title.
        bins:
            Number of histogram bins.
    """
    probs = y_prob.detach().cpu().numpy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(probs, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Predicted probability (UP class)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.2f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    # 2. Build datasets & loaders
    # -------------------------
    print("[evaluate_tft] Preparing datasets...")
    train_ds, val_ds, test_ds, scalers = prepare_datasets()
    print(f"[evaluate_tft] Train samples: {len(train_ds)}")
    print(f"[evaluate_tft] Val samples:   {len(val_ds)}")
    print(f"[evaluate_tft] Test samples:  {len(test_ds)}")

    # -------------------------
    # 2a. Class balance & pos_weight (mirror train_tft logic)
    # -------------------------
    train_labels_np = train_ds.labels.numpy()
    num_pos = float((train_labels_np == 1.0).sum())
    num_neg = float((train_labels_np == 0.0).sum())
    total = num_pos + num_neg

    if total == 0:
        raise ValueError("[evaluate_tft] Training set has zero samples; check your splits.")

    pos_frac = num_pos / total
    print(
        f"[evaluate_tft] Class balance (train): "
        f"negatives={int(num_neg)}, positives={int(num_pos)} "
        f"(pos_frac={pos_frac:.4f})"
    )

    # Same semantics as we used in train_tft:
    #   - TRAINING_CONFIG.pos_weight < 0  -> auto compute (#neg / #pos)
    #   - TRAINING_CONFIG.pos_weight == 1 -> no re-weighting
    #   - otherwise                      -> fixed manual value
    if TRAINING_CONFIG.pos_weight < 0:
        if num_pos == 0:
            raise ValueError(
                "[evaluate_tft] No positive samples in training set; cannot compute pos_weight."
            )
        pos_weight_value = num_neg / num_pos
        print(
            f"[evaluate_tft] Auto-computed pos_weight={pos_weight_value:.4f} "
            f"(= #neg / #pos on training labels)."
        )
    elif TRAINING_CONFIG.pos_weight == 1.0:
        pos_weight_value = 1.0
        print("[evaluate_tft] Using unweighted BCE (pos_weight=1.0 from config).")
    else:
        pos_weight_value = float(TRAINING_CONFIG.pos_weight)
        print(
            f"[evaluate_tft] Using manual pos_weight from config: "
            f"{pos_weight_value:.4f}"
        )

    batch_size = TRAINING_CONFIG.batch_size
    val_loader = create_eval_dataloader(val_ds, batch_size=batch_size)
    test_loader = create_eval_dataloader(test_ds, batch_size=batch_size)

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

    # Use the same loss style as in training (for reference)
    if pos_weight_value == 1.0:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("[evaluate_tft] Using BCEWithLogitsLoss with pos_weight=1.0 (no reweighting).")
    else:
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"[evaluate_tft] Using BCEWithLogitsLoss with pos_weight={pos_weight_value:.4f}")

    # Unique run ID for outputs
    run_id = time.strftime("tft_eval_%Y%m%d_%H%M%S")

    # =======================================================
    # 4. Threshold tuning on validation set (maximize F1)
    # =======================================================
    print("[evaluate_tft] Running inference on validation set for threshold tuning...")
    val_true, val_prob, val_loss = run_inference(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    # Grid of thresholds: 0.05, 0.10, ..., 0.85
    threshold_grid = [0.05 * i for i in range(1, 18)]

    best_threshold = None
    best_f1 = -1.0
    best_val_metrics = None

    for th in threshold_grid:
        m = utils.compute_classification_metrics(
            y_true=val_true,
            y_prob=val_prob,
            threshold=th,
        )
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = th
            best_val_metrics = m

    if best_threshold is None:
        raise RuntimeError("Threshold search failed – no thresholds evaluated.")

    # Attach loss for completeness
    best_val_metrics["loss"] = float(val_loss)

    print("\n[evaluate_tft] Validation threshold search (optimize F1):")
    print(f"  Best threshold on val: {best_threshold:.3f}")
    print(f"  Val F1 at best thres.: {best_f1:.4f}")
    print(f"  Val accuracy:          {best_val_metrics['accuracy']:.4f}")
    print(f"  Val precision:         {best_val_metrics['precision']:.4f}")
    print(f"  Val recall:            {best_val_metrics['recall']:.4f}")
    print(f"  Val loss:              {best_val_metrics['loss']:.4f}")

    # -------------------------
    # 4a. Plot validation probability distribution
    # -------------------------
    val_hist_path = os.path.join(PLOTS_DIR, f"{run_id}_val_prob_hist.png")
    plot_probability_histogram(
        y_prob=val_prob,
        out_path=val_hist_path,
        threshold=best_threshold,
        title="Predicted UP probabilities (validation)",
    )
    print(f"[evaluate_tft] Validation probability histogram saved to {val_hist_path}")

    # =======================================================
    # 5. Final evaluation on TEST set using tuned threshold
    # =======================================================
    print("\n[evaluate_tft] Running inference on test set with tuned threshold...")
    test_true, test_prob, test_loss = run_inference(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    test_metrics = utils.compute_classification_metrics(
        y_true=test_true,
        y_prob=test_prob,
        threshold=best_threshold,
    )
    test_metrics["loss"] = float(test_loss)
    test_metrics["threshold"] = float(best_threshold)
    # Also log the validation F1 used to pick this threshold
    test_metrics["val_f1_at_threshold"] = float(best_f1)
    # And the pos_weight we used
    test_metrics["pos_weight"] = float(pos_weight_value)

    print("\n[evaluate_tft] Test metrics (using tuned threshold):")
    print(f"  Test loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  Precision:      {test_metrics['precision']:.4f}")
    print(f"  Recall:         {test_metrics['recall']:.4f}")
    print(f"  F1-score:       {test_metrics['f1']:.4f}")
    print(f"  ROC AUC:        {test_metrics['auc']:.4f}")
    print(f"  Threshold used: {best_threshold:.3f}")

    # -------------------------
    # 5a. Plot test probability distribution
    # -------------------------
    test_hist_path = os.path.join(PLOTS_DIR, f"{run_id}_test_prob_hist.png")
    plot_probability_histogram(
        y_prob=test_prob,
        out_path=test_hist_path,
        threshold=best_threshold,
        title="Predicted UP probabilities (test)",
    )
    print(f"[evaluate_tft] Test probability histogram saved to {test_hist_path}")

    # -------------------------
    # 6. Save metrics & plots (TEST)
    # -------------------------
    # 6.1 Save metrics JSON
    metrics_path = os.path.join(EXPERIMENTS_DIR, f"{run_id}_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[evaluate_tft] Test metrics saved to {metrics_path}")

    # 6.2 Confusion matrix & ROC plots (on TEST set)
    y_true_np = test_true.cpu().numpy()
    y_prob_np = test_prob.cpu().numpy()
    y_pred_np = (y_prob_np >= best_threshold).astype(int)

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