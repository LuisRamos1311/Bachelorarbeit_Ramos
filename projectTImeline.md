Project timeline

Phase 0 – Setup & skeleton (low-stress start)
Goal: Have the empty project structure + basic placeholders.
Create:
    Folders: data/, models/, experiments/, plots/
    Files:
        config.py
        utils.py
        empty skeletons for:
            data_pipeline.py
            tft_model.py
            train_tft.py
            evaluate_tft.py
            trading_simulation.py
            sentiment_features.py (stub, just TODO for later)
Updates later:
    config.py will be updated in pretty much every later phase (adding hyperparameters, paths, etc.).
    utils.py will grow slowly as you see repeated code.


Phase 1 – Data pipeline (core foundation)
Goal: Be able to turn BTC CSVs into clean PyTorch datasets with up/down labels.
Work mainly in:
    data_pipeline.py
    config.py
    utils.py (for small helpers if needed)
Tasks:
    Implement:
        BTC loader (hourly/daily → daily OHLCV).
        Technical indicators.
        Train/val/test split logic.
        Scaling (fit on train, apply to others).
        Sequence creation for TFT (past window → next-day up/down).
        BTCTFTDataset class.
Updates to other files:
    Update config.py with:
        List of features.
        Date ranges.
        Sequence length.
    Add a tiny stub to data_pipeline.py where sentiment will be merged later:
        For now, it can just skip sentiment if none is provided.


Phase 2 – TFT model definition
Goal: Have a working TemporalFusionTransformer class that runs on dummy data.
Work mainly in:
    tft_model.py
    config.py
Tasks:
    Implement a simplified TFT:
        Input projection, LSTM encoder, attention, final dense layer.
    Make sure forward() accepts (batch, seq_len, n_features) and returns (batch, 1) logits.
Updates to other files:
    Update config.py with:
        input_size (number of features from data_pipeline.py).
        Model hyperparams (hidden_size, num_heads, dropout).
    You might write a tiny test block inside tft_model.py or in a scratch script to:
        Create random tensor,
        Run it through the model,
        Check the shapes.
Expands to advanced features!!

Phase 3 – Training script
Goal: Train TFT end-to-end on your BTC data.
Work mainly in:
    train_tft.py
    utils.py
    config.py
Tasks:
    In train_tft.py:
        Load config.
        Build train & validation DataLoaders via data_pipeline.py.
        Instantiate model from tft_model.py.
        Set up optimizer, loss (BCEWithLogitsLoss).
        Add training loop with:
            Loss logging,
            Basic metrics (accuracy, F1),
            Saving best model to models/.
    Use utils.py for:
        Setting random seed,
        Metric calculation,
        Plotting training curves into plots/.
Updates to other files:
    Might adjust data_pipeline.py slightly (batch shapes, feature order).
    Might adjust model hyperparams in config.py based on first runs.
Further Improvements
    Full training loop
        Add to train_tft.py:
            A loop over epoch in range(TRAINING_CONFIG.num_epochs):
                model.train() phase:
                    Loop over train_loader.
                    Forward pass: logits = model(x_past, x_future).
                    Compute loss, backprop, optimizer step.
                    Collect y_true and y_prob for metrics.
                model.eval() phase:
                    Loop over val_loader under torch.no_grad().
                    Compute validation loss and metrics.
                Call utils.compute_classification_metrics(...) for train & val each epoch.    
    Model saving and experiment history
        Track best validation metric (e.g. F1 or loss).
        Save the best model to models/ (e.g. tft_btc_best.pth).
        Accumulate history per epoch (loss, accuracy, F1, etc.).
        Save that history as a JSON in experiments/ so you can refer back to it.
        This is the TFT equivalent of what your LSTM train_model script did.
    extras for stability
        Not strictly required, but nice:
            Automatic pos_weight computation from training labels (instead of a fixed 1.0).
            Gradient clipping (e.g. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)) to stabilise training.
            LR scheduler (e.g. ReduceLROnPlateau or StepLR).
        These can be mentioned as enhancements / design choices in your thesis.


Phase 4 – Evaluation & plots
Goal: Compute final metrics on a held-out test period and generate figures for the thesis.
Work mainly in:
    evaluate_tft.py
    utils.py
    config.py
Tasks:
    In evaluate_tft.py:
        Load test DataLoader from data_pipeline.py.
        Load trained model from models/.
        Generate predictions and apply threshold.
        Compute:
            Accuracy, precision, recall, F1.
        Create and save confusion matrix and maybe ROC curve in plots/.
Updates to other files:
    You might move some metric/plot code from train_tft.py into utils.py so both train and eval reuse it.
    Can store test metrics summary (e.g., JSON) under experiments/.

After the first experiment the project only predicted ups when only 24% of them were actually ups.
1. First Experiment is going to be tuning the treshold to try to get a better result
    1. The TFT isn’t totally clueless (AUC ~0.68), but its sigmoid outputs are badly calibrated and sit mostly on one side of 0.5, 
    so with a fixed threshold we end up predicting only UP.
    2. The histograms strongly confirm our hypothesis that the main issue is calibration / bias of the output layer + class-distribution shift, not a complete lack of signal.
    3. The TFT model’s predicted probabilities are poorly calibrated: almost all samples receive a high UP probability around 0.6–0.7, 
    regardless of whether the actual next-day return is up or down.
    4. On the validation set, where UP days are the majority (~60%), 
    predicting UP for every day already yields a surprisingly high F1 (~0.75).
    5. The F1 curve is flat for all thresholds below ~0.6, so the grid search picks the first value in the grid, 0.05.
    6. On the test set, UP days are the minority (~24%), so the same “always UP” operating point leads to very poor accuracy and F1, 
    even though the ROC AUC (~0.685) indicates that the model still has some ranking ability.
    7. The histograms clearly show that the model does not spread probabilities across [0,1] in a meaningful way; 
    instead, it compresses everything into a narrow high-probability band, which makes threshold tuning ineffective and leads to trivial all-UP predictions.

What we’ve learned from Experiment 2a (pos_weight)
    1. Automatically setting pos_weight = #neg/#pos on the training labels yielded pos_weight ≈ 0.84, 
    since the training period actually contains more UP days than DOWN.
    2. Training with this reweighted BCE did not materially change validation F1 (still ~0.75 with an almost “always UP” classifier).
    3. On the test period, the same model produced balanced UP/DOWN predictions at an optimal threshold around 0.55, 
    but the ROC AUC dropped to ~0.46, indicating essentially random or slightly anti-correlated ranking.
    4. Therefore, class-weighting alone is not sufficient; the more fundamental problems lie in:
        the label definition (UP_THRESHOLD = 0.0),
        and the mismatch between the label distribution in the training/validation periods and the test period.
    That’s actually a very useful negative result: you can say “we tried standard class-weighting, 
    here’s why it didn’t help, and here’s why we changed the label design instead.”

Phase 5 – Trading simulation (optional but nice)
Goal: Show a simple backtest using TFT predictions.
Work mainly in:
    trading_simulation.py
    data_pipeline.py
    config.py
Tasks:
    Reuse data loading & scaling from data_pipeline.py.
    For a selected period (e.g. 2024):
        Build sliding windows day by day.
        Use the trained TFT to get probability of “up”.
        Apply a simple rule (e.g., long if p(up) > 0.6).
        Track portfolio vs buy-and-hold.
    Save equity curve plot under plots/.
Updates:
    Might add some small helper functions in utils.py for PnL calculations and plotting.


Future Phase – Reddit sentiment
This is after the main thesis code works.
Work in:
    sentiment_features.py
    data_pipeline.py
    config.py
You’ll:
    Implement sentiment loading and daily aggregation.
    Extend data_pipeline.py to join sentiment features into the feature set.
    Retrain and compare performance with/without sentiment.