After Phase 4 - Evaluation

After the first run of the model the project only predicted ups when only 24% of them were actually ups.
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


2. What we’ve learned from Experiment 2a (pos_weight)
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