# Final Model Selection (Experiment 9c vs 9d)

This README documents the final head-to-head comparison between **Experiment 9c** and **Experiment 9d**, and explains why **9c** is selected as the final “risk-averse long/cash” model.

The guiding idea throughout these experiments was *not* to beat buy-and-hold in every regime, but to build a **risk-aware strategy** that:
- **Participates meaningfully in bull markets** (go long when the model is confident),
- **Avoids large drawdowns in bear markets** (stay in cash when signal quality degrades),
- Uses only **causal / non-leaky information** (strict train/val/test splits and forward-looking horizons only).

---

## 1) Models Compared

### Experiment 9c (final choice)
- Uses TFT quantile forecasts (forecast band + median).
- Converts forecasts to a **long/cash signal** via a tuned threshold (τ) selected on validation.
- Behavior target: *risk-averse participation* — be in the market when conviction is high, otherwise stay out.

### Experiment 9d (final challenger)
- Same overall TFT + long/cash framing, but with the “9d upgrade” applied.
- In practice, the calibrated threshold in the final split produced an **extremely conservative exposure profile** (almost always cash), which reduced drawdowns but also prevented meaningful participation in the 2023–2024 upside.

---

## 2) Final Evaluation Protocol (the deciding test)

To avoid “cherry-picking” a single regime, the final decision was based on a **multi-regime test window** that spans late-cycle weakness + recovery + strong bull:

- **Train:** 2016-01-01 → 2020-12-31  
- **Validation:** 2021-01-01 → 2021-10-31  
- **Test:** 2021-11-01 → 2024-12-31  

This is the most important evaluation, because it measures whether a model can stay coherent when the market **changes regime multiple times**.

---

## 3) Head-to-Head Results (Test: 2021-11 → 2024-12)

| Model | Test Return (net) | Test Sharpe (net) | Max Drawdown (net) | Long-Rate (time in market) |
|------|-------------------:|------------------:|-------------------:|---------------------------:|
| **9c** | **+20.8%** | **0.35** | **0.335** | **27.9%** |
| **9d** | +1.7% | 0.10 | 0.168 | 2.0% |
| Buy & Hold | +51.8% | 0.52 | 0.767 | 100% |

**Interpretation**
- **9c** delivered a *meaningful positive return* while keeping drawdown far below buy-and-hold.  
- **9d** was **too conservative**: it spent ~98% of the time in cash, which capped drawdowns, but it also missed most of the upside in the strongest part of the cycle.
- Both models are “risk-averse,” but **9c is the only one that remained investable** over the full multi-regime horizon.

---

## 4) Why 9c is the final model

### 4.1 Generalization beats micro-optimizations
9d looked like a reasonable upgrade direction, but the final multi-regime test showed a key weakness: the validation-selected threshold produced **near-zero exposure** in the long test window, resulting in an equity curve that was essentially flat.

9c, on the other hand, maintained a **balanced exposure profile** (still risk-aware, but not frozen), which is exactly what this project aimed for:  
> “Go long in good years, hold cash in bad years — without over-trading.”

### 4.2 Better tradeoff: downside control *and* upside capture
9c does *not* try to be permanently invested. It accepts that it may underperform buy-and-hold in explosive bull phases, but it compensates by:
- reducing drawdown,
- reducing exposure during low-confidence regimes,
- still capturing enough upside to remain worthwhile over long horizons.

9d’s drawdown is lower, but it achieves that largely by **not participating**, which defeats the purpose of forecasting-driven positioning.

---

## 5) Supporting sanity checks (single-year regime tests)

Even before the final split, the project repeatedly validated the intended behavior:

### Bear regime (2022)
- 9c stayed close to flat while buy-and-hold suffered a deep drawdown (consistent with the “cash in bad years” objective).

### Bull regime (2024)
- 9c showed strong participation with high exposure, indicating the model can commit risk when confidence is high.

These regime-specific tests gave confidence that the framework is directionally correct; the final multi-regime test is what determined which variant generalizes better.

---

## 6) Final Decision

✅ **Experiment 9c is selected as the final model.**

Reason: **It is the best risk-averse model in the most realistic evaluation** (2021-11 → 2024-12), achieving a positive net return with materially reduced drawdown, while 9d became overly conservative and failed to participate in the upside.

---

## 7) Reproducibility Notes

To reproduce the decision test:
1. Set the train/val/test dates exactly as listed in Section 2.
2. Train each model on the train window.
3. Select τ on the validation window (same selection logic for both).
4. Evaluate on the full test window (2021-11 → 2024-12), including transaction cost + slippage assumptions.
5. Compare net equity curves + drawdown + exposure rate.