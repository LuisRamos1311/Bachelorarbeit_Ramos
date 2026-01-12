### Objective and Selection Logic

This thesis aims to build a **leak-free, uncertainty-aware forecasting pipeline** that can be translated into a **risk-averse long/cash allocation rule**. The goal is not to beat buy-and-hold in every regime, but to (i) participate meaningfully in bull markets when confidence is high and (ii) preserve capital in bear markets by moving to cash.

Both final candidates (9c and 9d) produce **quantile forecasts** of forward returns and generate a long signal using an uncertainty-adjusted score:

[
\text{score}=\frac{q_{0.5}}{(q_{0.9}-q_{0.1})+\epsilon}
]

A position is taken only if (q_{0.5}>0) and (\text{score}\ge \tau), where (\tau) is selected on the validation set to maximize risk-adjusted performance.

### Final Multi-Regime Evaluation Setup

To avoid selecting a model that only works in a single regime, the decisive comparison uses a long test period spanning multiple market phases:

* **Train:** 2016-01-01 → 2020-12-31
* **Validation:** 2021-01-01 → 2021-10-31
* **Test:** 2021-11-01 → 2024-12-31

All strategy metrics are reported **net of transaction costs and slippage**, alongside a buy-and-hold baseline.

### Results: Forecast Accuracy vs Trading Usefulness

On the test set, **9d slightly improves forecast metrics** relative to 9c (lower pinball loss and MAE@24h). However, the thesis outcome is determined by **strategy-level robustness**, not forecasting metrics alone.

The trading results show a clear difference in behaviour:

* **Experiment 9c:** long-rate **0.279**, net cumulative return **+0.208**, net Sharpe **0.345**, max drawdown **0.335**.
* **Experiment 9d:** long-rate **0.020**, net cumulative return **+0.017**, net Sharpe **0.104**, max drawdown **0.168**.
* **Buy & hold:** net cumulative return **+0.518**, max drawdown **0.767**.

In other words, 9d becomes **almost always cash** on the decisive long-horizon test. This reduces drawdowns, but largely by eliminating exposure, which prevents meaningful participation in the 2023–2024 bull run. In contrast, 9c remains risk-averse (far lower drawdown than buy-and-hold) while still taking enough risk to achieve a non-trivial net return.

### Conclusion: Why 9c is the Final Model

Although 9d improves forecast accuracy, it does not translate into superior decision-making under the fixed signal rule and validation-based threshold selection. Repeated attempts to stabilize 9d’s thresholding and calibration did not make it outperform 9c in practice. Therefore, **Experiment 9c is selected as the final model** because it best matches the thesis objective: a robust, risk-averse long/cash strategy that preserves capital in adverse regimes **without collapsing into near-zero participation** across a multi-regime evaluation window.