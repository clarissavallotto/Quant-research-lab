# Quantitative Research Lab

The project demonstrates a complete research loop around:

- **Pricing:** estimate a model fair value from current market state and forecast return.
- **Forecasting:** predict short-horizon mid-price returns.
- **Signal discovery:** test order-book and order-flow variables for incremental predictive information.
- **Method selection:** compare an interpretable statistical baseline, a nonlinear ensemble, and a configurable neural model.
- **Economic validation:** evaluate predictions after transaction costs, thresholds, position limits, turnover, and execution delay.
- **Production monitoring:** track rolling predictive performance, trading performance, prediction behavior, and feature distribution drift.
- **AI-enabled research:** use a small neural model as a hypothesis-testing tool rather than assuming more complexity is better.
- **Research / trading / engineering constraints:** encode assumptions that connect model output to executable trading decisions.
- **Scale awareness:** demonstrate the research logic locally while documenting how the data layer would scale to large distributed market-data systems.

The local experiment uses **synthetic market data** so it can run without proprietary exchange data. The resulting PnL and predictive metrics are therefore demonstrations of methodology, not evidence of a deployable trading edge.

## Quick start

The project can be run directly in Google Colab. After uploading or cloning the repository:

### 1. Install dependencies
```python
%cd /content/quant-research-lab
!pip install -q -r requirements.txt
```

### 2. Run the tests
```python
!pytest -q
```

The repository contains tests for data generation, feature construction, model configuration, evaluation, monitoring, the end-to-end experiment, and presentation artifact generation.

### 3. Run the complete experiment
```python
!python run_experiment.py
```

This executes the complete pipeline:

```text
Synthetic market data
        ↓
Causal feature engineering
        ↓
Purged chronological train / validation / test split
        ↓
Ridge / Random Forest / optional MLP
        ↓
Validation model comparison
        ↓
Economic model selection
        ↓
Out-of-sample forecasting
        ↓
Fair-value estimation
        ↓
Execution-aware backtest
        ↓
Production-style monitoring
        ↓
Feature drift analysis
        ↓
Research report + charts
```

### 4. Inspect the results

The experiment creates:

```text
artifacts/
├── report.json
├── monitoring.csv
├── test_predictions.csv
├── research_report.html
└── charts/
```

Open `artifacts/research_report.html` to see the complete research presentation.

To display the generated report inside Colab:

```python
from IPython.display import IFrame, display

display(
    IFrame(
        "artifacts/research_report.html",
        width="100%",
        height=1200,
    )
)
```

## Project structure

```text
quant-research-lab/
├── configs/
│   └── default.yaml
├── notebooks/
│   └── research_walkthrough.md
├── src/
│   └── qr_research/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── evaluation.py
│       ├── monitoring.py
│       ├── experiment.py
│       └── presentation.py
├── tests/
├── run_experiment.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Research question

The central hypothesis is:

> **Do order-book and order-flow features contain exploitable information about short-horizon mid-price returns, and does that information remain economically useful after realistic trading frictions?**

The project deliberately separates three related questions:

1. **Pricing:** what is a reasonable fair value given current book state and the model's expected return?
2. **Forecasting:** what is the expected return over the next `h` observations?
3. **Signal discovery:** which observable market-state variables contain incremental information about that future return?

## Feature engineering and causality

The feature set includes variables such as:

- bid/ask size imbalance,
- microprice,
- normalized microprice deviation,
- signed-flow statistics,
- fast-vs-slow flow momentum,
- recent volatility,
- spread,
- trade intensity.

Every predictive feature is constructed from information available at or before the prediction timestamp.

The project distinguishes two uses of microprice:

- **Raw microprice:** used as a market-state input to fair-value estimation.
- **Microprice deviation:** `(microprice - mid) / mid`, used as a normalized predictive feature and for drift monitoring so ordinary price trends are not mistaken for microstructure drift.

## Time-series validation and leakage control

Random train/test splitting would be inappropriate for this problem because future observations can influence labels and market conditions are time-dependent.

The experiment therefore uses chronological train / validation / test splits with an embargo equal to the forecast horizon:

```text
TRAIN | embargo | VALIDATION | embargo | TEST
```

This prevents boundary labels from crossing into the next split.

The project also uses causal feature construction rather than relying only on column-name checks. The intended research invariant is that changing observations after time `t` must not change features computed at `t`.

## Model selection

`src/qr_research/models.py` provides three candidate approaches:

- **Ridge regression:** interpretable statistical baseline for approximately linear relationships.
- **Random forest:** nonlinear model for interactions and threshold effects.
- **MLP:** small neural model for testing whether additional representation capacity produces incremental out-of-sample value.

The MLP is configurable:

```yaml
models:
  ridge: true
  random_forest: true
  mlp: true
```

For fast research iterations:

```yaml
models:
  ridge: true
  random_forest: true
  mlp: false
```

The same evaluation and presentation pipeline is used in both cases.

### Why model selection is economic

The project deliberately separates **statistical forecasting quality** from **economic trading performance**.

A model can have a better IC or RMSE while producing worse trading results after transaction costs, turnover and execution assumptions. Candidate models are therefore compared on both dimensions, and the final model is selected using validation economic performance.

This is intended to demonstrate a research principle:

> Predictive accuracy is evidence about a signal; it is not automatically evidence of tradable value.

## Regime design and monitoring

The synthetic generator supports a configurable relationship change through `regime_change` in `configs/default.yaml`.

The default is:

```yaml
regime_change: 0.70
```

With a 20% test set, this places the regime transition **inside the test period**, allowing the experiment to compare early and late out-of-sample behavior.

The monitoring layer calculates:

- rolling information coefficient,
- rolling directional accuracy,
- rolling strategy PnL,
- prediction behavior,
- feature distribution drift.

The current drift analysis explicitly compares the **early and late portions of the test period**. It should therefore be interpreted as evidence of changing market conditions, not as a complete regime-classification system.

A useful monitoring question is:

> Did performance deteriorate because the signal changed, because feature distributions moved, because the target relationship changed, or because trading constraints became less favorable?

## Economic evaluation

The backtest incorporates:

- transaction costs,
- prediction thresholds,
- position limits,
- turnover,
- one-period execution delay.

This connects statistical model output to a simplified trading decision and prevents the project from treating raw prediction accuracy as the final objective.

## Pricing / fair value

The pricing layer combines the current microprice with the forecast return:

```text
Current microprice + forecast return
                ↓
          model fair value
```

The resulting fair-value series is saved to `test_predictions.csv` and visualized against the market mid-price.

## Model-comparison presentation

Prediction quality and economic performance are intentionally split into **four independent charts** so that metrics with different units are never presented on a shared axis:

```text
04a_validation_ic.png
    Ridge
    Random Forest
    MLP

04b_validation_rmse.png
    Ridge
    Random Forest
    MLP

05a_validation_pnl.png
    Ridge
    Random Forest
    MLP

05b_validation_drawdown.png
    Ridge
    Random Forest
    MLP
```

This makes the distinction between statistical evidence and economic evidence visually explicit.

## Generated presentation artifacts

Running `python run_experiment.py` produces:

```text
artifacts/
├── report.json
├── monitoring.csv
├── test_predictions.csv
├── research_report.html
└── charts/
    ├── 01_mid_price.png
    ├── 02_order_book_imbalance.png
    ├── 03_signal_relationship.png
    ├── 04a_validation_ic.png
    ├── 04b_validation_rmse.png
    ├── 05a_validation_pnl.png
    ├── 05b_validation_drawdown.png
    ├── 06_prediction_vs_realized.png
    ├── 07_fair_value_vs_mid.png
    ├── 08_cumulative_pnl.png
    ├── 09_rolling_ic.png
    ├── 10_rolling_pnl.png
    └── 11_feature_drift.png
```

The HTML report presents the research workflow as:

```text
Hypothesis
   ↓
Signal evidence
   ↓
Model comparison
   ↓
Pricing / fair value
   ↓
Economic validation
   ↓
Production monitoring
   ↓
Next research question
```

## Scaling to large market-data systems

The local experiment is deliberately small enough to run on a laptop or Google Colab. The research logic is intended to remain the same when the data layer is replaced with a large-scale architecture:

```text
Exchange feeds
    ↓
Immutable event store
    ↓
Partition by date / venue / instrument
    ↓
Columnar storage
    ↓
Distributed feature computation
    ↓
Point-in-time feature tables
    ↓
Chronological experiment splits
    ↓
Model training
    ↓
Backtest / simulation
    ↓
Production model
    ↓
Monitoring + experiment registry
```

A production implementation would add distributed compute, incremental statistics, feature stores, data lineage, schema validation, point-in-time joins, experiment versioning, and stronger execution simulation.

## AI-enabled research workflow

Deep learning is treated as a hypothesis-testing tool rather than an automatic answer:

```text
Hypothesis
   ↓
Feature definition
   ↓
Leakage check
   ↓
Statistical baseline
   ↓
Nonlinear model
   ↓
Neural model
   ↓
Out-of-sample evidence
   ↓
Economic simulation
   ↓
Robustness / regime analysis
   ↓
Monitoring question
   ↓
Next experiment
```

The goal is to shorten the loop from hypothesis to evidence while retaining reproducible evaluation.

## Example research questions

After running the project, investigate:

1. Does order-book imbalance predict the next mid-price move?
2. Does adding trade-flow information improve the forecast?
3. Is the nonlinear model actually better after transaction costs?
4. Does the signal weaken after the synthetic regime transition?
5. Does feature drift precede model-performance degradation?
6. How sensitive is PnL to latency?
7. What happens when the prediction threshold is doubled?
8. Does a simpler model have better stability even if its validation RMSE is slightly worse?
9. Does volatility or spread condition the strength of the signal?

## Limitations

Synthetic data cannot reproduce real exchange microstructure, including:

- queue position,
- hidden liquidity,
- adverse selection,
- exchange-specific matching rules,
- market impact,
- cancellation dynamics,
- realistic latency distributions,
- cross-venue interactions.

The backtest is therefore a research-methodology demonstration rather than a claim of live profitability.
