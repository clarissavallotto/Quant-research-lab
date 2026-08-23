# Research Walkthrough

## Hypothesis

**H1:** short-horizon order-book imbalance and signed order flow contain incremental information about the next mid-price move.

### Step 1 — Define the target

Use:

\[
y_t = \log(M_{t+h}/M_t)
\]

where `M` is mid-price and `h` is a short horizon.

Why log returns? They are additive across time and make the target scale easier to interpret.

### Step 2 — Build causal features

Examples:

- bid/ask size imbalance,
- raw microprice for pricing,
- normalized microprice deviation for prediction and drift monitoring,
- normalized signed flow,
- fast-vs-slow flow momentum,
- recent volatility,
- spread,
- trade intensity.

Every feature must be computable using information available at or before `t`.

### Step 3 — Establish a baseline

Start with Ridge regression.

This asks whether the relationship is approximately linear and gives an interpretable reference point.

### Step 4 — Test nonlinear structure

Use a random forest.

If it improves out-of-sample prediction, investigate *which interactions* create the improvement instead of assuming the model is automatically better.

### Step 5 — Test deep learning

The optional MLP is deliberately small.

A useful question is not "can a neural network fit the data?" It is:

> "Does the neural network produce incremental, stable, cost-adjusted signal out of sample?"

If not, the extra complexity is not justified.

### Step 6 — Economic evaluation

Convert predictions into positions only when confidence exceeds a threshold.

Include:

- one-step execution delay,
- transaction costs,
- position limits,
- turnover.

This connects statistical evidence to trading performance.

### Step 7 — Production monitoring

Track rolling:

- information coefficient,
- directional accuracy,
- PnL,
- feature drift.

When performance deteriorates, distinguish among:

1. data quality failure,
2. feature drift,
3. target/regime shift,
4. model degradation,
5. execution-cost changes.

## Scaling this to petabytes

For a real market-data environment, I would change the data layer, not the research logic:

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

Important engineering controls include point-in-time correctness, reproducibility, schema validation, deterministic feature definitions, experiment metadata, and avoiding accidental leakage across partitions.
