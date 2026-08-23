from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def split_timewise(
    df: pd.DataFrame,
    validation_fraction=0.20,
    test_fraction=0.20,
    embargo=0,
):
    """Chronological train/validation/test split with an optional purge gap.

    The embargo prevents labels near a split boundary from using future
    observations that belong to the next split. For an h-step-ahead target,
    use ``embargo=h``.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")

    n = len(df)
    train_end = int(n * (1 - validation_fraction - test_fraction))
    val_end = int(n * (1 - test_fraction))

    train = df.iloc[: max(0, train_end - embargo)].copy()
    val = df.iloc[train_end + embargo : max(train_end + embargo, val_end - embargo)].copy()
    test = df.iloc[val_end + embargo :].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Embargo is too large for the requested split fractions")

    return train, val, test


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "ic": float(corr) if np.isfinite(corr) else 0.0,
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def backtest_signal(
    returns,
    predictions,
    transaction_cost_bps=1.0,
    threshold=0.00008,
    max_position=1.0,
):
    """Simple execution-aware backtest with one-step execution delay."""
    returns = np.asarray(returns)
    predictions = np.asarray(predictions)

    position = np.where(
        predictions > threshold,
        max_position,
        np.where(predictions < -threshold, -max_position, 0.0),
    )
    position = np.roll(position, 1)
    position[0] = 0.0

    turnover = np.abs(np.diff(np.r_[0.0, position]))
    costs = turnover * transaction_cost_bps / 10_000.0
    pnl = position * returns - costs

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    return {
        "mean_pnl_per_step": float(np.mean(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "annualization_free_sharpe": float(
            np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(len(pnl))
        ),
        "max_drawdown": float(np.min(drawdown)),
        "turnover": float(np.sum(turnover)),
        "hit_rate": float(np.mean(pnl > 0)),
    }


def fair_value(book_microprice, forecast_return):
    """Map a return forecast into a fair-value estimate."""
    return book_microprice * np.exp(forecast_return)
