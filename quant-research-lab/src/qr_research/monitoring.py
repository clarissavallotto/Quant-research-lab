from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_metrics(y_true, y_pred, pnl, window=1000):
    """Production-style rolling diagnostics."""
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    pnl = pd.Series(pnl)

    rolling_ic = y_true.rolling(window).corr(y_pred)
    rolling_acc = (
        (np.sign(y_true) == np.sign(y_pred)).astype(float).rolling(window).mean()
    )
    rolling_pnl = pnl.rolling(window).sum()

    return pd.DataFrame(
        {
            "rolling_ic": rolling_ic,
            "rolling_directional_accuracy": rolling_acc,
            "rolling_pnl": rolling_pnl,
        }
    )


def psi(reference, current, bins=10):
    """Population Stability Index for a simple drift diagnostic."""
    reference = np.asarray(reference)
    current = np.asarray(current)

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.maximum(ref_counts / max(len(reference), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(len(current), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift_report(reference_df, current_df, columns):
    return {
        col: psi(reference_df[col].dropna(), current_df[col].dropna())
        for col in columns
    }
