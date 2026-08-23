import numpy as np

from src.qr_research.monitoring import feature_drift_report, rolling_metrics


def test_rolling_metrics_have_expected_columns():
    y = np.arange(20, dtype=float)
    p = y + 0.1
    pnl = np.ones(20) * 0.001
    result = rolling_metrics(y, p, pnl, window=5)
    assert {"rolling_ic", "rolling_directional_accuracy", "rolling_pnl"}.issubset(result.columns)
    assert result["rolling_pnl"].iloc[-1] > 0


def test_feature_drift_is_finite():
    reference = {"x": np.random.default_rng(1).normal(size=1000)}
    current = {"x": np.random.default_rng(2).normal(size=1000)}
    import pandas as pd
    result = feature_drift_report(pd.DataFrame(reference), pd.DataFrame(current), ["x"])
    assert np.isfinite(result["x"])
