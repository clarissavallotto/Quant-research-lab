import numpy as np

from src.qr_research.evaluation import backtest_signal, regression_metrics


def test_metrics_are_finite():
    y = np.array([0.1, -0.2, 0.3, 0.0])
    p = np.array([0.2, -0.1, 0.1, 0.1])
    metrics = regression_metrics(y, p)
    assert all(np.isfinite(list(metrics.values())))


def test_costs_reduce_pnl():
    y = np.ones(20) * 0.001
    p = np.ones(20) * 0.001
    no_cost = backtest_signal(y, p, transaction_cost_bps=0)
    with_cost = backtest_signal(y, p, transaction_cost_bps=100)
    assert with_cost["total_pnl"] < no_cost["total_pnl"]
