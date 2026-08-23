from __future__ import annotations

import numpy as np
import pandas as pd


def generate_market_data(
    n_rows: int = 60_000,
    seed: int = 42,
    regime_change: float = 0.70,
) -> pd.DataFrame:
    """Generate synthetic L1 order-book/trade-flow observations.

    The second regime changes the relationship between order flow and returns,
    creating a realistic reason to monitor model performance over time.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    regime = (t >= int(n_rows * regime_change)).astype(int)

    mid = np.empty(n_rows)
    mid[0] = 100.0
    imbalance = np.clip(rng.normal(0, 0.35, n_rows), -1, 1)
    trade_sign = rng.choice([-1.0, 1.0], n_rows)
    trade_size = rng.lognormal(mean=0.0, sigma=0.7, size=n_rows)
    spread = np.clip(rng.normal(0.012, 0.003, n_rows), 0.004, 0.03)

    # Latent signal changes sign/strength in the second regime.
    latent = (
        0.00020 * imbalance
        + 0.00008 * np.tanh(trade_sign * np.log1p(trade_size))
        + 0.00012 * imbalance * (trade_size / (1 + trade_size))
    )
    latent *= np.where(regime == 0, 1.0, -0.55)

    noise = rng.normal(0, 0.00075, n_rows)
    returns = latent + noise
    mid[1:] = mid[0] * np.exp(np.cumsum(returns[1:]))

    bid = mid - spread / 2
    ask = mid + spread / 2
    bid_size = rng.lognormal(4.2 + 0.8 * imbalance, 0.45)
    ask_size = rng.lognormal(4.2 - 0.8 * imbalance, 0.45)

    # Signed volume creates an observable proxy for aggressive order flow.
    signed_volume = trade_sign * trade_size

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="ms"),
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "trade_sign": trade_sign,
            "trade_size": trade_size,
            "signed_volume": signed_volume,
            "regime": regime,
        }
    )


def iter_market_chunks(df: pd.DataFrame, chunk_size: int = 10_000):
    """Yield data chunks to mimic an out-of-core processing interface."""
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size].copy()
