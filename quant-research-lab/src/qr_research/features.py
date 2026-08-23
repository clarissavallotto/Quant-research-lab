from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, horizon: int = 5):
    """Build causal microstructure features and a future-return target."""
    x = df.copy()

    x["microprice"] = (
        x["ask"] * x["bid_size"] + x["bid"] * x["ask_size"]
    ) / (x["bid_size"] + x["ask_size"])

    # Price-level invariant version is preferable for modelling/monitoring.
    x["microprice_deviation"] = (
        x["microprice"] - x["mid"]
    ) / x["mid"]

    x["book_imbalance"] = (
        x["bid_size"] - x["ask_size"]
    ) / (x["bid_size"] + x["ask_size"])

    rolling_mean = x["signed_volume"].rolling(100, min_periods=20).mean()
    rolling_std = x["signed_volume"].rolling(100, min_periods=20).std()
    x["signed_flow_z"] = (x["signed_volume"] - rolling_mean) / rolling_std.replace(0, np.nan)

    x["flow_ema_fast"] = x["signed_volume"].ewm(span=20, adjust=False).mean()
    x["flow_ema_slow"] = x["signed_volume"].ewm(span=100, adjust=False).mean()
    x["flow_momentum"] = x["flow_ema_fast"] - x["flow_ema_slow"]

    x["mid_return_1"] = x["mid"].pct_change()
    x["mid_return_20"] = x["mid"].pct_change(20)
    x["realized_vol_100"] = x["mid_return_1"].rolling(100).std()
    x["trade_intensity"] = x["trade_size"].rolling(50).mean()
    x["signed_intensity"] = x["signed_volume"].abs().rolling(50).mean()

    future_mid = x["mid"].shift(-horizon)
    x["target_return"] = np.log(future_mid / x["mid"])

    feature_cols = [
        "book_imbalance",
        "signed_flow_z",
        "flow_momentum",
        "mid_return_1",
        "mid_return_20",
        "realized_vol_100",
        "spread",
        "trade_intensity",
        "signed_intensity",
        "microprice_deviation",
    ]

    x = x.dropna(subset=feature_cols + ["target_return"]).reset_index(drop=True)
    return x, feature_cols
