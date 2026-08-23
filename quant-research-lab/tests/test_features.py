import numpy as np
import pandas as pd

from src.qr_research.data import generate_market_data
from src.qr_research.features import build_features


def test_features_have_no_future_columns():
    raw = generate_market_data(2000, seed=1)
    data, cols = build_features(raw, horizon=5)
    assert len(data) > 100
    assert "target_return" not in cols
    assert all(np.isfinite(data[cols].to_numpy()).all(axis=0))


def test_target_is_forward_return():
    raw = generate_market_data(1000, seed=2)
    data, _ = build_features(raw, horizon=5)
    assert data["target_return"].abs().mean() > 0
