import pandas as pd

from src.qr_research.data import generate_market_data


def test_market_data_schema_and_size():
    data = generate_market_data(500, seed=1)
    assert len(data) == 500
    assert isinstance(data["timestamp"].dtype, pd.DatetimeTZDtype) is False
    assert {"mid", "bid", "ask", "bid_size", "ask_size", "signed_volume"}.issubset(data.columns)
    assert (data["ask"] > data["bid"]).all()
