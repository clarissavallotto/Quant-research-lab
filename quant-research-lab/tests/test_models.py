import numpy as np

from src.qr_research.data import generate_market_data
from src.qr_research.features import build_features
from src.qr_research.models import get_model_specs


def test_all_candidate_models_fit_and_predict():
    raw = generate_market_data(2500, seed=3)
    data, features = build_features(raw, horizon=5)
    X = data[features].iloc[:-1000]
    y = data["target_return"].iloc[:-1000]
    X_test = data[features].iloc[-1000:]

    for spec in get_model_specs(seed=3):
        spec.model.fit(X, y)
        pred = spec.model.predict(X_test)
        assert pred.shape == (len(X_test),)
        assert np.isfinite(pred).all()


def test_mlp_can_be_disabled_for_fast_iterations():
    specs = get_model_specs(seed=3, include_mlp=False)
    names = {spec.name for spec in specs}
    assert names == {"ridge", "random_forest"}
