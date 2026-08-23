from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    model: object


def get_model_specs(seed: int = 42, include_mlp: bool = True):
    """Return the enabled statistical, nonlinear and neural model classes.

    ``include_mlp=False`` is useful for fast research iterations when the
    neural model is not part of the current experiment.
    """
    specs = [
        ModelSpec(
            "ridge",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
        ),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=50,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "mlp",
            TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(64, 32),
                                activation="relu",
                                alpha=1e-4,
                                learning_rate_init=1e-3,
                                max_iter=150,
                                early_stopping=True,
                                validation_fraction=0.10,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
                transformer=StandardScaler(),
            ),
        ),
    ]

    if not include_mlp:
        specs = [spec for spec in specs if spec.name != "mlp"]

    return specs


def try_neural_model(seed: int = 42):
    """Backward-compatible optional PyTorch model factory."""
    try:
        import torch
        from torch import nn
    except ImportError:
        return None

    torch.manual_seed(seed)

    class MLP(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return MLP
