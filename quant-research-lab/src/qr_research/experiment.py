from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import generate_market_data
from .evaluation import (
    backtest_signal,
    fair_value,
    regression_metrics,
    split_timewise,
)
from .features import build_features
from .models import get_model_specs
from .monitoring import (
    feature_drift_report,
    rolling_metrics,
)


def run(config: dict, artifact_dir="artifacts"):

    seed = int(config["seed"])

    raw = generate_market_data(
        n_rows=int(config["n_rows"]),
        seed=seed,
        regime_change=float(config["regime_change"]),
    )

    data, feature_cols = build_features(
        raw,
        horizon=int(config["horizon"]),
    )

    train, val, test = split_timewise(
        data,
        validation_fraction=float(config["validation_fraction"]),
        test_fraction=float(config["test_fraction"]),
        embargo=int(config["horizon"]),
    )

    X_train = train[feature_cols]
    y_train = train["target_return"]

    X_val = val[feature_cols]
    y_val = val["target_return"]

    X_test = test[feature_cols]
    y_test = test["target_return"]

    results = {}
    fitted = {}

    # ---------------------------------------------------------
    # MODEL COMPARISON
    # ---------------------------------------------------------

    model_config = config.get(
        "models",
        {"ridge": True, "random_forest": True, "mlp": True},
    )
    enabled_names = {
        name for name, enabled in model_config.items() if bool(enabled)
    }

    specs = [
        spec
        for spec in get_model_specs(
            seed,
            include_mlp=("mlp" in enabled_names),
        )
        if spec.name in enabled_names
    ]

    if not specs:
        raise ValueError("At least one model must be enabled in config['models'].")

    for spec in specs:

        spec.model.fit(
            X_train,
            y_train,
        )

        val_pred = spec.model.predict(X_val)
        test_pred = spec.model.predict(X_test)

        val_metrics = regression_metrics(
            y_val,
            val_pred,
        )

        val_econ = backtest_signal(
            val["target_return"],
            val_pred,
            transaction_cost_bps=float(
                config["transaction_cost_bps"]
            ),
            threshold=float(
                config["prediction_threshold"]
            ),
            max_position=float(
                config["max_position"]
            ),
        )

        results[spec.name] = {
            "validation_prediction": val_metrics,
            "validation_economics": val_econ,
        }

        fitted[spec.name] = (
            spec.model,
            test_pred,
        )

    # ---------------------------------------------------------
    # MODEL SELECTION
    # ---------------------------------------------------------

    selected_name = max(
        results,
        key=lambda name:
        results[name]["validation_economics"]["total_pnl"],
    )

    selected_model, test_pred = fitted[selected_name]

    # ---------------------------------------------------------
    # TEST PERFORMANCE
    # ---------------------------------------------------------

    test_metrics = regression_metrics(
        y_test,
        test_pred,
    )

    test_econ = backtest_signal(
        y_test,
        test_pred,
        transaction_cost_bps=float(
            config["transaction_cost_bps"]
        ),
        threshold=float(
            config["prediction_threshold"]
        ),
        max_position=float(
            config["max_position"]
        ),
    )

    # ---------------------------------------------------------
    # POSITION / PNL SERIES
    # ---------------------------------------------------------

    position = np.where(
        test_pred
        > float(config["prediction_threshold"]),
        float(config["max_position"]),
        np.where(
            test_pred
            < -float(config["prediction_threshold"]),
            -float(config["max_position"]),
            0.0,
        ),
    )

    # One-step execution delay.
    position = np.roll(position, 1)
    position[0] = 0.0

    turnover = np.abs(
        np.diff(
            np.r_[0.0, position]
        )
    )

    pnl = (
        position
        * y_test.to_numpy()
        -
        turnover
        * float(
            config["transaction_cost_bps"]
        )
        / 10_000
    )

    # ---------------------------------------------------------
    # MONITORING
    # ---------------------------------------------------------

    monitor_window = min(
        int(config["monitor_window"]),
        max(50, len(test) // 4),
    )

    monitoring = rolling_metrics(
        y_test.to_numpy(),
        test_pred,
        pnl,
        window=monitor_window,
    )

    cut = len(test) // 2

    drift = feature_drift_report(
        test.iloc[:cut],
        test.iloc[cut:],
        feature_cols,
    )

    # ---------------------------------------------------------
    # FAIR VALUE / PRICING
    # ---------------------------------------------------------

    microprice = (
        test["mid"]
        + test["microprice_deviation"] * test["mid"]
    ).to_numpy()
    fair_value_estimate = fair_value(
        microprice,
        test_pred,
    )

    # ---------------------------------------------------------
    # RESEARCH CONCLUSIONS
    # ---------------------------------------------------------

    research_next_steps = [
        "Test whether the signal survives realistic latency and queue-position assumptions.",
        "Separate model decay from feature-distribution drift.",
        "Repeat the experiment across multiple market regimes and instruments.",
        "Compare incremental signal value against a simpler inventory-aware baseline.",
        "Investigate whether volatility and spread condition the strength of the signal.",
    ]

    report = {
        "research_question": (
            "Do order-book and order-flow features "
            "contain exploitable information about "
            "short-horizon returns?"
        ),

        "hypothesis": (
            "Order-book imbalance and signed order flow "
            "contain incremental information about future "
            "mid-price returns."
        ),

        "selected_model": selected_name,

        "feature_columns": feature_cols,

        "candidate_models": results,

        "test_prediction_metrics": test_metrics,

        "test_economics": test_econ,

        "feature_drift_early_to_late": drift,

        "pricing": {
            "fair_value_mean": float(np.mean(fair_value_estimate)),
            "fair_value_std": float(np.std(fair_value_estimate)),
            "mean_fair_value_minus_mid": float(np.mean(fair_value_estimate - test["mid"].to_numpy())),
        },

        "research_next_steps": research_next_steps,

        "summary": {
            "selected_model": selected_name,
            "test_ic": test_metrics["ic"],
            "test_directional_accuracy":
                test_metrics["directional_accuracy"],
            "test_total_pnl":
                test_econ["total_pnl"],
            "test_max_drawdown":
                test_econ["max_drawdown"],
            "test_turnover":
                test_econ["turnover"],
            "mean_fair_value_minus_mid":
                float(np.mean(fair_value_estimate - test["mid"].to_numpy())),
        },
    }

    # ---------------------------------------------------------
    # SAVE ARTIFACTS
    # ---------------------------------------------------------

    out = Path(artifact_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (out / "report.json").write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    monitoring.to_csv(
        out / "monitoring.csv",
        index=False,
    )

    test_output = test[
        [
            "timestamp",
            "mid",
            "book_imbalance",
            "signed_volume",
            "target_return",
            "microprice",
            "microprice_deviation",
        ]
    ].copy()

    test_output["prediction"] = test_pred
    test_output["fair_value"] = fair_value_estimate
    test_output["position"] = position
    test_output["pnl"] = pnl
    test_output["cumulative_pnl"] = np.cumsum(pnl)

    test_output.to_csv(
        out / "test_predictions.csv",
        index=False,
    )

    return {
        "report": report,
        "raw": raw,
        "data": data,
        "train": train,
        "validation": val,
        "test": test,
        "feature_cols": feature_cols,
        "results": results,
        "selected_model": selected_model,
        "test_predictions": test_pred,
        "fair_value": fair_value_estimate,
        "positions": position,
        "pnl": pnl,
        "monitoring": monitoring,
    }


def load_config(
    path="configs/default.yaml",
):
    import yaml

    return yaml.safe_load(
        Path(path).read_text()
    )
