from pathlib import Path

from src.qr_research.experiment import run


def test_end_to_end_experiment_writes_artifacts(tmp_path):
    config = {
        "seed": 7,
        "n_rows": 5000,
        "horizon": 5,
        "transaction_cost_bps": 1.0,
        "prediction_threshold": 0.00008,
        "max_position": 1.0,
        "regime_change": 0.60,
        "test_fraction": 0.20,
        "validation_fraction": 0.20,
        "monitor_window": 100,
    }
    result = run(config, artifact_dir=tmp_path)
    assert result["report"]["selected_model"] in {"ridge", "random_forest", "mlp"}
    assert (Path(tmp_path) / "report.json").exists()
    assert (Path(tmp_path) / "monitoring.csv").exists()
    assert (Path(tmp_path) / "test_predictions.csv").exists()
