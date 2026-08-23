from pathlib import Path

from src.qr_research.experiment import run
from src.qr_research.presentation import generate_research_report


def test_presentation_generates_html_and_charts(tmp_path):
    config = {
        "seed": 8,
        "n_rows": 5000,
        "horizon": 5,
        "transaction_cost_bps": 1.0,
        "prediction_threshold": 0.00008,
        "max_position": 1.0,
        "regime_change": 0.70,
        "test_fraction": 0.20,
        "validation_fraction": 0.20,
        "monitor_window": 100,
    }
    result = run(config, artifact_dir=tmp_path)
    report_path = generate_research_report(result, artifact_dir=tmp_path)
    assert report_path.exists()
    report_html = report_path.read_text(encoding="utf-8")
    for chart_name in [
        "04a_validation_ic.png",
        "04b_validation_rmse.png",
        "05a_validation_pnl.png",
        "05b_validation_drawdown.png",
        "06_prediction_vs_realized.png",
        "07_fair_value_vs_mid.png",
    ]:
        assert chart_name in report_html

    charts = list((Path(tmp_path) / "charts").glob("*.png"))
    assert len(charts) == 13
