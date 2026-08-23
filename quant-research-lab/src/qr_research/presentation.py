from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_charts(result: dict, artifact_dir: str | Path):
    """Create the visual research artifacts for an experiment."""

    artifact_dir = Path(artifact_dir)
    charts_dir = artifact_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    raw = result["raw"]
    data = result["data"]
    test = result["test"].copy()
    test_pred = np.asarray(result["test_predictions"])
    fair_value_estimate = np.asarray(result["fair_value"])
    pnl = np.asarray(result["pnl"])
    monitoring = result["monitoring"]
    report = result["report"]

    # ---------------------------------------------------------
    # 1. Mid-price
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        raw["timestamp"],
        raw["mid"],
        linewidth=1,
    )

    ax.set_title("Synthetic Mid-Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mid-price")
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "01_mid_price.png",
    )

    # ---------------------------------------------------------
    # 2. Order-book imbalance
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        data["timestamp"],
        data["book_imbalance"],
        linewidth=0.8,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title("Order-Book Imbalance")
    ax.set_xlabel("Time")
    ax.set_ylabel("Imbalance")
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "02_order_book_imbalance.png",
    )

    # ---------------------------------------------------------
    # 3. Signal relationship
    # ---------------------------------------------------------

    sample = data[
        [
            "book_imbalance",
            "target_return",
        ]
    ].dropna()

    if len(sample) > 5000:
        sample = sample.sample(
            5000,
            random_state=42,
        )

    ic = sample[
        "book_imbalance"
    ].corr(
        sample["target_return"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        sample["book_imbalance"],
        sample["target_return"],
        alpha=0.25,
        s=10,
    )

    ax.set_title(
        f"Order-Book Imbalance vs Future Return "
        f"(IC = {ic:.4f})"
    )

    ax.set_xlabel("Order-book imbalance")
    ax.set_ylabel("Future return")
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "03_signal_relationship.png",
    )

    # ---------------------------------------------------------
    # 4a. Validation IC
    # ---------------------------------------------------------

    results = report["candidate_models"]
    names = list(results.keys())
    x = np.arange(len(names))

    validation_ic = [
        results[name]["validation_prediction"]["ic"]
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, validation_ic, width=0.55)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Validation Information Coefficient")
    ax.set_ylabel("IC")
    ax.grid(axis="y", alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "04a_validation_ic.png",
    )

    # ---------------------------------------------------------
    # 4b. Validation RMSE
    # ---------------------------------------------------------

    validation_rmse = [
        results[name]["validation_prediction"]["rmse"]
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, validation_rmse, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Validation Root Mean Squared Error")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "04b_validation_rmse.png",
    )

    # ---------------------------------------------------------
    # 5a. Validation PnL
    # ---------------------------------------------------------

    validation_pnl = [
        results[name]["validation_economics"]["total_pnl"]
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, validation_pnl, width=0.55)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Validation Strategy PnL")
    ax.set_ylabel("PnL")
    ax.grid(axis="y", alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "05a_validation_pnl.png",
    )

    # ---------------------------------------------------------
    # 5b. Validation maximum drawdown
    # ---------------------------------------------------------

    validation_drawdown = [
        results[name]["validation_economics"]["max_drawdown"]
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, validation_drawdown, width=0.55)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Validation Maximum Drawdown")
    ax.set_ylabel("Maximum drawdown")
    ax.grid(axis="y", alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "05b_validation_drawdown.png",
    )

    # ---------------------------------------------------------
    # 5. Prediction vs realized
    # ---------------------------------------------------------

    y_true = test[
        "target_return"
    ].to_numpy()

    n = min(
        5000,
        len(y_true),
    )

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(y_true),
        n,
        replace=False,
    )

    corr = np.corrcoef(
        test_pred[indices],
        y_true[indices],
    )[0, 1]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        test_pred[indices],
        y_true[indices],
        alpha=0.25,
        s=10,
    )

    low = min(
        test_pred[indices].min(),
        y_true[indices].min(),
    )

    high = max(
        test_pred[indices].max(),
        y_true[indices].max(),
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        f"Predicted vs Realized Return "
        f"(IC = {corr:.4f})"
    )

    ax.set_xlabel("Predicted return")
    ax.set_ylabel("Realized return")
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "06_prediction_vs_realized.png",
    )

    # ---------------------------------------------------------
    # 6. Fair value vs market mid
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        test["timestamp"],
        test["mid"],
        linewidth=1,
        label="Market mid",
    )
    ax.plot(
        test["timestamp"],
        fair_value_estimate,
        linewidth=1,
        label="Model fair value",
    )

    ax.set_title("Model Fair Value vs Market Mid")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "07_fair_value_vs_mid.png",
    )

    # ---------------------------------------------------------
    # 6. Cumulative PnL
    # ---------------------------------------------------------

    cumulative_pnl = np.cumsum(pnl)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        test["timestamp"],
        cumulative_pnl,
        linewidth=1,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title("Cumulative Strategy PnL")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "08_cumulative_pnl.png",
    )

    # ---------------------------------------------------------
    # 7. Rolling IC
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        monitoring.index,
        monitoring["rolling_ic"],
        linewidth=1,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Production Monitoring — Rolling IC"
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel("Rolling IC")
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "09_rolling_ic.png",
    )

    # ---------------------------------------------------------
    # 8. Rolling PnL
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        monitoring.index,
        monitoring["rolling_pnl"],
        linewidth=1,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Production Monitoring — Rolling PnL"
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel("Rolling PnL")
    ax.grid(alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "10_rolling_pnl.png",
    )

    # ---------------------------------------------------------
    # 9. Feature drift
    # ---------------------------------------------------------

    drift = pd.Series(
        report[
            "feature_drift_early_to_late"
        ]
    ).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))

    drift.plot.barh(ax=ax)

    ax.set_title(
        "Feature Distribution Drift"
    )

    ax.set_xlabel(
        "Population Stability Index"
    )

    ax.grid(axis="x", alpha=0.2)

    _save_fig(
        fig,
        charts_dir / "11_feature_drift.png",
    )

    return charts_dir


def _format_metric(value):
    if isinstance(value, float):
        return f"{value:.5f}"

    return str(value)


def _table_from_dict(
    data: dict,
    columns=("Metric", "Value"),
):
    rows = []

    for key, value in data.items():
        rows.append(
            f"""
            <tr>
                <td>{key.replace("_", " ").title()}</td>
                <td>{_format_metric(value)}</td>
            </tr>
            """
        )

    return f"""
    <table>
        <thead>
            <tr>
                <th>{columns[0]}</th>
                <th>{columns[1]}</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def create_html_report(
    result: dict,
    artifact_dir: str | Path,
):
    """Generate a standalone HTML quantitative research report."""

    artifact_dir = Path(artifact_dir)

    report = result["report"]

    metrics = report[
        "test_prediction_metrics"
    ]

    economics = report[
        "test_economics"
    ]

    drift = report[
        "feature_drift_early_to_late"
    ]

    selected_model = report[
        "selected_model"
    ]

    charts = artifact_dir / "charts"

    model_rows = []

    for name, model_result in report[
        "candidate_models"
    ].items():

        prediction = model_result[
            "validation_prediction"
        ]

        economics_result = model_result[
            "validation_economics"
        ]

        model_rows.append(
            f"""
            <tr>
                <td>{name}</td>
                <td>{prediction["ic"]:.5f}</td>
                <td>{prediction["rmse"]:.6f}</td>
                <td>{prediction["directional_accuracy"]:.2%}</td>
                <td>{economics_result["total_pnl"]:.5f}</td>
            </tr>
            """
        )

    drift_rows = []

    for feature, value in sorted(
        drift.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        drift_rows.append(
            f"""
            <tr>
                <td>{feature}</td>
                <td>{value:.5f}</td>
            </tr>
            """
        )

    next_steps = "".join(
        f"<li>{step}</li>"
        for step in report[
            "research_next_steps"
        ]
    )

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>
Quantitative Research Report
</title>

<style>

body {{
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    max-width: 1200px;

    margin: 40px auto;

    padding: 0 30px;

    color: #222;

    line-height: 1.6;

    background: #fafafa;
}}

h1 {{
    margin-bottom: 5px;
}}

h2 {{
    margin-top: 45px;
}}

.subtitle {{
    color: #666;
    font-size: 18px;
}}

.card-grid {{
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);

    gap: 15px;

    margin: 30px 0;
}}

.card {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
}}

.card-title {{
    color: #777;
    font-size: 13px;
    text-transform: uppercase;
}}

.card-value {{
    font-size: 25px;
    font-weight: 600;
    margin-top: 5px;
}}

img {{
    width: 100%;
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin: 15px 0 30px 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
}}

th, td {{
    padding: 10px;
    border: 1px solid #ddd;
    text-align: left;
}}

th {{
    background: #f1f1f1;
}}

.conclusion {{
    background: white;
    border-left: 4px solid #222;
    padding: 20px;
    margin: 25px 0;
}}

</style>

</head>

<body>

<h1>
Quantitative Research Lab
</h1>

<p class="subtitle">
Order-flow signal discovery and short-horizon price forecasting
</p>

<h2>
Executive Summary
</h2>

<div class="card-grid">

<div class="card">
<div class="card-title">
Selected Model
</div>
<div class="card-value">
{selected_model}
</div>
</div>

<div class="card">
<div class="card-title">
Test IC
</div>
<div class="card-value">
{metrics["ic"]:.4f}
</div>
</div>

<div class="card">
<div class="card-title">
Total PnL
</div>
<div class="card-value">
{economics["total_pnl"]:.4f}
</div>
</div>

<div class="card">
<div class="card-title">
Max Drawdown
</div>
<div class="card-value">
{economics["max_drawdown"]:.4f}
</div>
</div>

</div>

<h2>
1. Research Question
</h2>

<p>
{report["research_question"]}
</p>

<h2>
2. Hypothesis
</h2>

<div class="conclusion">
{report["hypothesis"]}
</div>

<h2>
3. Market Data
</h2>

<img src="charts/01_mid_price.png">

<img src="charts/02_order_book_imbalance.png">

<h2>
4. Signal Discovery
</h2>

<img src="charts/03_signal_relationship.png">

<h2>
5. Model Comparison
</h2>

<table>

<thead>

<tr>
<th>Model</th>
<th>IC</th>
<th>RMSE</th>
<th>Directional Accuracy</th>
<th>Validation PnL</th>
</tr>

</thead>

<tbody>

{''.join(model_rows)}

</tbody>

</table>

<h3>Statistical prediction quality</h3>

<img src="charts/04a_validation_ic.png">

<img src="charts/04b_validation_rmse.png">

<h3>Economic trading performance</h3>

<img src="charts/05a_validation_pnl.png">

<img src="charts/05b_validation_drawdown.png">

<h2>
6. Out-of-Sample Prediction
</h2>

<img src="charts/06_prediction_vs_realized.png">

{_table_from_dict(metrics)}

<h2>
7. Pricing / Fair Value
</h2>

<img src="charts/07_fair_value_vs_mid.png">

{_table_from_dict(report["pricing"])}

<h2>
8. Economic Validation
</h2>

<img src="charts/08_cumulative_pnl.png">

{_table_from_dict(economics)}

<h2>
9. Production Monitoring
</h2>

<img src="charts/09_rolling_ic.png">

<img src="charts/10_rolling_pnl.png">

<h2>
10. Feature Drift
</h2>

<img src="charts/11_feature_drift.png">

<table>

<thead>
<tr>
<th>Feature</th>
<th>PSI</th>
</tr>
</thead>

<tbody>

{''.join(drift_rows)}

</tbody>

</table>

<h2>
11. Research Conclusion
</h2>

<div class="conclusion">

<p>
The experiment investigates whether order-book and order-flow
information contains exploitable short-horizon predictive power.
</p>

<p>
The selected model is <strong>{selected_model}</strong>,
chosen using out-of-sample economic performance rather than
prediction error alone.
</p>

<p>
The result should be interpreted together with transaction costs,
execution delay, turnover, drawdown and regime stability.
</p>

</div>

<h2>
12. Next Research Questions
</h2>

<ol>
{next_steps}
</ol>

</body>

</html>
"""

    report_path = (
        artifact_dir /
        "research_report.html"
    )

    report_path.write_text(
        html,
        encoding="utf-8",
    )

    return report_path


def generate_research_report(
    result: dict,
    artifact_dir="artifacts",
):
    """
    Complete presentation pipeline.

    Produces:

        artifacts/
        ├── charts/
        └── research_report.html
    """

    create_charts(
        result,
        artifact_dir,
    )

    return create_html_report(
        result,
        artifact_dir,
    )
