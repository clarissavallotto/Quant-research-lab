from pathlib import Path

from src.qr_research.experiment import (
    load_config,
    run,
)

from src.qr_research.presentation import (
    generate_research_report,
)


if __name__ == "__main__":

    config = load_config()

    artifact_dir = Path(
        "artifacts"
    )

    result = run(
        config,
        artifact_dir=artifact_dir,
    )

    report_path = generate_research_report(
        result,
        artifact_dir=artifact_dir,
    )

    report = result["report"]

    print()
    print("=" * 70)
    print("QUANTITATIVE RESEARCH EXPERIMENT")
    print("=" * 70)

    print(
        f"\nSelected model:"
        f" {report['selected_model']}"
    )

    print(
        f"Test IC:"
        f" {report['test_prediction_metrics']['ic']:.4f}"
    )

    print(
        f"Directional accuracy:"
        f" {report['test_prediction_metrics']['directional_accuracy']:.2%}"
    )

    print(
        f"Total PnL:"
        f" {report['test_economics']['total_pnl']:.5f}"
    )

    print(
        f"Maximum drawdown:"
        f" {report['test_economics']['max_drawdown']:.5f}"
    )

    print()
    print(
        f"Research report:"
        f" {report_path}"
    )
