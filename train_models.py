"""Train, compare, evaluate, and persist both project models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_CACHE = BASE_DIR / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(RUNTIME_CACHE / "joblib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from medical_pipeline import (
    data_quality_report,
    load_raw_data,
    save_json,
    save_model,
    train_demand_models,
    train_no_show_models,
)


DEFAULT_DATA = BASE_DIR / "Medical_appointment_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train no-show classification and daily demand forecasting models."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument(
        "--intervention-effectiveness",
        type=float,
        default=0.15,
        help="Assumed share of correctly targeted no-shows prevented (default: 0.15).",
    )
    return parser.parse_args()


def save_eda(data: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    missing, summary = data_quality_report(data)
    missing.to_csv(report_dir / "data_quality_missingness.csv", index=False)
    save_json(summary, report_dir / "data_quality_summary.json")

    daily = data.groupby("appointment_date_continuous").size().rename("appointments")
    no_show_daily = (
        data.assign(no_show_flag=data["no_show"].eq("yes").astype(int))
        .groupby("appointment_date_continuous")["no_show_flag"]
        .mean()
        .rename("no_show_rate")
    )
    pd.concat([daily, no_show_daily], axis=1).reset_index().to_csv(
        report_dir / "daily_appointments.csv", index=False
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(daily.index, daily.values, color="#24527A", linewidth=1)
    axes[0].set_title("Daily scheduled appointment volume")
    axes[0].set_ylabel("Appointments")
    axes[1].plot(no_show_daily.index, no_show_daily.values, color="#A23B72", linewidth=1)
    axes[1].axhline(data["no_show"].eq("yes").mean(), color="#666666", linestyle="--")
    axes[1].set_title("Daily no-show rate")
    axes[1].set_ylabel("Rate")
    axes[1].set_xlabel("Appointment date")
    fig.savefig(report_dir / "eda_daily_patterns.png", dpi=160)
    plt.close(fig)


def save_classification_outputs(
    artifact: dict, output: dict, model_dir: Path, report_dir: Path, effectiveness: float
) -> None:
    save_model(artifact, model_dir / "no_show_classifier.joblib")
    output["comparison"].to_csv(
        report_dir / "classification_model_comparison.csv", index=False
    )
    output["test_predictions"].to_csv(
        report_dir / "classification_test_predictions.csv", index=False
    )
    output["feature_importance"].to_csv(
        report_dir / "classification_feature_importance.csv", index=False
    )
    save_json(
        {
            "selected_model": artifact["model_name"],
            "metrics": output["test_metrics"],
            "temporal_split": output["split"],
        },
        report_dir / "classification_test_metrics.json",
    )

    predictions = output["test_predictions"]
    actual_no_shows = int(predictions["actual_no_show"].sum())
    true_positive_targets = int(
        (
            (predictions["actual_no_show"] == 1)
            & (predictions["predicted_no_show"] == 1)
        ).sum()
    )
    prevented = true_positive_targets * effectiveness
    impact = {
        "assumed_intervention_effectiveness": effectiveness,
        "test_set_actual_no_shows": actual_no_shows,
        "correctly_targeted_no_shows": true_positive_targets,
        "estimated_prevented_no_shows": prevented,
        "estimated_no_show_reduction_percent": (
            prevented / actual_no_shows if actual_no_shows else 0.0
        ),
        "note": "Scenario only. Replace the effectiveness assumption with pilot evidence.",
    }
    save_json(impact, report_dir / "business_impact_scenario.json")

    y_true = predictions["actual_no_show"].to_numpy()
    y_pred = predictions["predicted_no_show"].to_numpy()
    y_score = predictions["predicted_probability"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("No-show confusion matrix")
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[1])
    axes[1].set_title("No-show ROC curve")
    fig.savefig(report_dir / "classification_evaluation.png", dpi=160)
    plt.close(fig)


def save_forecast_outputs(
    artifact: dict, output: dict, model_dir: Path, report_dir: Path
) -> None:
    save_model(artifact, model_dir / "demand_forecaster.joblib")
    output["comparison"].to_csv(
        report_dir / "forecast_model_comparison.csv", index=False
    )
    output["specialty_metrics"].to_csv(
        report_dir / "forecast_specialty_metrics.csv", index=False
    )
    output["test_predictions"].to_csv(
        report_dir / "forecast_test_predictions_by_specialty.csv", index=False
    )
    output["overall_test_predictions"].to_csv(
        report_dir / "forecast_test_predictions_overall.csv", index=False
    )
    save_json(
        {
            "selected_model": artifact["model_name"],
            "metrics": output["test_metrics"],
            "test_start": output["test_start"],
            "test_end": output["test_end"],
            "backtest_type": artifact["backtest_type"],
        },
        report_dir / "forecast_test_metrics.json",
    )

    overall = output["overall_test_predictions"]
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.plot(overall["date"], overall["demand"], label="Actual", color="#24527A")
    ax.plot(
        overall["date"], overall["prediction"], label="Forecast", color="#E07A5F"
    )
    ax.set_title("Overall demand backtest")
    ax.set_ylabel("Appointments")
    ax.set_xlabel("Date")
    ax.legend()
    fig.savefig(report_dir / "forecast_backtest.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 <= args.intervention_effectiveness <= 1:
        raise ValueError("--intervention-effectiveness must be between 0 and 1.")

    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data.resolve()}")
    data = load_raw_data(args.data)
    print(f"Loaded {len(data):,} appointments across {data['appointment_date_continuous'].nunique()} dates.")

    print("Creating data-quality and EDA outputs...")
    save_eda(data, report_dir)

    print("Training three no-show classifiers...")
    classifier_artifact, classifier_output = train_no_show_models(data)
    save_classification_outputs(
        classifier_artifact,
        classifier_output,
        model_dir,
        report_dir,
        args.intervention_effectiveness,
    )
    print(
        "Selected classifier:",
        classifier_artifact["model_name"],
        classifier_output["test_metrics"],
    )

    print("Training three demand models plus a seasonal-naive benchmark...")
    demand_artifact, demand_output = train_demand_models(data)
    save_forecast_outputs(demand_artifact, demand_output, model_dir, report_dir)
    print(
        "Selected demand model:",
        demand_artifact["model_name"],
        demand_output["test_metrics"],
    )
    print(f"Models saved to {model_dir}")
    print(f"Evaluation outputs saved to {report_dir}")


if __name__ == "__main__":
    main()
