"""Training and inference utilities for the medical appointment project.

The module intentionally keeps feature engineering outside the Streamlit UI so the
same transformations are used during training and inference.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATE_COLUMN = "appointment_date_continuous"
TARGET_COLUMN = "no_show"

CLASSIFIER_NUMERIC_FEATURES = [
    "appointment_time",
    "age",
    "under_12_years_old",
    "over_60_years_old",
    "patient_needs_companion",
    "average_temp_day",
    "average_rain_day",
    "max_temp_day",
    "max_rain_day",
    "rainy_day_before",
    "storm_day_before",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "Scholarship",
    "SMS_received",
    "appointment_year",
    "appointment_month",
    "appointment_day",
    "appointment_day_of_week",
    "appointment_week_of_year",
    "is_weekend",
    "temperature_range",
    "log_average_rain",
    "day_of_year_sin",
    "day_of_year_cos",
]

CLASSIFIER_CATEGORICAL_FEATURES = [
    "specialty",
    "gender",
    "disability",
    "place_group",
    "appointment_shift",
    "rain_intensity",
    "heat_intensity",
]

RAW_CLASSIFIER_NUMERIC_FEATURES = [
    "appointment_time",
    "age",
    "under_12_years_old",
    "over_60_years_old",
    "patient_needs_companion",
    "average_temp_day",
    "average_rain_day",
    "max_temp_day",
    "max_rain_day",
    "rainy_day_before",
    "storm_day_before",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "Scholarship",
    "SMS_received",
]

DEMAND_LAGS = (1, 2, 7, 14, 28)
DEMAND_WINDOWS = (7, 14, 28)
DEMAND_NUMERIC_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "day_of_year",
    "is_weekend",
    "trend",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    *[f"lag_{lag}" for lag in DEMAND_LAGS],
    *[name for window in DEMAND_WINDOWS for name in (f"rolling_mean_{window}", f"rolling_std_{window}")],
]
DEMAND_CATEGORICAL_FEATURES = ["specialty"]


@dataclass(frozen=True)
class TemporalSplit:
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def _plain(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    return value


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def load_raw_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    if DATE_COLUMN not in data or TARGET_COLUMN not in data:
        raise ValueError(f"CSV must contain '{DATE_COLUMN}' and '{TARGET_COLUMN}'.")
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    if data[DATE_COLUMN].isna().any():
        bad = int(data[DATE_COLUMN].isna().sum())
        raise ValueError(f"{bad} appointment dates could not be parsed.")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("string").str.strip().str.lower()
    invalid_targets = sorted(set(data[TARGET_COLUMN].dropna()) - {"yes", "no"})
    if invalid_targets:
        raise ValueError(f"Unexpected no_show values: {invalid_targets}")
    return data


def _ascii_upper(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    normalized = unicodedata.normalize("NFKD", str(value).strip())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).upper()


def normalize_place(value: Any) -> str:
    """Collapse known municipalities and quarantine synthetic/high-cardinality values."""
    text = _ascii_upper(value).replace(".", " ")
    text = " ".join(text.split())
    if not text:
        return "unknown"

    aliases = {
        "ITAJAI": "itajai",
        "B CAMBORIU": "balneario camboriu",
        "BALN CAMBORIU": "balneario camboriu",
        "BALNEARIO CAMBORIU": "balneario camboriu",
        "CAMBORIU": "camboriu",
        "NAVEGANTES": "navegantes",
        "ITAPEMA": "itapema",
        "BOMBINHAS": "bombinhas",
        "PENHA": "penha",
        "PORTO BELO": "porto belo",
        "BALN PICARRAS": "balneario picarras",
        "BALNEARIO PICARRAS": "balneario picarras",
        "ILHOTA": "ilhota",
        "LUIZ ALVES": "luiz alves",
        "MONTENEGRO": "montenegro",
    }
    return aliases.get(text, "other_or_anonymized")


def _clean_category(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )
    return cleaned.astype(object).where(cleaned.notna(), np.nan)


def prepare_classifier_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    required = set(RAW_CLASSIFIER_NUMERIC_FEATURES) | {
        "specialty",
        "gender",
        "disability",
        "place",
        "appointment_shift",
        "rain_intensity",
        "heat_intensity",
        DATE_COLUMN,
    }
    for column in required:
        if column not in frame:
            frame[column] = np.nan

    date_values = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if date_values.isna().any():
        raise ValueError("Every appointment needs a valid appointment date.")

    for column in RAW_CLASSIFIER_NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    derived_under_12 = (frame["age"] < 12).astype(float)
    derived_over_60 = (frame["age"] > 60).astype(float)
    frame["under_12_years_old"] = frame["under_12_years_old"].fillna(derived_under_12)
    frame["over_60_years_old"] = frame["over_60_years_old"].fillna(derived_over_60)

    frame["specialty"] = _clean_category(frame["specialty"])
    frame["gender"] = _clean_category(frame["gender"])
    frame["disability"] = _clean_category(frame["disability"])
    frame["appointment_shift"] = _clean_category(frame["appointment_shift"])
    frame["rain_intensity"] = _clean_category(frame["rain_intensity"])
    frame["heat_intensity"] = _clean_category(frame["heat_intensity"])
    frame["place_group"] = frame["place"].map(normalize_place)

    frame["appointment_year"] = date_values.dt.year
    frame["appointment_month"] = date_values.dt.month
    frame["appointment_day"] = date_values.dt.day
    frame["appointment_day_of_week"] = date_values.dt.dayofweek
    frame["appointment_week_of_year"] = date_values.dt.isocalendar().week.astype(int)
    frame["is_weekend"] = (date_values.dt.dayofweek >= 5).astype(int)
    frame["temperature_range"] = frame["max_temp_day"] - frame["average_temp_day"]
    frame["log_average_rain"] = np.log1p(frame["average_rain_day"].clip(lower=0))
    radians = 2 * np.pi * date_values.dt.dayofyear / 365.25
    frame["day_of_year_sin"] = np.sin(radians)
    frame["day_of_year_cos"] = np.cos(radians)
    return frame[CLASSIFIER_NUMERIC_FEATURES + CLASSIFIER_CATEGORICAL_FEATURES]


def build_classifier_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=25,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, CLASSIFIER_NUMERIC_FEATURES),
            ("categorical", categorical, CLASSIFIER_CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def temporal_split_dates(
    dates: Iterable[pd.Timestamp], test_fraction: float = 0.20, validation_fraction: float = 0.15
) -> TemporalSplit:
    unique_dates = pd.DatetimeIndex(pd.Series(dates).dropna().sort_values().unique())
    if len(unique_dates) < 60:
        raise ValueError("At least 60 distinct dates are required for temporal validation.")
    test_count = max(1, int(math.ceil(len(unique_dates) * test_fraction)))
    development_dates = unique_dates[:-test_count]
    validation_count = max(1, int(math.ceil(len(development_dates) * validation_fraction)))
    return TemporalSplit(
        train_dates=development_dates[:-validation_count],
        validation_dates=development_dates[-validation_count:],
        test_dates=unique_dates[-test_count:],
    )


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def classification_metrics(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def train_no_show_models(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    dates = pd.to_datetime(data[DATE_COLUMN])
    split = temporal_split_dates(dates)
    features = prepare_classifier_frame(data)
    target = data[TARGET_COLUMN].map({"no": 0, "yes": 1}).astype(int)

    train_mask = dates.isin(split.train_dates)
    validation_mask = dates.isin(split.validation_dates)
    test_mask = dates.isin(split.test_dates)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    comparison: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_classifier_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(features.loc[train_mask], target.loc[train_mask])
        validation_probability = pipeline.predict_proba(features.loc[validation_mask])[:, 1]
        threshold = _best_f1_threshold(target.loc[validation_mask], validation_probability)
        metrics = classification_metrics(
            target.loc[validation_mask], validation_probability, threshold
        )
        comparison.append({"model": name, **metrics})
        fitted[name] = pipeline
        thresholds[name] = threshold

    comparison_frame = pd.DataFrame(comparison).sort_values(
        ["f1", "roc_auc"], ascending=False
    )
    best_name = str(comparison_frame.iloc[0]["model"])
    best_threshold = thresholds[best_name]

    development_mask = train_mask | validation_mask
    final_pipeline = clone(fitted[best_name])
    final_pipeline.fit(features.loc[development_mask], target.loc[development_mask])
    test_probability = final_pipeline.predict_proba(features.loc[test_mask])[:, 1]
    test_metrics = classification_metrics(
        target.loc[test_mask], test_probability, best_threshold
    )

    test_positions = np.flatnonzero(test_mask.to_numpy())
    sample_rng = np.random.default_rng(RANDOM_STATE)
    if len(test_positions) > 5000:
        test_positions = sample_rng.choice(test_positions, size=5000, replace=False)
    importance = permutation_importance(
        final_pipeline,
        features.iloc[test_positions],
        target.iloc[test_positions],
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    specialty_values = sorted(_clean_category(data["specialty"]).dropna().unique().tolist())
    artifact = {
        "pipeline": final_pipeline,
        "threshold": best_threshold,
        "model_name": best_name,
        "numeric_features": CLASSIFIER_NUMERIC_FEATURES,
        "categorical_features": CLASSIFIER_CATEGORICAL_FEATURES,
        "specialties": specialty_values,
        "training_date_min": dates.min(),
        "training_date_max": dates.max(),
        "test_metrics": test_metrics,
    }
    output = {
        "comparison": comparison_frame,
        "test_metrics": test_metrics,
        "test_predictions": pd.DataFrame(
            {
                "row_index": data.index[test_mask],
                "appointment_date": dates.loc[test_mask].to_numpy(),
                "actual_no_show": target.loc[test_mask].to_numpy(),
                "predicted_probability": test_probability,
                "predicted_no_show": (test_probability >= best_threshold).astype(int),
            }
        ),
        "feature_importance": importance_frame,
        "split": {
            "train_start": split.train_dates.min(),
            "train_end": split.train_dates.max(),
            "validation_start": split.validation_dates.min(),
            "validation_end": split.validation_dates.max(),
            "test_start": split.test_dates.min(),
            "test_end": split.test_dates.max(),
        },
    }
    return artifact, output


def predict_no_show(artifact: dict[str, Any], appointment: pd.DataFrame) -> np.ndarray:
    features = prepare_classifier_frame(appointment)
    return artifact["pipeline"].predict_proba(features)[:, 1]


def build_daily_demand_panel(data: pd.DataFrame) -> pd.DataFrame:
    working = data[[DATE_COLUMN, "specialty"]].copy()
    working[DATE_COLUMN] = pd.to_datetime(working[DATE_COLUMN])
    working["specialty"] = _clean_category(working["specialty"]).fillna("unknown")
    dates = pd.date_range(working[DATE_COLUMN].min(), working[DATE_COLUMN].max(), freq="D")
    specialties = sorted(working["specialty"].unique().tolist())
    full_index = pd.MultiIndex.from_product(
        [dates, specialties], names=["date", "specialty"]
    )
    counts = (
        working.groupby([DATE_COLUMN, "specialty"], observed=True)
        .size()
        .rename("demand")
    )
    counts.index.names = ["date", "specialty"]
    return counts.reindex(full_index, fill_value=0).reset_index()


def _calendar_features(frame: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["date"])
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["month"] = dates.dt.month
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["day_of_year"] = dates.dt.dayofyear
    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    result["trend"] = (dates - pd.Timestamp(start_date)).dt.days
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_year_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    return result


def add_demand_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["specialty", "date"]).copy()
    start_date = pd.Timestamp(frame["date"].min())
    frame = _calendar_features(frame, start_date)
    grouped = frame.groupby("specialty", observed=True)["demand"]
    for lag in DEMAND_LAGS:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in DEMAND_WINDOWS:
        shifted = grouped.shift(1)
        frame[f"rolling_mean_{window}"] = (
            shifted.groupby(frame["specialty"], observed=True)
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        frame[f"rolling_std_{window}"] = (
            shifted.groupby(frame["specialty"], observed=True)
            .rolling(window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )
    return frame.dropna(subset=DEMAND_NUMERIC_FEATURES).reset_index(drop=True)


def build_demand_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, DEMAND_NUMERIC_FEATURES),
            ("categorical", categorical, DEMAND_CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1.0))),
        "r2": r2_score(y_true, y_pred),
    }


def _demand_evaluation(
    test_frame: pd.DataFrame, predictions: np.ndarray
) -> dict[str, Any]:
    detail = test_frame[["date", "specialty", "demand"]].copy()
    detail["prediction"] = np.clip(predictions, 0, None)
    overall = detail.groupby("date", as_index=False)[["demand", "prediction"]].sum()
    specialty_metrics = []
    for specialty, group in detail.groupby("specialty", observed=True):
        specialty_metrics.append(
            {"specialty": specialty, **regression_metrics(group["demand"], group["prediction"])}
        )
    return {
        "overall": regression_metrics(overall["demand"], overall["prediction"]),
        "specialty_macro": pd.DataFrame(specialty_metrics).mean(numeric_only=True).to_dict(),
        "specialty_metrics": specialty_metrics,
        "detail": detail,
        "overall_predictions": overall,
    }


def train_demand_models(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    panel = build_daily_demand_panel(data)
    supervised = add_demand_features(panel)
    unique_dates = pd.DatetimeIndex(sorted(supervised["date"].unique()))
    test_count = max(28, int(math.ceil(len(unique_dates) * 0.20)))
    test_dates = unique_dates[-test_count:]
    train_mask = ~supervised["date"].isin(test_dates)
    test_mask = supervised["date"].isin(test_dates)

    feature_columns = DEMAND_NUMERIC_FEATURES + DEMAND_CATEGORICAL_FEATURES
    x_train = supervised.loc[train_mask, feature_columns]
    y_train = supervised.loc[train_mask, "demand"]
    x_test = supervised.loc[test_mask, feature_columns]
    test_frame = supervised.loc[test_mask].copy()

    candidates = {
        "poisson_regression": PoissonRegressor(alpha=0.5, max_iter=1000),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    comparison: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_demand_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = np.clip(pipeline.predict(x_test), 0, None)
        evaluation = _demand_evaluation(test_frame, predictions)
        comparison.append(
            {
                "model": name,
                **{f"overall_{k}": v for k, v in evaluation["overall"].items()},
                **{
                    f"specialty_macro_{k}": v
                    for k, v in evaluation["specialty_macro"].items()
                },
            }
        )
        fitted[name] = pipeline
        evaluations[name] = evaluation

    baseline_predictions = test_frame["lag_7"].to_numpy()
    baseline_evaluation = _demand_evaluation(test_frame, baseline_predictions)
    comparison.append(
        {
            "model": "seasonal_naive_7_day",
            **{
                f"overall_{k}": v
                for k, v in baseline_evaluation["overall"].items()
            },
            **{
                f"specialty_macro_{k}": v
                for k, v in baseline_evaluation["specialty_macro"].items()
            },
        }
    )

    comparison_frame = pd.DataFrame(comparison).sort_values(
        ["overall_mae", "overall_mape"], ascending=True
    )
    eligible = comparison_frame[comparison_frame["model"] != "seasonal_naive_7_day"]
    best_name = str(eligible.iloc[0]["model"])
    best_evaluation = evaluations[best_name]

    final_pipeline = clone(fitted[best_name])
    final_pipeline.fit(supervised[feature_columns], supervised["demand"])
    artifact = {
        "pipeline": final_pipeline,
        "model_name": best_name,
        "feature_columns": feature_columns,
        "numeric_features": DEMAND_NUMERIC_FEATURES,
        "categorical_features": DEMAND_CATEGORICAL_FEATURES,
        "history": panel,
        "specialties": sorted(panel["specialty"].unique().tolist()),
        "training_start": pd.Timestamp(panel["date"].min()),
        "training_end": pd.Timestamp(panel["date"].max()),
        "test_metrics": best_evaluation["overall"],
        "backtest_type": "rolling one-day-ahead with actual lag history",
    }
    output = {
        "comparison": comparison_frame,
        "test_metrics": best_evaluation["overall"],
        "specialty_metrics": pd.DataFrame(best_evaluation["specialty_metrics"]),
        "test_predictions": best_evaluation["detail"],
        "overall_test_predictions": best_evaluation["overall_predictions"],
        "test_start": test_dates.min(),
        "test_end": test_dates.max(),
    }
    return artifact, output


def _future_demand_rows(
    forecast_date: pd.Timestamp,
    history: pd.DataFrame,
    specialties: list[str],
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for specialty in specialties:
        series = (
            history.loc[history["specialty"] == specialty, ["date", "demand"]]
            .sort_values("date")
            .set_index("date")["demand"]
        )
        row: dict[str, Any] = {"date": forecast_date, "specialty": specialty}
        for lag in DEMAND_LAGS:
            row[f"lag_{lag}"] = float(series.iloc[-lag])
        for window in DEMAND_WINDOWS:
            values = series.iloc[-window:]
            row[f"rolling_mean_{window}"] = float(values.mean())
            row[f"rolling_std_{window}"] = float(values.std(ddof=1))
        rows.append(row)
    return _calendar_features(pd.DataFrame(rows), start_date)


def forecast_demand(artifact: dict[str, Any], horizon_days: int) -> pd.DataFrame:
    if not 1 <= int(horizon_days) <= 365:
        raise ValueError("horizon_days must be between 1 and 365.")
    history = artifact["history"].copy()
    history["date"] = pd.to_datetime(history["date"])
    specialties = list(artifact["specialties"])
    last_date = pd.Timestamp(history["date"].max())
    predictions: list[pd.DataFrame] = []

    for offset in range(1, int(horizon_days) + 1):
        forecast_date = last_date + pd.Timedelta(days=offset)
        features = _future_demand_rows(
            forecast_date,
            history,
            specialties,
            pd.Timestamp(artifact["training_start"]),
        )
        predicted = np.clip(
            artifact["pipeline"].predict(features[artifact["feature_columns"]]),
            0,
            None,
        )
        day_result = features[["date", "specialty"]].copy()
        day_result["forecast"] = predicted
        predictions.append(day_result)
        history = pd.concat(
            [
                history,
                day_result.rename(columns={"forecast": "demand"})[
                    ["date", "specialty", "demand"]
                ],
            ],
            ignore_index=True,
        )

    detail = pd.concat(predictions, ignore_index=True)
    total = detail.groupby("date", as_index=False)["forecast"].sum()
    total["specialty"] = "all specialties"
    return pd.concat([detail, total], ignore_index=True).sort_values(
        ["date", "specialty"]
    )


def data_quality_report(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = (
        data.isna().sum().rename("missing_rows").to_frame().assign(
            missing_percent=lambda x: x["missing_rows"] / len(data)
        )
    )
    weather_columns = [
        "average_temp_day",
        "average_rain_day",
        "max_temp_day",
        "max_rain_day",
    ]
    per_day_unique = data.groupby(DATE_COLUMN)[weather_columns].nunique(dropna=True)
    weather_inconsistent = (per_day_unique > 1).sum().to_dict()
    summary = {
        "rows": len(data),
        "columns": len(data.columns),
        "date_min": data[DATE_COLUMN].min(),
        "date_max": data[DATE_COLUMN].max(),
        "unique_dates": data[DATE_COLUMN].nunique(),
        "duplicate_rows": int(data.duplicated().sum()),
        "no_show_rate": float(data[TARGET_COLUMN].eq("yes").mean()),
        "place_unique_values": int(data["place"].nunique(dropna=True)),
        "weather_columns_with_multiple_values_on_same_date": weather_inconsistent,
    }
    return missing.reset_index(names="column"), summary


def save_model(artifact: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_model(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
