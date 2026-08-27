"""Core data, validation, training, and prediction logic for the Streamlit app.

This module deliberately has no Streamlit imports so it can be tested as a
normal Python module before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42

FEATURES = [
    "Previous_CGPA",
    "Average_Score",
    "Attendance_Pct",
    "Study_Hours_Per_Day",
    "Sleep_Hours",
]

FEATURE_LABELS = {
    "Previous_CGPA": "Previous CGPA",
    "Average_Score": "Average Score",
    "Attendance_Pct": "Attendance Rate (%)",
    "Study_Hours_Per_Day": "Study Hours per Day",
    "Sleep_Hours": "Sleep Hours per Day",
}

FEATURE_RANGES = {
    "Previous_CGPA": (0.0, 4.0),
    "Average_Score": (0.0, 100.0),
    "Attendance_Pct": (0.0, 100.0),
    "Study_Hours_Per_Day": (0.0, 24.0),
    "Sleep_Hours": (0.0, 24.0),
}

CLASS_ORDER = ["At Risk", "Average", "Good", "Excellent"]
CLASS_COLORS = {
    "At Risk": "#dc2626",
    "Average": "#d97706",
    "Good": "#2563eb",
    "Excellent": "#16a34a",
}


@dataclass(frozen=True)
class DatasetResult:
    frame: pd.DataFrame
    source: str
    note: str


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    evaluation: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray]
    best_model: str
    errors: dict[str, str]


def performance_category(value: float) -> str:
    """Convert a CGPA value into the project's four target classes."""
    value = float(value)
    if value >= 3.50:
        return "Excellent"
    if value >= 3.00:
        return "Good"
    if value >= 2.50:
        return "Average"
    return "At Risk"


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and enrich a raw student dataset."""
    required = FEATURES + ["Final_CGPA"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    frame = raw.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).copy()

    for feature, (low, high) in FEATURE_RANGES.items():
        frame = frame[frame[feature].between(low, high, inclusive="both")]
    frame = frame[frame["Final_CGPA"].between(0.0, 4.0, inclusive="both")]

    if len(frame) < 100:
        raise ValueError("Dataset has fewer than 100 valid rows after validation.")

    frame["Performance_Category"] = frame["Final_CGPA"].map(performance_category)
    present = set(frame["Performance_Category"].unique())
    missing_classes = [label for label in CLASS_ORDER if label not in present]
    if missing_classes:
        raise ValueError("Dataset does not contain target class(es): " + ", ".join(missing_classes))

    return frame.reset_index(drop=True)


def make_fallback_dataset(size: int = 1800) -> pd.DataFrame:
    """Build deterministic demo data so a missing CSV cannot crash deployment."""
    rng = np.random.default_rng(RANDOM_STATE)
    previous = np.clip(rng.normal(3.05, 0.52, size), 1.0, 4.0)
    attendance = np.clip(rng.normal(84.0, 11.5, size), 40.0, 100.0)
    study = np.clip(rng.normal(4.3, 2.2, size), 0.1, 12.0)
    sleep = np.clip(rng.normal(7.0, 1.3, size), 3.5, 10.5)
    social = np.clip(rng.normal(8.0, 2.8, size), 0.0, 20.0)
    subjects = rng.integers(4, 9, size=size)
    score = np.clip(
        20.0 + 15.5 * previous + 0.14 * attendance + 1.8 * study + rng.normal(0.0, 7.0, size),
        35.0,
        100.0,
    )
    final = np.clip(
        0.50 * previous
        + 0.018 * score
        + 0.0035 * attendance
        + 0.033 * study
        + rng.normal(0.0, 0.21, size)
        - 0.60,
        1.0,
        4.0,
    )
    majors = np.array(["Computing", "Business", "Engineering", "Psychology", "Science", "Arts"])

    return pd.DataFrame(
        {
            "Student_ID": [f"DEMO{index:05d}" for index in range(1, size + 1)],
            "Gender": rng.choice(["Female", "Male"], size=size),
            "Age": rng.integers(18, 25, size=size),
            "Major": rng.choice(majors, size=size),
            "Attendance_Pct": np.round(attendance, 1),
            "Study_Hours_Per_Day": np.round(study, 1),
            "Previous_CGPA": np.round(previous, 2),
            "Sleep_Hours": np.round(sleep, 1),
            "Social_Hours_Week": np.round(social, 1),
            "Final_CGPA": np.round(final, 2),
            "Number_of_Subjects": subjects,
            "Average_Score": np.round(score, 2),
        }
    )


def load_dataset(project_root: Path | str) -> DatasetResult:
    """Load the repository CSV, with a safe deterministic fallback."""
    project_root = Path(project_root)
    candidates = [
        project_root / "Student_data.csv",
        project_root / "student_data.csv",
        project_root / "dataset" / "Student_data.csv",
        project_root / "dataset" / "student_data.csv",
    ]
    failures: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            frame = prepare_dataset(pd.read_csv(path))
            return DatasetResult(frame, "Repository CSV", path.name)
        except Exception as exc:  # Continue safely to the next candidate.
            failures.append(f"{path.name}: {exc}")

    fallback = prepare_dataset(make_fallback_dataset())
    note = "Deterministic built-in fallback dataset"
    if failures:
        note += "; repository CSV was invalid"
    return DatasetResult(fallback, "Built-in fallback", note)


def build_estimators() -> dict[str, Any]:
    """Return three defensible classifiers with probability estimates."""
    calibrated_svm = CalibratedClassifierCV(
        estimator=SVC(C=2.0, kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
        method="sigmoid",
        cv=3,
    )
    return {
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "SVM": calibrated_svm,
        "ANN": MLPClassifier(
            hidden_layer_sizes=(48, 24),
            activation="relu",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    }


def train_model_suite(frame: pd.DataFrame) -> TrainingResult:
    """Train KNN, SVM, and ANN on one reproducible stratified split."""
    X = frame[FEATURES].copy()
    y = frame["Performance_Category"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models: dict[str, Pipeline] = {}
    matrices: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    metrics: list[dict[str, float | str]] = []

    for name, estimator in build_estimators().items():
        try:
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
            pipeline.fit(X_train, y_train)
            predicted = pipeline.predict(X_test)
            models[name] = pipeline
            matrices[name] = confusion_matrix(y_test, predicted, labels=CLASS_ORDER)
            metrics.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, predicted),
                    "Precision": precision_score(y_test, predicted, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, predicted, average="weighted", zero_division=0),
                    "Weighted F1": f1_score(y_test, predicted, average="weighted", zero_division=0),
                    "Macro F1": f1_score(y_test, predicted, average="macro", zero_division=0),
                }
            )
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    if not models:
        raise RuntimeError("All machine-learning models failed to train: " + repr(errors))

    evaluation = pd.DataFrame(metrics).sort_values(
        ["Macro F1", "Accuracy", "Weighted F1"], ascending=False
    ).reset_index(drop=True)
    best_model = str(evaluation.loc[0, "Model"])
    return TrainingResult(models, evaluation, matrices, best_model, errors)


def validate_batch(raw: pd.DataFrame, maximum_rows: int = 10_000) -> tuple[pd.DataFrame, list[str]]:
    """Validate uploaded batch data and return a numeric copy plus clear errors."""
    errors: list[str] = []
    missing = [feature for feature in FEATURES if feature not in raw.columns]
    if missing:
        return raw.copy(), ["Missing required column(s): " + ", ".join(missing)]
    if raw.empty:
        return raw.copy(), ["The uploaded file contains no student records."]
    if len(raw) > maximum_rows:
        errors.append(f"Maximum batch size is {maximum_rows:,} students; received {len(raw):,}.")

    cleaned = raw.copy()
    for feature, (low, high) in FEATURE_RANGES.items():
        numeric = pd.to_numeric(cleaned[feature], errors="coerce")
        invalid_numeric = int(numeric.isna().sum())
        if invalid_numeric:
            errors.append(f"{FEATURE_LABELS[feature]} has {invalid_numeric} missing or non-numeric value(s).")
            continue
        invalid_range = int((~numeric.between(low, high, inclusive="both")).sum())
        if invalid_range:
            errors.append(
                f"{FEATURE_LABELS[feature]} has {invalid_range} value(s) outside {low:g}–{high:g}."
            )
        cleaned[feature] = numeric
    return cleaned, errors


def predict_model_details(
    models: dict[str, Pipeline], row: pd.DataFrame, best_model: str
) -> tuple[str, float, pd.DataFrame]:
    """Return the selected prediction plus transparent per-model details."""
    details: list[dict[str, float | str]] = []
    for name, model in models.items():
        prediction = str(model.predict(row[FEATURES])[0])
        probability = model.predict_proba(row[FEATURES])[0]
        details.append(
            {
                "Model": name,
                "Prediction": prediction,
                "Confidence": float(np.max(probability)),
            }
        )
    detail_frame = pd.DataFrame(details)
    selected = detail_frame.loc[detail_frame["Model"] == best_model].iloc[0]
    return str(selected["Prediction"]), float(selected["Confidence"]), detail_frame


def predict_batch(
    models: dict[str, Pipeline], batch: pd.DataFrame, best_model: str
) -> pd.DataFrame:
    """Append all model predictions and a final best-model decision."""
    output = batch.copy()
    X = output[FEATURES]
    for name, model in models.items():
        output[f"{name}_Prediction"] = model.predict(X)
        output[f"{name}_Confidence"] = model.predict_proba(X).max(axis=1)
    output["Final_Prediction"] = output[f"{best_model}_Prediction"]
    output["Final_Confidence"] = output[f"{best_model}_Confidence"]
    output["Selected_Model"] = best_model
    return output


def feature_relevance(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate two complementary, reproducible feature-relevance measures."""
    class_codes = frame["Performance_Category"].map(
        {"At Risk": 0, "Average": 1, "Good": 2, "Excellent": 3}
    )
    information = mutual_info_classif(frame[FEATURES], class_codes, random_state=RANDOM_STATE)
    spearman = frame[FEATURES].apply(lambda series: series.corr(frame["Final_CGPA"], method="spearman"))
    result = pd.DataFrame(
        {
            "Feature Key": FEATURES,
            "Feature": [FEATURE_LABELS[feature] for feature in FEATURES],
            "Mutual Information": information,
            "Spearman Correlation": [spearman[feature] for feature in FEATURES],
        }
    )
    return result.sort_values(
        ["Mutual Information", "Spearman Correlation"], ascending=False
    ).reset_index(drop=True)


def student_recommendations(row: pd.Series, category: str) -> list[str]:
    """Create concise, input-driven academic support recommendations."""
    tips: list[str] = []
    if float(row["Attendance_Pct"]) < 80.0:
        tips.append("Improve attendance and schedule catch-up sessions for missed lessons.")
    if float(row["Study_Hours_Per_Day"]) < 2.0:
        tips.append("Build a consistent daily study plan with short, focused revision blocks.")
    if float(row["Sleep_Hours"]) < 6.0:
        tips.append("Aim for a more consistent sleep routine to support concentration and memory.")
    if float(row["Average_Score"]) < 65.0:
        tips.append("Prioritise weaker subjects and seek lecturer or peer-tutoring support.")
    if float(row["Previous_CGPA"]) < 2.75:
        tips.append("Set a short-term CGPA target and review progress every two weeks.")
    if category == "At Risk" and len(tips) < 4:
        tips.append("Arrange an early academic-adviser meeting and agree on a monitored support plan.")
    if not tips:
        tips.append("Maintain the current learning routine and continue monitoring performance.")
    return tips[:4]
