"""Core machine-learning functions for the four-feature student performance system."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Student_data.csv"
RANDOM_STATE = 42
TARGET = "Final_CGPA"

# Exactly four ML inputs are used everywhere in the final system.
FEATURES = [
    "Average_Score",
    "Attendance_Pct",
    "Study_Hours_Per_Day",
    "Previous_CGPA",
]
FEATURE_LABELS = {
    "Average_Score": "Average Score",
    "Attendance_Pct": "Attendance Percentage",
    "Study_Hours_Per_Day": "Study Hours per Day",
    "Previous_CGPA": "Previous CGPA",
}
FEATURE_RANGES = {
    "Average_Score": (0.0, 100.0),
    "Attendance_Pct": (0.0, 100.0),
    "Study_Hours_Per_Day": (0.0, 24.0),
    "Previous_CGPA": (0.0, 4.0),
}
CLASS_ORDER = ["At Risk", "Average", "Good", "Excellent"]
CLASS_COLORS = {
    "At Risk": "#ef4444",
    "Average": "#f59e0b",
    "Good": "#2563eb",
    "Excellent": "#16a34a",
}


@dataclass
class ModelSuite:
    models: dict
    evaluation: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray]
    classes: list[str]
    best_model: str
    feature_correlations: pd.Series
    train_size: int
    test_size: int


def performance_category(final_cgpa: float) -> str:
    if final_cgpa >= 3.50:
        return "Excellent"
    if final_cgpa >= 3.00:
        return "Good"
    if final_cgpa >= 2.50:
        return "Average"
    return "At Risk"


def _fallback_dataset(rows: int = 5000) -> pd.DataFrame:
    """Deterministic fallback keeps the Streamlit interface available if the CSV is missing."""
    rng = np.random.default_rng(RANDOM_STATE)
    previous = np.clip(rng.normal(3.0, 0.55, rows), 1.0, 4.0)
    attendance = np.clip(rng.normal(82, 10, rows), 35, 100)
    study = np.clip(rng.normal(3.6, 1.7, rows), 0, 12)
    average = np.clip(16 + previous * 16 + attendance * 0.12 + study * 1.4 + rng.normal(0, 7, rows), 30, 100)
    final = np.clip(
        0.61 * previous + 0.012 * average + 0.0028 * attendance + 0.018 * study + rng.normal(0, 0.20, rows),
        1.0,
        4.0,
    )
    return pd.DataFrame(
        {
            "Student_ID": [f"F{i:05d}" for i in range(1, rows + 1)],
            "Average_Score": average.round(2),
            "Attendance_Pct": attendance.round(1),
            "Study_Hours_Per_Day": study.round(1),
            "Previous_CGPA": previous.round(2),
            "Final_CGPA": final.round(2),
        }
    )


def load_dataset() -> tuple[pd.DataFrame, str]:
    source = "repository dataset"
    try:
        df = pd.read_csv(DATA_PATH)
        required = FEATURES + [TARGET]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))

        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[required].isna().any().any():
            raise ValueError("Selected training columns contain missing or non-numeric values")
    except Exception:
        df = _fallback_dataset()
        source = "deterministic fallback dataset"

    df = df.copy()
    df["Performance_Category"] = df[TARGET].apply(performance_category)
    return df, source


def _algorithms() -> dict[str, object]:
    return {
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance", p=2),
        "SVM": SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=900,
            random_state=RANDOM_STATE,
        ),
    }


def train_model_suite(df: pd.DataFrame) -> ModelSuite:
    X = df[FEATURES].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Performance_Category"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models: dict[str, dict] = {}
    confusion_matrices: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for name, algorithm in _algorithms().items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", algorithm)])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, prediction, average="weighted", zero_division=0),
            }
        )
        models[name] = {"model": pipeline, "label_encoder": encoder}
        confusion_matrices[name] = confusion_matrix(y_test, prediction)

    evaluation = pd.DataFrame(rows).sort_values(["Accuracy", "F1 Score"], ascending=False).reset_index(drop=True)
    correlations = (
        df[FEATURES + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
    )

    return ModelSuite(
        models=models,
        evaluation=evaluation,
        confusion_matrices=confusion_matrices,
        classes=encoder.classes_.tolist(),
        best_model=str(evaluation.iloc[0]["Model"]),
        feature_correlations=correlations,
        train_size=len(X_train),
        test_size=len(X_test),
    )


def predict_model_details(input_df: pd.DataFrame, suite: ModelSuite) -> pd.DataFrame:
    rows = []
    X = input_df[FEATURES]
    for name, bundle in suite.models.items():
        encoded = bundle["model"].predict(X)
        labels = bundle["label_encoder"].inverse_transform(encoded)
        probabilities = bundle["model"].predict_proba(X)
        rows.append(
            {
                "Model": name,
                "Prediction": str(labels[0]),
                "Confidence": float(probabilities[0].max()),
            }
        )
    return pd.DataFrame(rows)


def validate_batch(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        return ["Missing required column(s): " + ", ".join(missing)]

    for feature in FEATURES:
        minimum, maximum = FEATURE_RANGES[feature]
        values = pd.to_numeric(df[feature], errors="coerce")
        invalid = values.isna() | (values < minimum) | (values > maximum)
        if invalid.any():
            excel_rows = (df.index[invalid] + 2).tolist()[:10]
            errors.append(
                f"{feature} must be between {minimum:g} and {maximum:g}. Invalid Excel row(s): "
                + ", ".join(map(str, excel_rows))
            )
    return errors


def predict_batch(df: pd.DataFrame, suite: ModelSuite) -> pd.DataFrame:
    result = df.copy()
    X = result[FEATURES].apply(pd.to_numeric)

    for name, bundle in suite.models.items():
        encoded = bundle["model"].predict(X)
        labels = bundle["label_encoder"].inverse_transform(encoded)
        probabilities = bundle["model"].predict_proba(X)
        result[f"{name}_Prediction"] = labels
        result[f"{name}_Confidence"] = probabilities.max(axis=1)

    best = suite.best_model
    result["Final_Prediction"] = result[f"{best}_Prediction"]
    result["Final_Confidence"] = result[f"{best}_Confidence"]
    return result


def student_recommendation(category: str) -> str:
    return {
        "Excellent": "Maintain the current learning routine and continue challenging yourself with higher-level academic goals.",
        "Good": "Performance is strong. Keep attendance and study habits consistent to move toward Excellent.",
        "Average": "Use a structured study schedule, improve attendance where possible and seek lecturer feedback early.",
        "At Risk": "Early academic intervention is recommended. Review study habits, attendance and assessment performance with a lecturer or advisor.",
    }.get(category, "Review the prediction with a lecturer or academic advisor.")
