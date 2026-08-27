from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "Student_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

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

CLASS_ORDER = ["At Risk", "Average", "Good", "Excellent"]

def category_from_cgpa(final_cgpa: float) -> str:
    value = float(final_cgpa)
    if value >= 3.50:
        return "Excellent"
    if value >= 3.00:
        return "Good"
    if value >= 2.50:
        return "Average"
    return "At Risk"

RANDOM_STATE = 42

CANDIDATE_NUMERIC_FEATURES = [
    "Previous_CGPA",
    "Average_Score",
    "Attendance_Pct",
    "Study_Hours_Per_Day",
    "Sleep_Hours",
    "Social_Hours_Week",
    "Age",
    "Number_of_Subjects",
]


def build_models() -> dict[str, Pipeline]:
    """Return stable, pre-selected model configurations."""
    return {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=21, weights="distance")),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=5.0, gamma=0.05, probability=True, random_state=RANDOM_STATE)),
        ]),
        "ANN": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    alpha=0.0001,
                    max_iter=700,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }


def validate_training_data(df: pd.DataFrame) -> None:
    required = set(FEATURES + ["Final_CGPA"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    for column in required:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{column} contains missing or non-numeric values.")


def create_feature_selection_report(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    available = [column for column in CANDIDATE_NUMERIC_FEATURES if column in df.columns]
    X = df[available].apply(pd.to_numeric, errors="coerce")

    ordinal = y.map({"At Risk": 0, "Average": 1, "Good": 2, "Excellent": 3}).astype(int)
    mutual_info = mutual_info_classif(X, ordinal, random_state=RANDOM_STATE)

    rows = []
    for column, mi in zip(available, mutual_info):
        spearman = X[column].corr(ordinal, method="spearman")
        rows.append({
            "Feature": column,
            "Friendly_Name": FEATURE_LABELS.get(column, column.replace("_", " ")),
            "Mutual_Information": float(mi),
            "Abs_Spearman": float(abs(spearman)) if pd.notna(spearman) else 0.0,
            "Selected": column in FEATURES,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["Mutual_Information", "Abs_Spearman"], ascending=False)
        .reset_index(drop=True)
    )


def save_confusion_matrix(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(matrix)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name.lower()}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    validate_training_data(df)
    y_labels = df["Final_CGPA"].map(category_from_cgpa)
    class_to_int = {label: index for index, label in enumerate(CLASS_ORDER)}
    y = y_labels.map(class_to_int).astype(int)
    X = df[FEATURES].copy()

    feature_report = create_feature_selection_report(df, y_labels)
    feature_report.to_csv(RESULTS_DIR / "feature_selection.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    evaluation_rows = []

    for name, model in build_models().items():
        print(f"Training and evaluating {name}...")

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=1,
        )

        evaluation_model = clone(model)
        evaluation_model.fit(X_train, y_train)
        prediction = evaluation_model.predict(X_test).astype(int)
        y_test_labels = y_test.map(lambda value: CLASS_ORDER[int(value)])
        prediction_labels = np.array([CLASS_ORDER[int(value)] for value in prediction])

        evaluation_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
            "F1_Score": f1_score(y_test, prediction, average="weighted", zero_division=0),
            "Macro_F1": f1_score(y_test, prediction, average="macro", zero_division=0),
            "CV_F1_Mean": float(cv_scores.mean()),
            "CV_F1_Std": float(cv_scores.std()),
        })

        report = pd.DataFrame(
            classification_report(
                y_test_labels,
                prediction_labels,
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        report.to_csv(RESULTS_DIR / f"{name.lower()}_classification_report.csv")
        save_confusion_matrix(name, y_test_labels, prediction_labels)

        # Refit on all available records for the production artifact.
        production_model = clone(model)
        production_model.fit(X, y)
        joblib.dump(
            {
                "model": production_model,
                "features": FEATURES,
                "classes": CLASS_ORDER,
                "class_order": CLASS_ORDER,
                "trained_rows": len(df),
                "sklearn_version_note": "Use the pinned scikit-learn version from requirements.txt.",
            },
            MODEL_DIR / f"{name.lower()}_model.joblib",
        )

    evaluation = (
        pd.DataFrame(evaluation_rows)
        .sort_values(["F1_Score", "Accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    evaluation.to_csv(RESULTS_DIR / "evaluation.csv", index=False)

    best = evaluation.iloc[0]
    metadata = {
        "project_version": "2.0.0",
        "dataset_rows": int(len(df)),
        "features": FEATURES,
        "feature_labels": {feature: FEATURE_LABELS[feature] for feature in FEATURES},
        "target": "Performance_Category derived from Final_CGPA",
        "classes": CLASS_ORDER,
        "cgpa_ranges": {
            "At Risk": "Below 2.50",
            "Average": "2.50 - 2.99",
            "Good": "3.00 - 3.49",
            "Excellent": "3.50 - 4.00",
        },
        "best_model": str(best["Model"]),
        "best_accuracy": float(best["Accuracy"]),
        "best_f1": float(best["F1_Score"]),
        "split": "80% train / 20% stratified holdout test",
        "selection_metric": "Weighted F1 Score",
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(evaluation.to_string(index=False))
    print(f"\nBest model: {metadata['best_model']} | F1={metadata['best_f1']:.3f} | Accuracy={metadata['best_accuracy']:.3f}")


if __name__ == "__main__":
    main()
