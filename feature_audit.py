"""Reproduce the evidence used to select the four final predictors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from student_model import FEATURES, feature_relevance, load_dataset


ROOT = Path(__file__).resolve().parent


def macro_f1(frame: pd.DataFrame, features: list[str]) -> float:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=2.0, kernel="rbf", class_weight="balanced", random_state=42)),
        ]
    )
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        frame[features],
        frame["Performance_Category"],
        cv=split,
        scoring="f1_macro",
        n_jobs=-1,
    )
    return float(scores.mean())


def main() -> None:
    frame = load_dataset(ROOT).frame
    audit = feature_relevance(frame)
    print("PROJECT NUMERIC FEATURE AUDIT")
    print(
        audit[
            [
                "Feature",
                "Pearson Correlation",
                "Spearman Correlation",
                "Mutual Information",
                "Decision",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

    sets = {
        "Previous CGPA only": ["Previous_CGPA"],
        "+ Average Score": ["Previous_CGPA", "Average_Score"],
        "+ Attendance": ["Previous_CGPA", "Average_Score", "Attendance_Pct"],
        "Selected four": FEATURES,
        "+ Number Subjects": FEATURES + ["Number_of_Subjects"],
    }
    results = {name: macro_f1(frame, features) for name, features in sets.items()}
    print("\nFIVE-FOLD SVM MACRO F1")
    for name, score in results.items():
        print(f"{name:20s} {score:.6f}")

    assert results["+ Average Score"] > results["Previous CGPA only"]
    assert results["+ Attendance"] > results["+ Average Score"]
    assert results["Selected four"] > results["+ Attendance"]
    assert results["Selected four"] > results["+ Number Subjects"]
    print("\nFEATURE AUDIT PASSED")


if __name__ == "__main__":
    main()
