"""Fail-fast verification for data, models, predictions, and deployment structure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from student_model import (
    CLASS_ORDER,
    FEATURES,
    feature_relevance,
    load_dataset,
    predict_batch,
    predict_model_details,
    train_model_suite,
    validate_batch,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("1/6 Checking deployment files...")
    required_files = [
        "app.py",
        "student_model.py",
        "Student_data.csv",
        "requirements.txt",
        "runtime.txt",
        ".streamlit/config.toml",
    ]
    for name in required_files:
        path = ROOT / name
        assert path.is_file(), f"Missing deployment file: {name}"
        print(f"  OK: {name}")

    print("2/6 Validating dataset...")
    result = load_dataset(ROOT)
    frame = result.frame
    assert result.source == "Repository CSV", f"Unexpected data source: {result.source}"
    assert len(frame) == 5_000, f"Expected 5,000 records, found {len(frame):,}"
    assert set(CLASS_ORDER) == set(frame["Performance_Category"].unique())
    assert not frame[FEATURES + ["Final_CGPA"]].isna().any().any()
    print(f"  OK: {len(frame):,} valid records and all four target classes")

    print("3/6 Checking feature evidence...")
    relevance = feature_relevance(frame)
    strongest = set(relevance.head(2)["Feature Key"])
    assert strongest == {"Previous_CGPA", "Average_Score"}, strongest
    print("  OK: Previous CGPA and Average Score are the two strongest selected signals")

    print("4/6 Training and evaluating models...")
    training = train_model_suite(frame)
    assert set(training.models) == {"KNN", "SVM", "ANN"}, training.errors
    assert training.evaluation["Accuracy"].between(0.0, 1.0).all()
    assert training.evaluation["Macro F1"].between(0.0, 1.0).all()
    for row in training.evaluation.itertuples(index=False):
        print(
            f"  OK: {row.Model} accuracy={row.Accuracy:.4f} "
            f"weighted_f1={getattr(row, '_4'):.4f} macro_f1={getattr(row, '_5'):.4f}"
        )

    print("5/6 Testing individual and batch prediction paths...")
    sample = pd.DataFrame(
        [
            {
                "Student_ID": "TEST001",
                "Student_Name": "Test Student",
                "Previous_CGPA": 3.20,
                "Average_Score": 74.5,
                "Attendance_Pct": 88.0,
                "Study_Hours_Per_Day": 3.5,
                "Sleep_Hours": 7.0,
            }
        ]
    )
    cleaned, errors = validate_batch(sample)
    assert not errors, errors
    prediction, confidence, detail = predict_model_details(
        training.models, cleaned, training.best_model
    )
    assert prediction in CLASS_ORDER
    assert 0.0 <= confidence <= 1.0
    assert len(detail) == 3
    batch_result = predict_batch(training.models, cleaned, training.best_model)
    assert batch_result.loc[0, "Final_Prediction"] in CLASS_ORDER
    assert 0.0 <= float(batch_result.loc[0, "Final_Confidence"]) <= 1.0
    print(f"  OK: individual={prediction}, batch={batch_result.loc[0, 'Final_Prediction']}")

    print("6/6 Checking removal of legacy deployment dependencies...")
    combined = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ["app.py", "student_model.py"]
    ).lower()
    forbidden = [
        "application could not load its data/model artifacts",
        "run python train_model.py once",
        "from src.core",
        "dataset/student_data.csv'",
    ]
    for phrase in forbidden:
        assert phrase not in combined, f"Legacy deployment text remains: {phrase}"
    print("  OK: no legacy src/model/dataset hard dependency")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
