from pathlib import Path
import sys
import pandas as pd

from student_model import FEATURES, load_dataset, predict_batch, predict_model_details, train_model_suite, validate_batch

ROOT = Path(__file__).resolve().parent
EXPECTED_FEATURES = ["Average_Score", "Attendance_Pct", "Study_Hours_Per_Day", "Previous_CGPA"]


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


for filename in ["app.py", "student_model.py", "Student_data.csv", "requirements.txt", "runtime.txt"]:
    if not (ROOT / filename).exists():
        fail(f"Missing {filename}")
print("[OK] Required deployment files exist")

if FEATURES != EXPECTED_FEATURES:
    fail(f"Final feature list must be exactly {EXPECTED_FEATURES}; got {FEATURES}")
print("[OK] Exact four-feature definition is locked")

app_text = (ROOT / "app.py").read_text(encoding="utf-8")
legacy_terms = [
    'Input Features\",\"5',
    'Required Features\",\"5',
    "five features",
    "Sleep_Hours",
    "Sleep Hours",
    "Number_of_Subjects",
    "Number of Subjects",
]
for term in legacy_terms:
    if term.lower() in app_text.lower():
        fail(f"Legacy five-feature UI term remains in app.py: {term}")
for marker in ['Input Features\",\"4', 'Required Features\",\"4', "Exactly 4 ML Features"]:
    if marker not in app_text:
        fail(f"Expected four-feature UI marker missing: {marker}")
print("[OK] Prediction Hub, Batch UI and feature pages are locked to four inputs")

df, source = load_dataset()
if source != "repository dataset":
    fail("Repository dataset did not pass validation")
if len(df) != 5000:
    fail(f"Expected 5,000 rows, found {len(df):,}")
if df[FEATURES + ["Final_CGPA"]].isna().any().any():
    fail("Selected training columns contain missing values")
print("[OK] Repository dataset and selected features are valid")

suite = train_model_suite(df)
if set(suite.evaluation["Model"]) != {"KNN", "SVM", "ANN"}:
    fail("Expected KNN, SVM and ANN")
if not suite.evaluation["Accuracy"].between(0, 1).all():
    fail("Invalid model accuracy result")
print("[OK] KNN, SVM and ANN train successfully")

sample = pd.DataFrame([[80.0, 90.0, 4.0, 3.4]], columns=FEATURES)
details = predict_model_details(sample, suite)
if len(details) != 3 or not details["Confidence"].between(0, 1).all():
    fail("Individual prediction path failed")
print("[OK] Individual four-feature prediction path works")

batch = pd.DataFrame([
    {"Student_ID": "S001", "Student_Name": "Test A", "Average_Score": 80, "Attendance_Pct": 90, "Study_Hours_Per_Day": 4, "Previous_CGPA": 3.4},
    {"Student_ID": "S002", "Student_Name": "Test B", "Average_Score": 60, "Attendance_Pct": 70, "Study_Hours_Per_Day": 2, "Previous_CGPA": 2.4},
])
if validate_batch(batch):
    fail("Valid batch data was rejected")
result = predict_batch(batch, suite)
if len(result) != 2 or "Final_Prediction" not in result:
    fail("Batch prediction path failed")
print("[OK] Batch validation and four-feature prediction paths work")

print("\nALL CHECKS PASSED")
print(suite.evaluation.to_string(index=False))
print("\nFeature correlations with Final CGPA:")
print(suite.feature_correlations.to_string())
