"""Run this locally to verify the project data and three ML models before deployment."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Student_data.csv"
FEATURES = ["Previous_CGPA", "Average_Score", "Attendance_Pct", "Study_Hours_Per_Day", "Sleep_Hours"]

def category(v):
    if v >= 3.5: return "Excellent"
    if v >= 3.0: return "Good"
    if v >= 2.5: return "Average"
    return "At Risk"

def main():
    print("1/4 Checking project files...")
    for name in ["app.py", "requirements.txt", "Student_data.csv"]:
        path = ROOT / name
        if not path.exists():
            raise SystemExit(f"FAILED: {name} is missing")
        print(f"  OK: {name}")

    print("2/4 Checking dataset...")
    df = pd.read_csv(DATA)
    required = FEATURES + ["Final_CGPA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("FAILED: missing columns: " + ", ".join(missing))
    print(f"  OK: {len(df):,} rows; required columns present")

    print("3/4 Training models...")
    work = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    X = work[FEATURES]
    y = work["Final_CGPA"].map(category)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "SVM": SVC(C=2.0, probability=True, random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=350, early_stopping=False, random_state=42),
    }
    for name, clf in models.items():
        model = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        print(f"  OK: {name} accuracy={acc:.4f}")

    print("4/4 Deployment-path check...")
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    forbidden = ["from src.core", "Run python train_model.py once", "application could not load its data/model artifacts"]
    for text in forbidden:
        if text.lower() in app_text.lower():
            raise SystemExit(f"FAILED: old deployment code still present: {text}")
    print("  OK: no old src/model/dataset hard-dependency messages")
    print("\nALL CHECKS PASSED")

if __name__ == "__main__":
    main()
