"""Optional local verification script. The Streamlit app trains models automatically."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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
    if not DATA.exists():
        raise SystemExit("Student_data.csv is missing beside train_model.py")
    df = pd.read_csv(DATA)
    required = FEATURES + ["Final_CGPA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(missing))
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    y = df["Final_CGPA"].map(category)
    X = df[FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "SVM": SVC(C=2.0, probability=True, random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=350, early_stopping=False, random_state=42),
    }
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rows.append((name, accuracy_score(yte,pred), f1_score(yte,pred,average="weighted")))
    print("Model verification completed successfully")
    for name, acc, f1 in sorted(rows, key=lambda x:x[1], reverse=True):
        print(f"{name}: Accuracy={acc:.4f}, Weighted F1={f1:.4f}")

if __name__ == "__main__":
    main()
