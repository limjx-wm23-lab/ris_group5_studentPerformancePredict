from __future__ import annotations

from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent

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

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

st.markdown(
    """
<style>
.stApp {background: linear-gradient(180deg,#f8fbff 0%,#f8fafc 55%,#f7f5ff 100%);}
.block-container {max-width:1180px;padding-top:1.1rem;padding-bottom:2rem;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0b1220 0%,#111827 55%,#172554 100%);}
[data-testid="stSidebar"] * {color:white;}
.hero {padding:1.35rem 1.5rem;border-radius:24px;background:linear-gradient(135deg,#0f172a,#1e3a8a 62%,#6d28d9);color:white;box-shadow:0 18px 45px rgba(30,58,138,.18);margin-bottom:1rem;}
.hero h1 {margin:0;color:white;font-size:2rem;}.hero p {margin:.55rem 0 0;color:#dbeafe;}
.card {background:white;border:1px solid #e2e8f0;border-radius:18px;padding:1rem 1.05rem;box-shadow:0 10px 30px rgba(15,23,42,.05);margin-bottom:.85rem;}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;border-radius:16px;padding:.75rem .85rem;}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button {border-radius:12px!important;font-weight:800!important;}
.feature-box {background:#eef2ff;border:1px solid #dbeafe;border-radius:14px;padding:.8rem;text-align:center;font-weight:800;color:#3730a3;min-height:76px;display:flex;align-items:center;justify-content:center;}
.badge {display:inline-block;padding:.28rem .55rem;border-radius:999px;background:#eef2ff;border:1px solid #dbeafe;color:#3730a3;font-size:.78rem;font-weight:700;margin:.1rem;}
</style>
""",
    unsafe_allow_html=True,
)


def performance_category(value: float) -> str:
    value = float(value)
    if value >= 3.50:
        return "Excellent"
    if value >= 3.00:
        return "Good"
    if value >= 2.50:
        return "Average"
    return "At Risk"


def make_fallback_dataset(n: int = 1800) -> pd.DataFrame:
    """Deterministic built-in dataset so deployment can still run if the CSV is missing."""
    rng = np.random.default_rng(42)
    previous = np.clip(rng.normal(3.0, 0.55, n), 1.0, 4.0)
    attendance = np.clip(rng.normal(83, 10, n), 45, 100)
    study = np.clip(rng.normal(3.2, 1.25, n), 0.2, 9.0)
    sleep = np.clip(rng.normal(7.0, 1.25, n), 3.5, 10.5)
    score = np.clip(20 + 16 * previous + 0.15 * attendance + 2.0 * study + rng.normal(0, 7, n), 35, 100)
    final = np.clip(
        0.52 * previous + 0.018 * score + 0.0035 * attendance + 0.035 * study + rng.normal(0, 0.20, n) - 0.65,
        1.0,
        4.0,
    )
    return pd.DataFrame(
        {
            "Student_ID": [f"DEMO{i:05d}" for i in range(1, n + 1)],
            "Attendance_Pct": np.round(attendance, 1),
            "Study_Hours_Per_Day": np.round(study, 1),
            "Previous_CGPA": np.round(previous, 2),
            "Sleep_Hours": np.round(sleep, 1),
            "Final_CGPA": np.round(final, 2),
            "Average_Score": np.round(score, 2),
        }
    )


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    required = FEATURES + ["Final_CGPA"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = raw.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    if len(df) < 100:
        raise ValueError("Dataset has fewer than 100 valid rows after cleaning.")
    for col in FEATURES:
        lo, hi = FEATURE_RANGES[col]
        df = df[df[col].between(lo, hi, inclusive="both")]
    df = df[df["Final_CGPA"].between(0, 4, inclusive="both")]
    if len(df) < 100:
        raise ValueError("Dataset has too few valid rows after range validation.")
    df["Performance_Category"] = df["Final_CGPA"].map(performance_category)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, str, str]:
    candidates = [
        ROOT / "Student_data.csv",
        ROOT / "student_data.csv",
        ROOT / "dataset" / "Student_data.csv",
        ROOT / "dataset" / "student_data.csv",
    ]
    problems = []
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                return prepare_dataset(pd.read_csv(path)), "GitHub CSV", str(path.name)
            except Exception as exc:
                problems.append(f"{path.name}: {exc}")
    fallback = prepare_dataset(make_fallback_dataset())
    note = "Built-in fallback dataset"
    if problems:
        note += " (repository CSV was invalid)"
    return fallback, "Built-in fallback", note


def build_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "SVM": SVC(C=2.0, kernel="rbf", probability=True, random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=350, early_stopping=False, random_state=42),
    }


@st.cache_resource(show_spinner="Training KNN, SVM and ANN models...")
def train_models(df_signature: str, csv_payload: bytes):
    # csv_payload is used for stable Streamlit caching across reruns.
    df = pd.read_csv(BytesIO(csv_payload))
    X = df[FEATURES].copy()
    y = df["Performance_Category"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {}
    rows = []
    matrices = {}
    errors = {}

    for name, classifier in build_models().items():
        try:
            model = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            models[name] = model
            matrices[name] = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0),
                }
            )
        except Exception as exc:
            errors[name] = str(exc)

    if not models:
        raise RuntimeError("All machine-learning models failed to train: " + str(errors))

    evaluation = pd.DataFrame(rows).sort_values(["Accuracy", "F1 Score"], ascending=False).reset_index(drop=True)
    best_model = str(evaluation.loc[0, "Model"])
    return models, evaluation, matrices, best_model, errors


def dataset_to_cache_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def validate_batch(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return ["Missing required column(s): " + ", ".join(missing)]
    if df.empty:
        return ["The uploaded file contains no records."]
    if len(df) > 10000:
        errors.append("Maximum batch size is 10,000 students.")
    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.isna().any():
            errors.append(f"{feature} contains missing or non-numeric values.")
            continue
        lo, hi = FEATURE_RANGES[feature]
        if (~values.between(lo, hi, inclusive="both")).any():
            errors.append(f"{feature} must be between {lo:g} and {hi:g}.")
    return errors


def predict_one(models, row: pd.DataFrame, best_model: str):
    details = []
    for name, model in models.items():
        pred = str(model.predict(row[FEATURES])[0])
        conf = float(np.max(model.predict_proba(row[FEATURES])[0])) if hasattr(model, "predict_proba") else np.nan
        details.append({"Model": name, "Prediction": pred, "Confidence": conf})
    detail = pd.DataFrame(details)
    final_row = detail[detail["Model"] == best_model].iloc[0]
    return str(final_row["Prediction"]), float(final_row["Confidence"]), detail


def predict_batch(models, batch_df: pd.DataFrame, best_model: str) -> pd.DataFrame:
    out = batch_df.copy()
    X = out[FEATURES].apply(pd.to_numeric)
    for name, model in models.items():
        out[f"{name}_Prediction"] = model.predict(X)
        if hasattr(model, "predict_proba"):
            out[f"{name}_Confidence"] = model.predict_proba(X).max(axis=1)
    out["Final_Prediction"] = out[f"{best_model}_Prediction"]
    out["Final_Confidence"] = out[f"{best_model}_Confidence"]
    out["Best_Model"] = best_model
    return out


def make_template() -> bytes:
    sample = pd.DataFrame(
        {
            "Student_ID": ["ID05001", "ID05002", "ID05003"],
            "Student_Name": ["Example Student A", "Example Student B", "Example Student C"],
            "Previous_CGPA": [3.20, 2.70, 3.65],
            "Average_Score": [74.5, 62.0, 84.0],
            "Attendance_Pct": [88.0, 76.0, 94.0],
            "Study_Hours_Per_Day": [3.5, 2.0, 4.5],
            "Sleep_Hours": [7.0, 6.5, 7.5],
        }
    )
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        sample.to_excel(writer, index=False, sheet_name="Batch Input")
        ws = writer.sheets["Batch Input"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = __import__("openpyxl").styles.Font(color="FFFFFF", bold=True)
            cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="1E3A8A")
        for col, width in {"A":16,"B":24,"C":16,"D":16,"E":18,"F":22,"G":16}.items():
            ws.column_dimensions[col].width = width
    return out.getvalue()


def make_result_excel(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prediction Results")
        summary = df["Final_Prediction"].value_counts().reindex(CLASS_ORDER, fill_value=0).rename_axis("Category").reset_index(name="Students")
        summary["Percentage"] = summary["Students"] / max(len(df), 1)
        summary.to_excel(writer, index=False, sheet_name="Summary")
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = __import__("openpyxl").styles.Font(color="FFFFFF", bold=True)
                cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="1E3A8A")
            for cells in ws.columns:
                width = min(max(max(len(str(c.value or "")) for c in cells) + 2, 12), 28)
                ws.column_dimensions[cells[0].column_letter].width = width
    return out.getvalue()


def recommendations(row: pd.Series, category: str) -> list[str]:
    tips = []
    if float(row["Attendance_Pct"]) < 80: tips.append("Improve attendance and review missed lessons.")
    if float(row["Study_Hours_Per_Day"]) < 2: tips.append("Build a consistent daily study schedule.")
    if float(row["Sleep_Hours"]) < 6: tips.append("Aim for a more regular sleep schedule.")
    if float(row["Average_Score"]) < 65: tips.append("Focus revision on weaker subjects and seek academic support.")
    if float(row["Previous_CGPA"]) < 2.75: tips.append("Set short-term CGPA targets and monitor progress regularly.")
    if not tips:
        tips.append("Maintain the current learning routine and continue monitoring progress.")
    return tips[:4]


# ----------------------
# Safe startup
# ----------------------
try:
    df, data_source, data_note = get_dataset()
except Exception:
    df = prepare_dataset(make_fallback_dataset())
    data_source, data_note = "Built-in fallback", "Automatic emergency fallback"

payload = dataset_to_cache_bytes(df)
try:
    models, evaluation, matrices, best_model, model_errors = train_models(str(len(df)), payload)
except Exception as exc:
    st.error("Model initialization failed. Please reboot the app after dependencies finish installing.")
    st.code(str(exc))
    st.stop()

# ----------------------
# Navigation
# ----------------------
st.sidebar.markdown("# 🎓 Student AI")
st.sidebar.caption("Performance Prediction System")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Individual Prediction", "Batch Prediction", "Model Evaluation", "Feature Analysis", "Dataset Explorer", "About"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: {data_source}")
st.sidebar.caption(f"Students: {len(df):,}")
st.sidebar.caption(f"Best model: {best_model}")
st.sidebar.caption("Version 4.0 · Deployment-proof")

if data_source != "GitHub CSV":
    st.warning("Student_data.csv was not found or was invalid. The app is still running with a built-in fallback dataset. Upload Student_data.csv beside app.py in GitHub to use the project dataset.")

if page == "Overview":
    st.markdown('<div class="hero"><h1>Student Performance Prediction System</h1><p>Machine-learning decision support using KNN, SVM and ANN for early academic performance screening.</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Selected Features", "5")
    c3.metric("Models Trained", str(len(models)))
    c4.metric("Best Model", best_model)

    st.markdown("### Five Selected Features")
    cols = st.columns(5)
    for col, feat in zip(cols, FEATURES):
        col.markdown(f'<div class="feature-box">{FEATURE_LABELS[feat]}</div>', unsafe_allow_html=True)

    st.markdown("### Performance Categories")
    counts = df["Performance_Category"].value_counts().reindex(CLASS_ORDER, fill_value=0).reset_index()
    counts.columns = ["Category", "Students"]
    fig = px.bar(counts, x="Category", y="Students", color="Category", color_discrete_map=CLASS_COLORS, text="Students")
    fig.update_layout(showlegend=False, height=390, margin=dict(l=20,r=20,t=25,b=20))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Individual Prediction":
    st.markdown('<div class="hero"><h1>Individual Prediction</h1><p>Enter one student’s academic and study-habit information.</p></div>', unsafe_allow_html=True)
    with st.form("single_prediction"):
        c1,c2 = st.columns(2)
        previous = c1.number_input("Previous CGPA", 0.0, 4.0, 3.00, 0.01)
        average = c2.number_input("Average Score", 0.0, 100.0, 70.0, 0.5)
        attendance = c1.number_input("Attendance Rate (%)", 0.0, 100.0, 85.0, 0.5)
        study = c2.number_input("Study Hours per Day", 0.0, 24.0, 3.0, 0.5)
        sleep = c1.number_input("Sleep Hours per Day", 0.0, 24.0, 7.0, 0.5)
        submitted = st.form_submit_button("Predict Student Performance", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Previous_CGPA": previous,
            "Average_Score": average,
            "Attendance_Pct": attendance,
            "Study_Hours_Per_Day": study,
            "Sleep_Hours": sleep,
        }])
        final, confidence, detail = predict_one(models, row, best_model)
        st.success(f"Final Prediction: {final}")
        a,b,c = st.columns(3)
        a.metric("Final Category", final)
        b.metric("Confidence", f"{confidence:.1%}")
        c.metric("Selected Model", best_model)
        display = detail.copy()
        display["Confidence"] = display["Confidence"].map(lambda x: f"{x:.1%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.markdown("### Recommended Actions")
        for tip in recommendations(row.iloc[0], final):
            st.write("• " + tip)

elif page == "Batch Prediction":
    st.markdown('<div class="hero"><h1>Batch Prediction</h1><p>Upload up to 10,000 students in CSV or Excel format.</p></div>', unsafe_allow_html=True)
    st.download_button("⬇️ Download Batch Template", make_template(), "student_batch_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    uploaded = st.file_uploader("Upload completed CSV or Excel file", type=["csv","xlsx"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            errors = validate_batch(batch)
            if errors:
                st.error("The uploaded file has validation errors.")
                for error in errors: st.write("• " + error)
            else:
                st.success(f"{len(batch):,} student records loaded successfully.")
                st.dataframe(batch.head(50), hide_index=True, use_container_width=True)
                result = predict_batch(models, batch, best_model)
                st.markdown("### Prediction Results")
                st.dataframe(result.head(200), hide_index=True, use_container_width=True)
                st.download_button("⬇️ Download Prediction Report", make_result_excel(result), "student_prediction_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        except Exception as exc:
            st.error("Unable to process the uploaded file.")
            st.code(str(exc))

elif page == "Model Evaluation":
    st.markdown('<div class="hero"><h1>Model Evaluation</h1><p>Compare KNN, SVM and ANN using the same held-out test set.</p></div>', unsafe_allow_html=True)
    show = evaluation.copy()
    for c in ["Accuracy","Precision","Recall","F1 Score"]: show[c] = show[c].map(lambda x: f"{x:.2%}")
    st.dataframe(show, hide_index=True, use_container_width=True)
    chart = evaluation.melt(id_vars="Model", value_vars=["Accuracy","Precision","Recall","F1 Score"], var_name="Metric", value_name="Score")
    fig = px.bar(chart, x="Model", y="Score", color="Metric", barmode="group", range_y=[0,1], title="Model Performance Comparison")
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion Matrices")
    tabs = st.tabs(list(models.keys()))
    for tab, name in zip(tabs, models.keys()):
        with tab:
            cm = matrices[name]
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=CLASS_ORDER, y=CLASS_ORDER, colorscale="Blues", showscale=True, text=cm, texttemplate="%{text}"))
            fig_cm.update_layout(title=f"{name} Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=500)
            st.plotly_chart(fig_cm, use_container_width=True)

elif page == "Feature Analysis":
    st.markdown('<div class="hero"><h1>Feature Analysis</h1><p>Review the relationship and information value of the five selected predictors.</p></div>', unsafe_allow_html=True)
    X = df[FEATURES]
    y = df["Performance_Category"].map({"At Risk":0,"Average":1,"Good":2,"Excellent":3})
    mi = mutual_info_classif(X, y, random_state=42)
    corr = X.apply(lambda s: s.corr(df["Final_CGPA"], method="spearman"))
    feat = pd.DataFrame({"Feature":[FEATURE_LABELS[f] for f in FEATURES], "Mutual Information":mi, "Spearman with Final CGPA":[corr[f] for f in FEATURES]}).sort_values("Mutual Information", ascending=False)
    st.dataframe(feat, hide_index=True, use_container_width=True)
    fig = px.bar(feat, x="Mutual Information", y="Feature", orientation="h", title="Feature Relevance Ranking")
    fig.update_layout(yaxis={"categoryorder":"total ascending"}, height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Previous CGPA and Average Score are the strongest predictors in this dataset. Attendance and Study Hours add useful behavioural/engagement information. Sleep Hours is retained as a relevant learning-habit feature for broader student context.")

elif page == "Dataset Explorer":
    st.markdown('<div class="hero"><h1>Dataset Explorer</h1><p>Browse the student dataset used by the deployed application.</p></div>', unsafe_allow_html=True)
    st.caption(f"Current source: {data_source} · {data_note}")
    st.dataframe(df, hide_index=True, use_container_width=True, height=560)
    st.download_button("⬇️ Download Current Dataset", df.to_csv(index=False).encode("utf-8"), "current_student_dataset.csv", "text/csv")

else:
    st.markdown('<div class="hero"><h1>About the System</h1><p>Student Performance Prediction System — deployment-proof Streamlit edition.</p></div>', unsafe_allow_html=True)
    st.markdown("""
### Project Purpose
This application predicts student academic performance into **Excellent, Good, Average, or At Risk** categories.

### Machine Learning Models
- K-Nearest Neighbours (KNN)
- Support Vector Machine (SVM)
- Artificial Neural Network (ANN / MLP)

### Selected Features
1. Previous CGPA
2. Average Score
3. Attendance Rate
4. Study Hours per Day
5. Sleep Hours per Day

### Deployment Design
This version does **not** require `src/`, `models/`, `results/`, or `dataset/` folders. If `Student_data.csv` is present beside `app.py`, it is used automatically. If it is missing, the application still starts using a deterministic built-in fallback dataset instead of crashing.
""")
