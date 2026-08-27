from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

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

FEATURE_RANGES = {
    "Previous_CGPA": (0.0, 4.0),
    "Average_Score": (0.0, 100.0),
    "Attendance_Pct": (0.0, 100.0),
    "Study_Hours_Per_Day": (0.0, 24.0),
    "Sleep_Hours": (0.0, 24.0),
}

CLASS_ORDER = ["At Risk", "Average", "Good", "Excellent"]
CGPA_RANGES = {
    "At Risk": "Below 2.50",
    "Average": "2.50 – 2.99",
    "Good": "3.00 – 3.49",
    "Excellent": "3.50 – 4.00",
}


def category_from_cgpa(final_cgpa: float) -> str:
    """Convert Final CGPA into one of the four project performance classes."""
    value = float(final_cgpa)
    if value >= 3.50:
        return "Excellent"
    if value >= 3.00:
        return "Good"
    if value >= 2.50:
        return "Average"
    return "At Risk"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Final_CGPA" not in df.columns:
        raise ValueError("Dataset must contain Final_CGPA.")
    df = df.copy()
    df["Performance_Category"] = df["Final_CGPA"].map(category_from_cgpa)
    return df


def load_metadata(path: Path | None = None) -> dict:
    path = path or (MODEL_DIR / "metadata.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_models(model_dir: Path = MODEL_DIR) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for name in ["KNN", "SVM", "ANN"]:
        path = model_dir / f"{name.lower()}_model.joblib"
        models[name] = joblib.load(path)
    return models


def validate_feature_frame(
    frame: pd.DataFrame,
    features: Iterable[str] = FEATURES,
) -> list[str]:
    """Validate batch input and return human-readable errors."""
    features = list(features)
    errors: list[str] = []

    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
        return errors

    if frame.empty:
        errors.append("The uploaded file contains no student records.")
        return errors

    if len(frame) > 10000:
        errors.append("A maximum of 10,000 student records can be processed at one time.")

    for feature in features:
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        invalid_numeric = numeric.isna()
        if invalid_numeric.any():
            rows = (invalid_numeric.to_numpy().nonzero()[0] + 2).tolist()
            errors.append(
                f"{feature} contains missing or non-numeric values at row(s): "
                + ", ".join(map(str, rows[:10]))
            )
            continue

        minimum, maximum = FEATURE_RANGES[feature]
        outside = ~numeric.between(minimum, maximum, inclusive="both")
        if outside.any():
            rows = (outside.to_numpy().nonzero()[0] + 2).tolist()
            errors.append(
                f"{feature} must be between {minimum:g} and {maximum:g}. "
                "Invalid row(s): " + ", ".join(map(str, rows[:10]))
            )

    return errors


def predict_frame(
    models: dict[str, dict],
    frame: pd.DataFrame,
    best_model: str,
    features: Iterable[str] = FEATURES,
) -> pd.DataFrame:
    """Run all models and return predictions/confidences plus final prediction."""
    features = list(features)
    result = frame.copy()
    X = result[features].apply(pd.to_numeric, errors="raise")

    for model_name, bundle in models.items():
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        raw_prediction = model.predict(X)
        class_order = bundle.get("class_order") if isinstance(bundle, dict) else None
        if class_order is not None:
            labels = [class_order[int(value)] for value in raw_prediction]
        else:
            labels = raw_prediction
        result[f"{model_name}_Prediction"] = labels

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            result[f"{model_name}_Confidence"] = probabilities.max(axis=1)
        else:
            result[f"{model_name}_Confidence"] = pd.NA

    best_model = str(best_model).upper()
    result["Final_Prediction"] = result[f"{best_model}_Prediction"]
    result["Final_Confidence"] = result[f"{best_model}_Confidence"]
    result["Best_Model"] = best_model
    return result


def recommendation_for_student(row: pd.Series, predicted_category: str) -> list[str]:
    """Create simple, transparent academic support suggestions."""
    suggestions: list[str] = []

    if float(row["Attendance_Pct"]) < 80:
        suggestions.append("Improve class attendance and follow up on missed lessons.")
    if float(row["Study_Hours_Per_Day"]) < 2:
        suggestions.append("Increase consistent daily study time using a weekly study plan.")
    if float(row["Sleep_Hours"]) < 6:
        suggestions.append("Aim for a more regular sleep schedule to support study consistency.")
    if float(row["Average_Score"]) < 65:
        suggestions.append("Review weaker subjects and seek targeted academic support.")
    if float(row["Previous_CGPA"]) < 2.75:
        suggestions.append("Set short-term CGPA improvement targets and monitor progress frequently.")

    if not suggestions:
        if predicted_category == "Excellent":
            suggestions.append("Maintain the current study, attendance and revision routine.")
        elif predicted_category == "Good":
            suggestions.append("Maintain current progress and focus on weaker assessment areas.")
        else:
            suggestions.append("Monitor academic progress regularly and seek guidance when needed.")

    return suggestions[:4]


def make_prediction_workbook(result: pd.DataFrame) -> bytes:
    """Create a professional Excel report for batch prediction downloads."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    main_columns = [
        column
        for column in [
            "Student_ID",
            "Student_Name",
            *FEATURES,
            "Final_Prediction",
            "Final_Confidence",
        ]
        if column in result.columns
    ]
    comparison_columns = [
        column
        for column in [
            "Student_ID",
            "Student_Name",
            "KNN_Prediction",
            "KNN_Confidence",
            "SVM_Prediction",
            "SVM_Confidence",
            "ANN_Prediction",
            "ANN_Confidence",
            "Best_Model",
        ]
        if column in result.columns
    ]

    friendly = {**FEATURE_LABELS,
        "Student_ID": "Student ID",
        "Student_Name": "Student Name",
        "Final_Prediction": "Final Prediction",
        "Final_Confidence": "Final Confidence",
        "KNN_Prediction": "KNN Prediction",
        "KNN_Confidence": "KNN Confidence",
        "SVM_Prediction": "SVM Prediction",
        "SVM_Confidence": "SVM Confidence",
        "ANN_Prediction": "ANN Prediction",
        "ANN_Confidence": "ANN Confidence",
        "Best_Model": "Best Model",
    }

    main_df = result[main_columns].rename(columns=friendly)
    compare_df = result[comparison_columns].rename(columns=friendly)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="Prediction Results", index=False)
        compare_df.to_excel(writer, sheet_name="Model Comparison", index=False)

        counts = (
            result["Final_Prediction"]
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
        )
        summary_df = pd.DataFrame({
            "Performance Category": CLASS_ORDER,
            "Students": [int(counts[c]) for c in CLASS_ORDER],
            "Percentage": [float(counts[c]) / len(result) if len(result) else 0 for c in CLASS_ORDER],
            "Recommended Action": [
                "Early intervention and close monitoring",
                "Academic guidance and targeted support",
                "Maintain progress and improve weak areas",
                "Maintain strong performance",
            ],
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        wb = writer.book
        header_fill = PatternFill("solid", fgColor="1E3A8A")
        final_fill = PatternFill("solid", fgColor="5B21B6")
        white_font = Font(color="FFFFFF", bold=True)
        border_side = Side(style="thin", color="D7DEE8")
        border = Border(left=border_side, right=border_side, top=border_side, bottom=border_side)
        class_fills = {
            "At Risk": PatternFill("solid", fgColor="FEE2E2"),
            "Average": PatternFill("solid", fgColor="FEF3C7"),
            "Good": PatternFill("solid", fgColor="DBEAFE"),
            "Excellent": PatternFill("solid", fgColor="DCFCE7"),
        }

        for sheet_name in ["Prediction Results", "Model Comparison", "Summary"]:
            ws = wb[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            ws.sheet_view.showGridLines = False

            for cell in ws[1]:
                cell.fill = final_fill if cell.value in {"Final Prediction", "Final Confidence"} else header_fill
                cell.font = white_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = border

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.border = border
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    if cell.value in class_fills:
                        cell.fill = class_fills[cell.value]

            for column_cells in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells[: min(len(column_cells), 100)])
                ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max(max_len + 2, 12), 28)

        # Apply percentage formatting to confidence and summary percentage columns.
        for ws_name in ["Prediction Results", "Model Comparison"]:
            ws = wb[ws_name]
            for cell in ws[1]:
                if "Confidence" in str(cell.value):
                    for row_idx in range(2, ws.max_row + 1):
                        ws.cell(row_idx, cell.column).number_format = "0.0%"

        summary_ws = wb["Summary"]
        for cell in summary_ws[1]:
            if cell.value == "Percentage":
                for row_idx in range(2, summary_ws.max_row + 1):
                    summary_ws.cell(row_idx, cell.column).number_format = "0.0%"

    output.seek(0)
    return output.getvalue()


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f172a 0%, #172554 100%);}
[data-testid="stSidebar"] * {color: white;}
.hero {
    border-radius: 24px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #6d28d9 100%);
    color: white; box-shadow: 0 16px 38px rgba(15,23,42,.18);
}
.hero h1 {color: white; margin: 0 0 .4rem 0; font-size: 2rem;}
.hero p {margin: 0; color: #dbeafe;}
.soft-card {
    background: rgba(255,255,255,.98); border: 1px solid #e2e8f0; border-radius: 18px;
    padding: 1rem 1.1rem; box-shadow: 0 10px 28px rgba(15,23,42,.05); margin-bottom: .8rem;
}
.feature-chip {display:inline-block; padding:.32rem .62rem; border-radius:999px; background:#eef2ff; color:#3730a3; margin:.18rem; font-size:.82rem; font-weight:700;}
.pred-excellent {background:#dcfce7;border:1px solid #86efac;padding:1rem;border-radius:16px;}
.pred-good {background:#dbeafe;border:1px solid #93c5fd;padding:1rem;border-radius:16px;}
.pred-average {background:#fef3c7;border:1px solid #fcd34d;padding:1rem;border-radius:16px;}
.pred-risk {background:#fee2e2;border:1px solid #fca5a5;padding:1rem;border-radius:16px;}
.small-note {color:#64748b;font-size:.85rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_metadata():
    return load_metadata()


@st.cache_data
def get_evaluation() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "evaluation.csv")


@st.cache_data
def get_feature_selection() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "feature_selection.csv")


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_prediction_banner(category: str, confidence: float, model_name: str) -> None:
    css_class = {
        "Excellent": "pred-excellent",
        "Good": "pred-good",
        "Average": "pred-average",
        "At Risk": "pred-risk",
    }[category]
    st.markdown(
        f"""
<div class="{css_class}">
    <div style="font-size:.82rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;">Final Prediction</div>
    <div style="font-size:1.8rem;font-weight:900;margin:.15rem 0;">{category}</div>
    <div><b>Confidence:</b> {confidence:.1%} &nbsp; | &nbsp; <b>Selected model:</b> {model_name}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


try:
    df = get_data()
    models = get_models()
    metadata = get_metadata()
    evaluation = get_evaluation()
    feature_selection = get_feature_selection()
except Exception as exc:
    st.error("The application could not load its data/model artifacts.")
    st.code(str(exc))
    st.info("Run `python train_model.py` once, then restart the Streamlit app.")
    st.stop()

best_model = str(metadata["best_model"]).upper()
best_row = evaluation.loc[evaluation["Model"].str.upper() == best_model].iloc[0]

with st.sidebar:
    st.markdown("## 🎓 Student AI")
    st.caption("Performance Prediction System v2.0")
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Prediction",
            "Model Evaluation",
            "Feature Analysis",
            "Dataset Explorer",
            "About",
        ],
    )
    st.divider()
    st.markdown(f"**Best model:** {best_model}")
    st.markdown(f"**Holdout accuracy:** {best_row['Accuracy']:.1%}")
    st.markdown(f"**Weighted F1:** {best_row['F1_Score']:.1%}")
    st.caption("For academic decision-support and demonstration purposes.")


if page == "Dashboard":
    hero(
        "Student Performance Prediction System",
        "A clean, reproducible machine-learning dashboard for early academic performance screening.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Selected Features", len(FEATURES))
    c3.metric("Models Compared", len(models))
    c4.metric("Best Model", best_model)

    st.markdown("### Five selected input features")
    st.markdown(
        "".join(f'<span class="feature-chip">{FEATURE_LABELS[f]}</span>' for f in FEATURES),
        unsafe_allow_html=True,
    )
    st.caption(
        "The previous Number of Subjects feature was removed because it showed almost no relationship with the target in this dataset. "
        "Sleep Hours was added as a more relevant student-habit variable."
    )

    left, right = st.columns([1, 1])
    with left:
        counts = df["Performance_Category"].value_counts().reindex(CLASS_ORDER, fill_value=0).reset_index()
        counts.columns = ["Category", "Students"]
        fig = px.bar(counts, x="Category", y="Students", text="Students", title="Student Performance Distribution")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=390, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        chart_df = evaluation.melt(
            id_vars="Model",
            value_vars=["Accuracy", "F1_Score", "Macro_F1", "CV_F1_Mean"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(chart_df, x="Model", y="Score", color="Metric", barmode="group", title="Model Performance")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(height=390, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Prediction categories are derived from Final CGPA: At Risk < 2.50, Average 2.50–2.99, Good 3.00–3.49, and Excellent 3.50–4.00."
    )


elif page == "Prediction":
    hero(
        "Prediction Centre",
        "Predict one student or upload up to 10,000 records for batch analysis.",
    )

    mode = st.radio("Prediction mode", ["Individual Prediction", "Batch Prediction"], horizontal=True)

    if mode == "Individual Prediction":
        st.markdown("### Student details")
        identity1, identity2 = st.columns(2)
        with identity1:
            student_id = st.text_input("Student ID (optional)", placeholder="e.g. ID05001")
        with identity2:
            student_name = st.text_input("Student Name (optional)", placeholder="e.g. Alex Tan")

        with st.form("individual_prediction_form"):
            row1a, row1b, row1c = st.columns(3)
            previous_cgpa = row1a.number_input("Previous CGPA", min_value=0.0, max_value=4.0, value=3.00, step=0.01)
            average_score = row1b.number_input("Average Score", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
            attendance = row1c.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.5)

            row2a, row2b = st.columns(2)
            study_hours = row2a.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
            sleep_hours = row2b.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

            submitted = st.form_submit_button("Predict Student Performance", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([{
                "Student_ID": student_id.strip() or "N/A",
                "Student_Name": student_name.strip() or "N/A",
                "Previous_CGPA": previous_cgpa,
                "Average_Score": average_score,
                "Attendance_Pct": attendance,
                "Study_Hours_Per_Day": study_hours,
                "Sleep_Hours": sleep_hours,
            }])
            prediction = predict_frame(models, input_df, best_model)
            final_category = prediction.loc[0, "Final_Prediction"]
            final_confidence = float(prediction.loc[0, "Final_Confidence"])

            show_prediction_banner(final_category, final_confidence, best_model)
            st.markdown("### Model comparison")
            comparison = pd.DataFrame({
                "Model": ["KNN", "SVM", "ANN"],
                "Prediction": [prediction.loc[0, f"{name}_Prediction"] for name in ["KNN", "SVM", "ANN"]],
                "Confidence": [float(prediction.loc[0, f"{name}_Confidence"]) for name in ["KNN", "SVM", "ANN"]],
            })
            st.dataframe(
                comparison.style.format({"Confidence": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Suggested academic actions")
            for suggestion in recommendation_for_student(input_df.iloc[0], final_category):
                st.write(f"• {suggestion}")
            st.caption("Recommendations are transparent rule-based suggestions and are not part of the ML model itself.")

    else:
        st.markdown("### Batch prediction")
        st.write(
            "Required columns: **Previous_CGPA, Average_Score, Attendance_Pct, Study_Hours_Per_Day, Sleep_Hours**. "
            "Student_ID and Student_Name are optional."
        )

        template_path = Path(__file__).resolve().parent / "templates" / "student_batch_template.xlsx"
        if template_path.exists():
            st.download_button(
                "Download Batch Excel Template",
                data=template_path.read_bytes(),
                file_name="student_batch_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                batch_df = read_uploaded_table(uploaded_file)
                errors = validate_feature_frame(batch_df)
                if errors:
                    st.error("Please fix the following issue(s) before prediction:")
                    for error in errors:
                        st.write(f"• {error}")
                else:
                    st.success(f"{len(batch_df):,} student records are valid and ready.")
                    st.dataframe(batch_df.head(50), hide_index=True, use_container_width=True)
                    if st.button("Predict All Students", type="primary", use_container_width=True):
                        st.session_state["batch_result"] = predict_frame(models, batch_df, best_model)
            except Exception as exc:
                st.error(f"Unable to read the uploaded file: {exc}")

        if "batch_result" in st.session_state:
            result = st.session_state["batch_result"]
            st.markdown("### Batch results")
            c1, c2, c3, c4, c5 = st.columns(5)
            counts = result["Final_Prediction"].value_counts().reindex(CLASS_ORDER, fill_value=0)
            c1.metric("Total", f"{len(result):,}")
            c2.metric("At Risk", int(counts["At Risk"]))
            c3.metric("Average", int(counts["Average"]))
            c4.metric("Good", int(counts["Good"]))
            c5.metric("Excellent", int(counts["Excellent"]))

            show_columns = [
                column for column in [
                    "Student_ID", "Student_Name", *FEATURES,
                    "Final_Prediction", "Final_Confidence", "Best_Model",
                ] if column in result.columns
            ]
            st.dataframe(
                result[show_columns].head(1000).style.format({"Final_Confidence": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
            if len(result) > 1000:
                st.caption("The on-screen table is limited to the first 1,000 rows. The Excel download contains all records.")

            output_bytes = make_prediction_workbook(result)
            st.download_button(
                "Download Complete Prediction Report",
                data=output_bytes,
                file_name="student_performance_prediction_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


elif page == "Model Evaluation":
    hero(
        "Model Evaluation",
        "Compare KNN, SVM and ANN using a stratified holdout test and five-fold cross-validation.",
    )

    show = evaluation.copy()
    percentage_cols = ["Accuracy", "Precision", "Recall", "F1_Score", "Macro_F1", "CV_F1_Mean", "CV_F1_Std"]
    st.dataframe(show.style.format({column: "{:.2%}" for column in percentage_cols}), hide_index=True, use_container_width=True)

    chart_df = evaluation.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1_Score", "Macro_F1", "CV_F1_Mean"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(chart_df, x="Model", y="Score", color="Metric", barmode="group", title="Evaluation Metric Comparison")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion matrices")
    tabs = st.tabs(["KNN", "SVM", "ANN"])
    for tab, model_name in zip(tabs, ["KNN", "SVM", "ANN"]):
        with tab:
            image_path = RESULTS_DIR / f"{model_name.lower()}_confusion_matrix.png"
            if image_path.exists():
                st.image(str(image_path), caption=f"{model_name} confusion matrix")

    st.info(
        f"The final app uses **{best_model}** because it achieved the highest weighted F1 score on the holdout evaluation. "
        "Weighted F1 is used as the selection metric because the four performance categories are not equally sized."
    )


elif page == "Feature Analysis":
    hero(
        "Feature Analysis",
        "Explain why these five inputs were selected and how they relate to student performance in this dataset.",
    )

    st.markdown("### Selected features")
    selected = feature_selection[feature_selection["Selected"]].copy()
    selected["Feature"] = selected["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    st.dataframe(
        selected[["Feature", "Mutual_Information", "Abs_Spearman"]].style.format({
            "Mutual_Information": "{:.3f}",
            "Abs_Spearman": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    ranking = feature_selection.copy()
    ranking["Display"] = ranking["Feature"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ")))
    fig = px.bar(
        ranking.sort_values("Mutual_Information"),
        x="Mutual_Information",
        y="Display",
        orientation="h",
        title="Candidate Feature Ranking by Mutual Information",
        text="Mutual_Information",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Why the feature set changed")
    st.write(
        "The old version used Number of Subjects, but the dataset analysis found that it carried almost no useful relationship with the target. "
        "The rebuilt version keeps the strongest academic indicators and replaces that weak input with Sleep Hours, a relevant student-habit variable."
    )
    st.caption(
        "Feature ranking shows association/predictive information in this dataset; it does not prove that a feature causes a student's academic outcome."
    )


elif page == "Dataset Explorer":
    hero(
        "Dataset Explorer",
        "Inspect the 5,000-record project dataset, filter students and review descriptive statistics.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    filter_category = st.multiselect("Performance Category", CLASS_ORDER, default=CLASS_ORDER)
    filtered = df[df["Performance_Category"].isin(filter_category)].copy()

    if "Major" in filtered.columns:
        majors = sorted(filtered["Major"].astype(str).unique().tolist())
        selected_majors = st.multiselect("Major", majors, default=majors)
        filtered = filtered[filtered["Major"].isin(selected_majors)]

    st.dataframe(filtered, hide_index=True, use_container_width=True, height=520)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records.")

    st.markdown("### Descriptive statistics for selected ML features")
    stats = df[FEATURES].describe().T.reset_index().rename(columns={"index": "Feature"})
    stats["Feature"] = stats["Feature"].map(FEATURE_LABELS)
    st.dataframe(stats, hide_index=True, use_container_width=True)


elif page == "About":
    hero(
        "About the Project",
        "Methodology, dataset source, prediction classes and reproducibility notes.",
    )

    st.markdown("### Project objective")
    st.write(
        "This system demonstrates supervised machine learning for student performance classification and early academic decision support. "
        "It compares KNN, SVM and ANN, then uses the best-performing model for the final prediction while still showing all three model outputs."
    )

    st.markdown("### Dataset source")
    st.write("University Student Performance & Habits Dataset by Robiul Hasan Jisan (Kaggle).")
    st.code("https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset")
    st.caption(
        "Average_Score is an engineered project field included in the supplied project CSV; it should not be described as an original Kaggle column."
    )

    st.markdown("### Target classes")
    target_table = pd.DataFrame({"Performance Category": CLASS_ORDER, "Final CGPA Range": [CGPA_RANGES[c] for c in CLASS_ORDER]})
    st.dataframe(target_table, hide_index=True, use_container_width=True)

    st.markdown("### Reproducibility")
    st.write(
        "The training script uses a fixed random state, an 80/20 stratified holdout split, five-fold stratified cross-validation, StandardScaler pipelines, "
        "saved evaluation outputs, and pinned package versions. Running `python train_model.py` regenerates all models and result files."
    )

    st.warning(
        "This project is an academic prototype. Predictions should support—not replace—teacher judgement, student consultation, and other contextual information."
    )
