"""Student Performance Prediction System — four-feature A++ edition."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openpyxl.styles import Alignment, Font, PatternFill

from student_model import (
    CLASS_COLORS,
    CLASS_ORDER,
    FEATURES,
    FEATURE_LABELS,
    feature_relevance,
    load_dataset,
    predict_batch,
    predict_model_details,
    student_recommendations,
    train_model_suite,
    validate_batch,
)


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {--navy:#0f172a;--blue:#2563eb;--violet:#7c3aed;--line:#dbe4f0;}
.stApp {background:linear-gradient(180deg,#f8fbff 0%,#f8fafc 48%,#f5f3ff 100%);}
.block-container {max-width:1220px;padding-top:1.25rem;padding-bottom:2.5rem;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#07111f 0%,#111827 55%,#172554 100%);}
[data-testid="stSidebar"] * {color:#f8fafc;}
.hero {padding:1.45rem 1.6rem;border-radius:24px;background:linear-gradient(135deg,#0f172a,#1e3a8a 58%,#6d28d9);color:white;box-shadow:0 18px 48px rgba(30,58,138,.18);margin-bottom:1rem;}
.hero h1 {margin:0;color:white;font-size:2.05rem;line-height:1.2;}
.hero p {margin:.55rem 0 0;color:#dbeafe;font-size:1rem;}
.section-card {background:rgba(255,255,255,.94);border:1px solid #dbe4f0;border-radius:18px;padding:1rem 1.1rem;box-shadow:0 10px 28px rgba(15,23,42,.05);margin:.5rem 0 1rem;}
.feature-card {min-height:92px;border:1px solid #dbeafe;border-radius:16px;padding:.8rem;background:linear-gradient(145deg,#ffffff,#eef2ff);display:flex;align-items:center;justify-content:center;text-align:center;font-weight:800;color:#312e81;box-shadow:0 6px 18px rgba(49,46,129,.06);}
.method-step {border-left:4px solid #4f46e5;background:#f8fafc;border-radius:0 14px 14px 0;padding:.75rem .9rem;margin:.55rem 0;}
.pill {display:inline-block;padding:.28rem .6rem;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:.78rem;font-weight:750;margin:.12rem;}
[data-testid="stMetric"] {background:white;border:1px solid #dbe4f0;border-radius:16px;padding:.72rem .85rem;box-shadow:0 7px 20px rgba(15,23,42,.04);}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button {border-radius:12px!important;font-weight:800!important;}
</style>
""",
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_dataset() -> tuple[pd.DataFrame, str, str]:
    result = load_dataset(ROOT)
    return result.frame, result.source, result.note


@st.cache_resource(show_spinner="Training KNN, SVM and ANN models...")
def cached_training(dataset_hash: str, csv_payload: bytes):
    del dataset_hash  # It remains part of the cache key.
    frame = pd.read_csv(BytesIO(csv_payload))
    return train_model_suite(frame)


@st.cache_data(show_spinner=False)
def cached_relevance(csv_payload: bytes) -> pd.DataFrame:
    return feature_relevance(pd.read_csv(BytesIO(csv_payload)))


@st.cache_data(show_spinner=False)
def make_batch_template() -> bytes:
    sample = pd.DataFrame(
        {
            "Student_ID": ["ID05001", "ID05002", "ID05003", "ID05004", "ID05005"],
            "Student_Name": ["Aiman Hakim", "Nur Aisyah", "Lim Wei Jian", "Siti Hajar", "Arjun Kumar"],
            "Number_of_Subjects": [5, 6, 5, 4, 6],
            "Average_Score": [74.5, 62.0, 84.0, 55.5, 70.0],
            "Attendance_Pct": [88.0, 76.0, 94.0, 68.0, 85.0],
            "Study_Hours_Per_Day": [3.5, 2.0, 4.5, 1.5, 3.0],
            "Previous_CGPA": [3.20, 2.70, 3.65, 2.35, 3.05],
        }
    )
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sample.to_excel(writer, index=False, sheet_name="Batch Input")
        worksheet = writer.sheets["Batch Input"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill("solid", fgColor="1E3A8A")
            cell.alignment = Alignment(horizontal="center")
        widths = {"A": 16, "B": 24, "C": 20, "D": 17, "E": 20, "F": 23, "G": 17}
        for column, width in widths.items():
            worksheet.column_dimensions[column].width = width
    return output.getvalue()


def make_result_workbook(result: pd.DataFrame) -> bytes:
    output = BytesIO()
    summary = (
        result["Final_Prediction"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename_axis("Category")
        .reset_index(name="Students")
    )
    summary["Percentage"] = summary["Students"] / max(len(result), 1)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Prediction Results")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(color="FFFFFF", bold=True)
                cell.fill = PatternFill("solid", fgColor="1E3A8A")
                cell.alignment = Alignment(horizontal="center")
            for cells in worksheet.columns:
                width = min(max(max(len(str(cell.value or "")) for cell in cells) + 2, 12), 30)
                worksheet.column_dimensions[cells[0].column_letter].width = width
        for cell in writer.sheets["Summary"]["C"][1:]:
            cell.number_format = "0.0%"
    return output.getvalue()


try:
    dataset, data_source, data_note = cached_dataset()
except Exception as exc:
    st.error("The application could not initialize its dataset safely.")
    st.exception(exc)
    st.stop()

payload = dataset.to_csv(index=False).encode("utf-8")
signature = sha256(payload).hexdigest()
try:
    training = cached_training(signature, payload)
except Exception as exc:
    st.error("Model initialization failed. Please check the deployment logs and dependency installation.")
    st.exception(exc)
    st.stop()

models = training.models
evaluation = training.evaluation
matrices = training.confusion_matrices
best_model = training.best_model


st.sidebar.markdown("# 🎓 Student AI")
st.sidebar.caption("Performance Prediction System")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Individual Prediction",
        "Batch Prediction",
        "Model Evaluation",
        "Feature Analysis",
        "Dataset Explorer",
        "Methodology & About",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {data_source}")
st.sidebar.caption(f"Valid students: {len(dataset):,}")
st.sidebar.caption(f"Selected model: {best_model}")
st.sidebar.caption("Four-Feature A++ · v7.0")

if data_source != "Repository CSV":
    st.warning(
        "Student_data.csv was missing or invalid. The app remains operational using a deterministic "
        "fallback dataset; place a valid Student_data.csv beside app.py to restore project data."
    )
if training.errors:
    st.warning("One or more optional models could not train: " + "; ".join(training.errors))


if page == "Overview":
    hero(
        "Student Performance Prediction System",
        "Professional academic screening with KNN, SVM and ANN — rebuilt from the four-feature final concept.",
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Student Records", f"{len(dataset):,}")
    metric_columns[1].metric("Selected Features", str(len(FEATURES)))
    metric_columns[2].metric("Models Trained", str(len(models)))
    metric_columns[3].metric("Best Hold-out Model", best_model)

    st.markdown("### Four Selected Features")
    feature_columns = st.columns(len(FEATURES))
    for column, feature in zip(feature_columns, FEATURES):
        column.markdown(
            f'<div class="feature-card">{FEATURE_LABELS[feature]}</div>', unsafe_allow_html=True
        )

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("### Performance Distribution")
        distribution = (
            dataset["Performance_Category"]
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
            .rename_axis("Category")
            .reset_index(name="Students")
        )
        figure = px.bar(
            distribution,
            x="Category",
            y="Students",
            color="Category",
            color_discrete_map=CLASS_COLORS,
            text="Students",
            category_orders={"Category": CLASS_ORDER},
        )
        figure.update_layout(showlegend=False, height=400, margin=dict(l=15, r=15, t=20, b=15))
        st.plotly_chart(figure, width="stretch")
    with right:
        st.markdown("### How to Use the System")
        st.markdown(
            """
<div class="method-step"><b>1 · Individual</b><br>Enter one student's four evidence-backed academic indicators.</div>
<div class="method-step"><b>2 · Batch</b><br>Upload up to 10,000 students using the downloadable template.</div>
<div class="method-step"><b>3 · Evaluate</b><br>Compare the same held-out test metrics for KNN, SVM and ANN.</div>
<div class="method-step"><b>4 · Act</b><br>Use predictions as an early-support signal, not as a final academic decision.</div>
""",
            unsafe_allow_html=True,
        )
        st.info("Number of Subjects remains useful profile information but is excluded from prediction because its correlation is approximately zero.")

elif page == "Individual Prediction":
    hero("Individual Prediction", "Enter one student profile and compare all three model decisions.")
    with st.form("individual_prediction"):
        first, second = st.columns(2)
        number_of_subjects = first.number_input("Number of Subjects (profile only)", 1, 12, 5, 1)
        average_score = second.number_input("Average Score", 0.0, 100.0, 70.0, 0.5)
        attendance = first.number_input("Attendance Rate (%)", 0.0, 100.0, 85.0, 0.5)
        study = second.number_input("Study Hours per Day", 0.0, 24.0, 3.0, 0.5)
        previous = first.number_input("Previous CGPA", 0.0, 4.0, 3.00, 0.01)
        submitted = st.form_submit_button("Predict Student Performance", width="stretch")

    if submitted:
        input_row = pd.DataFrame(
            [
                {
                    "Number_of_Subjects": number_of_subjects,
                    "Average_Score": average_score,
                    "Previous_CGPA": previous,
                    "Attendance_Pct": attendance,
                    "Study_Hours_Per_Day": study,
                }
            ]
        )
        final_prediction, confidence, details = predict_model_details(models, input_row, best_model)
        st.success(f"Final prediction: {final_prediction}")
        metrics = st.columns(3)
        metrics[0].metric("Performance Category", final_prediction)
        metrics[1].metric("Model Confidence", f"{confidence:.1%}")
        metrics[2].metric("Selected Model", best_model)

        display = details.copy()
        display["Confidence"] = display["Confidence"].map(lambda value: f"{value:.1%}")
        st.markdown("### Transparent Model Comparison")
        st.dataframe(display, hide_index=True, width="stretch")
        agreement = details["Prediction"].nunique() == 1
        if agreement:
            st.info("All trained models agree on this prediction.")
        else:
            st.info(f"Models disagree; the final result follows {best_model}, selected by Macro F1 then accuracy.")

        st.markdown("### Recommended Support Actions")
        for recommendation in student_recommendations(input_row.iloc[0], final_prediction):
            st.write("• " + recommendation)

elif page == "Batch Prediction":
    hero("Batch Prediction", "Validate and predict up to 10,000 students from CSV or Excel.")
    st.download_button(
        "⬇️ Download Validated Batch Template",
        make_batch_template(),
        "student_batch_prediction_template.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    uploaded = st.file_uploader("Upload completed CSV or Excel file", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            raw_batch = (
                pd.read_csv(uploaded)
                if uploaded.name.lower().endswith(".csv")
                else pd.read_excel(uploaded)
            )
            cleaned_batch, validation_errors = validate_batch(raw_batch)
            if validation_errors:
                st.error("The uploaded file did not pass validation.")
                for error in validation_errors:
                    st.write("• " + error)
            else:
                st.success(f"{len(cleaned_batch):,} student records passed validation.")
                result = predict_batch(models, cleaned_batch, best_model)
                summary = (
                    result["Final_Prediction"]
                    .value_counts()
                    .reindex(CLASS_ORDER, fill_value=0)
                    .rename_axis("Category")
                    .reset_index(name="Students")
                )
                summary_figure = px.bar(
                    summary,
                    x="Category",
                    y="Students",
                    color="Category",
                    color_discrete_map=CLASS_COLORS,
                    text="Students",
                    category_orders={"Category": CLASS_ORDER},
                )
                summary_figure.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(summary_figure, width="stretch")
                st.markdown("### Prediction Results")
                st.caption(f"Showing {min(len(result), 500):,} of {len(result):,} predicted records.")
                st.dataframe(result.head(500), hide_index=True, width="stretch", height=520)
                st.download_button(
                    "⬇️ Download Complete Prediction Report",
                    make_result_workbook(result),
                    "student_prediction_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
        except Exception as exc:
            st.error("The uploaded file could not be processed safely.")
            st.exception(exc)

elif page == "Model Evaluation":
    hero("Model Evaluation", "A fair comparison using one reproducible 80/20 stratified hold-out split.")
    display = evaluation.copy()
    metric_names = ["Accuracy", "Precision", "Recall", "Weighted F1", "Macro F1"]
    for metric in metric_names:
        display[metric] = display[metric].map(lambda value: f"{value:.2%}")
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("The selected model is ranked by Macro F1, then Accuracy and Weighted F1.")

    melted = evaluation.melt(
        id_vars="Model", value_vars=metric_names, var_name="Metric", value_name="Score"
    )
    comparison = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 1],
        title="Hold-out Performance Comparison",
    )
    comparison.update_layout(height=450)
    st.plotly_chart(comparison, width="stretch")

    st.markdown("### Confusion Matrices")
    tabs = st.tabs(list(models))
    for tab, model_name in zip(tabs, models):
        with tab:
            matrix = matrices[model_name]
            matrix_figure = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=CLASS_ORDER,
                    y=CLASS_ORDER,
                    colorscale="Blues",
                    text=matrix,
                    texttemplate="%{text}",
                    hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Students=%{z}<extra></extra>",
                )
            )
            matrix_figure.update_layout(
                title=f"{model_name} Confusion Matrix",
                xaxis_title="Predicted Category",
                yaxis_title="Actual Category",
                height=500,
            )
            st.plotly_chart(matrix_figure, width="stretch")

elif page == "Feature Analysis":
    hero("Feature Analysis", "Correlation and ablation evidence for the final four-feature model design.")
    relevance = cached_relevance(payload)
    st.dataframe(relevance.drop(columns="Feature Key"), hide_index=True, width="stretch")
    relevance_figure = px.bar(
        relevance,
        x="Mutual Information",
        y="Feature",
        orientation="h",
        title="Feature Relevance Ranking",
        color="Spearman Correlation",
        color_continuous_scale="Blues",
    )
    relevance_figure.update_layout(yaxis={"categoryorder": "total ascending"}, height=440)
    st.plotly_chart(relevance_figure, width="stretch")
    st.markdown(
        """
<div class="section-card">
<b>Selection rationale</b><br>
Previous CGPA and Average Score provide the strongest academic signal. Attendance and Study Hours add
engagement and learning-effort information and improve five-fold cross-validation. Number of Subjects is kept
as profile information but excluded from the model because its correlation is approximately zero. Sleep Hours,
Social Hours and Age are also excluded because their relationships are negligible. Gender and Major are excluded
to avoid unnecessary demographic influence.
</div>
""",
        unsafe_allow_html=True,
    )

elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Inspect the exact validated records used by the deployed application.")
    st.caption(f"Source: {data_source} · {data_note}")
    selected_categories = st.multiselect(
        "Filter performance categories", CLASS_ORDER, default=CLASS_ORDER
    )
    filtered = dataset[dataset["Performance_Category"].isin(selected_categories)]
    st.metric("Visible Records", f"{len(filtered):,}")
    st.dataframe(filtered, hide_index=True, width="stretch", height=580)
    st.download_button(
        "⬇️ Download Filtered Dataset",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_student_dataset.csv",
        "text/csv",
    )

else:
    hero("Methodology & About", "A reproducible machine-learning workflow for responsible early academic support.")
    st.markdown(
        """
### Project Purpose
The system classifies students into **Excellent, Good, Average, or At Risk** to support early intervention.
It is a decision-support tool and must not be used as the sole basis for grades, disciplinary action, or access to education.

### Reproducible Workflow
1. Validate the 5,000-record dataset and numeric ranges.
2. Convert Final CGPA into four documented performance categories.
3. Use an 80/20 stratified split with random state 42.
4. Standardise all four evidence-backed predictors inside each model pipeline.
5. Train KNN, calibrated SVM, and ANN on the same training data.
6. Select the final model by Macro F1, then accuracy and Weighted F1.
7. Present all model predictions, confidence and input-driven support recommendations.

### Performance Categories
<span class="pill">Excellent: CGPA ≥ 3.50</span>
<span class="pill">Good: 3.00–3.49</span>
<span class="pill">Average: 2.50–2.99</span>
<span class="pill">At Risk: below 2.50</span>

### Deployment Protection
The application reads `Student_data.csv` from the repository root. If that file is missing or invalid, a
deterministic built-in fallback keeps the interface operational. Models are trained automatically and cached;
no `models/`, `src/`, `results/`, or `dataset/` folder is required.

### Dataset Source
Jisan, R. H. (n.d.). *University student performance & habits dataset* [Data set]. Kaggle.  
https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset

The project starts from the 5,000-row Kaggle dataset and uses the supplied processed project CSV. `Average_Score`
and `Number_of_Subjects` are project-engineered extensions and are not described as original Kaggle columns.
Only `Average_Score` is used by the model; `Number_of_Subjects` is profile/report information only.

### Academic Integrity and Limitations
Predictions reflect patterns in the supplied dataset, not guaranteed future outcomes. Human review, contextual
information and regular model monitoring remain necessary. Demographic variables are shown only in the
Dataset Explorer and are excluded from prediction.
""",
        unsafe_allow_html=True,
    )
