"""Student Performance Prediction System — final four-feature Streamlit application."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from student_model import (
    CLASS_COLORS,
    CLASS_ORDER,
    FEATURES,
    FEATURE_LABELS,
    load_dataset,
    predict_batch,
    predict_model_details,
    student_recommendation,
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
:root {--navy:#0f172a;--blue:#2563eb;--violet:#7c3aed;--muted:#64748b;}
.stApp {background:radial-gradient(circle at 8% 8%,rgba(59,130,246,.17),transparent 32%),radial-gradient(circle at 94% 12%,rgba(124,58,237,.13),transparent 29%),linear-gradient(180deg,#f8fbff 0%,#eef4ff 48%,#faf8ff 100%);}
.block-container {max-width:1180px;padding-top:1.1rem;padding-bottom:2.7rem;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#081225 0%,#111827 52%,#172554 100%);border-right:1px solid rgba(255,255,255,.08);}
[data-testid="stSidebar"] * {color:white;}
[data-testid="stSidebar"] div[role="radiogroup"] label {border-radius:14px;padding:.68rem .78rem;margin-bottom:.25rem;border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.035);transition:.16s ease;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:rgba(255,255,255,.1);transform:translateX(2px);}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background:linear-gradient(90deg,#2563eb,#7c3aed);box-shadow:0 10px 28px rgba(37,99,235,.24);}
.hero {padding:1.45rem 1.55rem;border-radius:24px;color:white;background:radial-gradient(circle at 92% 10%,rgba(99,102,241,.36),transparent 27%),linear-gradient(135deg,#0b1220 0%,#172554 60%,#312e81 100%);box-shadow:0 22px 55px rgba(15,23,42,.20);margin-bottom:1.1rem;}
.hero .eyebrow {font-size:.77rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#bfdbfe;}.hero h1{margin:.18rem 0 .34rem;font-size:2rem;letter-spacing:-.04em}.hero p{margin:0;color:#dbeafe;max-width:900px;font-size:.95rem}
.feature-card{background:white;border:1px solid #e2e8f0;border-radius:18px;padding:1rem;min-height:160px;box-shadow:0 10px 28px rgba(15,23,42,.055)}.feature-number{width:34px;height:34px;display:flex;align-items:center;justify-content:center;border-radius:10px;color:white;font-weight:800;background:linear-gradient(135deg,#2563eb,#7c3aed);margin-bottom:.65rem}.feature-title{font-size:1.01rem;font-weight:800;color:#172033;margin-bottom:.30rem}.feature-desc{font-size:.85rem;color:#64748b;line-height:1.5}
.result-card{border-radius:22px;padding:1.2rem 1.3rem;background:linear-gradient(135deg,#0f172a,#1e3a8a);color:white;box-shadow:0 16px 40px rgba(30,58,138,.20)}.result-card .big{font-size:2rem;font-weight:900;letter-spacing:-.03em}.result-card .small{color:#bfdbfe;font-size:.87rem}
.cgpa-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;margin:.75rem 0 1.05rem}.cgpa-box{padding:.72rem .80rem;border-radius:14px;border:1px solid #e2e8f0;background:rgba(255,255,255,.86)}.cgpa-box b{display:block;color:#172033;margin-bottom:.15rem}.cgpa-box span{font-size:.82rem;color:#64748b}
.sidebar-foot{margin-top:2.1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,.12);color:#b9c6df;font-size:.72rem;line-height:1.6}
div[data-testid="stMetric"]{background:rgba(255,255,255,.88);border:1px solid rgba(148,163,184,.28);padding:.72rem .82rem;border-radius:16px;box-shadow:0 8px 22px rgba(15,23,42,.05)}.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:750}[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid #e2e8f0}
@media(max-width:850px){.cgpa-grid{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:1.55rem}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Training KNN, SVM and ANN...")
def get_suite():
    data, _ = load_dataset()
    return train_model_suite(data)


def hero(title: str, subtitle: str, eyebrow: str = "RIS Group 5 • BMCS2003 Artificial Intelligence") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def cgpa_guide() -> None:
    st.markdown(
        """
<div class="cgpa-grid">
<div class="cgpa-box"><b>🌟 Excellent</b><span>Final CGPA 3.50 – 4.00</span></div>
<div class="cgpa-box"><b>✅ Good</b><span>Final CGPA 3.00 – 3.49</span></div>
<div class="cgpa-box"><b>📘 Average</b><span>Final CGPA 2.50 – 2.99</span></div>
<div class="cgpa-box"><b>⚠️ At Risk</b><span>Final CGPA below 2.50</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def autosize(ws) -> None:
    for cells in ws.columns:
        letter = get_column_letter(cells[0].column)
        ws.column_dimensions[letter].width = min(max(len(str(c.value or "")) for c in cells) + 3, 42)


def excel_bytes(dataframe: pd.DataFrame, title: str) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Prediction Results"
    dark = PatternFill("solid", fgColor="172554")
    alt = PatternFill("solid", fgColor="EFF6FF")
    thin = Side(style="thin", color="D9E2F1")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, len(dataframe.columns)))
    cell = ws.cell(1, 1, title)
    cell.fill = dark
    cell.font = Font(size=15, bold=True, color="FFFFFF")
    cell.alignment = Alignment(horizontal="center")

    for j, column in enumerate(dataframe.columns, 1):
        h = ws.cell(3, j, column)
        h.fill = dark
        h.font = Font(color="FFFFFF", bold=True)
        h.alignment = Alignment(horizontal="center")

    for i, row in enumerate(dataframe.itertuples(index=False), 4):
        for j, value in enumerate(row, 1):
            c = ws.cell(i, j, value)
            c.border = Border(bottom=thin)
            if i % 2 == 0:
                c.fill = alt
            if isinstance(value, float) and "Confidence" in str(dataframe.columns[j - 1]):
                c.number_format = "0.0%"
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = ws.dimensions
    autosize(ws)
    output = BytesIO()
    wb.save(output)
    return output.getvalue()


def batch_template() -> bytes:
    sample = pd.DataFrame(
        [
            {"Student_ID":"S001","Student_Name":"Ali Tan","Average_Score":82.0,"Attendance_Pct":91.0,"Study_Hours_Per_Day":4.5,"Previous_CGPA":3.45},
            {"Student_ID":"S002","Student_Name":"Mei Ling","Average_Score":67.0,"Attendance_Pct":78.0,"Study_Hours_Per_Day":2.5,"Previous_CGPA":2.85},
        ]
    )
    return excel_bytes(sample, "Student Batch Prediction Template — 4 ML Features")


def confusion_figure(matrix, classes, model_name):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=classes,
            y=classes,
            colorscale="Blues",
            showscale=True,
            text=matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=f"{model_name} Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


df, data_source = get_dataset()
suite = get_suite()
evaluation = suite.evaluation
best_model = suite.best_model
best_accuracy = float(evaluation.iloc[0]["Accuracy"])

st.sidebar.markdown("## 🎓 Student AI")
st.sidebar.caption("Final 4-Feature Edition")
choice = st.sidebar.radio(
    "Navigation",
    ["🏠 Home","🎯 Individual Prediction","📂 Batch Prediction","📊 Model Evaluation","⭐ Feature Analysis","📈 Dataset Explorer","ℹ️ About"],
    label_visibility="collapsed",
)
page = choice.split(" ", 1)[1]
st.sidebar.markdown(
    f'<div class="sidebar-foot"><b>ML Inputs: 4</b><br>Average Score<br>Attendance Percentage<br>Study Hours per Day<br>Previous CGPA<br><br><b>Best Model:</b> {best_model}<br><b>Accuracy:</b> {best_accuracy:.1%}<br><br>RIS Group 5 • 2026</div>',
    unsafe_allow_html=True,
)

if page == "Home":
    hero("Student Performance Prediction System", "A supervised machine learning prototype that uses exactly four academically relevant features to classify performance as Excellent, Good, Average or At Risk.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Students",f"{len(df):,}");c2.metric("ML Features","4");c3.metric("Best Model",best_model);c4.metric("Best Accuracy",f"{best_accuracy:.1%}")
    if data_source != "repository dataset":
        st.warning("The repository CSV could not be used, so a deterministic fallback dataset is active for this session.")

    st.markdown("### Four Final Prediction Features")
    cards=[
        ("Average Score","Recent overall assessment performance and one of the strongest signals in the dataset."),
        ("Attendance Percentage","Measures participation and consistency in attending scheduled learning activities."),
        ("Study Hours per Day","Represents daily academic effort, preparation and independent study."),
        ("Previous CGPA","Historical academic achievement and the strongest selected predictor of Final CGPA."),
    ]
    cols=st.columns(4)
    for i,(title,desc) in enumerate(cards,1):
        cols[i-1].markdown(f'<div class="feature-card"><div class="feature-number">{i}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>',unsafe_allow_html=True)

    st.markdown("### Model Performance Overview")
    chart=evaluation.melt(id_vars="Model",value_vars=["Accuracy","Precision","Recall","F1 Score"],var_name="Metric",value_name="Score")
    fig=px.bar(chart,x="Model",y="Score",color="Metric",barmode="group",text_auto=".1%",range_y=[0,1])
    fig.update_yaxes(tickformat=".0%");fig.update_layout(height=420,margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    cgpa_guide()
    st.info("The four selected inputs are the strongest meaningful academic features in this project dataset. Number of Subjects is not used because its relationship with Final CGPA is approximately zero.")

elif page == "Individual Prediction":
    hero("Individual Student Prediction","Enter exactly four academic indicators. The system compares KNN, SVM and ANN and uses the best-performing model for the final decision-support result.","Prediction Module • Single Student")
    cgpa_guide()
    with st.form("individual_form"):
        st.markdown("### Student Information")
        a,b=st.columns(2)
        student_id=a.text_input("Student ID (optional)",placeholder="e.g. 23WMR12345")
        student_name=b.text_input("Student Name (optional)",placeholder="e.g. Alex Tan")
        st.markdown("### Four Prediction Features")
        left,right=st.columns(2)
        average=left.number_input("1. Average Score (%)",0.0,100.0,75.0,1.0)
        attendance=right.number_input("2. Attendance Percentage (%)",0.0,100.0,85.0,0.5)
        study=left.number_input("3. Study Hours per Day",0.0,24.0,3.0,0.5)
        previous=right.number_input("4. Previous CGPA",0.0,4.0,3.0,0.01,format="%.2f")
        submitted=st.form_submit_button("🚀 Predict Student Performance",type="primary",use_container_width=True)

    if submitted:
        X=pd.DataFrame([{"Average_Score":average,"Attendance_Pct":attendance,"Study_Hours_Per_Day":study,"Previous_CGPA":previous}])[FEATURES]
        comparison=predict_model_details(X,suite)
        best_row=comparison[comparison["Model"]==best_model].iloc[0]
        final=str(best_row["Prediction"]);confidence=float(best_row["Confidence"])
        emoji={"Excellent":"🌟","Good":"✅","Average":"📘","At Risk":"⚠️"}.get(final,"🎓")
        st.markdown("### Prediction Result")
        st.markdown(f'<div class="result-card"><div class="small">Final prediction using {best_model}</div><div class="big">{emoji} {final}</div><div class="small">Confidence: {confidence:.1%}</div></div>',unsafe_allow_html=True)
        st.write(student_recommendation(final))
        show=comparison.copy();show["Confidence"]=show["Confidence"].map(lambda x:f"{x:.1%}")
        st.markdown("#### Three-Model Comparison");st.dataframe(show,hide_index=True,use_container_width=True)
        export=pd.DataFrame([{"Student_ID":student_id or "N/A","Student_Name":student_name or "N/A","Average_Score":average,"Attendance_Pct":attendance,"Study_Hours_Per_Day":study,"Previous_CGPA":previous,"Final_Prediction":final,"Final_Confidence":confidence,"Best_Model":best_model,"Recommendation":student_recommendation(final)}])
        st.download_button("⬇️ Download Prediction Report",excel_bytes(export,"Individual Student Prediction Report"),"individual_student_prediction.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)

elif page == "Batch Prediction":
    hero("Batch Student Prediction","Upload multiple records, validate the four required features, identify at-risk students and export a complete Excel report.","Prediction Module • Multiple Students")
    c1,c2=st.columns(2)
    c1.download_button("⬇️ Download 4-Feature Excel Template",batch_template(),"student_batch_prediction_template.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    c2.info("Required: Average_Score, Attendance_Pct, Study_Hours_Per_Day, Previous_CGPA")
    uploaded=st.file_uploader("Upload .xlsx or .csv file",type=["xlsx","csv"])
    if uploaded is not None:
        try:
            batch_df=pd.read_csv(uploaded) if Path(uploaded.name).suffix.lower()==".csv" else pd.read_excel(uploaded)
        except Exception as exc:
            st.error("Unable to read the uploaded file.");st.exception(exc);st.stop()
        st.markdown("### Uploaded Data Preview");st.dataframe(batch_df.head(20),hide_index=True,use_container_width=True);st.caption(f"{len(batch_df):,} record(s) loaded.")
        errors=validate_batch(batch_df)
        if errors:
            for error in errors: st.error(error)
        elif len(batch_df)==0:
            st.warning("The uploaded file contains no student records.")
        elif st.button("🚀 Run Batch Prediction",type="primary",use_container_width=True):
            result=predict_batch(batch_df,suite);result["Recommendation"]=result["Final_Prediction"].map(student_recommendation);st.session_state["batch_result"]=result

    if "batch_result" in st.session_state:
        result=st.session_state["batch_result"]
        st.markdown("### Batch Prediction Dashboard")
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Total Students",f"{len(result):,}");m2.metric("At Risk",f"{int((result['Final_Prediction']=='At Risk').sum()):,}");m3.metric("Excellent",f"{int((result['Final_Prediction']=='Excellent').sum()):,}");m4.metric("Avg. Confidence",f"{float(result['Final_Confidence'].mean()):.1%}")
        counts=result["Final_Prediction"].value_counts().reindex(CLASS_ORDER,fill_value=0).rename_axis("Category").reset_index(name="Students")
        fig=px.bar(counts,x="Category",y="Students",text="Students",color="Category",color_discrete_map=CLASS_COLORS);fig.update_layout(height=350,showlegend=False,margin=dict(l=10,r=10,t=20,b=10));st.plotly_chart(fig,use_container_width=True)
        f1,f2=st.columns(2);category=f1.selectbox("Performance Category",["All"]+CLASS_ORDER);search=f2.text_input("Search Student ID / Name")
        filtered=result.copy()
        if category!="All":filtered=filtered[filtered["Final_Prediction"]==category]
        if search.strip():
            cols=[c for c in ["Student_ID","Student_Name"] if c in filtered.columns]
            if cols:
                mask=pd.Series(False,index=filtered.index)
                for c in cols: mask|=filtered[c].astype(str).str.contains(search.strip(),case=False,na=False)
                filtered=filtered[mask]
        display=filtered.copy()
        for col in [c for c in display.columns if "Confidence" in c]:display[col]=display[col].map(lambda x:f"{float(x):.1%}")
        st.caption(f"Showing {len(filtered):,} of {len(result):,} predicted records.");st.dataframe(display,hide_index=True,use_container_width=True,height=540)
        d1,d2=st.columns(2);d1.download_button("⬇️ Download Complete Predicted Excel",excel_bytes(result,"Batch Student Prediction Report"),"student_batch_predictions.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
        if d2.button("🗑️ Clear Batch Result",use_container_width=True):del st.session_state["batch_result"];st.rerun()

elif page == "Model Evaluation":
    hero("Model Evaluation Dashboard","Compare KNN, SVM and ANN using Accuracy, Precision, Recall and F1 Score, then inspect each confusion matrix.","Machine Learning Module • Evaluation")
    st.success(f"🏆 {best_model} is the best-performing model with {best_accuracy:.2%} accuracy on the fixed 20% hold-out test set.")
    chart=evaluation.melt(id_vars="Model",value_vars=["Accuracy","Precision","Recall","F1 Score"],var_name="Metric",value_name="Score")
    fig=px.bar(chart,x="Model",y="Score",color="Metric",barmode="group",text_auto=".1%");fig.update_yaxes(range=[0,1],tickformat=".0%");fig.update_layout(height=420,margin=dict(l=10,r=10,t=20,b=10));st.plotly_chart(fig,use_container_width=True)
    formatted=evaluation.copy()
    for col in ["Accuracy","Precision","Recall","F1 Score"]:formatted[col]=formatted[col].map(lambda x:f"{x:.2%}")
    st.dataframe(formatted,hide_index=True,use_container_width=True)
    tabs=st.tabs(["KNN","SVM","ANN"])
    for tab,name in zip(tabs,["KNN","SVM","ANN"]):
        with tab:
            st.plotly_chart(confusion_figure(suite.confusion_matrices[name],suite.classes,name),use_container_width=True)
            row=evaluation[evaluation["Model"]==name].iloc[0];a,b,c,d=st.columns(4);a.metric("Accuracy",f"{row['Accuracy']:.2%}");b.metric("Precision",f"{row['Precision']:.2%}");c.metric("Recall",f"{row['Recall']:.2%}");d.metric("F1 Score",f"{row['F1 Score']:.2%}")

elif page == "Feature Analysis":
    hero("Four-Feature Selection Analysis","The final model uses only four academically meaningful variables, supported by correlation evidence from the 5,000-record project dataset.","Data Analysis Module • Feature Selection")
    corr=suite.feature_correlations.rename("Correlation").rename_axis("Feature").reset_index();corr["Feature Label"]=corr["Feature"].map(FEATURE_LABELS);corr=corr.sort_values("Correlation")
    fig=px.bar(corr,x="Correlation",y="Feature Label",orientation="h",text=corr["Correlation"].map(lambda x:f"{x:.3f}"),range_x=[0,1]);fig.update_layout(height=380,yaxis_title="",margin=dict(l=10,r=10,t=20,b=10));st.plotly_chart(fig,use_container_width=True)
    rationale=pd.DataFrame([
        ["Previous CGPA",suite.feature_correlations.get("Previous_CGPA",0),"Strongest historical indicator of academic achievement."],
        ["Average Score",suite.feature_correlations.get("Average_Score",0),"Directly reflects recent assessment performance."],
        ["Attendance Percentage",suite.feature_correlations.get("Attendance_Pct",0),"Captures learning participation and consistency."],
        ["Study Hours per Day",suite.feature_correlations.get("Study_Hours_Per_Day",0),"Captures academic effort and preparation."],
    ],columns=["Feature","Correlation with Final CGPA","Why it matters"]);rationale["Correlation with Final CGPA"]=rationale["Correlation with Final CGPA"].map(lambda x:f"{x:.3f}");st.dataframe(rationale,hide_index=True,use_container_width=True)
    numeric=[c for c in df.select_dtypes(include="number").columns if c!="Final_CGPA"]
    all_corr=df[numeric+["Final_CGPA"]].corr(numeric_only=True)["Final_CGPA"].drop("Final_CGPA").sort_values(ascending=False).rename("Correlation with Final CGPA").reset_index().rename(columns={"index":"Available Numerical Attribute"})
    st.markdown("### Correlation Check Across Available Numerical Attributes");st.dataframe(all_corr,hide_index=True,use_container_width=True)
    st.info("Number of Subjects is intentionally excluded from the final model because its correlation with Final CGPA is approximately 0.005, while the four selected features have stronger academic relevance.")

elif page == "Dataset Explorer":
    hero("Dataset Explorer","Browse the 5,000 student records used for preprocessing, model training, testing and feature analysis.","Dataset Module • 5,000 Student Records")
    c1,c2,c3,c4=st.columns(4);c1.metric("Rows",f"{len(df):,}");c2.metric("Columns",len(df.columns));c3.metric("Missing Values",int(df.isna().sum().sum()));c4.metric("Selected Features","4")
    search=st.text_input("Search Student ID",placeholder="e.g. ID00001");explorer=df.copy()
    if search.strip() and "Student_ID" in explorer.columns:explorer=explorer[explorer["Student_ID"].astype(str).str.contains(search.strip(),case=False,na=False)]
    st.dataframe(explorer,hide_index=True,use_container_width=True,height=540);st.caption(f"Showing {len(explorer):,} record(s).")
    stats=df[FEATURES].describe().T.reset_index().rename(columns={"index":"Feature"});stats["Feature"]=stats["Feature"].map(FEATURE_LABELS);st.markdown("### Descriptive Statistics of the Four Inputs");st.dataframe(stats,hide_index=True,use_container_width=True)

else:
    hero("About the Project","Student Performance Prediction is an academic decision-support prototype developed with Python, Streamlit and Scikit-learn.","Project Documentation • Final 4-Feature Version")
    st.markdown("### Project Objective");st.write("Predict student academic performance using four relevant inputs and compare three supervised machine learning algorithms: K-Nearest Neighbours (KNN), Support Vector Machine (SVM) and Artificial Neural Network (ANN).")
    st.markdown("### Final Machine Learning Inputs")
    for i,feature in enumerate(FEATURES,1):st.write(f"{i}. **{FEATURE_LABELS[feature]}**")
    st.markdown("### Target Classes");cgpa_guide()
    st.markdown("### Dataset Source");st.write("University Student Performance & Habits Dataset by Robiul Hasan Jisan (Kaggle). The project CSV contains an engineered Average Score field for this academic prototype. Average Score must not be described as an original Kaggle column.")
    st.markdown("**APA:** Jisan, R. H. (n.d.). *University student performance & habits dataset* [Data set]. Kaggle.")
    st.code("https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset",language=None)
    st.markdown("### Reproducible Method")
    st.write(f"80/20 stratified train-test split with random state 42. Training rows: {suite.train_size:,}; testing rows: {suite.test_size:,}. All three models use the same four input features and StandardScaler pipeline.")
    st.warning("This system is an educational prototype. Predictions should support, not replace, lecturer judgement or formal academic assessment.")
