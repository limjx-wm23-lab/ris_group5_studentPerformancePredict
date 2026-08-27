# Student Performance Prediction System — A++ Final Edition

A deployment-safe Streamlit machine-learning system for early student-performance screening using **KNN, SVM, and ANN**.

## Key strengths

- Uses 5,000 validated university student records with no missing values.
- Selects five defensible, numeric and actionable features.
- Excludes gender, age and major from prediction to reduce unnecessary demographic influence.
- Compares three models on the same reproducible stratified hold-out set.
- Selects the final model by Macro F1, followed by accuracy and Weighted F1.
- Supports transparent individual prediction and up to 10,000 batch records.
- Generates downloadable Excel templates and complete prediction reports.
- Includes deterministic fallback data so a missing CSV does not crash deployment.
- Requires no pre-generated model files or hidden folders.

## Five selected features

1. Previous CGPA
2. Average Score
3. Attendance Rate (%)
4. Study Hours per Day
5. Sleep Hours per Day

Previous CGPA and Average Score are the strongest academic signals. Attendance and Study Hours add engagement and effort information. Sleep Hours is retained as an actionable wellbeing indicator. Number of Subjects, Social Hours, age, gender and major are not used by the prediction models.

## Target classes

| Category | Final CGPA rule |
|---|---:|
| Excellent | 3.50–4.00 |
| Good | 3.00–3.49 |
| Average | 2.50–2.99 |
| At Risk | Below 2.50 |

## Application modules

- Overview dashboard
- Individual Prediction
- Batch Prediction with validation and Excel download
- Model Evaluation with five metrics and confusion matrices
- Feature Analysis
- Dataset Explorer
- Methodology, ethics and limitations

## Reproducible methodology

1. Validate schema, missing data and numeric ranges.
2. Map Final CGPA into four documented target classes.
3. Create an 80/20 stratified split using random state 42.
4. Standardise predictors inside every model pipeline.
5. Train KNN, calibrated SVM and ANN on the same training set.
6. Compare Accuracy, Precision, Recall, Weighted F1 and Macro F1.
7. Use the highest-ranked hold-out model for the final prediction while showing all model decisions.

## Reproduced hold-out results

The verification script retrains every model from the repository dataset. With the pinned dependencies,
80/20 stratified split and random state 42, the current results are:

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| SVM | 78.30% | 78.25% | 76.97% |
| KNN | 77.60% | 77.56% | 76.15% |
| ANN | 77.50% | 77.40% | 75.31% |

SVM is selected by the documented Macro F1 ranking rule. These are hold-out results, not claims of
performance on unseen institutions or future cohorts.

## Local verification

Python 3.12 is recommended.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_project.py
.venv/bin/python smoke_test.py
.venv/bin/streamlit run app.py
```

On Windows, replace `.venv/bin/python` with `.venv\\Scripts\\python.exe`.

The automated smoke test renders all seven application pages and submits the individual prediction form.
The verification script also exercises dataset validation, feature evidence, all three models, individual
prediction and batch prediction.

## Streamlit Community Cloud

- Repository: `limjx-wm23-lab/ris_group5_studentPerformancePredict`
- Branch: `main`
- Main file path: `app.py`

The app automatically loads `Student_data.csv` from the repository root and trains the models at startup. If the CSV is missing or invalid, the interface remains available through a deterministic fallback dataset.

## Dataset source

The project dataset is based on the **University Student Performance & Habits Dataset**, credited to **Robiul Hasan Jisan** on Kaggle:

https://www.kaggle.com/datasets/asifxzaman/university-students-performance-and-study-habits2026

The repository CSV contains additional prepared fields used by this application, including `Number_of_Subjects` and `Average_Score`.

## Responsible-use statement

This application is an academic decision-support prototype. Predictions are probabilistic and must not be used as the sole basis for grades, disciplinary decisions, scholarships, admissions, or access to education. Human review and contextual evidence remain essential.
