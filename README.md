# Student Performance Prediction System — Four-Feature A++ Final Edition

A deployment-safe Streamlit machine-learning system for early student-performance screening using **KNN, SVM, and ANN**.

## Key strengths

- Uses the supplied 5,000-row processed project dataset based on the cited Kaggle source.
- Selects four predictors with meaningful correlation and cross-validation evidence.
- Excludes gender and major for fairness, and excludes unrelated numeric variables.
- Compares three models on the same reproducible stratified hold-out set.
- Selects the final model by Macro F1, followed by accuracy and Weighted F1.
- Supports transparent individual prediction and up to 10,000 batch records.
- Generates downloadable Excel templates and complete prediction reports.
- Includes deterministic fallback data so a missing CSV does not crash deployment.
- Requires no pre-generated model files or hidden folders.

## Four selected model features

1. Average Score
2. Attendance Rate (%)
3. Study Hours per Day
4. Previous CGPA

All four have a measurable relationship with Final CGPA: Previous CGPA has Pearson r=0.879, Average Score
r=0.834, Attendance r=0.303 and Study Hours r=0.231. The feature audit also demonstrates incremental
five-fold SVM Macro F1 improvement. `Number_of_Subjects` remains visible in profiles and reports but is not a
model feature because its correlation is approximately zero. Gender and Major are excluded to avoid unnecessary
demographic influence. Average Score and Number of Subjects are project-engineered extensions and are not
described as original Kaggle columns.

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

## Feature audit summary

| Project variable | Pearson r with Final CGPA | Decision |
|---|---:|---|
| Previous CGPA | 0.879 | Included |
| Average Score | 0.834 | Included — project-engineered |
| Attendance Percentage | 0.303 | Included |
| Study Hours per Day | 0.231 | Included |
| Number of Subjects | 0.005 | Profile only — project-engineered |
| Age | -0.012 | Excluded |
| Social Hours per Week | -0.005 | Excluded |
| Sleep Hours | 0.003 | Excluded |

The app's Feature Analysis page reports Pearson correlation, Spearman correlation and mutual information for
every numeric candidate. The documented verification script checks the exact source schema and SHA-256.

## Reproduced hold-out results

The verification script retrains every model from the processed project CSV. With the pinned dependencies,
80/20 stratified split and random state 42, the current results are regenerated during verification.

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| ANN | 81.20% | 81.05% | 80.16% |
| KNN | 79.20% | 79.12% | 78.23% |
| SVM | 78.70% | 78.69% | 77.46% |

ANN is selected by the documented Macro F1 ranking rule. These are hold-out results, not claims of
performance on unseen institutions or future cohorts.

## Local verification

Python 3.12 is recommended.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_project.py
.venv/bin/python feature_audit.py
.venv/bin/python smoke_test.py
.venv/bin/streamlit run app.py
```

On Windows, replace `.venv/bin/python` with `.venv\\Scripts\\python.exe`.

The feature audit reproduces the correlations and five-fold ablation evidence. The automated smoke test renders
all seven application pages and submits the individual prediction form. The verification script also exercises
source fidelity, all three models, individual prediction and 10,000-record batch prediction.

## Streamlit Community Cloud

- Repository: `limjx-wm23-lab/ris_group5_studentPerformancePredict`
- Branch: `main`
- Main file path: `app.py`

The app automatically loads `Student_data.csv` from the repository root and trains the models at startup. If the CSV is missing or invalid, the interface remains available through a deterministic fallback dataset.

## Dataset source and attribution

Jisan, R. H. (n.d.). *University student performance & habits dataset* [Data set]. Kaggle.

https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset

The original Kaggle dataset contains 5,000 rows and 10 columns. The supplied project version adds two clearly
documented engineered fields: `Number_of_Subjects` and `Average_Score`.

- Records: 5,000
- Project columns: 12
- Processed CSV SHA-256: `bfadc93d30f7341ed83f5eb1cd793d4cc69e47a4fb532522469929cafd77e9c6`
- License reported by Kaggle: CC BY-NC-SA 4.0

## Responsible-use statement

This application is an academic decision-support prototype. Predictions are probabilistic and must not be used as the sole basis for grades, disciplinary decisions, scholarships, admissions, or access to education. Human review and contextual evidence remain essential.
