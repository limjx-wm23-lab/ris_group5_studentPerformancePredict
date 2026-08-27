# 🎓 Student Performance Prediction System — Final 4-Feature Edition

A complete Streamlit machine-learning application for academic performance classification using **KNN, SVM and ANN**.

The final application uses **exactly four ML input features everywhere**: model training, individual prediction, batch prediction, correlation analysis, feature analysis and downloadable templates.

## Final 4 ML Features

| # | Feature | Purpose |
|---|---|---|
| 1 | Average Score | Current academic assessment performance |
| 2 | Attendance Percentage | Learning participation and consistency |
| 3 | Study Hours per Day | Daily study effort |
| 4 | Previous CGPA | Historical academic achievement |

No fifth input is required by the prediction pipeline.

## Target Classes

| Category | Final CGPA |
|---|---:|
| Excellent | 3.50–4.00 |
| Good | 3.00–3.49 |
| Average | 2.50–2.99 |
| At Risk | Below 2.50 |

## Application Modules

- **Home** — KPI dashboard, four final feature cards and model overview.
- **Prediction** — premium Prediction Hub with Individual and Batch modes.
- **Model Results** — KNN/SVM/ANN metrics and confusion matrices.
- **Correlation** — correlation matrix and ranking for the exact four model inputs.
- **Dataset** — explorer for the 5,000 student records.
- **Feature Analysis** — exactly four feature rows using Mutual Information, Pearson and Spearman analysis.
- **About** — methodology, target classes and responsible-use statement.

## Current Reproduced Results

Using the repository dataset with an 80/20 stratified split and random state 42:

| Model | Accuracy |
|---|---:|
| ANN | 81.2% |
| SVM | 79.7% |
| KNN | 79.2% |

## Batch Prediction Required Columns

```text
Average_Score
Attendance_Pct
Study_Hours_Per_Day
Previous_CGPA
```

Optional identification columns:

```text
Student_ID
Student_Name
```

## Verification

```bash
python -m pip install -r requirements.txt
python verify_project.py
python smoke_test.py
streamlit run app.py
```

The verification script checks the exact four-feature definition, dataset validity, all three model paths, Individual Prediction and Batch Prediction. The Streamlit smoke test renders every top-level application page.

## Streamlit Community Cloud

- Repository: `limjx-wm23-lab/ris_group5_studentPerformancePredict`
- Branch: `main`
- Main file path: `app.py`
- Python runtime: 3.12

If an older interface is still visible after a GitHub update, use **Manage app → Reboot app** so the deployment reloads the latest `main` branch.

## Responsible Use

This project is an academic decision-support prototype. Predictions are probabilistic and should not be the sole basis for high-impact educational decisions. Human review remains essential.
