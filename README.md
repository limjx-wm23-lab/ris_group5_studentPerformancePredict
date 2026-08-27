# 🎓 Student Performance Prediction System — Final 4-Feature Edition

A Streamlit Artificial Intelligence project that predicts university student performance using **exactly four academically relevant features** and compares **KNN, SVM and ANN**.

## Final Machine Learning Features

| Feature | Purpose | Correlation with Final CGPA |
|---|---|---:|
| Previous CGPA | Historical academic achievement | 0.879 |
| Average Score | Recent assessment performance | 0.834 |
| Attendance Percentage | Learning participation and consistency | 0.303 |
| Study Hours per Day | Academic effort and preparation | 0.231 |

These are the only four model inputs used throughout the UI, individual prediction, batch prediction and model training.

`Number_of_Subjects` is **not** used by the final model because its correlation with Final CGPA in the project dataset is approximately **0.005**. Keeping it as an ML feature would add almost no useful predictive relationship.

> **Important dataset note:** `Average_Score` is an engineered field in the project CSV. It must not be described as an original Kaggle column.

## Target Classes

| Category | Final CGPA |
|---|---:|
| Excellent | 3.50 – 4.00 |
| Good | 3.00 – 3.49 |
| Average | 2.50 – 2.99 |
| At Risk | Below 2.50 |

## Machine Learning Models

- K-Nearest Neighbours (KNN)
- Support Vector Machine (SVM)
- Artificial Neural Network (ANN / MLPClassifier)

All models use the same 80/20 stratified train-test split (`random_state=42`) and a `StandardScaler` pipeline for fair comparison. The application automatically selects the best-performing model by hold-out accuracy for the final result while displaying all three model predictions.

Current reproduced results from the 5,000-record project dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| ANN | ~81.2% | ~81.0% | ~81.2% | ~81.1% |
| SVM | ~79.7% | ~79.6% | ~79.7% | ~79.6% |
| KNN | ~79.2% | ~79.1% | ~79.2% | ~79.1% |

Exact values are reproduced automatically by `verify_project.py` in the deployment environment.

## Application Modules

1. **Home** – project KPIs, four final features, model comparison and CGPA guide.
2. **Individual Prediction** – four-feature form, three-model comparison, confidence and Excel export.
3. **Batch Prediction** – Excel/CSV upload, strict validation, at-risk identification, dashboard and Excel export.
4. **Model Evaluation** – Accuracy, Precision, Recall, F1 Score and confusion matrices.
5. **Feature Analysis** – feature-selection evidence using correlations with Final CGPA.
6. **Dataset Explorer** – browse all 5,000 student records and descriptive statistics.
7. **About** – project objective, target classes, methodology and dataset reference.

## Required Batch Columns

```text
Average_Score
Attendance_Pct
Study_Hours_Per_Day
Previous_CGPA
```

`Student_ID` and `Student_Name` may be included as profile information but are not machine learning inputs.

## Local Run

```bash
python -m pip install -r requirements.txt
python verify_project.py
python smoke_test.py
streamlit run app.py
```

## Streamlit Community Cloud

- Repository: `limjx-wm23-lab/ris_group5_studentPerformancePredict`
- Branch: `main`
- Main file path: `app.py`
- Python: 3.12

Models are trained and cached automatically from `Student_data.csv` when the app starts, so separate `.joblib` files are not required for deployment.

## Dataset Source

Jisan, R. H. (n.d.). *University student performance & habits dataset* [Data set]. Kaggle.

https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset

The repository dataset contains 5,000 student records. The project uses selected and prepared fields for model development and demonstration.

## Responsible Use

This application is an educational decision-support prototype. Predictions are probabilistic and should support, not replace, lecturer judgement or formal academic assessment.
