# Student Performance Prediction System v2.1 — Streamlit Cloud Safe

A supervised machine-learning application that predicts student academic performance using KNN, SVM and ANN.

## Important deployment fix

This build is **self-contained**. `app.py` does not import `src.core`, so it will not fail with `ModuleNotFoundError: No module named src` when deployed on Streamlit Community Cloud.

## Five selected features

1. Previous CGPA
2. Average Score
3. Attendance Rate (%)
4. Study Hours per Day
5. Sleep Hours per Day

## Performance categories

- At Risk: Final CGPA below 2.50
- Average: 2.50–2.99
- Good: 3.00–3.49
- Excellent: 3.50–4.00

## GitHub structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── Student_data.csv
├── models/
│   ├── knn_model.joblib
│   ├── svm_model.joblib
│   ├── ann_model.joblib
│   └── metadata.json
├── results/
│   ├── evaluation.csv
│   ├── feature_selection.csv
│   └── *_confusion_matrix.png
├── templates/
│   └── student_batch_template.xlsx
└── .streamlit/
    └── config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The pretrained models are already included. You do **not** need to run `train_model.py` before launching the app. Run it only if you intentionally want to regenerate all models and evaluation results.

## Streamlit Community Cloud

1. Upload **all files and folders** in this project to the root of your GitHub repository.
2. In Streamlit Community Cloud choose your repository.
3. Set the main file path to `app.py`.
4. Deploy or reboot the app.

Do not upload only `app.py`; the `dataset`, `models`, and `results` folders are required.
