# Student Performance Prediction System — FINAL Deployment-Proof Build

A Streamlit machine-learning application that predicts student performance using KNN, SVM and ANN.

## Five selected features
1. Previous CGPA
2. Average Score
3. Attendance Rate (%)
4. Study Hours per Day
5. Sleep Hours per Day

## Target classes
- Excellent: Final CGPA 3.50–4.00
- Good: Final CGPA 3.00–3.49
- Average: Final CGPA 2.50–2.99
- At Risk: Final CGPA below 2.50

## GitHub deployment
Upload these files to the ROOT of the repository:

- `app.py`
- `Student_data.csv`
- `requirements.txt`
- `README.md`
- `train_model.py`

Streamlit Community Cloud main file path: `app.py`

## Important deployment protection
The app does not require `src/`, `models/`, `results/`, or `dataset/` folders. It automatically uses `Student_data.csv` when present. If the CSV is missing or invalid, the app still starts using a deterministic built-in fallback dataset, so a missing-data path does not crash the deployment.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional model verification:
```bash
python train_model.py
```
