# Deployment Checklist — Final 4-Feature Edition

## Streamlit Cloud

- Repository: `limjx-wm23-lab/ris_group5_studentPerformancePredict`
- Branch: `main`
- Main file path: `app.py`
- Python runtime: 3.12

## Final feature consistency

- [x] Average Score
- [x] Attendance Percentage
- [x] Study Hours per Day
- [x] Previous CGPA
- [x] No fifth ML feature is used anywhere in prediction or model training.

## Quality checks

- [x] 5,000-row repository dataset validated.
- [x] KNN, SVM and ANN train on the same four inputs.
- [x] Individual prediction path verified.
- [x] Batch validation and prediction path verified.
- [x] Seven Streamlit pages covered by AppTest in GitHub Actions.
- [x] Excel export/template generated dynamically by the app.
- [x] Number of Subjects excluded from final ML input due negligible correlation.

## Verification commands

```bash
python -m pip install -r requirements.txt
python verify_project.py
python smoke_test.py
streamlit run app.py
```
