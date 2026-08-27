# Final Deployment Checklist — 4-Feature Edition

## Streamlit Cloud Settings

- Repository: `limjx-wm23-lab/ris_group5_studentPerformancePredict`
- Branch: `main`
- Main file path: `app.py`
- Python: 3.12

## Final ML Input Definition

1. `Average_Score`
2. `Attendance_Pct`
3. `Study_Hours_Per_Day`
4. `Previous_CGPA`

## UI Checks

- [x] Home shows `ML Features = 4`.
- [x] Prediction Hub shows `Input Features = 4`.
- [x] Individual Prediction contains four numbered ML inputs.
- [x] Batch Prediction shows `Required Features = 4`.
- [x] Batch template requires exactly the same four ML input columns.
- [x] Correlation page analyses only the four selected ML inputs.
- [x] Feature Analysis table contains exactly four feature rows.
- [x] About page lists exactly four ML inputs.

## Automated Verification

```bash
python -m pip install -r requirements.txt
python verify_project.py
python smoke_test.py
```

Expected result:

```text
ALL CHECKS PASSED
```

## If Old UI Is Still Visible

In Streamlit Community Cloud, use **Manage app → Reboot app** and confirm the deployment is using `main` and `app.py`.
