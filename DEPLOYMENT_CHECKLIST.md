# Deployment Checklist

## Streamlit Community Cloud settings

- Repository: `limjx-wm23-lab/ris_group5_studentPerformancePredict`
- Branch: `main`
- Main file path: `app.py`
- Python runtime: 3.12 (`runtime.txt`)

## Required root files

- `app.py`
- `student_model.py`
- `Student_data.csv`
- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`

## Pre-deployment verification

```bash
python -m pip install -r requirements.txt
python verify_project.py
python smoke_test.py
streamlit run app.py
```

The health endpoint should return `ok`:

```bash
curl http://127.0.0.1:8501/_stcore/health
```

Expected final messages:

- `ALL CHECKS PASSED`
- `STREAMLIT APPTEST PASSED: 7 pages and individual form submission`

## Important

The app trains and caches all three models automatically. Do not run a separate training command on Streamlit Cloud. No `src/`, `models/`, `results/`, or `dataset/` directory is required.
