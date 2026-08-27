"""Render every top-level Streamlit page with Streamlit's official AppTest framework."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for filename in ["app.py", "student_model.py", "verify_project.py"]:
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / filename)], check=True)
subprocess.run([sys.executable, str(ROOT / "verify_project.py")], check=True)

from streamlit.testing.v1 import AppTest

pages = [
    "🏠 Home",
    "🎯 Prediction",
    "📊 Model Results",
    "🔗 Correlation",
    "📈 Dataset",
    "⭐ Feature Analysis",
    "ℹ️ About",
]

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
assert not at.exception, at.exception
for page in pages[1:]:
    at.sidebar.radio[0].set_value(page).run(timeout=120)
    assert not at.exception, f"Streamlit exception on {page}: {at.exception}"

print("STREAMLIT APPTEST PASSED: 7 top-level pages rendered")
