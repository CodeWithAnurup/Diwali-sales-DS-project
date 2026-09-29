"""Entry point alias for Streamlit Cloud."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__" or "streamlit" in sys.modules:
    app_path = Path(__file__).resolve().parent / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")
