@echo off
cd /d "%~dp0"
echo Starting Diwali Sales Analytics Platform...
call ".\venv\Scripts\activate.bat"
streamlit run app.py
pause
