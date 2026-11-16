@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.ps1 first.
  pause
  exit /b 1
)
set PYTHONDONTWRITEBYTECODE=1
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" -X utf8 web\serve.py
pause
