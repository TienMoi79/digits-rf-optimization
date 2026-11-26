@echo off
setlocal
cd /d "%~dp0.."
set PYTHONDONTWRITEBYTECODE=1
if exist ".venv\Scripts\python.exe" goto venv
where py >nul 2>&1
if not errorlevel 1 goto launcher
where python >nul 2>&1
if not errorlevel 1 goto systempython
echo Install Python 3.12, then double-click this file again.
echo The web demo needs only Python's standard library.
pause
exit /b 1
:venv
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" -X utf8 web\serve.py
goto done
:launcher
start "" "http://127.0.0.1:8765"
py -3 -X utf8 web\serve.py
goto done
:systempython
start "" "http://127.0.0.1:8765"
python -X utf8 web\serve.py
:done
pause
endlocal
