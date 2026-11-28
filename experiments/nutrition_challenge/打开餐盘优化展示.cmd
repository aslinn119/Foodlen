@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m challenge_external.desktop_demo --nutritionverse
) else (
  python -m challenge_external.desktop_demo --nutritionverse
)
if errorlevel 1 pause
