@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m challenge_external.desktop_demo
) else (
  python -m challenge_external.desktop_demo
)
if errorlevel 1 pause
