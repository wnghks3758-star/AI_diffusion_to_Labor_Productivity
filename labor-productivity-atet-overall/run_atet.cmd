@echo off
setlocal
set "ATET_ROOT=%~dp0"
if exist "%ATET_ROOT%.venv\Scripts\python.exe" (
  "%ATET_ROOT%.venv\Scripts\python.exe" "%ATET_ROOT%run_atet.py" %*
) else if exist "%ATET_ROOT%..\ACR\Scripts\python.exe" (
  "%ATET_ROOT%..\ACR\Scripts\python.exe" "%ATET_ROOT%run_atet.py" %*
) else (
  python "%ATET_ROOT%run_atet.py" %*
)
set "ATET_EXIT=%ERRORLEVEL%"
if not "%ATET_EXIT%"=="0" echo Run finished with code %ATET_EXIT%. See saved output or error above.
pause
exit /b %ATET_EXIT%
