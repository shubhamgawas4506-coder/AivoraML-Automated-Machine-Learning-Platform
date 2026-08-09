@echo off
echo ================================================
echo   Starting AivoraML ML Platform
echo ================================================
echo.

cd /d "%~dp0.."

set "AIVORAML_PYTHON=python"
if exist "%CD%\venv\Scripts\python.exe" set "AIVORAML_PYTHON=%CD%\venv\Scripts\python.exe"

echo Checking Python...
"%AIVORAML_PYTHON%" --version
if errorlevel 1 (
    echo ERROR: Python was not found. Create venv using the README steps.
    pause
    exit /b 1
)

echo.
echo Starting FastAPI server...
echo Landing Page: http://localhost:8000
echo ML Platform:  http://localhost:8000/app
echo.
echo Press CTRL+C to stop the server
echo.

"%AIVORAML_PYTHON%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

pause
