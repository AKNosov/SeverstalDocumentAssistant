@echo off
chcp 1241 >nul
echo ==========================================
echo    Document Assistant - Streamlit App
echo ==========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

if not exist venv (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist app.py (
    echo [ERROR] app.py not found in current directory!
    echo Current directory: %cd%
    dir *.py
    pause
    exit /b 1
)


echo.
echo ==========================================
echo    [4/4] Starting application...
echo    Open browser: http://localhost:8401
echo    Press Ctrl+C to stop
echo ==========================================
echo.

streamlit run app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start application
    echo Check if streamlit is installed: pip show streamlit
    pause
)

pause