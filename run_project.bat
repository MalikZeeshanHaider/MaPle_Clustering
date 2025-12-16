@echo off
REM ============================================================================
REM MaPle Clustering Project - One-Click Launcher
REM ============================================================================
REM This batch file starts both backend and frontend automatically
REM Author: MaPle Project Team
REM Date: December 16, 2025
REM ============================================================================

echo.
echo ============================================================================
echo    MaPle Clustering Project - Automated Launcher
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python detected
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Check if requirements are installed
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies from requirements.txt...
    echo This may take a few minutes...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed successfully
    echo.
) else (
    echo [OK] Dependencies already installed
    echo.
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo [INFO] Creating .env file from .env.example...
    copy .env.example .env >nul
    echo [OK] .env file created
    echo.
)

REM Create data directory if it doesn't exist
if not exist "data\" (
    echo [INFO] Creating data directory...
    mkdir data
    echo [OK] Data directory created
    echo.
)

echo ============================================================================
echo    Starting Backend and Frontend
echo ============================================================================
echo.
echo [INFO] Backend will run at: http://localhost:8000
echo [INFO] Frontend will run at: http://localhost:8501
echo.
echo [IMPORTANT] Two terminal windows will open:
echo   1. Backend Server (FastAPI)
echo   2. Frontend App (Streamlit)
echo.
echo Press Ctrl+C in each window to stop the servers
echo.
echo Starting in 3 seconds...
timeout /t 3 >nul

REM Start backend in a new window
echo [INFO] Starting Backend Server...
start "MaPle Backend Server" cmd /k "venv\Scripts\activate.bat && python -m backend.main"

REM Wait a moment for backend to initialize
timeout /t 5 >nul

REM Start frontend in a new window
echo [INFO] Starting Frontend App...
start "MaPle Frontend App" cmd /k "venv\Scripts\activate.bat && streamlit run frontend/app.py"

echo.
echo ============================================================================
echo    Project Started Successfully!
echo ============================================================================
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8501
echo API Docs: http://localhost:8000/docs
echo.
echo The browser should open automatically for the Streamlit app.
echo If not, manually navigate to http://localhost:8501
echo.
echo To stop the servers:
echo   - Press Ctrl+C in the backend window
echo   - Press Ctrl+C in the frontend window
echo   - Or close both terminal windows
echo.
echo ============================================================================
echo.

REM Open browser to frontend after a delay
timeout /t 8 >nul
start http://localhost:8501

echo [INFO] Project is running. Check the opened windows and browser.
echo.
echo Press any key to exit this launcher window...
pause >nul
