@echo off
echo ========================================
echo   DIALOGUE-X Translation Platform
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if FFmpeg is installed (required for pydub)
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: FFmpeg is not installed
    echo pydub requires FFmpeg for audio conversion
    echo Download from: https://ffmpeg.org/download.html
    echo.
    pause
)

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Starting backend server...
start "Backend Server" cmd /k "cd backend && python server.py"

REM Wait for backend to start
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Starting frontend server...
start "Frontend Server" cmd /k "cd frontend && python -m http.server 8080"

timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo   System Started Successfully!
echo ========================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:8080
echo.
echo Opening browser...

timeout /t 2 /nobreak >nul
start http://localhost:8080

echo.
echo System is running with synchronized video delay!
echo Close the server windows to stop all services.
pause
