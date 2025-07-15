@echo off
REM FramePack Quick Start Script for Windows

echo ============================================================
echo 🎬 FramePack - AI Video Generation Framework
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "framepack_env\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo ✅ Activating virtual environment...
call framepack_env\Scripts\activate.bat

REM Show menu
echo.
echo Please choose an application to start:
echo 1. FramePack Video Generation (Main)
echo 2. RMBG Background Processing
echo 3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🎬 Starting FramePack Video Generation...
    python scripts\framepack_app.py --server 127.0.0.1 --inbrowser
) else if "%choice%"=="2" (
    echo.
    echo 🎨 Starting RMBG Background Processing...
    python scripts\rmbg_app.py --server 127.0.0.1 --inbrowser
) else if "%choice%"=="3" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

echo.
echo Application closed.
pause
