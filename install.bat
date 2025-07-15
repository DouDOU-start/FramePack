@echo off
REM FramePack Installation Script for Windows
REM This script sets up FramePack environment on Windows systems

echo ============================================================
echo 🎬 FramePack - AI Video Generation Framework
echo ============================================================
echo Setting up FramePack on Windows...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment
echo.
echo 📦 Creating virtual environment...
python -m venv framepack_env
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo ✅ Activating virtual environment...
call framepack_env\Scripts\activate.bat

REM Run Python installation script
echo.
echo 🚀 Running installation script...
python install.py

if errorlevel 1 (
    echo ❌ Installation failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🎉 Installation completed successfully!
echo ============================================================
echo.
echo To use FramePack:
echo 1. Activate the environment: framepack_env\Scripts\activate.bat
echo 2. FramePack视频生成: python scripts\framepack_app.py --server 127.0.0.1 --inbrowser
echo 3. RMBG背景处理: python scripts\rmbg_app.py --server 127.0.0.1 --inbrowser
echo.
echo Press any key to exit...
pause >nul
