#!/bin/bash
# FramePack Installation Script for Linux/macOS
# This script sets up FramePack environment on Unix-like systems

set -e  # Exit on any error

echo "============================================================"
echo "🎬 FramePack - AI Video Generation Framework"
echo "============================================================"
echo "Setting up FramePack on $(uname -s)..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9+ first."
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "   CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "   macOS: brew install python3"
    exit 1
fi

echo "✅ Python found"
python3 --version

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    echo "❌ pip not found. Please install pip first."
    echo "   Ubuntu/Debian: sudo apt install python3-pip"
    exit 1
fi

# Create virtual environment
echo
echo "📦 Creating virtual environment..."
python3 -m venv framepack_env

# Activate virtual environment
echo "✅ Activating virtual environment..."
source framepack_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Run Python installation script
echo
echo "🚀 Running installation script..."
python install.py

echo
echo "============================================================"
echo "🎉 Installation completed successfully!"
echo "============================================================"
echo
echo "To use FramePack:"
echo "1. Activate the environment: source framepack_env/bin/activate"
echo "2. Run the application: python demo_gradio.py --server 127.0.0.1 --inbrowser"
echo
echo "For server deployment, use: python demo_gradio.py --server 0.0.0.0 --port 7860"
echo
