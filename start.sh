#!/bin/bash
# FramePack Quick Start Script for Linux/macOS

echo "============================================================"
echo "🎬 FramePack - AI Video Generation Framework"
echo "============================================================"
echo

# Check if virtual environment exists
if [ ! -f "framepack_env/bin/activate" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run install.sh first to set up the environment."
    echo
    exit 1
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source framepack_env/bin/activate

# Show menu
echo
echo "Please choose an application to start:"
echo "1. FramePack Video Generation (Main)"
echo "2. RMBG Background Processing"
echo "3. Exit"
echo
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo
        echo "🎬 Starting FramePack Video Generation..."
        python scripts/framepack_app.py --server 127.0.0.1 --inbrowser
        ;;
    2)
        echo
        echo "🎨 Starting RMBG Background Processing..."
        python scripts/rmbg_app.py --server 127.0.0.1 --inbrowser
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo
echo "Application closed."
