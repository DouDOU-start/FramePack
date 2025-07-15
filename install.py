#!/usr/bin/env python3
"""
FramePack Installation Script

This script helps users set up the FramePack environment automatically.
It checks system requirements, installs dependencies, and verifies the installation.

Author: FramePack Project
License: See LICENSE file
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_header():
    """Print installation header"""
    print("=" * 60)
    print("🎬 FramePack - AI Video Generation Framework")
    print("=" * 60)
    print("This script will help you set up FramePack on your system.")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("📋 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Not compatible")
        print("   Please install Python 3.9 or higher")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 6:
                print("✅ GPU memory sufficient for FramePack")
                return True
            else:
                print("⚠️  GPU memory may be insufficient (minimum 6GB recommended)")
                return False
        else:
            print("❌ No CUDA-compatible GPU found")
            print("   FramePack requires NVIDIA GPU with CUDA support")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - will check after installation")
        return None

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print("\n🎥 Checking FFmpeg...")
    if shutil.which("ffmpeg"):
        print("✅ FFmpeg is installed")
        return True
    else:
        print("⚠️  FFmpeg not found")
        system = platform.system().lower()
        if system == "windows":
            print("   Please download FFmpeg from: https://ffmpeg.org/download.html")
            print("   And add it to your system PATH")
        elif system == "linux":
            print("   Install with: sudo apt install ffmpeg (Ubuntu/Debian)")
            print("   Or: sudo yum install ffmpeg (CentOS/RHEL)")
        elif system == "darwin":
            print("   Install with: brew install ffmpeg")
        return False

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("\n🔥 Installing PyTorch with CUDA support...")
    
    # Check CUDA version
    cuda_version = "cu126"  # Default to CUDA 12.6
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            if "CUDA Version: 11." in output:
                cuda_version = "cu118"
                print("   Detected CUDA 11.x - using cu118 index")
            else:
                print("   Using CUDA 12.6 index (default)")
    except FileNotFoundError:
        print("   nvidia-smi not found, using default CUDA version")
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_version}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_requirements():
    """Install project requirements"""
    print("\n📦 Installing project dependencies...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["outputs", "hf_download", "inputs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    try:
        import torch
        import gradio
        import PIL
        import cv2
        import numpy
        import transformers
        import diffusers
        
        print("✅ All core packages imported successfully")
        
        # Check GPU again after PyTorch installation
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main installation function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg()
    
    # Install PyTorch
    if not install_pytorch():
        print("\n❌ Installation failed at PyTorch step")
        sys.exit(1)
    
    # Check GPU after PyTorch installation
    gpu_ok = check_gpu()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Installation failed at requirements step")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed")
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Installation completed!")
    print("=" * 60)
    
    if gpu_ok:
        print("✅ GPU support: Available")
    else:
        print("⚠️  GPU support: Limited or unavailable")
    
    if ffmpeg_ok:
        print("✅ Video processing: Available")
    else:
        print("⚠️  Video processing: Limited (install FFmpeg for full support)")
    
    print("\n🚀 Next steps:")
    print("1. FramePack视频生成: python scripts/framepack_app.py --server 127.0.0.1 --inbrowser")
    print("2. RMBG背景处理: python scripts/rmbg_app.py --server 127.0.0.1 --inbrowser")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()
