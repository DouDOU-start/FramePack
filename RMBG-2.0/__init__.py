"""
RMBG-2.0 - Enhanced Background Processing System

A unified and optimized background processing system that consolidates
functionality from multiple files into a clean, modular architecture.

Features:
- Unified model management
- Efficient video processing
- Batch processing capabilities
- Memory optimization
- Progress tracking
- Temporary file management

Usage:
    from RMBG_2_0 import BackgroundProcessor
    
    processor = BackgroundProcessor(model_path)
    result = processor.remove_background_image(input_path, output_path)
"""

__version__ = "2.0.0"
__author__ = "FramePack Project Enhanced"

# Import main classes
from .background_processor import BackgroundProcessor, get_processor
from .processors.image_processor import ImageProcessor
from .processors.video_processor import VideoProcessor

# Import core utilities
from .core.model_manager import ModelManager
from .core.video_utils import VideoUtils
from .core.color_utils import ColorUtils

# Import utility functions
from .utils.temp_file_manager import get_temp_manager, create_temp_workspace

# Convenience functions
from .background_processor import (
    remove_background,
    replace_background,
    process_video
)

__all__ = [
    # Main classes
    'BackgroundProcessor',
    'ImageProcessor', 
    'VideoProcessor',
    
    # Core utilities
    'ModelManager',
    'VideoUtils',
    'ColorUtils',
    
    # Convenience functions
    'get_processor',
    'get_temp_manager',
    'create_temp_workspace',
    'remove_background',
    'replace_background',
    'process_video'
]