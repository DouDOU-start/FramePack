"""
Main background processing module
Provides unified interface for all background operations
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, List, Callable

# Import processors
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor


class BackgroundProcessor:
    """
    Unified background processor for all operations
    Provides a single interface for image and video processing
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize background processor
        
        Args:
            model_path: Path to the model directory
            device: Device to use for processing
        """
        self.model_path = model_path
        self.device = device
        self.image_processor = None
        self.video_processor = None
        
        # Initialize processors lazily
        self._init_image_processor()
        self._init_video_processor()
    
    def _init_image_processor(self):
        """Initialize image processor if not already initialized"""
        if self.image_processor is None:
            self.image_processor = ImageProcessor(self.model_path, self.device)
    
    def _init_video_processor(self):
        """Initialize video processor if not already initialized"""
        if self.video_processor is None:
            self.video_processor = VideoProcessor(self.model_path, self.device)
    
    # Image processing methods
    def remove_background_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Remove background from an image
        
        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)
            
        Returns:
            Path to output image
        """
        self._init_image_processor()
        return self.image_processor.remove_background(image_path, output_path)
    
    def replace_background_image(self, image_path: str, background: Union[str, Tuple[int, int, int]],
                                output_path: Optional[str] = None) -> str:
        """
        Replace background of an image
        
        Args:
            image_path: Path to input image
            background: Background color or RGB tuple
            output_path: Path to output image (optional)
            
        Returns:
            Path to output image
        """
        self._init_image_processor()
        return self.image_processor.replace_background(image_path, background, output_path)
    
    def change_background_color_image(self, image_path: str, color: str,
                                     output_path: Optional[str] = None) -> str:
        """
        Change background color of an image
        
        Args:
            image_path: Path to input image
            color: Background color
            output_path: Path to output image (optional)
            
        Returns:
            Path to output image
        """
        self._init_image_processor()
        return self.image_processor.change_background_color(image_path, color, output_path)
    
    # Video processing methods
    def remove_background_video(self, video_path: str, output_path: str,
                               resolution: Optional[Tuple[int, int]] = None,
                               fps: Optional[float] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Remove background from video
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            resolution: Output resolution (optional)
            fps: Output FPS (optional)
            progress_callback: Progress callback function
            
        Returns:
            Path to output video
        """
        self._init_video_processor()
        return self.video_processor.remove_background_from_video(
            video_path, output_path, resolution, fps, progress_callback
        )
    
    def replace_background_video(self, video_path: str, background: Union[str, Tuple[int, int, int]],
                                output_path: str, resolution: Optional[Tuple[int, int]] = None,
                                fps: Optional[float] = None,
                                progress_callback: Optional[Callable] = None) -> str:
        """
        Replace background of video
        
        Args:
            video_path: Path to input video
            background: Background color or RGB tuple
            output_path: Path to output video
            resolution: Output resolution (optional)
            fps: Output FPS (optional)
            progress_callback: Progress callback function
            
        Returns:
            Path to output video
        """
        self._init_video_processor()
        return self.video_processor.process_video_background(
            video_path, output_path, background, resolution, fps, progress_callback
        )
    
    def add_background_to_video(self, video_path: str, background: Union[str, Tuple[int, int, int]],
                               output_path: str, resolution: Optional[Tuple[int, int]] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Add background to video (for videos with alpha channel)
        
        Args:
            video_path: Path to input video
            background: Background color or RGB tuple
            output_path: Path to output video
            resolution: Output resolution (optional)
            progress_callback: Progress callback function
            
        Returns:
            Path to output video
        """
        self._init_video_processor()
        return self.video_processor.add_background_to_video(
            video_path, background, output_path, resolution, progress_callback
        )
    
    # Batch processing methods
    def process_image_batch(self, image_paths: List[str], output_dir: str,
                           background: Union[str, Tuple[int, int, int]] = None) -> List[str]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory for output images
            background: Background to apply (optional)
            
        Returns:
            List of output image paths
        """
        self._init_image_processor()
        return self.image_processor.process_batch(image_paths, background, output_dir)
    
    def process_video_batch(self, video_paths: List[str], output_dir: str,
                           background: Union[str, Tuple[int, int, int]] = None,
                           resolution: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Process multiple videos in batch
        
        Args:
            video_paths: List of input video paths
            output_dir: Directory for output videos
            background: Background to apply (optional)
            resolution: Output resolution (optional)
            
        Returns:
            List of output video paths
        """
        self._init_video_processor()
        return self.video_processor.batch_process_videos(video_paths, output_dir, background, resolution)
    
    # Utility methods
    def get_video_info(self, video_path: str) -> Optional[dict]:
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary or None
        """
        self._init_video_processor()
        return self.video_processor.get_video_info(video_path)
    
    def cleanup(self):
        """Clean up resources"""
        if self.image_processor:
            self.image_processor.cleanup()
        if self.video_processor:
            self.video_processor.cleanup()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()


# Global processor instance
_global_processor = None


def get_processor(model_path: str, device: Optional[str] = None) -> BackgroundProcessor:
    """
    Get the global background processor instance
    
    Args:
        model_path: Path to the model directory
        device: Device to use for processing
        
    Returns:
        BackgroundProcessor instance
    """
    global _global_processor
    if _global_processor is None:
        _global_processor = BackgroundProcessor(model_path, device)
    return _global_processor


# Convenience functions for easy usage
def remove_background(image_path: str, model_path: str, output_path: Optional[str] = None) -> str:
    """
    Remove background from image (convenience function)
    
    Args:
        image_path: Path to input image
        model_path: Path to model directory
        output_path: Path to output image (optional)
        
    Returns:
        Path to output image
    """
    processor = get_processor(model_path)
    result = processor.remove_background_image(image_path, output_path)
    processor.cleanup()
    return result


def replace_background(image_path: str, background: Union[str, Tuple[int, int, int]], 
                     model_path: str, output_path: Optional[str] = None) -> str:
    """
    Replace background of image (convenience function)
    
    Args:
        image_path: Path to input image
        background: Background color or RGB tuple
        model_path: Path to model directory
        output_path: Path to output image (optional)
        
    Returns:
        Path to output image
    """
    processor = get_processor(model_path)
    result = processor.replace_background_image(image_path, background, output_path)
    processor.cleanup()
    return result


def process_video(video_path: str, output_path: str, model_path: str,
                 background: Union[str, Tuple[int, int, int]] = None,
                 resolution: Optional[Tuple[int, int]] = None,
                 fps: Optional[float] = None) -> str:
    """
    Process video background (convenience function)
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        model_path: Path to model directory
        background: Background to apply (optional)
        resolution: Output resolution (optional)
        fps: Output FPS (optional)
        
    Returns:
        Path to output video
    """
    processor = get_processor(model_path)
    if background:
        result = processor.replace_background_video(video_path, background, output_path, resolution, fps)
    else:
        result = processor.remove_background_video(video_path, output_path, resolution, fps)
    processor.cleanup()
    return result