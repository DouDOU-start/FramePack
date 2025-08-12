"""
Unified video processor for background operations
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, Tuple, List, Callable
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import core modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
from core.model_manager import ModelManager
from core.video_utils import VideoUtils
from core.color_utils import ColorUtils
from utils.temp_file_manager import get_temp_manager, create_temp_workspace
from processors.image_processor import ImageProcessor


class VideoProcessor:
    """
    Unified video processor for background operations
    Consolidates functionality from multiple video processing files
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize video processor
        
        Args:
            model_path: Path to the model directory
            device: Device to use for processing
        """
        self.model_manager = ModelManager(model_path, device)
        self.temp_manager = get_temp_manager()
        self.image_processor = ImageProcessor(model_path, device)
    
    def process_video_background(self, video_path: str, output_path: str,
                               background: Union[str, Tuple[int, int, int], Image.Image, str] = None,
                               resolution: Optional[Tuple[int, int]] = None,
                               fps: Optional[float] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Process video background removal or replacement
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            background: Background to apply (None for transparent)
            resolution: Output resolution (optional)
            fps: Output FPS (optional)
            progress_callback: Progress callback function
            
        Returns:
            Path to output video
        """
        # Check if FFmpeg is available
        if not VideoUtils.check_ffmpeg():
            raise RuntimeError("FFmpeg is required for video processing")
        
        # Create workspace
        workspace = create_temp_workspace()
        
        try:
            # Extract video info
            video_info = VideoUtils.get_video_info(video_path)
            if not video_info:
                raise RuntimeError("Could not get video information")
            
            # Use original resolution and FPS if not specified
            if resolution is None:
                resolution = (video_info['width'], video_info['height'])
            if fps is None:
                fps = video_info['fps']
            
            # Extract frames
            input_frames_dir = os.path.join(workspace, 'input_frames')
            if progress_callback:
                progress_callback(0, "Extracting frames...")
            
            frames = VideoUtils.extract_frames(video_path, input_frames_dir)
            if not frames:
                raise RuntimeError("Could not extract frames from video")
            
            # Process frames
            processed_frames_dir = os.path.join(workspace, 'processed_frames')
            os.makedirs(processed_frames_dir, exist_ok=True)
            
            total_frames = len(frames)
            processed_count = 0
            
            if progress_callback:
                progress_callback(5, "Processing frames...")
            
            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_frame = {}
                
                for i, frame_path in enumerate(frames):
                    future = executor.submit(
                        self._process_frame,
                        frame_path,
                        processed_frames_dir,
                        background,
                        i
                    )
                    future_to_frame[future] = i
                
                for future in as_completed(future_to_frame):
                    try:
                        future.result()
                        processed_count += 1
                        progress = 5 + (processed_count / total_frames) * 80
                        
                        if progress_callback:
                            progress_callback(progress, f"Processing frame {processed_count}/{total_frames}")
                    
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue
            
            # Create output video
            if progress_callback:
                progress_callback(90, "Creating output video...")
            
            if background is None:
                # Create video with alpha channel
                success = VideoUtils.create_video_with_alpha(
                    processed_frames_dir,
                    output_path,
                    fps=fps
                )
            else:
                # Create regular video (帧已完成合成，这里仅负责编码；添加 -y 避免阻塞由 core.video_utils 内部处理)
                success = VideoUtils.create_video_from_frames(
                    processed_frames_dir,
                    output_path,
                    fps=fps,
                    resolution=resolution
                )
            
            if not success:
                raise RuntimeError("Could not create output video")
            
            if progress_callback:
                progress_callback(100, "Processing complete!")
            
            return output_path
            
        finally:
            # Clean up workspace
            self.temp_manager.cleanup_dir(workspace)
    
    def _process_frame(self, frame_path: str, output_dir: str, 
                      background: Union[str, Tuple[int, int, int], Image.Image, str],
                      frame_index: int) -> str:
        """
        Process a single frame
        
        Args:
            frame_path: Path to input frame
            output_dir: Directory for processed frames
            background: Background to apply
            frame_index: Frame index for naming
            
        Returns:
            Path to processed frame
        """
        try:
            if background is None:
                # Remove background (transparent)
                output_path = self.image_processor.remove_background(frame_path)
            else:
                # Replace background
                output_path = self.image_processor.replace_background(frame_path, background)
            
            # Move to output directory with proper naming
            output_filename = f"frame_{frame_index:06d}.png"
            final_path = os.path.join(output_dir, output_filename)
            
            # If it's a different format, convert to PNG
            if not output_path.endswith('.png'):
                img = Image.open(output_path)
                img.save(final_path, 'PNG')
                self.temp_manager.cleanup_file(output_path)
            else:
                os.rename(output_path, final_path)
            
            return final_path
            
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
            raise
    
    def add_background_to_video(self, video_path: str, background: Union[str, Image.Image],
                               output_path: str, resolution: Optional[Tuple[int, int]] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Add background to a video (assuming video has alpha channel)
        
        Args:
            video_path: Path to input video with alpha
            background: Background to apply
            output_path: Path to output video
            resolution: Output resolution (optional)
            progress_callback: Progress callback function
            
        Returns:
            Path to output video
        """
        return self.process_video_background(
            video_path, output_path, background, resolution, None, progress_callback
        )
    
    def remove_background_from_video(self, video_path: str, output_path: str,
                                   resolution: Optional[Tuple[int, int]] = None,
                                   fps: Optional[float] = None,
                                   progress_callback: Optional[Callable] = None) -> str:
        """
        Remove background from video (create transparent video)
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            resolution: Output resolution (optional)
            fps: Output FPS (optional)
            progress_callback: Progress callback function
            
        Returns:
            Path to output video
        """
        return self.process_video_background(
            video_path, output_path, None, resolution, fps, progress_callback
        )
    
    def batch_process_videos(self, video_paths: List[str], output_dir: str,
                            background: Union[str, Tuple[int, int, int], Image.Image, str] = None,
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
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, video_path in enumerate(video_paths):
            try:
                # Generate output filename
                input_name = Path(video_path).stem
                if background is None:
                    output_filename = f"{input_name}_transparent.webm"
                else:
                    output_filename = f"{input_name}_processed.mp4"
                
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"Processing video {i+1}/{len(video_paths)}: {input_name}")
                
                # Process video
                result_path = self.process_video_background(
                    video_path, output_path, background, resolution
                )
                
                results.append(result_path)
                print(f"✅ Completed: {input_name}")
                
            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                continue
        
        return results
    
    def get_video_info(self, video_path: str) -> Optional[dict]:
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary or None
        """
        return VideoUtils.get_video_info(video_path)
    
    def cleanup(self):
        """Clean up resources"""
        self.model_manager.cleanup()
        self.image_processor.cleanup()
        self.temp_manager.cleanup_all()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()


# Convenience functions for backward compatibility
def process_video_background(video_path: str, output_path: str, model_path: str,
                           background: Union[str, Tuple[int, int, int], Image.Image] = None,
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
    processor = VideoProcessor(model_path)
    result = processor.process_video_background(video_path, output_path, background, resolution, fps)
    processor.cleanup()
    return result


def remove_background_from_video(video_path: str, output_path: str, model_path: str,
                                resolution: Optional[Tuple[int, int]] = None,
                                fps: Optional[float] = None) -> str:
    """
    Remove background from video (convenience function)
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        model_path: Path to model directory
        resolution: Output resolution (optional)
        fps: Output FPS (optional)
        
    Returns:
        Path to output video
    """
    processor = VideoProcessor(model_path)
    result = processor.remove_background_from_video(video_path, output_path, resolution, fps)
    processor.cleanup()
    return result