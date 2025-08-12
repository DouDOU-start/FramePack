"""
Video utilities for FFmpeg operations and video information extraction
"""

import subprocess
import json
import os
import shutil
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class VideoUtils:
    """
    Unified video utilities for FFmpeg operations
    Eliminates duplicate FFmpeg code across multiple files
    """
    
    @staticmethod
    def get_video_info(video_path: str) -> Optional[Dict]:
        """
        Get comprehensive video information using ffprobe
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        if not os.path.exists(video_path):
            return None
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
                '-show_streams', video_path
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=True,
            )
            stdout_text = result.stdout or ''
            data = json.loads(stdout_text)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            # Extract information
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(data.get('format', {}).get('duration', 0))
            fps = eval(video_stream.get('r_frame_rate', '30/1'))
            
            return {
                'width': width,
                'height': height,
                'duration': duration,
                'fps': fps,
                'frame_count': int(duration * fps),
                'codec': video_stream.get('codec_name', 'unknown'),
                'format': data.get('format', {}).get('format_name', 'unknown')
            }
            
        except Exception as e:
            err = ''
            try:
                err = result.stderr if 'result' in locals() else ''
            except Exception:
                pass
            print(f"Error getting video info: {e} | stderr: {err}")
            return None
    
    @staticmethod
    def get_video_framerate(video_path: str) -> Optional[float]:
        """
        Get video framerate
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Framerate as float or None if failed
        """
        info = VideoUtils.get_video_info(video_path)
        return info.get('fps') if info else None
    
    @staticmethod
    def get_video_resolution(video_path: str) -> Optional[Tuple[int, int]]:
        """
        Get video resolution
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (width, height) or None if failed
        """
        info = VideoUtils.get_video_info(video_path)
        return (info['width'], info['height']) if info else None
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, 
                     start_time: float = 0, end_time: Optional[float] = None,
                     fps: Optional[float] = None) -> List[str]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save frames
            start_time: Start time in seconds
            end_time: End time in seconds (None for entire video)
            fps: Frames per second to extract (None for original fps)
            
        Returns:
            List of extracted frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = ['ffmpeg', '-i', video_path]
        
        if start_time > 0:
            cmd.extend(['-ss', str(start_time)])
        
        if end_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        
        if fps is not None:
            cmd.extend(['-r', str(fps)])
        
        cmd.extend([
            '-q:v', '2',  # High quality
            os.path.join(output_dir, 'frame_%06d.png')
        ])
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
            )
            
            # Get list of extracted frames
            frames = []
            for frame_file in sorted(os.listdir(output_dir)):
                if frame_file.startswith('frame_') and frame_file.endswith('.png'):
                    frames.append(os.path.join(output_dir, frame_file))
            
            return frames
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frames: {e}")
            return []
    
    @staticmethod
    def create_video_from_frames(frame_dir: str, output_path: str,
                               fps: float = 30, codec: str = 'libx264',
                               bitrate: str = '10M', resolution: Optional[Tuple[int, int]] = None) -> bool:
        """
        Create video from frames
        
        Args:
            frame_dir: Directory containing frames
            output_path: Path for output video
            fps: Frames per second
            codec: Video codec
            bitrate: Video bitrate
            resolution: Output resolution (width, height)
            
        Returns:
            True if successful, False otherwise
        """
        # Find frames
        frames = []
        for frame_file in sorted(os.listdir(frame_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frames.append(os.path.join(frame_dir, frame_file))
        
        if not frames:
            print("No frames found")
            return False
        
        # Overwrite if exists and reduce interactivity to avoid hanging
        cmd = ['ffmpeg', '-y', '-r', str(fps), '-i', os.path.join(frame_dir, 'frame_%06d.png')]
        
        if resolution:
            cmd.extend(['-vf', f'scale={resolution[0]}:{resolution[1]}'])
        
        cmd.extend([
            '-c:v', codec,
            '-b:v', bitrate,
            '-pix_fmt', 'yuv420p',
            output_path
        ])
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}")
            return False
    
    @staticmethod
    def create_video_with_alpha(frame_dir: str, output_path: str,
                              fps: float = 30, codec: str = 'libvpx-vp9') -> bool:
        """
        Create video with alpha channel from frames
        
        Args:
            frame_dir: Directory containing frames with alpha
            output_path: Path for output video
            fps: Frames per second
            codec: Video codec (should support alpha)
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            'ffmpeg', '-y', '-r', str(fps), '-i', os.path.join(frame_dir, 'frame_%06d.png'),
            '-c:v', codec,
            '-pix_fmt', 'yuva420p',
            '-auto-alt-ref', '0',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating video with alpha: {e}")
            return False
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """
        Check if FFmpeg and FFprobe are available
        
        Returns:
            True if both are available, False otherwise
        """
        return (shutil.which('ffmpeg') is not None and 
                shutil.which('ffprobe') is not None)