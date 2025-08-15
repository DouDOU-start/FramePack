#!/usr/bin/env python3
"""
Video Cropping Module for RMBG-2.0

Provides functionality to crop videos to specified dimensions and aspect ratios.
Supports WebM format and various output formats.

Author: FramePack Project
"""

import cv2
import os
import tempfile
import shutil
import subprocess
import json
from typing import Tuple, Optional, Callable, Dict, Any
import numpy as np

class VideoCropper:
    """
    Video cropping utility that supports various formats including WebM.
    Provides flexible cropping options including center crop, custom region crop,
    and aspect ratio based cropping.
    """
    
    def __init__(self):
        """Initialize the video cropper."""
        self.supported_formats = ['webm', 'mp4', 'mov', 'avi', 'mkv']
        
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information or None if failed
        """
        try:
            ffprobe_path = shutil.which("ffprobe")
            if not ffprobe_path:
                return None
                
            cmd = [
                ffprobe_path, "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", video_path
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
            
            # Extract video stream info
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
                
            info = {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': float(video_stream.get('duration', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'format': data.get('format', {}).get('format_name', 'unknown')
            }
            
            return info
            
        except Exception as e:
            err = ''
            try:
                err = result.stderr if 'result' in locals() else ''
            except Exception:
                pass
            print(f"Error getting video info: {e} | stderr: {err}")
            return None
    
    def calculate_crop_region(self, 
                            video_width: int, 
                            video_height: int,
                            crop_type: str = "center",
                            target_width: Optional[int] = None,
                            target_height: Optional[int] = None,
                            crop_x: Optional[int] = None,
                            crop_y: Optional[int] = None,
                            aspect_ratio: Optional[str] = None) -> Tuple[int, int, int, int]:
        """
        Calculate the crop region based on specified parameters.
        
        Args:
            video_width: Original video width
            video_height: Original video height
            crop_type: Type of crop ("center", "custom", "aspect_ratio")
            target_width: Target crop width
            target_height: Target crop height
            crop_x: X coordinate for custom crop
            crop_y: Y coordinate for custom crop
            aspect_ratio: Target aspect ratio (e.g., "16:9", "1:1")
            
        Returns:
            Tuple of (x, y, width, height) for crop region
        """
        
        if crop_type == "center":
            # Center crop to target dimensions
            if target_width and target_height:
                crop_width = min(target_width, video_width)
                crop_height = min(target_height, video_height)
            else:
                # Default to smaller dimension for square crop
                crop_width = crop_height = min(video_width, video_height)
                
            x = (video_width - crop_width) // 2
            y = (video_height - crop_height) // 2
            
        elif crop_type == "custom":
            # Custom position crop
            if crop_x is not None and crop_y is not None and target_width and target_height:
                x = max(0, min(crop_x, video_width - target_width))
                y = max(0, min(crop_y, video_height - target_height))
                crop_width = min(target_width, video_width - x)
                crop_height = min(target_height, video_height - y)
            else:
                # Fallback to center crop
                crop_width = crop_height = min(video_width, video_height)
                x = (video_width - crop_width) // 2
                y = (video_height - crop_height) // 2
                
        elif crop_type == "aspect_ratio" and aspect_ratio:
            # Crop to maintain aspect ratio
            if ":" in aspect_ratio:
                ratio_w, ratio_h = map(float, aspect_ratio.split(":"))
                target_ratio = ratio_w / ratio_h
            else:
                target_ratio = float(aspect_ratio)
                
            video_ratio = video_width / video_height
            
            if video_ratio > target_ratio:
                # Video is wider, crop width
                crop_height = video_height
                crop_width = int(video_height * target_ratio)
                x = (video_width - crop_width) // 2
                y = 0
            else:
                # Video is taller, crop height
                crop_width = video_width
                crop_height = int(video_width / target_ratio)
                x = 0
                y = (video_height - crop_height) // 2
        else:
            # Default: no crop (return full video dimensions)
            x, y = 0, 0
            crop_width, crop_height = video_width, video_height
            
        return x, y, crop_width, crop_height
    
    def crop_video(self,
                   input_path: str,
                   output_path: str,
                   crop_type: str = "center",
                   target_width: Optional[int] = None,
                   target_height: Optional[int] = None,
                   crop_x: Optional[int] = None,  
                   crop_y: Optional[int] = None,
                   aspect_ratio: Optional[str] = None,
                   output_format: str = "webm",
                   quality: str = "medium",
                   progress_callback: Optional[Callable] = None) -> str:
        """
        Crop a video file.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            crop_type: Type of crop ("center", "custom", "aspect_ratio")
            target_width: Target crop width
            target_height: Target crop height
            crop_x: X coordinate for custom crop
            crop_y: Y coordinate for custom crop
            aspect_ratio: Target aspect ratio
            output_format: Output format (webm, mp4, mov)
            quality: Quality setting (low, medium, high)
            progress_callback: Optional progress callback function
            
        Returns:
            Path to output video file
        """
        
        # Check if ffmpeg is available
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise Exception("FFmpeg not found. Please install FFmpeg to process videos.")
            
        # Get video information
        video_info = self.get_video_info(input_path)
        if not video_info:
            raise Exception("Could not get video information")
            
        video_width = video_info['width']
        video_height = video_info['height']
        
        # Calculate crop region
        crop_x_calc, crop_y_calc, crop_width, crop_height = self.calculate_crop_region(
            video_width, video_height, crop_type, target_width, target_height,
            crop_x, crop_y, aspect_ratio
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build ffmpeg command
        cmd = [ffmpeg_path, "-y", "-i", input_path]
        
        # Add crop filter
        crop_filter = f"crop={crop_width}:{crop_height}:{crop_x_calc}:{crop_y_calc}"
        cmd.extend(["-vf", crop_filter])
        
        # Set quality and codec based on output format
        if output_format.lower() == "webm":
            if quality == "high":
                cmd.extend(["-c:v", "libvpx-vp9", "-crf", "15", "-b:v", "0"])
            elif quality == "low":
                cmd.extend(["-c:v", "libvpx", "-crf", "35", "-b:v", "1M"])
            else:  # medium
                cmd.extend(["-c:v", "libvpx-vp9", "-crf", "25", "-b:v", "0"])
                
            if video_info.get('codec') == 'vp8' or video_info.get('codec') == 'vp9':
                # Maintain audio if present
                cmd.extend(["-c:a", "libopus"])
            else:
                cmd.extend(["-c:a", "copy"])
                
        elif output_format.lower() == "mp4":
            if quality == "high":
                cmd.extend(["-c:v", "libx264", "-crf", "18"])
            elif quality == "low":
                cmd.extend(["-c:v", "libx264", "-crf", "28"])
            else:  # medium
                cmd.extend(["-c:v", "libx264", "-crf", "23"])
            cmd.extend(["-c:a", "aac"])
            
        elif output_format.lower() == "mov":
            if quality == "high":
                cmd.extend(["-c:v", "libx264", "-crf", "18"])
            elif quality == "low":
                cmd.extend(["-c:v", "libx264", "-crf", "28"])
            else:  # medium
                cmd.extend(["-c:v", "libx264", "-crf", "23"])
            cmd.extend(["-c:a", "aac"])
            
        # Add output path
        cmd.append(output_path)
        
        try:
            if progress_callback:
                # Run with progress monitoring
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='ignore',
                )
                
                # Monitor progress (basic implementation)
                total_frames = int(video_info.get('duration', 0) * video_info.get('fps', 25))
                current_frame = 0
                
                for line in process.stderr:
                    if 'frame=' in line:
                        try:
                            frame_part = line.split('frame=')[1].split()[0]
                            current_frame = int(frame_part)
                            if total_frames > 0:
                                progress = min(100, (current_frame / total_frames) * 100)
                                progress_callback(progress)
                        except:
                            pass
                
                process.wait()
                if process.returncode != 0:
                    raise Exception(f"FFmpeg failed with return code {process.returncode}")
                    
            else:
                # Run without progress monitoring
                _ = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    check=True,
                )
                
            # Verify output file was created
            if not os.path.exists(output_path):
                raise Exception("Output file was not created")
                
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr}" if e.stderr else str(e)
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"Video cropping failed: {str(e)}")
    
    def get_crop_preview(self, 
                        input_path: str,
                        crop_type: str = "center",
                        target_width: Optional[int] = None,
                        target_height: Optional[int] = None,
                        crop_x: Optional[int] = None,
                        crop_y: Optional[int] = None,
                        aspect_ratio: Optional[str] = None,
                        frame_time: float = 1.0) -> Optional[np.ndarray]:
        """
        Get a preview of the crop region from a specific frame.
        
        Args:
            input_path: Path to input video
            crop_type: Type of crop
            target_width: Target crop width
            target_height: Target crop height
            crop_x: X coordinate for custom crop
            crop_y: Y coordinate for custom crop
            aspect_ratio: Target aspect ratio
            frame_time: Time position to extract frame (seconds)
            
        Returns:
            Cropped frame as numpy array or None if failed
        """
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return None
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Seek to specific frame
            target_frame = min(int(frame_time * fps), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
                
            # Calculate crop region
            crop_x_calc, crop_y_calc, crop_width, crop_height = self.calculate_crop_region(
                video_width, video_height, crop_type, target_width, target_height,
                crop_x, crop_y, aspect_ratio
            )
            
            # Crop frame
            cropped_frame = frame[crop_y_calc:crop_y_calc + crop_height, 
                                crop_x_calc:crop_x_calc + crop_width]
            
            return cropped_frame
            
        except Exception as e:
            print(f"Error getting crop preview: {e}")
            return None

    def get_frame(self, video_path: str, frame_time: float = 1.0) -> Optional[np.ndarray]:
        """
        Extract a single frame from a video.

        Args:
            video_path: Path to input video
            frame_time: Time position to extract frame (seconds)

        Returns:
            Frame as numpy array (RGB) or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            target_frame = min(int(frame_time * fps), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None

            # Convert BGR (OpenCV default) to RGB for Gradio
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"Error getting frame: {e}")
            return None

    def generate_preview_frame(self,
                               input_path: str,
                               canvas_w: int,
                               canvas_h: int,
                               video_scale_w: Optional[int] = None,
                               video_scale_h: Optional[int] = None,
                               pos_x: int = 0,
                               pos_y: int = 0,
                               bg_color: str = "#FFFFFF",
                               bg_transparent: bool = False,
                               sizing_mode: str = "裁剪 (Crop)") -> Optional[np.ndarray]:
        """
        Generates a single preview frame by composing a video frame onto a canvas.
        Uses Pillow and OpenCV for fast image manipulation.
        """
        from PIL import Image, ImageDraw, ImageColor, ImageOps

        # Get a frame from the video (e.g., at 0.5s to avoid potential black frames at the start)
        frame_np = self.get_frame(input_path, frame_time=0.5)
        if frame_np is None:
            return None

        frame_pil = Image.fromarray(frame_np)

        # Use provided video dimensions or fallback to original
        if not video_scale_w or video_scale_w <= 0:
            video_scale_w = frame_pil.width
        if not video_scale_h or video_scale_h <= 0:
            video_scale_h = frame_pil.height

        # Process the frame based on the sizing mode
        if "拉伸" in sizing_mode: # Stretch
            processed_frame = frame_pil.resize((video_scale_w, video_scale_h), Image.Resampling.LANCZOS)
        elif "缩放" in sizing_mode: # Scale (Contain/Fit)
            # Create a new transparent image with the target dimensions
            processed_frame = Image.new("RGBA", (video_scale_w, video_scale_h), (0, 0, 0, 0))
            # Create a thumbnail of the original frame that fits within the target dimensions
            thumbnail_frame = frame_pil.copy()
            thumbnail_frame.thumbnail((video_scale_w, video_scale_h), Image.Resampling.LANCZOS)
            # Paste the thumbnail into the center of the transparent frame
            paste_x = (video_scale_w - thumbnail_frame.width) // 2
            paste_y = (video_scale_h - thumbnail_frame.height) // 2
            processed_frame.paste(thumbnail_frame, (paste_x, paste_y))
        else: # Default to Crop (裁剪/Cover)
            processed_frame = ImageOps.fit(frame_pil, (video_scale_w, video_scale_h), Image.Resampling.LANCZOS)

        # Create the background canvas
        canvas_mode = "RGBA"
        if bg_transparent:
            canvas = Image.new(canvas_mode, (canvas_w, canvas_h), (0, 0, 0, 0))
            # Draw checkerboard for visualization
            draw = ImageDraw.Draw(canvas)
            tile_size = 16
            for i in range(0, canvas_w, tile_size):
                for j in range(0, canvas_h, tile_size):
                    if (i // tile_size + j // tile_size) % 2 == 0:
                        draw.rectangle([i, j, i + tile_size, j + tile_size], fill=(230, 230, 230, 255))
                    else:
                        draw.rectangle([i, j, i + tile_size, j + tile_size], fill=(255, 255, 255, 255))
        else:
            try:
                # This needs the same robust parsing as in main.py
                if bg_color.strip().lower().startswith('rgba'):
                    parts = bg_color.strip().lower().replace('rgba(', '').replace(')', '').split(',')
                    r, g, b = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
                    color_rgb = (r, g, b)
                else:
                    color_rgb = ImageColor.getrgb(bg_color)
            except Exception:
                color_rgb = (255, 255, 255) # Fallback to white
            canvas = Image.new(canvas_mode, (canvas_w, canvas_h), color_rgb)

        # Paste the processed frame onto the canvas
        # Pillow's paste handles RGBA pasting correctly
        canvas.paste(processed_frame, (pos_x, pos_y), processed_frame if processed_frame.mode == 'RGBA' else None)

        # Convert back to numpy array for Gradio
        return np.array(canvas.convert("RGB"))

    def create_complex_crop_video(self,
                               input_path: str,
                               output_path: str,
                               output_format: str = "webm",
                               video_scale_w: int = 640,
                               video_scale_h: int = 480,
                               canvas_w: int = 640,
                               canvas_h: int = 480,
                               pos_x: int = 0,
                               pos_y: int = 0,
                               bg_color: str = "transparent",
                               sizing_mode: str = "裁剪 (Crop)",
                               quality: str = "medium",
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Creates a video by scaling the input video and overlaying it on a canvas.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            output_format: Output format (webm, mp4)
            video_scale_w: Width to scale the video to
            video_scale_h: Height to scale the video to
            canvas_w: Width of the background canvas
            canvas_h: Height of the background canvas
            pos_x: X position of the video on the canvas
            pos_y: Y position of the video on the canvas
            bg_color: Background color ('transparent', 'black', '#RRGGBB', etc.)
            quality: Quality setting (low, medium, high)
            progress_callback: Optional progress callback function

        Returns:
            Path to output video file
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise Exception("FFmpeg not found. Please install FFmpeg to process videos.")

        video_info = self.get_video_info(input_path)
        if not video_info:
            raise Exception("Could not get video information")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if video_scale_w > canvas_w or video_scale_h > canvas_h:
             # video resolution is constrained by canvas resolution
             video_scale_w = min(video_scale_w, canvas_w)
             video_scale_h = min(video_scale_h, canvas_h)

        # Define the background color source. Expects 'transparent' or a hex code.
        if bg_color == "transparent":
            color_source = f"color=c=black@0.0:s={canvas_w}x{canvas_h}"
        else:
            # bg_color should be a sanitized hex string
            color_source = f"color=c={bg_color}:s={canvas_w}x{canvas_h}"

        # Add duration to the color source, which is required by some ffmpeg versions
        if video_info.get('duration'):
             color_source += f":d={video_info['duration']}"

        # Define the pixel format for the final output
        pix_fmt = "yuva420p" if output_format.lower() == 'webm' and bg_color == "transparent" else "yuv420p"

        # 1. Define the video scaling filter based on mode
        if "拉伸" in sizing_mode: # Stretch
            scale_filter = f"scale={video_scale_w}:{video_scale_h}"
        elif "缩放" in sizing_mode: # Scale (Contain/Fit)
            scale_filter = f"scale={video_scale_w}:{video_scale_h}:force_original_aspect_ratio=decrease"
        else: # Default to Crop (裁剪/Cover)
            scale_filter = f"scale={video_scale_w}:{video_scale_h}:force_original_aspect_ratio=increase,crop={video_scale_w}:{video_scale_h}"

        # 2. Scale the video and pad it to the target video box size.
        # This creates a video stream of size video_scale_w x video_scale_h.
        # The padding is transparent, so the main canvas will show through for "Scale" mode.
        # We add a +2 over-pad and then crop back to size to avoid 1px rounding errors in ffmpeg's scale filter.
        video_processing_chain = (
            f"{scale_filter},"
            f"pad={video_scale_w}+2:{video_scale_h}+2:-1:-1:color=black@0.0,"
            f"crop={video_scale_w}:{video_scale_h}"
        )

        # 3. Create the main canvas and overlay the processed video stream onto it.
        vf_command = (
            f"{color_source}[bg];"
            f"[0:v]{video_processing_chain}[fg];"
            f"[bg][fg]overlay=x={pos_x}:y={pos_y}"
        )

        cmd = [ffmpeg_path, "-y", "-i", input_path, "-vf", vf_command, "-map", "0:a?"]

        # Codec and quality settings
        if output_format.lower() == "webm":
            cmd.extend(["-c:v", "libvpx-vp9"])
            if quality == "high":
                cmd.extend(["-crf", "15", "-b:v", "0"])
            elif quality == "low":
                cmd.extend(["-crf", "35", "-b:v", "1M"])
            else: # medium
                cmd.extend(["-crf", "25", "-b:v", "0"])
            cmd.extend(["-c:a", "libopus"])
        elif output_format.lower() == "mp4":
            cmd.extend(["-c:v", "libx264"])
            if quality == "high":
                cmd.extend(["-crf", "18"])
            elif quality == "low":
                cmd.extend(["-crf", "28"])
            else: # medium
                cmd.extend(["-crf", "23"])
            cmd.extend(["-c:a", "aac"])
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        cmd.append(output_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=True,
            )
            if not os.path.exists(output_path):
                raise Exception(f"Output file was not created. FFmpeg stderr: {result.stderr}")
            return output_path
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr}" if e.stderr else str(e)
            raise Exception(error_msg)