#!/usr/bin/env python3
"""
RMBG-2.0 Gradio Web Application

A comprehensive web interface for RMBG-2.0 background processing capabilities.
Provides image background removal, color replacement, and video processing.

Author: FramePack Project
License: See LICENSE file
"""

import os
import sys
import gradio as gr
import torch
import numpy as np
import argparse
import tempfile
import shutil
from PIL import Image
import warnings
import logging

# Suppress specific warnings and errors
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
logging.getLogger("h11").setLevel(logging.CRITICAL)

# Add RMBG-2.0 to path
rmbg_path = os.path.join(os.path.dirname(__file__), '..', 'RMBG-2.0')
sys.path.append(rmbg_path)

from remove_bg import BackgroundRemover
from change_bg_color import BackgroundColorChanger

# Global variables for model instances
bg_remover = None
bg_color_changer = None
model_path = "./hf_download/RMBG-2.0"

def initialize_models():
    """Initialize the RMBG-2.0 models"""
    global bg_remover, bg_color_changer
    
    if not os.path.exists(model_path):
        return False, f"Model path not found: {model_path}"
    
    try:
        print("Initializing RMBG-2.0 models...")
        bg_remover = BackgroundRemover(model_path)
        bg_color_changer = BackgroundColorChanger(model_path)
        print("Models initialized successfully!")
        return True, "Models loaded successfully"
    except Exception as e:
        return False, f"Failed to load models: {str(e)}"

def remove_background(input_image):
    """Remove background from image"""
    if input_image is None:
        return None, "Please upload an image"
    
    if bg_remover is None:
        success, message = initialize_models()
        if not success:
            return None, message
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Ensure RGB format
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Get mask using the processor
        mask = bg_remover.processor.get_mask(input_image)
        
        # Apply the mask to create transparent background
        input_image.putalpha(mask)
        
        return input_image, "Background removed successfully!"
        
    except Exception as e:
        return None, f"Error removing background: {str(e)}"

def change_background_color(input_image, bg_color):
    """Change background color of image"""
    if input_image is None:
        return None, "Please upload an image"
    
    if bg_color_changer is None:
        success, message = initialize_models()
        if not success:
            return None, message
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Ensure RGB format
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Get mask using the processor
        mask = bg_color_changer.processor.get_mask(input_image)
        
        # Create new background with specified color
        background = Image.new('RGB', input_image.size, bg_color)
        
        # Composite the original image onto the new background using the mask
        result_image = Image.composite(input_image, background, mask)
        
        return result_image, f"Background changed to {bg_color} successfully!"
        
    except Exception as e:
        return None, f"Error changing background color: {str(e)}"

def process_video_background(input_video, output_format, keep_frames):
    """Process video to remove background"""
    if input_video is None:
        return None, "Please upload a video"
    
    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            return None, "FFmpeg not found. Please install FFmpeg to process videos."
        
        # Process video using the existing process_video function
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Import and use the process_video function
        from process_video import process_video
        
        process_video(
            video_path=input_video,
            output_dir=output_dir,
            model_path=model_path,
            device=device,
            keep_frames=keep_frames,
            output_format=output_format
        )
        
        # Find the output video file
        output_files = [f for f in os.listdir(output_dir) if f.endswith(f'.{output_format}')]
        if output_files:
            output_video_path = os.path.join(output_dir, output_files[0])
            # Copy to a permanent location
            final_output = f"./outputs/processed_video.{output_format}"
            os.makedirs("./outputs", exist_ok=True)
            shutil.copy2(output_video_path, final_output)
            return final_output, f"Video processed successfully! Output format: {output_format}"
        else:
            return None, "Video processing failed - no output file generated"
            
    except Exception as e:
        return None, f"Error processing video: {str(e)}"
    finally:
        # Clean up temporary directory
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

# Color presets for easy selection
COLOR_PRESETS = {
    "White": "#FFFFFF",
    "Black": "#000000", 
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "Purple": "#800080",
    "Orange": "#FFA500",
    "Pink": "#FFC0CB",
    "Gray": "#808080"
}

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="RMBG-2.0 Background Processing", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 RMBG-2.0 Background Processing Tool")
        gr.Markdown("Remove backgrounds, change background colors, or process videos with transparent backgrounds using RMBG-2.0")
        
        with gr.Tabs():
            # Tab 1: Background Removal
            with gr.TabItem("🗑️ Remove Background"):
                with gr.Row():
                    with gr.Column():
                        input_img_remove = gr.Image(
                            label="Upload Image", 
                            type="numpy",
                            height=400
                        )
                        remove_btn = gr.Button("Remove Background", variant="primary")
                        
                    with gr.Column():
                        output_img_remove = gr.Image(
                            label="Result (Transparent Background)", 
                            height=400
                        )
                        status_remove = gr.Textbox(
                            label="Status", 
                            interactive=False
                        )
                
                remove_btn.click(
                    fn=remove_background,
                    inputs=[input_img_remove],
                    outputs=[output_img_remove, status_remove]
                )
            
            # Tab 2: Background Color Change
            with gr.TabItem("🎨 Change Background Color"):
                with gr.Row():
                    with gr.Column():
                        input_img_color = gr.Image(
                            label="Upload Image", 
                            type="numpy",
                            height=400
                        )
                        
                        with gr.Row():
                            color_preset = gr.Dropdown(
                                choices=list(COLOR_PRESETS.keys()),
                                label="Color Presets",
                                value="White"
                            )
                            custom_color = gr.ColorPicker(
                                label="Custom Color",
                                value="#FFFFFF"
                            )
                        
                        change_color_btn = gr.Button("Change Background Color", variant="primary")
                        
                    with gr.Column():
                        output_img_color = gr.Image(
                            label="Result", 
                            height=400
                        )
                        status_color = gr.Textbox(
                            label="Status", 
                            interactive=False
                        )
                
                def update_color_from_preset(preset):
                    return COLOR_PRESETS.get(preset, "#FFFFFF")
                
                color_preset.change(
                    fn=update_color_from_preset,
                    inputs=[color_preset],
                    outputs=[custom_color]
                )
                
                change_color_btn.click(
                    fn=change_background_color,
                    inputs=[input_img_color, custom_color],
                    outputs=[output_img_color, status_color]
                )
            
            # Tab 3: Video Processing
            with gr.TabItem("🎬 Video Background Removal"):
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(
                            label="Upload Video",
                            height=400
                        )
                        
                        with gr.Row():
                            output_format = gr.Radio(
                                choices=["webm", "mov"],
                                label="Output Format",
                                value="webm",
                                info="WebM for web use, MOV for professional editing"
                            )
                            keep_frames = gr.Checkbox(
                                label="Keep Individual Frames",
                                value=False,
                                info="Save processed frames as PNG files"
                            )
                        
                        process_video_btn = gr.Button("Process Video", variant="primary")
                        
                    with gr.Column():
                        output_video = gr.Video(
                            label="Processed Video (Transparent Background)",
                            height=400
                        )
                        status_video = gr.Textbox(
                            label="Status", 
                            interactive=False
                        )
                
                process_video_btn.click(
                    fn=process_video_background,
                    inputs=[input_video, output_format, keep_frames],
                    outputs=[output_video, status_video]
                )
        
        # Footer with information
        gr.Markdown("""
        ---
        ### 📝 Usage Notes:
        - **Background Removal**: Creates images with transparent backgrounds (PNG format)
        - **Color Change**: Replaces background with solid colors
        - **Video Processing**: Requires FFmpeg installation for video processing
        - **Supported Formats**: Images (JPG, PNG, etc.), Videos (MP4, MOV, etc.)
        
        ### ⚙️ Requirements:
        - RMBG-2.0 model files in `./hf_download/RMBG-2.0/`
        - FFmpeg for video processing
        - CUDA GPU recommended for faster processing
        """)
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMBG-2.0 Gradio Interface")
    parser.add_argument('--share', action='store_true', help='Share the interface publicly')
    parser.add_argument("--server", type=str, default='127.0.0.1', help='Server address')
    parser.add_argument("--port", type=int, default=7860, help='Server port')
    parser.add_argument("--inbrowser", action='store_true', help='Open in browser automatically')
    parser.add_argument("--model-path", type=str, default="./hf_download/RMBG-2.0", help='Path to RMBG-2.0 model')

    args = parser.parse_args()

    # Update model path if provided
    model_path = args.model_path

    # Initialize models on startup
    print("🚀 Starting RMBG-2.0 Gradio Interface...")
    success, message = initialize_models()
    if not success:
        print(f"Warning: {message}")
        print("Models will be loaded when first used.")

    # Create and launch interface
    try:
        demo = create_gradio_interface()
        print(f"🌐 Starting server at http://{args.server}:{args.port}")

        # Use queue to handle concurrent requests better
        demo.queue(max_size=10)

        demo.launch(
            server_name=args.server,
            server_port=args.port,
            share=args.share,
            inbrowser=args.inbrowser,
            prevent_thread_lock=False,
            show_error=False,  # Don't show errors in interface to avoid protocol issues
            quiet=False  # Keep some logging for debugging
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {str(e)}")
        sys.exit(1)
