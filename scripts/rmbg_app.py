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

def process_video_background(input_video, output_format, keep_frames, output_framerate, resolution_setting, custom_width, custom_height, aspect_ratio_setting, custom_aspect_width, custom_aspect_height, progress=gr.Progress()):
    """Process video to remove background"""
    if input_video is None:
        return None, "请上传一个视频"

    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            return None, "未找到FFmpeg。请安装FFmpeg以处理视频。"

        # Get original video resolution to maintain aspect ratio
        def get_video_info(video_path):
            """Get video resolution using ffprobe"""
            import subprocess
            try:
                ffprobe_path = shutil.which("ffprobe")
                if not ffprobe_path:
                    return None, None

                result = subprocess.run([
                    ffprobe_path, "-v", "quiet", "-print_format", "json",
                    "-show_streams", video_path
                ], capture_output=True, text=True, check=True)

                import json
                data = json.loads(result.stdout)
                for stream in data['streams']:
                    if stream['codec_type'] == 'video':
                        width = int(stream['width'])
                        height = int(stream['height'])
                        return width, height
                return None, None
            except:
                return None, None

        original_width, original_height = get_video_info(input_video)
        original_aspect_ratio = None
        if original_width and original_height:
            original_aspect_ratio = original_width / original_height
        
        # Step 1: Determine base resolution from resolution_setting
        base_width = None
        base_height = None

        if resolution_setting == "Original":
            base_width = original_width
            base_height = original_height
        elif resolution_setting == "Custom":
            if custom_width and custom_height and custom_width > 0 and custom_height > 0:
                base_width = int(custom_width)
                base_height = int(custom_height)
            else:
                return None, "请输入有效的自定义宽度和高度值"
        else:
            # Parse preset resolution (e.g., "1920x1080")
            preset_width, preset_height = map(int, resolution_setting.split('x'))
            base_width = preset_width
            base_height = preset_height

        if not base_width or not base_height: # Fallback
            if original_width and original_height:
                base_width, base_height = original_width, original_height
            else:
                return None, "无法确定视频的原始尺寸"

        # Step 2: Determine target aspect ratio and calculate final dimensions
        target_aspect_ratio = None
        final_width = base_width
        final_height = base_height

        if aspect_ratio_setting == "Custom":
            if custom_aspect_width and custom_aspect_height and custom_aspect_width > 0 and custom_aspect_height > 0:
                target_aspect_ratio = custom_aspect_width / custom_aspect_height
            else:
                return None, "请输入有效的自定义宽高比值"
        elif aspect_ratio_setting != "Original": # Preset aspect ratio
            aspect_parts = aspect_ratio_setting.split(':')
            if len(aspect_parts) == 2:
                aspect_w, aspect_h = map(float, aspect_parts)
                target_aspect_ratio = aspect_w / aspect_h

        if target_aspect_ratio:
            # 以基础高度为准，根据新的宽高比调整宽度
            final_height = base_height 
            final_width = int(final_height * target_aspect_ratio)

        # 最终输出分辨率
        resolution = (final_height, final_width)

        # Step 3: Calculate padding_params to fit original content into the final canvas
        padding_params = None
        if original_aspect_ratio:
            final_aspect_ratio = final_width / final_height
            
            # 如果内容和画布的宽高比不同，则需要填充
            if abs(original_aspect_ratio - final_aspect_ratio) > 0.01:
                
                if original_aspect_ratio > final_aspect_ratio:
                    # 内容比画布宽，适配宽度，上下填充
                    scale_width = final_width
                    scale_height = int(final_width / original_aspect_ratio)
                    pad_top = (final_height - scale_height) // 2
                    pad_bottom = final_height - scale_height - pad_top
                    padding_params = {
                        'scale_width': scale_width, 'scale_height': scale_height,
                        'pad_left': 0, 'pad_right': 0,
                        'pad_top': pad_top, 'pad_bottom': pad_bottom
                    }
                else:
                    # 内容比画布高，适配高度，左右填充
                    scale_height = final_height
                    scale_width = int(final_height * original_aspect_ratio)
                    pad_left = (final_width - scale_width) // 2
                    pad_right = final_width - scale_width - pad_left
                    padding_params = {
                        'scale_width': scale_width, 'scale_height': scale_height,
                        'pad_left': pad_left, 'pad_right': pad_right,
                        'pad_top': 0, 'pad_bottom': 0
                    }

        # Process video using the existing process_video function
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Import and use the process_video function
        from process_video import process_video
        
        # 定义进度回调函数来更新Gradio进度条
        def update_progress(percent):
            progress(percent / 100)

        framerate_to_pass = None
        if output_framerate != "Original":
            framerate_to_pass = str(output_framerate)
        
        process_video(
            video_path=input_video,
            output_dir=output_dir,
            model_path=model_path,
            device=device,
            keep_frames=keep_frames,
            output_format=output_format,
            resolution=resolution,
            padding_params=padding_params,
            progress_callback=update_progress,
            output_framerate=framerate_to_pass
        )
        
        # Find the output video file
        output_files = [f for f in os.listdir(output_dir) if f.endswith(f'.{output_format}')]
        if output_files:
            output_video_path = os.path.join(output_dir, output_files[0])
            # Copy to a permanent location
            final_output = f"./outputs/processed_video.{output_format}"
            os.makedirs("./outputs", exist_ok=True)
            shutil.copy2(output_video_path, final_output)

            # Verify actual output file format
            actual_format = output_format
            try:
                import subprocess
                ffprobe_path = shutil.which("ffprobe")
                if ffprobe_path:
                    verify_cmd = [ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_format", final_output]
                    result = subprocess.run(verify_cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        import json
                        format_info = json.loads(result.stdout)
                        format_name = format_info.get('format', {}).get('format_name', '')
                        if output_format == 'mov' and 'mov' in format_name:
                            actual_format = f"{output_format} ✅"
                        elif output_format == 'mov' and 'mp4' in format_name:
                            actual_format = f"{output_format} ⚠️ (实际为MP4)"
                        elif output_format == 'webm' and 'webm' in format_name:
                            actual_format = f"{output_format} ✅"
            except:
                pass

            # Create status message with resolution and aspect ratio info
            status_msg = f"视频处理成功! 输出格式: {actual_format}"

            # Add resolution info
            status_msg += f" | 分辨率: {final_width}x{final_height}"

            # Add aspect ratio and framerate info
            current_aspect_ratio = final_width / final_height
            status_msg += f" | 宽高比: {current_aspect_ratio:.3f}:1 ({aspect_ratio_setting})"
            status_msg += f" | 帧率: {output_framerate} FPS"


            # Add padding info
            if padding_params:
                if padding_params['pad_top'] > 0 or padding_params['pad_bottom'] > 0:
                    status_msg += " | 处理方式: 上下填充以保持原始内容比例"
                elif padding_params['pad_left'] > 0 or padding_params['pad_right'] > 0:
                    status_msg += " | 处理方式: 左右填充以保持原始内容比例"
            else:
                status_msg += " | 处理方式: 保持原始宽高比"

            # 对于MOV格式，提供下载链接以避免Gradio自动转换
            if output_format == 'mov':
                return final_output, final_output, status_msg + " | 💡 MOV文件请使用下载链接获取原始格式"
            else:
                return final_output, None, status_msg
        else:
            return None, None, "视频处理失败 - 未生成输出文件"
            
    except Exception as e:
        return None, None, f"处理视频时出错: {str(e)}"
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
    
    with gr.Blocks(title="RMBG-2.0 背景处理工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 RMBG-2.0 背景处理工具")
        gr.Markdown("移除背景、更换背景颜色或处理带有透明背景的视频，使用 RMBG-2.0")
        
        with gr.Tabs():
            # Tab 1: Background Removal
            with gr.TabItem("🗑️ 移除背景"):
                with gr.Row():
                    with gr.Column():
                        input_img_remove = gr.Image(
                            label="上传图片", 
                            type="numpy",
                            height=400
                        )
                        remove_btn = gr.Button("移除背景", variant="primary")
                        
                    with gr.Column():
                        output_img_remove = gr.Image(
                            label="结果（透明背景）", 
                            height=400
                        )
                        status_remove = gr.Textbox(
                            label="状态", 
                            interactive=False
                        )
                
                remove_btn.click(
                    fn=remove_background,
                    inputs=[input_img_remove],
                    outputs=[output_img_remove, status_remove]
                )
            
            # Tab 2: Background Color Change
            with gr.TabItem("🎨 更换背景颜色"):
                with gr.Row():
                    with gr.Column():
                        input_img_color = gr.Image(
                            label="上传图片", 
                            type="numpy",
                            height=400
                        )
                        
                        with gr.Row():
                            color_preset = gr.Dropdown(
                                choices=list(COLOR_PRESETS.keys()),
                                label="颜色预设",
                                value="White"
                            )
                            custom_color = gr.ColorPicker(
                                label="自定义颜色",
                                value="#FFFFFF"
                            )
                        
                        change_color_btn = gr.Button("更换背景颜色", variant="primary")
                        
                    with gr.Column():
                        output_img_color = gr.Image(
                            label="结果", 
                            height=400
                        )
                        status_color = gr.Textbox(
                            label="状态", 
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
            with gr.TabItem("🎬 视频背景移除"):
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(
                            label="上传视频",
                            height=400
                        )
                        
                        with gr.Row():
                            output_format = gr.Radio(
                                choices=["webm", "mov"],
                                label="输出格式",
                                value="webm",
                                info="WebM用于网页，MOV用于专业编辑"
                            )
                            keep_frames = gr.Checkbox(
                                label="保留单独帧",
                                value=False,
                                info="将处理后的帧保存为PNG文件"
                            )

                        output_framerate_selector = gr.Radio(
                            choices=["Original", "24", "25", "30", "60"],
                            label="输出帧率 (FPS)",
                            value="Original",
                            info="选择输出视频的帧率"
                        )

                        # Resolution settings
                        with gr.Group():
                            gr.Markdown("### 📐 输出分辨率")
                            resolution_setting = gr.Radio(
                                choices=[
                                    "Original",
                                    "3840x2160",  # 4K UHD
                                    "2560x1440",  # 2K QHD
                                    "1920x1080",  # Full HD
                                    "1280x720",   # HD
                                    "854x480",    # 480p
                                    "640x360",    # 360p
                                    "Custom"
                                ],
                                label="分辨率",
                                value="Original",
                                info="选择输出视频分辨率（将根据宽高比进行填充调整）"
                            )

                            with gr.Row(visible=False) as custom_resolution_row:
                                custom_width = gr.Number(
                                    label="目标宽度",
                                    value=1920,
                                    minimum=1,
                                    maximum=4096,
                                    step=1,
                                    info="将根据宽高比进行填充调整"
                                )
                                custom_height = gr.Number(
                                    label="目标高度",
                                    value=1080,
                                    minimum=1,
                                    maximum=4096,
                                    step=1,
                                    info="将根据宽高比进行填充调整"
                                )

                            # Show/hide custom resolution inputs based on selection
                            def toggle_custom_resolution(choice):
                                return gr.update(visible=(choice == "Custom"))

                            resolution_setting.change(
                                fn=toggle_custom_resolution,
                                inputs=[resolution_setting],
                                outputs=[custom_resolution_row]
                            )

                        # Aspect ratio settings
                        with gr.Group():
                            gr.Markdown("### 📏 输出宽高比")
                            aspect_ratio_setting = gr.Radio(
                                choices=[
                                    "Original",
                                    "16:9",      # Widescreen
                                    "9:16",      # Vertical/Mobile
                                    "4:3",       # Traditional TV
                                    "3:4",       # Vertical traditional
                                    "21:9",      # Ultra-wide
                                    "1:1",       # Square
                                    "2:1",       # Cinema
                                    "Custom"
                                ],
                                label="宽高比",
                                value="Original",
                                info="选择输出视频宽高比（将使用填充而非剪裁）"
                            )

                            with gr.Row(visible=False) as custom_aspect_row:
                                custom_aspect_width = gr.Number(
                                    label="宽度比例",
                                    value=16,
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    info="宽度比例值（例如16:9中的16）"
                                )
                                custom_aspect_height = gr.Number(
                                    label="高度比例",
                                    value=9,
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    info="高度比例值（例如16:9中的9）"
                                )

                            # Show/hide custom aspect ratio inputs based on selection
                            def toggle_custom_aspect_ratio(choice):
                                return gr.update(visible=(choice == "Custom"))

                            aspect_ratio_setting.change(
                                fn=toggle_custom_aspect_ratio,
                                inputs=[aspect_ratio_setting],
                                outputs=[custom_aspect_row]
                            )
                        
                        process_video_btn = gr.Button("处理视频", variant="primary")
                        
                    with gr.Column():
                        output_video = gr.Video(
                            label="处理后的视频（透明背景）",
                            height=400
                        )

                        # 添加下载链接组件（特别是对于MOV格式）
                        download_link = gr.File(
                            label="下载原始格式视频",
                            visible=False
                        )

                        status_video = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                
                # 使用进度条功能调用处理函数
                process_video_btn.click(
                    fn=process_video_background,
                    inputs=[input_video, output_format, keep_frames, output_framerate_selector, resolution_setting, custom_width, custom_height, aspect_ratio_setting, custom_aspect_width, custom_aspect_height],
                    outputs=[output_video, download_link, status_video],
                    show_progress=True
                )

                # 根据输出格式显示/隐藏下载链接
                def toggle_download_link(format_choice):
                    if format_choice == 'mov':
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)

                output_format.change(
                    fn=toggle_download_link,
                    inputs=[output_format],
                    outputs=[download_link]
                )
        
        # Footer with information
        gr.Markdown("""
        ---
        ### 📝 使用说明:
        - **背景移除**: 创建带有透明背景的图像（PNG格式）
        - **颜色更换**: 用纯色替换背景
        - **视频处理**: 需要安装FFmpeg才能处理视频
        - **支持格式**: 图像（JPG, PNG等），视频（MP4, MOV等）
        - **MOV格式**: 由于浏览器兼容性，MOV文件请使用下载链接获取原始格式

        ### 🎞️ 帧率设置:
        - **Original**: 使用原始视频的帧率。
        - **24 FPS**: 电影标准帧率，具有电影感。
        - **30 FPS**: 标准视频帧率，适用于大多数网络视频。
        - **60 FPS**: 高帧率，画面更流畅，适合游戏录像或慢动作。

        ### 📐 分辨率设置:
        - **原始**: 保持原始视频分辨率（当宽高比也为原始时）
        - **预设分辨率**: 目标分辨率将根据所选宽高比进行填充调整
          - **4K UHD (3840x2160)**: 超高清，需要强大的GPU
          - **2K QHD (2560x1440)**: 四倍高清，质量和性能的良好平衡
          - **Full HD (1920x1080)**: 标准高清
          - **HD (1280x720)**: 标准清晰度，处理更快
          - **480p/360p**: 更低分辨率，用于快速处理或小文件
        - **自定义**: 设置目标宽度和高度（将根据所选宽高比进行填充调整）
        - **注意**: 更高的分辨率需要更多的处理时间和GPU内存

        ### 📏 宽高比设置:
        - **原始**: 保持原始视频宽高比（无剪裁或填充）
        - **16:9**: 标准宽屏格式（现代视频最常见）
        - **9:16**: 垂直格式（适合移动/社交媒体）
        - **4:3**: 传统电视格式
        - **3:4**: 垂直传统格式
        - **21:9**: 超宽电影格式
        - **1:1**: 正方形格式（适合Instagram帖子）
        - **2:1**: 电影格式
        - **自定义**: 设置自己的宽高比（宽度:高度比）
        - **重要**: 改变宽高比可能会对视频内容进行填充，但不会剪裁

        ### ⚙️ 要求:
        - RMBG-2.0 模型文件位于 `./hf_download/RMBG-2.0/`
        - 需要安装 FFmpeg 处理视频
        - 推荐使用 CUDA GPU 以加快处理速度
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
