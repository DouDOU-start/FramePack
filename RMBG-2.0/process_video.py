import os
import sys
import subprocess
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as tv_transforms
from transformers import AutoModelForImageSegmentation

def get_video_framerate(video_path: str) -> str:
    """使用 ffprobe 获取视频的帧率，如果失败则返回默认值。"""
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        print("警告: 未找到 ffprobe。无法确定原始帧率，将默认使用 30fps。")
        return "30"
    try:
        command = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"警告: 使用 ffprobe 获取帧率失败: {e}。将默认使用 30fps。")
        return "30"

def get_video_resolution(video_path: str) -> tuple[int, int] | None:
    """使用 ffprobe 获取视频的分辨率 (height, width)，如果失败则返回 None。"""
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        print("警告: 未找到 ffprobe。无法确定原始分辨率。")
        return None
    try:
        command = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            video_path,
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return height, width
    except Exception as e:
        print(f"警告: 使用 ffprobe 获取分辨率失败: {e}。")
        return None

def process_video(video_path, output_dir, model_path, device, keep_frames, output_format="webm", resolution=None, output_aspect_ratio=None, padding_params=None, progress_callback=None, output_framerate=None):
    """
    处理视频：提取帧、移除背景、保存处理后的帧，并最终合成为具有透明背景的视频。
    
    参数:
        video_path: 输入视频路径
        output_dir: 输出目录
        model_path: RMBG-2.0模型路径
        device: 处理设备 (CPU/GPU)
        keep_frames: 是否保留处理后的帧
        output_format: 输出视频格式 ('webm'/'mov')
        resolution: 目标分辨率 (height, width)
        output_aspect_ratio: 输出宽高比 (w_ratio, h_ratio)
        padding_params: 填充参数字典，优先级最高
        progress_callback: 进度回调函数 (接受0-100的进度值)
        output_framerate: 输出视频的帧率
    """
    # --- 1. Setup Directories ---
    temp_dir = os.path.join(output_dir, "temp_raw_frames")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"临时帧目录: {temp_dir}")
    print(f"最终输出目录: {output_dir}")

    # --- 2. Extract Frames using FFmpeg ---
    print("开始使用 FFmpeg 提取视频帧...")
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("错误: 未在您的系统中找到 ffmpeg。请确保已安装并将其添加至系统路径。")
        return

    # 获取视频原始帧率以用于后续合成
    original_framerate = get_video_framerate(video_path)
    framerate_to_use = output_framerate if output_framerate else original_framerate
    print(f"检测到视频原始帧率: {original_framerate}，输出将使用帧率: {framerate_to_use}")

    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-i", video_path,
                os.path.join(temp_dir, "frame_%06d.png"),
            ],
            check=True, capture_output=True, text=True
        )
        print("视频帧提取成功。")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 运行失败，错误信息:\n{e.stderr}")
        return

    # --- 3. Load RMBG-2.0 Model ---
    print(f"正在从 '{model_path}' 加载 RMBG-2.0 模型...")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        model.to(device=device)
        model.eval()

        # 确定模型输入尺寸 (需要是32的倍数)
        model_input_size = None
        original_resolution = get_video_resolution(video_path)
        
        # 1. 首先确定原始分辨率
        if original_resolution:
            print(f"检测到视频原始分辨率: {original_resolution[1]}x{original_resolution[0]}")
            original_size = original_resolution
        else:
            print("无法检测到视频分辨率，将默认使用 1024x1024。")
            original_size = (1024, 1024)
            
        # 2. 确定目标分辨率 (用于最终输出)
        target_output_size = original_size
        if resolution:
            target_output_size = resolution
            print(f"将使用指定的目标分辨率: {resolution[1]}x{resolution[0]}")
            
        # 3. 为模型计算合适的输入尺寸 (32的倍数)
        h, w = target_output_size
        model_h = (h // 32) * 32
        model_w = (w // 32) * 32
        if model_h == 0 or model_w == 0:  # 避免分辨率过小导致尺寸为0
            model_h = max(((h - 1) // 32 + 1) * 32, 32)
            model_w = max(((w - 1) // 32 + 1) * 32, 32)
        
        model_input_size = (model_h, model_w)
        if model_input_size != target_output_size:
            print(f"为满足模型要求，模型处理分辨率调整为: {model_w}x{model_h}")
        
        # 设置模型变换
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        transform = tv_transforms.Compose([
            tv_transforms.Resize(model_input_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std)
        ])
        print("RMBG-2.0 模型加载成功。")
    except Exception as e:
        print(f"加载 RMBG-2.0 模型失败: {e}")
        return

    # --- 4. Process Each Frame ---
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    print(f"找到 {total_frames} 帧，开始逐帧处理...")
    
    # 如果存在填充参数，输出一次填充信息
    padding_info_printed = False
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="处理帧")):
        frame_path = os.path.join(temp_dir, frame_file)
        
        # 更新进度
        if progress_callback:
            progress = int((idx / total_frames) * 90)  # 预留10%给视频合成
            progress_callback(progress)
            
        try:
            # 1. 读取原始帧
            image_pil = Image.open(frame_path).convert("RGB")
            original_size = image_pil.size  # (width, height)
            
            # 2. 运行模型生成蒙版
            input_tensor = transform(image_pil).unsqueeze(0).to(device=device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        preds = model(input_tensor)[-1]
                else:
                    preds = model(input_tensor)[-1]
                
                preds = preds.sigmoid().to(dtype=torch.float32).cpu()

            # 将蒙版调整回原始图像尺寸
            mask = tv_transforms.ToPILImage()(preds[0].squeeze()).resize(original_size, Image.LANCZOS)
            image_pil.putalpha(mask)
            
            # 3. 统一处理分辨率和宽高比
            # 创建最终输出图像 (使用统一的填充逻辑)
            final_image = image_pil
            
            # 如果有明确的填充参数，优先使用
            if padding_params:
                if not padding_info_printed:
                    print(f"使用填充模式处理宽高比差异")
                    padding_info_printed = True
                    
                # 先调整到缩放尺寸
                scale_width = padding_params['scale_width']
                scale_height = padding_params['scale_height']
                
                # 缩放到填充参数指定的内容尺寸
                if image_pil.size != (scale_width, scale_height):
                    image_pil = image_pil.resize((scale_width, scale_height), Image.LANCZOS)
                
                # 创建目标尺寸的透明画布
                target_width = scale_width + padding_params['pad_left'] + padding_params['pad_right']
                target_height = scale_height + padding_params['pad_top'] + padding_params['pad_bottom']
                
                # 只在第一帧输出详细信息
                if idx == 0:
                    print(f"输出尺寸: {target_width}x{target_height}, 内容尺寸: {scale_width}x{scale_height}")
                    if padding_params['pad_top'] > 0 or padding_params['pad_bottom'] > 0:
                        print(f"添加上下填充: 上={padding_params['pad_top']}, 下={padding_params['pad_bottom']}")
                    if padding_params['pad_left'] > 0 or padding_params['pad_right'] > 0:
                        print(f"添加左右填充: 左={padding_params['pad_left']}, 右={padding_params['pad_right']}")
                
                # 创建透明画布并粘贴内容
                final_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
                paste_pos = (padding_params['pad_left'], padding_params['pad_top'])
                final_image.paste(image_pil, paste_pos, image_pil)
            
            # 处理resolution (无填充参数但有指定分辨率)
            elif resolution:
                target_width, target_height = resolution[1], resolution[0]  # resolution为(height, width)
                
                # 如果原始宽高比与目标不同，创建填充参数
                original_width, original_height = image_pil.size
                original_aspect = original_width / original_height
                target_aspect = target_width / target_height
                
                if abs(original_aspect - target_aspect) > 0.01:
                    # 优先保持内容完整，使用填充
                    if original_aspect > target_aspect:
                        # 原始比例更宽，使用上下填充
                        content_width = target_width
                        content_height = int(target_width / original_aspect)
                        pad_top = (target_height - content_height) // 2
                        pad_bottom = target_height - content_height - pad_top
                        
                        if not padding_info_printed and idx == 0:
                            print(f"使用自动填充: 内容尺寸={content_width}x{content_height}, 上填充={pad_top}, 下填充={pad_bottom}")
                            padding_info_printed = True
                        
                        # 先缩放内容
                        image_pil = image_pil.resize((content_width, content_height), Image.LANCZOS)
                        
                        # 创建透明画布并粘贴
                        final_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
                        final_image.paste(image_pil, (0, pad_top), image_pil)
                    else:
                        # 原始比例更高，使用左右填充
                        content_height = target_height
                        content_width = int(target_height * original_aspect)
                        pad_left = (target_width - content_width) // 2
                        pad_right = target_width - content_width - pad_left
                        
                        if not padding_info_printed and idx == 0:
                            print(f"使用自动填充: 内容尺寸={content_width}x{content_height}, 左填充={pad_left}, 右填充={pad_right}")
                            padding_info_printed = True
                        
                        # 先缩放内容
                        image_pil = image_pil.resize((content_width, content_height), Image.LANCZOS)
                        
                        # 创建透明画布并粘贴
                        final_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
                        final_image.paste(image_pil, (pad_left, 0), image_pil)
                else:
                    # 宽高比相同，直接缩放
                    if not padding_info_printed and idx == 0:
                        print(f"直接缩放到目标分辨率: {target_width}x{target_height}")
                        padding_info_printed = True
                    final_image = image_pil.resize((target_width, target_height), Image.LANCZOS)
            
            # 处理output_aspect_ratio (无填充参数，无指定分辨率，但有指定宽高比)
            elif output_aspect_ratio:
                w, h = image_pil.size
                target_w_ratio, target_h_ratio = output_aspect_ratio
                target_aspect = target_w_ratio / target_h_ratio
                current_aspect = w / h
                
                if abs(current_aspect - target_aspect) > 0.01:
                    if not padding_info_printed and idx == 0:
                        print(f"调整宽高比从 {current_aspect:.3f} 到 {target_aspect:.3f}")
                        padding_info_printed = True
                    
                    if current_aspect > target_aspect:
                        # 当前图像比目标"宽"，保持宽度并增加高度
                        new_w = w
                        new_h = int(w / target_aspect)
                        paste_pos = (0, (new_h - h) // 2)
                    else:
                        # 当前图像比目标"高"，保持高度并增加宽度
                        new_h = h
                        new_w = int(h * target_aspect)
                        paste_pos = ((new_w - w) // 2, 0)
                    
                    # 创建透明画布并粘贴
                    canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                    canvas.paste(image_pil, paste_pos, image_pil)
                    final_image = canvas
            
            # 保存处理后的帧
            output_path = os.path.join(output_dir, frame_file)
            final_image.save(output_path)
            
        except Exception as e:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"处理帧 '{frame_file}' 时出错: {e}")

    # --- 5. Synthesize Transparent Video ---
    print(f"\n开始合成具有透明背景的 {output_format.upper()} 视频...")
    output_video_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_transparent.{output_format}"
    output_video_path = os.path.join(output_dir, output_video_filename)
    input_frames_pattern = os.path.join(output_dir, "frame_%06d.png")

    try:
        command = [
            ffmpeg_path,
            "-framerate", framerate_to_use,
            "-i", input_frames_pattern,
        ]

        if output_format == 'webm':
            command.extend([
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-b:v", "0",
                "-crf", "30",
                "-auto-alt-ref", "0",
            ])
        elif output_format == 'mov':
            # 使用 ProRes 编码器，支持 alpha 通道
            command.extend([
                "-c:v", "prores_ks",
                "-pix_fmt", "yuva444p10le",
                "-profile:v", "4444", 
                "-qscale:v", "9" # qscale 范围 1-31, 数字越小质量越高
            ])

        command.extend([
            "-y",
            output_video_path,
        ])
        print(f"运行 FFmpeg 命令: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"成功创建透明视频: {output_video_path}")
        
        # 更新进度到100%
        if progress_callback:
            progress_callback(100)
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 视频合成失败，错误信息:\n{e.stderr}")
        # Even if synthesis fails, continue to cleanup
    
    # --- 6. Cleanup ---
    if not keep_frames:
        print("正在清理已处理的 PNG 帧...")
        for frame_file in tqdm(frame_files, desc="清理帧"):
            file_to_delete = os.path.join(output_dir, frame_file)
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)

    print("正在清理临时文件...")
    shutil.rmtree(temp_dir)
    print("处理完成！")


'''
使用示例：
python process_video.py --input "../inputs/华佗.mp4" --output "../outputs/华佗" --model-path "../hf_download/RMBG-2.0" --gpu 0 --keep-frames
python process_video.py --input "../inputs/华佗.mp4" --output "../outputs/华佗" --model-path "../hf_download/RMBG-2.0" --gpu 0 --output-format mov
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 RMBG-2.0 对视频进行逐帧抠图，并合成为透明背景的视频。")
    parser.add_argument("--input", type=str, required=True, help="输入视频文件的路径。")
    parser.add_argument("--output", type=str, required=True, help="保存输出视频和（可选的）处理后PNG帧的目录。")
    parser.add_argument("--model-path", type=str, default="../hf_download/RMBG-2.0", help="RMBG-2.0 模型所在的路径。")
    parser.add_argument("--gpu", type=int, default=0, help="要使用的GPU索引。如果为-1，则使用CPU。")
    parser.add_argument(
        "--keep-frames",
        action="store_true", # 当出现这个参数时，其值为 True
        help="如果设置此项，将在视频合成后保留所有处理过的PNG帧。默认不保留。"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=['webm', 'mov'],
        default='webm',
        help="输出视频的格式。可选 'webm' 或 'mov'。默认为 'webm'。"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help="指定处理视频时的分辨率 (e.g., --resolution 1920 1080)。如果未指定，则自动检测原始分辨率。"
    )
    parser.add_argument(
        "--output-aspect-ratio",
        type=float,
        nargs=2,
        metavar=('W_RATIO', 'H_RATIO'),
        help="设置输出视频的宽高比，例如 --output-aspect-ratio 1 4。多余部分将以透明填充。"
    )
    parser.add_argument(
        "--output-framerate",
        type=int,
        help="指定输出视频的帧率。如果未指定，则回退到使用输入视频的帧率。"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入视频文件不存在: {args.input}")
        sys.exit(1)

    resolution_arg = None
    if args.resolution:
        resolution_arg = (args.resolution[1], args.resolution[0]) # (height, width) for torchvision

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"将使用 GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("将使用 CPU。")
    
    process_video(args.input, args.output, args.model_path, device, args.keep_frames, args.output_format, resolution_arg, args.output_aspect_ratio, output_framerate=args.output_framerate)