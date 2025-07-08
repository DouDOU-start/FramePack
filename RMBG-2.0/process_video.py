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

def process_video(video_path, output_dir, model_path, device, keep_frames):
    """
    处理视频：提取帧、移除背景、保存处理后的帧，并最终合成为具有透明背景的视频。
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
    framerate = get_video_framerate(video_path)
    print(f"检测到视频帧率: {framerate}")

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

        image_size = (1024, 1024)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        transform = tv_transforms.Compose([
            tv_transforms.Resize(image_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std)
        ])
        print("RMBG-2.0 模型加载成功。")
    except Exception as e:
        print(f"加载 RMBG-2.0 模型失败: {e}")
        return

    # --- 4. Process Each Frame ---
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
    print(f"找到 {len(frame_files)} 帧，开始逐帧处理...")

    for frame_file in tqdm(frame_files, desc="处理帧"):
        frame_path = os.path.join(temp_dir, frame_file)
        try:
            image_pil = Image.open(frame_path).convert("RGB")
            input_tensor = transform(image_pil).unsqueeze(0).to(device=device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        preds = model(input_tensor)[-1]
                else:
                    preds = model(input_tensor)[-1]
                
                preds = preds.sigmoid().to(dtype=torch.float32).cpu()

            mask = tv_transforms.ToPILImage()(preds[0].squeeze()).resize(image_pil.size, Image.LANCZOS)
            image_pil.putalpha(mask)
            output_path = os.path.join(output_dir, frame_file)
            image_pil.save(output_path)
        except Exception as e:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"处理帧 '{frame_file}' 时出错: {e}")

    # --- 5. Synthesize Transparent Video ---
    print("\n开始合成具有透明背景的 WebM 视频...")
    output_video_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_transparent.webm"
    output_video_path = os.path.join(output_dir, output_video_filename)
    input_frames_pattern = os.path.join(output_dir, "frame_%06d.png")

    try:
        command = [
            ffmpeg_path,
            "-framerate", framerate,
            "-i", input_frames_pattern,
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-b:v", "0",
            "-crf", "30",
            "-auto-alt-ref", "0",
            "-y",
            output_video_path,
        ]
        print(f"运行 FFmpeg 命令: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"成功创建透明视频: {output_video_path}")
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
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 RMBG-2.0 对视频进行逐帧抠图，并合成为透明背景的 WebM 视频。")
    parser.add_argument("--input", type=str, required=True, help="输入视频文件的路径。")
    parser.add_argument("--output", type=str, required=True, help="保存输出视频和（可选的）处理后PNG帧的目录。")
    parser.add_argument("--model-path", type=str, default="../hf_download/RMBG-2.0", help="RMBG-2.0 模型所在的路径。")
    parser.add_argument("--gpu", type=int, default=0, help="要使用的GPU索引。如果为-1，则使用CPU。")
    parser.add_argument(
        "--keep-frames",
        action="store_true", # 当出现这个参数时，其值为 True
        help="如果设置此项，将在视频合成后保留所有处理过的PNG帧。默认不保留。"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入视频文件不存在: {args.input}")
        sys.exit(1)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"将使用 GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("将使用 CPU。")
    
    process_video(args.input, args.output, args.model_path, device, args.keep_frames)