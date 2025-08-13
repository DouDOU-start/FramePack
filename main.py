from diffusers_helper.hf_login import login

import os
import sys
import shutil

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

"""
RMBG-2.0 integration imports
Because the package directory contains a dash, import modules via sys.path.
"""
sys.path.append(os.path.join(os.path.dirname(__file__), 'RMBG-2.0'))
try:
    import background_processor as rmbg_bg
    from video_cropper import VideoCropper as RMBGVideoCropper
except Exception as _rmbg_import_err:
    rmbg_bg = None
    RMBGVideoCropper = None
    print(f"[RMBG] Import warning: {_rmbg_import_err}")


def main():
    global high_vram, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer, stream, outputs_folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument("--server", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, required=False)
    parser.add_argument("--inbrowser", action='store_true')
    args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

    print(args)

    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    print(f'Free VRAM {free_mem_gb} GB')
    print(f'High-VRAM Mode: {high_vram}')

    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')

    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    if not high_vram:
        # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)

    stream = AsyncStream()

    outputs_folder = './outputs/'
    os.makedirs(outputs_folder, exist_ok=True)

    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


# ============================
# RMBG-2.0 integration helpers
# ============================

def _ensure_outputs_dir(subdir: str) -> str:
    base = os.path.join('./outputs', subdir)
    os.makedirs(base, exist_ok=True)
    return base


def _normalize_video_input(value):
    """Accept gr.Video input variants and return a file path string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # gradio may pass dict-like with 'name' or 'path'
    try:
        if isinstance(value, dict):
            return value.get('name') or value.get('path') or value.get('video') or None
    except Exception:
        pass
    return None


# RMBG lifecycle cache
_rmbg_loaded_key = None  # (model_path, device_str)


def _get_rmbg_model_path() -> str:
    env_path = os.environ.get('RMBG_MODEL_PATH')
    if env_path and os.path.isdir(env_path):
        return env_path
    local_candidate = os.path.join(os.path.dirname(__file__), 'hf_download', 'RMBG-2.0')
    if os.path.isdir(local_candidate):
        return local_candidate
    return 'briaai/RMBG-2.0'


def _resolve_device(device_sel: str | None) -> str | None:
    if not device_sel or device_sel == 'auto':
        return None
    if device_sel == 'cuda':
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return 'cuda:0'
        except Exception:
            pass
        return 'cuda'  # fallback
    return device_sel


def rmbg_load(device_sel: str) -> str:
    global _rmbg_loaded_key
    if rmbg_bg is None:
        return 'RMBG 模块不可用'
    model_path = _get_rmbg_model_path()
    device = _resolve_device(device_sel)
    key = (model_path, device or 'auto')
    try:
        # 预检：模型路径
        if model_path and model_path.strip() and os.path.isdir(model_path) is False and model_path.find('/') == -1:
            # 像 'briaai/RMBG-2.0' 这样的远程ID在离线环境会失败
            return f'❌ 本地未找到模型目录：{model_path}。请设置环境变量 RMBG_MODEL_PATH 指向本地模型目录。'

        # 预检：设备可用性
        if device and device.startswith('cuda') and not torch.cuda.is_available():
            return '❌ CUDA 不可用：未检测到可用 GPU 或 CUDA 驱动。'

        if getattr(rmbg_bg, '_global_processor', None) is not None:
            if _rmbg_loaded_key == key:
                return f'✅ 已加载：{model_path} ({device or "auto"})'
            # 设备或模型变化，需要卸载后重载
            try:
                rmbg_bg._global_processor.cleanup()
            except Exception:
                pass
            rmbg_bg._global_processor = None
        # 加载
        rmbg_bg.get_processor(model_path, device)
        _rmbg_loaded_key = key
        return f'✅ 已加载：{model_path} ({device or "auto"})'
    except Exception as e:
        return f'❌ 加载失败：{e}'


def rmbg_unload() -> str:
    global _rmbg_loaded_key
    if rmbg_bg is None:
        return ''
    try:
        if getattr(rmbg_bg, '_global_processor', None) is not None:
            rmbg_bg._global_processor.cleanup()
            rmbg_bg._global_processor = None
            _rmbg_loaded_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return '🧹 已卸载 RMBG 模型，释放显存'
    except Exception as e:
        return f'⚠️ 卸载提示：{e}'
    return ''


def _get_rmbg_model_path() -> str:
    # 优先环境变量
    env_path = os.environ.get('RMBG_MODEL_PATH')
    if env_path and os.path.isdir(env_path):
        return env_path
    # 尝试本地缓存目录
    local_candidate = os.path.join(os.path.dirname(__file__), 'hf_download', 'RMBG-2.0')
    if os.path.isdir(local_candidate):
        return local_candidate
    # 回退到 HF 模型ID（如需完全离线，请设置 RMBG_MODEL_PATH）
    return 'briaai/RMBG-2.0'


def rmbg_process_image(image_path: str, background_color: str, device_sel: str, background_image_path: str | None = None):
    """Image background remove/replace using RMBG-2.0.

    Returns a file path for display/download.
    """
    if rmbg_bg is None:
        raise RuntimeError('RMBG-2.0 未正确导入，无法处理图像。')

    assert image_path, '请上传图片文件'
    model_id_or_path = _get_rmbg_model_path()
    device = None if device_sel == 'auto' else device_sel

    processor = rmbg_bg.get_processor(model_id_or_path, device)
    try:
        out_dir = _ensure_outputs_dir('rmbg_images')
        out_path = os.path.join(out_dir, f"image_{os.path.basename(image_path)}")

        if background_image_path:
            try:
                bg_img = Image.open(background_image_path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f'背景图片加载失败：{e}')
            result = processor.replace_background_image(image_path, bg_img, out_path)
        elif background_color and background_color.strip():
            result = processor.replace_background_image(image_path, background_color.strip(), out_path)
        else:
            result = processor.remove_background_image(image_path, out_path)

        return result
    finally:
        processor.cleanup()


def rmbg_process_video(video_path: str, background_color: str,
                       res_w: float | None, res_h: float | None, fps: float | None, device_sel: str,
                       background_image_path: str | None = None):
    """Video background remove/replace using RMBG-2.0.

    Returns (output_video_path, status_text)
    """
    if rmbg_bg is None:
        raise RuntimeError('RMBG-2.0 未正确导入，无法处理视频。')

    video_path = _normalize_video_input(video_path)
    assert video_path, '请上传视频文件'
    model_id_or_path = _get_rmbg_model_path()
    device = None if device_sel == 'auto' else device_sel

    processor = rmbg_bg.get_processor(model_id_or_path, device)
    try:
        out_dir = _ensure_outputs_dir('rmbg_videos')
        stem = os.path.splitext(os.path.basename(video_path))[0]
        # 如果仅抠图透明，使用支持 alpha 的 webm；有背景颜色或背景图片则输出 mp4
        use_bg_image = bool(background_image_path)
        is_transparent = not (use_bg_image or (background_color and background_color.strip()))
        ext = '.webm' if is_transparent else '.mp4'
        out_path = os.path.join(out_dir, f"{stem}_rmbg{ext}")

        resolution = None
        if res_w and res_h and res_w > 0 and res_h > 0:
            resolution = (int(res_w), int(res_h))

        fps_val = float(fps) if fps and fps > 0 else None

        def _progress(pct, msg):
            # This simple callback does not stream to Gradio; keep for future live updates.
            print(f"[RMBG][Video] {pct:.1f}% {msg}")

        if is_transparent:
            result = processor.remove_background_video(
                video_path, out_path, resolution, fps_val, _progress
            )
        else:
            if use_bg_image:
                try:
                    bg_img = Image.open(background_image_path).convert('RGB')
                except Exception as e:
                    raise RuntimeError(f'背景图片加载失败：{e}')
                result = processor.replace_background_video(
                    video_path, bg_img, out_path, resolution, fps_val, _progress
                )
            else:
                result = processor.replace_background_video(
                    video_path, background_color.strip(), out_path, resolution, fps_val, _progress
                )

        return result, '处理完成'
    finally:
        processor.cleanup()


from PIL import ImageColor

# ============================
# FFmpeg Canvas Cropping Helpers
# ============================

def _convert_color_to_ffmpeg_hex(color_spec: str) -> str:
    """Converts a color specifier from Gradio to an FFmpeg-compatible hex string."""
    if not color_spec:
        return '#FFFFFF'  # Default to white

    color_spec = color_spec.strip().lower()

    if color_spec.startswith('rgba'):
        try:
            parts = color_spec.replace('rgba(', '').replace(')', '').split(',')
            r = int(float(parts[0]))
            g = int(float(parts[1]))
            b = int(float(parts[2]))
            # Ensure values are within 0-255 range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, IndexError):
            return '#FFFFFF'  # Fallback

    # It might already be a hex string or a named color that ffmpeg understands.
    # We will return it as is.
    return color_spec

def _create_bg_image(width, height, color_spec, transparent):
    """Creates a background image for the editor."""
    if transparent:
        # Create a checkerboard pattern for transparency visualization
        bg = Image.new('RGBA', (int(width), int(height)), (0, 0, 0, 0))
        pixels = bg.load()
        for i in range(int(width)):
            for j in range(int(height)):
                if (i // 16 + j // 16) % 2 == 0:
                    pixels[i, j] = (230, 230, 230, 255)
                else:
                    pixels[i, j] = (255, 255, 255, 255)
        return bg
    else:
        color_rgb = None
        try:
            # Handle rgba() strings from the color picker, which may contain floats
            if color_spec.strip().lower().startswith('rgba'):
                parts = color_spec.strip().lower().replace('rgba(', '').replace(')', '').split(',')
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                color_rgb = (r, g, b)
            else:
                # This handles hex codes (#RRGGBB) and named colors (e.g., "white")
                color_rgb = ImageColor.getrgb(color_spec)
        except (ValueError, IndexError):
            # Fallback to white if parsing fails for any reason
            color_rgb = (255, 255, 255)

        return Image.new("RGB", (int(width), int(height)), color_rgb)

def _update_crop_editor(video_path, canvas_w, canvas_h, bg_color_spec, bg_transparent):
    """
    Callback to update the ImageEditor when video or canvas settings change.
    """
    video_path = _normalize_video_input(video_path)

    # Create a background image based on settings
    bg = _create_bg_image(canvas_w, canvas_h, bg_color_spec, bg_transparent)

    if not video_path:
        # No video, just return a blank canvas
        return gr.update(value={"background": bg, "layers": [], "composite": None})

    # A video is present, extract a frame
    cropper = RMBGVideoCropper()
    frame = cropper.get_frame(video_path)

    if frame is None:
        # This can happen if the video is invalid
        return gr.update(value={"background": bg, "layers": [], "composite": None})

    # Convert numpy frame to PIL Image before passing to editor
    frame_pil = Image.fromarray(frame)

    # Return the background and the video frame as a new layer
    return gr.update(value={"background": bg, "layers": [frame_pil], "composite": None})

def _ffmpeg_canvas_crop_video(video_path, editor_data, canvas_w, canvas_h, bg_color_spec, bg_transparent, output_format, quality):
    """
    The main processing function for the new cropping UI.
    """
    video_path = _normalize_video_input(video_path)
    if not video_path or not editor_data or not editor_data['layers']:
        raise gr.Error("请上传视频并将视频帧拖到画布上进行编辑。")

    # Extract geometry from the ImageEditor's output
    # The layer is a PIL Image. Geometry is in the .info dict, but only if the user interacts.
    layer_image = editor_data['layers'][0]
    layer_info = layer_image.info

    # Use .get() to provide default values if keys are missing (i.e., user didn't move/resize).
    pos_x = layer_info.get('left', 0)
    pos_y = layer_info.get('top', 0)
    video_scale_w = layer_info.get('width', layer_image.width)
    video_scale_h = layer_info.get('height', layer_image.height)

    cropper = RMBGVideoCropper()
    out_dir = _ensure_outputs_dir('rmbg_crops')
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_canvas_crop.{output_format}")

    if bg_transparent:
        bg_color_for_ffmpeg = "transparent"
    else:
        bg_color_for_ffmpeg = _convert_color_to_ffmpeg_hex(bg_color_spec)

    try:
        result = cropper.create_complex_crop_video(
            input_path=video_path,
            output_path=out_path,
            output_format=output_format,
            video_scale_w=int(video_scale_w),
            video_scale_h=int(video_scale_h),
            canvas_w=int(canvas_w),
            canvas_h=int(canvas_h),
            pos_x=int(pos_x),
            pos_y=int(pos_y),
            bg_color=bg_color_for_ffmpeg,
            quality=quality,
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"视频裁剪失败: {e}")

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('## FramePack 工具集')
    with gr.Tabs():
        # ============= Tab 1: FramePack 视频生成 =============
        with gr.TabItem('视频生成'):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources='upload', type="numpy", label="输入图片", height=320)
                    prompt = gr.Textbox(label="提示词", value='')
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label='快速提示', samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)
                    with gr.Row():
                        start_button = gr.Button(value="开始生成")
                        end_button = gr.Button(value="结束生成", interactive=False)
                    with gr.Group():
                        use_teacache = gr.Checkbox(label='使用 TeaCache', value=True, info='更快，但可能稍微影响手部细节')
                        with gr.Accordion('高级设置', open=False):
                            n_prompt = gr.Textbox(label="反向提示词", value="", visible=False)
                            seed = gr.Number(label="随机种子", value=31337, precision=0)
                            total_second_length = gr.Slider(label="视频总时长（秒）", minimum=1, maximum=120, value=5, step=0.1)
                            latent_window_size = gr.Slider(label="潜变量窗口（固定）", minimum=1, maximum=33, value=9, step=1, visible=False)
                            steps = gr.Slider(label="步数", minimum=1, maximum=100, value=25, step=1, info='不建议修改')
                            cfg = gr.Slider(label="CFG（固定）", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                            gs = gr.Slider(label="蒸馏 CFG", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='不建议修改')
                            rs = gr.Slider(label="CFG Re-Scale（固定）", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                            gpu_memory_preservation = gr.Slider(label="GPU 预留显存（GB）（越大越慢）", minimum=6, maximum=128, value=6, step=0.1, info="OOM 则调大；数值越大速度越慢")
                            mp4_crf = gr.Slider(label="MP4 压缩（0=无压缩；值越低质量越好）", minimum=0, maximum=100, value=16, step=1)
                with gr.Column():
                    with gr.Group():
                        result_video = gr.Video(label="生成视频", autoplay=True, show_share_button=False, height=512, loop=True)
                        preview_image = gr.Image(label="采样过程预览", height=200, visible=False)
                    with gr.Group():
                        gr.Markdown('提示：反向采样会先生成结尾动作，起始动作稍后呈现。')
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            # gr.Markdown('——')
            ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf]
            start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
            end_button.click(fn=end_process)
        # ============= Tab 2: RMBG-2.0 =============
        with gr.TabItem('RMBG-2.0'):
            gr.Markdown('### 背景处理工具')
            with gr.Group():
                with gr.Row():
                    rmbg_device = gr.Dropdown(choices=['auto', 'cuda', 'cpu'], value='auto', label='设备', scale=1)
                    btn_load = gr.Button('加载模型', variant='primary', scale=0)
                    btn_unload = gr.Button('卸载模型', scale=0)
                rmbg_status = gr.Markdown('')
            def _on_rmbg_tab_enter(device):
                return rmbg_load(device)
            def _on_rmbg_tab_leave():
                return rmbg_unload()
            def _on_rmbg_device_change(device):
                unload_msg = rmbg_unload()
                load_msg = rmbg_load(device)
                sep = '\n\n' if unload_msg and load_msg else ''
                return f"{unload_msg}{sep}{load_msg}"
            with gr.Tabs():
                # 子页：图片抠图 / 背景替换
                with gr.TabItem('图片抠图 / 背景替换'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                rmbg_image_input = gr.Image(sources='upload', type='filepath', label='上传图片')
                            with gr.Group():
                                gr.Markdown('背景设定（图片优先于颜色）')
                                with gr.Row():
                                    rmbg_bg_color = gr.Textbox(label='背景颜色', placeholder='如：white, #00ff00, rgb(0,255,0)')
                                    color_quick = gr.Radio(choices=['white','black','green','blue','red','gray'], label='常用颜色', value=None)
                                rmbg_bg_img = gr.Image(sources='upload', type='filepath', label='背景图片（可选）', height=160)
                            with gr.Row():
                                rmbg_image_btn = gr.Button('处理图像', variant='primary')
                                btn_clear_img = gr.Button('清空')
                        with gr.Column(scale=1):
                            rmbg_image_output = gr.Image(type='filepath', label='结果图片', height=360)
                    def _set_color(c):
                        return c or ''
                    color_quick.change(fn=_set_color, inputs=[color_quick], outputs=[rmbg_bg_color])
                    def _clear_img():
                        return gr.update(value=None)
                    btn_clear_img.click(fn=_clear_img, outputs=[rmbg_image_output])
                    rmbg_image_btn.click(
                        fn=rmbg_process_image,
                        inputs=[rmbg_image_input, rmbg_bg_color, rmbg_device, rmbg_bg_img],
                        outputs=[rmbg_image_output],
                    )
                # 子页：视频抠图 / 背景替换
                with gr.TabItem('视频抠图 / 背景替换'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                rmbg_video_input = gr.Video(label='上传视频', height=240)
                            with gr.Group():
                                gr.Markdown('背景设定（图片优先于颜色）')
                                with gr.Row():
                                    rmbg_bg_color_v = gr.Textbox(label='背景颜色', placeholder='如：white, #00ff00, rgb(0,255,0)')
                                    color_quick_v = gr.Radio(choices=['white','black','green','blue','red','gray'], label='常用颜色', value=None)
                                rmbg_bg_img_v = gr.Image(sources='upload', type='filepath', label='背景图片（可选）', height=160)
                            with gr.Accordion('输出参数（可选）', open=False):
                                with gr.Row():
                                    rmbg_res_w = gr.Number(label='输出宽度', value=None)
                                    rmbg_res_h = gr.Number(label='输出高度', value=None)
                                    rmbg_fps = gr.Number(label='输出FPS', value=None)
                            with gr.Row():
                                rmbg_video_btn = gr.Button('开始处理视频', variant='primary')
                                btn_clear_vid = gr.Button('清空')
                        with gr.Column(scale=1):
                            rmbg_video_status = gr.Markdown('')
                            rmbg_video_output = gr.Video(label='处理后视频', autoplay=False, height=360)
                    color_quick_v.change(fn=_set_color, inputs=[color_quick_v], outputs=[rmbg_bg_color_v])
                    def _clear_vid():
                        return gr.update(value=None), gr.update(value='')
                    btn_clear_vid.click(fn=_clear_vid, outputs=[rmbg_video_output, rmbg_video_status])
                    def _rmbg_video_wrapper(video_path, model_id, bg_color, w, h, fps, device):
                        try:
                            out, status = rmbg_process_video(video_path, model_id, bg_color, w, h, fps, device)
                            return out, status
                        except Exception as e:
                            return None, f"❌ 错误：{e}"
                    def _rmbg_video_wrapper_with_bg_img(video_path, bg_color_v, w, h, fps, device, bg_img):
                        try:
                            out, status = rmbg_process_video(video_path, bg_color_v, w, h, fps, device, bg_img)
                            return out, status
                        except Exception as e:
                            return None, f"❌ 错误：{e}"
                    rmbg_video_btn.click(
                        fn=_rmbg_video_wrapper_with_bg_img,
                        inputs=[rmbg_video_input, rmbg_bg_color_v, rmbg_res_w, rmbg_res_h, rmbg_fps, rmbg_device, rmbg_bg_img_v],
                        outputs=[rmbg_video_output, rmbg_video_status],
                    )
            # 绑定手动加载/卸载按钮
            btn_load.click(fn=rmbg_load, inputs=[rmbg_device], outputs=[rmbg_status])
            btn_unload.click(fn=lambda: rmbg_unload(), outputs=[rmbg_status])
            # 自动进入/离开 Tab 的加载/卸载（通过交互触发，避免空转）
            # 当用户更改设备或点击任一子页控件时触发一次加载
            rmbg_device.change(fn=_on_rmbg_device_change, inputs=[rmbg_device], outputs=[rmbg_status])
        # ============= Tab 3: 视频裁剪（FFmpeg） =============
        with gr.TabItem('视频裁剪（FFmpeg）') as tab_ffmpeg_crop:
            ff_ok = (shutil.which('ffmpeg') is not None) and (shutil.which('ffprobe') is not None)
            ff_tip = '✅ 已检测到 FFmpeg/FFprobe' if ff_ok else '⚠️ 未检测到 FFmpeg/FFprobe，请先安装并加入系统 PATH'
            gr.Markdown(ff_tip)

            gr.Markdown("""
            ### 使用指南
            1.  **上传视频**: 点击下方“上传视频”区域，选择一个视频文件。
            2.  **调整画布与背景**: 使用滑块和颜色选择器设置画布尺寸和背景。
            3.  **编辑视频位置与尺寸**:
                *   视频的第一帧会自动加载到右侧的编辑器中。
                *   **移动**: 直接在编辑器中拖动视频。
                *   **缩放**: 首先**点击**视频（会出现边框），然后拖动边角或边缘的控制点来调整大小。
            4.  **生成视频**: 设置好输出格式和画质后，点击“生成视频”按钮。
            """)

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ff_video_in = gr.Video(label='1. 上传视频', sources='upload', height=240)
                    with gr.Accordion('2. 画布和背景设置', open=True):
                        with gr.Row():
                            ff_canvas_w = gr.Slider(label='画布宽度', minimum=256, maximum=2048, value=1280, step=16, interactive=True)
                            ff_canvas_h = gr.Slider(label='画布高度', minimum=256, maximum=2048, value=720, step=16, interactive=True)
                        with gr.Row():
                            ff_bg_color = gr.ColorPicker(label='背景颜色', value='#FFFFFF', interactive=True)
                            ff_bg_transparent = gr.Checkbox(label='透明背景', value=False, interactive=True)

                    with gr.Accordion('4. 输出设置', open=True):
                        with gr.Row():
                            ff_out_fmt = gr.Dropdown(choices=['webm', 'mp4'], value='webm', label='输出格式')
                            ff_quality = gr.Dropdown(choices=['low', 'medium', 'high'], value='medium', label='画质')

                    ff_crop_btn = gr.Button('生成视频', variant='primary')

                with gr.Column(scale=2):
                    gr.Markdown("### 3. 编辑区域")
                    ff_editor = gr.ImageEditor(
                        label="在下方画布中调整视频位置和尺寸",
                        height=600,
                        type="pil",
                        interactive=True
                    )

            with gr.Row():
                ff_cropped_video = gr.Video(label='裁剪结果', height=360, interactive=False, width='100%')

            # Event handlers for the new cropper
            editor_inputs = [ff_video_in, ff_canvas_w, ff_canvas_h, ff_bg_color, ff_bg_transparent]

            # When the tab is selected, initialize the editor
            tab_ffmpeg_crop.select(
                fn=_update_crop_editor,
                inputs=editor_inputs,
                outputs=[ff_editor]
            )

            # Any change in inputs should trigger an editor update
            for comp in editor_inputs:
                comp.change(
                    fn=_update_crop_editor,
                    inputs=editor_inputs,
                    outputs=[ff_editor]
                )

            ff_crop_btn.click(
                fn=_ffmpeg_canvas_crop_video,
                inputs=[ff_video_in, ff_editor, ff_canvas_w, ff_canvas_h, ff_bg_color, ff_bg_transparent, ff_out_fmt, ff_quality],
                outputs=[ff_cropped_video],
            )
 


if __name__ == '__main__':
    main()