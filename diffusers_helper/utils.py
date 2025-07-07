import os
import cv2
import json
import random
import glob
import torch
import einops
import numpy as np
import datetime
import torchvision

import safetensors.torch as sf
from PIL import Image


def min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def d_resize(x, y):
    H, W, C = y.shape
    new_min = min(H, W)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (W, H), interpolation=interpolation)
    return y


def resize_and_center_crop(image, target_width, target_height):
    # 检查是否有Alpha通道
    has_alpha = image.shape[2] == 4
    
    if has_alpha:
        # 分离RGB和Alpha通道
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]  # 注意：这里不保留额外的维度
        
        # 处理RGB部分
        h, w = rgb.shape[:2]
        
        # 计算调整大小后的尺寸
        if h * target_width > w * target_height:  # 如果原始图像更高
            new_h = int(target_height * w / target_width)
            new_w = w
            top = (h - new_h) // 2
            left = 0
        else:  # 如果原始图像更宽
            new_h = h
            new_w = int(target_width * h / target_height)
            top = 0
            left = (w - new_w) // 2
        
        # 裁剪
        rgb_cropped = rgb[top:top + new_h, left:left + new_w]
        alpha_cropped = alpha[top:top + new_h, left:left + new_w]
        
        # 调整大小
        rgb_resized = cv2.resize(rgb_cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        alpha_resized = cv2.resize(alpha_cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 确保alpha_resized是3D数组，最后一维为1
        alpha_resized = alpha_resized[:, :, np.newaxis]
        
        # 合并通道
        result = np.concatenate([rgb_resized, alpha_resized], axis=2)
        return result
    else:
        # 原始的RGB处理逻辑
        h, w = image.shape[:2]
        
        # 计算调整大小后的尺寸
        if h * target_width > w * target_height:  # 如果原始图像更高
            new_h = int(target_height * w / target_width)
            new_w = w
            top = (h - new_h) // 2
            left = 0
        else:  # 如果原始图像更宽
            new_h = h
            new_w = int(target_width * h / target_height)
            top = 0
            left = (w - new_w) // 2
        
        # 裁剪并调整大小
        cropped = image[top:top + new_h, left:left + new_w]
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        return resized


def resize_and_center_crop_pytorch(image, target_width, target_height):
    B, C, H, W = image.shape

    if H == target_height and W == target_width:
        return image

    scale_factor = max(target_width / W, target_height / H)
    resized_width = int(round(W * scale_factor))
    resized_height = int(round(H * scale_factor))

    resized = torch.nn.functional.interpolate(image, size=(resized_height, resized_width), mode='bilinear', align_corners=False)

    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    cropped = resized[:, :, top:top + target_height, left:left + target_width]

    return cropped


def resize_without_crop(image, target_width, target_height):
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


def just_crop(image, w, h):
    if h == image.shape[0] and w == image.shape[1]:
        return image

    original_height, original_width = image.shape[:2]
    k = min(original_height / h, original_width / w)
    new_width = int(round(w * k))
    new_height = int(round(h * k))
    x_start = (original_width - new_width) // 2
    y_start = (original_height - new_height) // 2
    cropped_image = image[y_start:y_start + new_height, x_start:x_start + new_width]
    return cropped_image


def write_to_json(data, file_path):
    temp_file_path = file_path + ".tmp"
    with open(temp_file_path, 'wt', encoding='utf-8') as temp_file:
        json.dump(data, temp_file, indent=4)
    os.replace(temp_file_path, file_path)
    return


def read_from_json(file_path):
    with open(file_path, 'rt', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_active_parameters(m):
    return {k: v for k, v in m.named_parameters() if v.requires_grad}


def cast_training_params(m, dtype=torch.float32):
    result = {}
    for n, param in m.named_parameters():
        if param.requires_grad:
            param.data = param.to(dtype)
            result[n] = param
    return result


def separate_lora_AB(parameters, B_patterns=None):
    parameters_normal = {}
    parameters_B = {}

    if B_patterns is None:
        B_patterns = ['.lora_B.', '__zero__']

    for k, v in parameters.items():
        if any(B_pattern in k for B_pattern in B_patterns):
            parameters_B[k] = v
        else:
            parameters_normal[k] = v

    return parameters_normal, parameters_B


def set_attr_recursive(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)
    return


def print_tensor_list_size(tensors):
    total_size = 0
    total_elements = 0

    if isinstance(tensors, dict):
        tensors = tensors.values()

    for tensor in tensors:
        total_size += tensor.nelement() * tensor.element_size()
        total_elements += tensor.nelement()

    total_size_MB = total_size / (1024 ** 2)
    total_elements_B = total_elements / 1e9

    print(f"Total number of tensors: {len(tensors)}")
    print(f"Total size of tensors: {total_size_MB:.2f} MB")
    print(f"Total number of parameters: {total_elements_B:.3f} billion")
    return


@torch.no_grad()
def batch_mixture(a, b=None, probability_a=0.5, mask_a=None):
    batch_size = a.size(0)

    if b is None:
        b = torch.zeros_like(a)

    if mask_a is None:
        mask_a = torch.rand(batch_size) < probability_a

    mask_a = mask_a.to(a.device)
    mask_a = mask_a.reshape((batch_size,) + (1,) * (a.dim() - 1))
    result = torch.where(mask_a, a, b)
    return result


@torch.no_grad()
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


@torch.no_grad()
def supress_lower_channels(m, k, alpha=0.01):
    data = m.weight.data.clone()

    assert int(data.shape[1]) >= k

    data[:, :k] = data[:, :k] * alpha
    m.weight.data = data.contiguous().clone()
    return m


def freeze_module(m):
    if not hasattr(m, '_forward_inside_frozen_module'):
        m._forward_inside_frozen_module = m.forward
    m.requires_grad_(False)
    m.forward = torch.no_grad()(m.forward)
    return m


def get_latest_safetensors(folder_path):
    safetensors_files = glob.glob(os.path.join(folder_path, '*.safetensors'))

    if not safetensors_files:
        raise ValueError('No file to resume!')

    latest_file = max(safetensors_files, key=os.path.getmtime)
    latest_file = os.path.abspath(os.path.realpath(latest_file))
    return latest_file


def generate_random_prompt_from_tags(tags_str, min_length=3, max_length=32):
    tags = tags_str.split(', ')
    tags = random.sample(tags, k=min(random.randint(min_length, max_length), len(tags)))
    prompt = ', '.join(tags)
    return prompt


def interpolate_numbers(a, b, n, round_to_int=False, gamma=1.0):
    numbers = a + (b - a) * (np.linspace(0, 1, n) ** gamma)
    if round_to_int:
        numbers = np.round(numbers).astype(int)
    return numbers.tolist()


def uniform_random_by_intervals(inclusive, exclusive, n, round_to_int=False):
    edges = np.linspace(0, 1, n + 1)
    points = np.random.uniform(edges[:-1], edges[1:])
    numbers = inclusive + (exclusive - inclusive) * points
    if round_to_int:
        numbers = np.round(numbers).astype(int)
    return numbers.tolist()


def soft_append_bcthw(history, current, overlap=0):
    """
    将当前帧与历史帧软连接，处理重叠区域
    
    history: [b, c, t, h, w] 历史帧
    current: [b, c, t, h, w] 当前帧
    overlap: 重叠帧数
    """
    if overlap <= 0:
        return torch.cat([history, current], dim=2)
    
    assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
    assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
    
    # 确保通道数匹配
    if current.shape[1] != history.shape[1]:
        # 如果通道数不同，可能是一个有Alpha通道而另一个没有
        # 在这种情况下，我们需要为没有Alpha通道的张量添加一个
        if current.shape[1] == 3 and history.shape[1] == 4:
            b, c, t, h, w = current.shape
            alpha_channel = torch.ones((b, 1, t, h, w), device=current.device, dtype=current.dtype)
            current = torch.cat([current, alpha_channel], dim=1)
        elif current.shape[1] == 4 and history.shape[1] == 3:
            b, c, t, h, w = history.shape
            alpha_channel = torch.ones((b, 1, t, h, w), device=history.device, dtype=history.dtype)
            history = torch.cat([history, alpha_channel], dim=1)
    
    # 创建线性权重用于混合
    weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
    
    # 混合重叠区域
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output.to(history)


def save_bcthw_as_mp4(x, output_filename, fps=10, crf=0):
    b, c, t, h, w = x.shape

    per_row = b
    for p in [6, 5, 4, 3, 2]:
        if b % p == 0:
            per_row = p
            break

    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    torchvision.io.write_video(output_filename, x, fps=fps, video_codec='libx264', options={'crf': str(int(crf))})
    return x


def save_bcthw_as_png(x, output_filename):
    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, 'b c t h w -> c (b h) (t w)')
    torchvision.io.write_png(x, output_filename)
    return output_filename


def save_bchw_as_png(x, output_filename):
    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, 'b c h w -> c h (b w)')
    torchvision.io.write_png(x, output_filename)
    return output_filename


def add_tensors_with_padding(tensor1, tensor2):
    if tensor1.shape == tensor2.shape:
        return tensor1 + tensor2

    shape1 = tensor1.shape
    shape2 = tensor2.shape

    new_shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))

    padded_tensor1 = torch.zeros(new_shape)
    padded_tensor2 = torch.zeros(new_shape)

    padded_tensor1[tuple(slice(0, s) for s in shape1)] = tensor1
    padded_tensor2[tuple(slice(0, s) for s in shape2)] = tensor2

    result = padded_tensor1 + padded_tensor2
    return result


def print_free_mem():
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    free_mem_mb = free_mem / (1024 ** 2)
    total_mem_mb = total_mem / (1024 ** 2)
    print(f"Free memory: {free_mem_mb:.2f} MB")
    print(f"Total memory: {total_mem_mb:.2f} MB")
    return


def print_gpu_parameters(device, state_dict, log_count=1):
    summary = {"device": device, "keys_count": len(state_dict)}

    logged_params = {}
    for i, (key, tensor) in enumerate(state_dict.items()):
        if i >= log_count:
            break
        logged_params[key] = tensor.flatten()[:3].tolist()

    summary["params"] = logged_params

    print(str(summary))
    return


def visualize_txt_as_img(width, height, text, font_path='font/DejaVuSans.ttf', size=18):
    from PIL import Image, ImageDraw, ImageFont

    txt = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype(font_path, size=size)

    if text == '':
        return np.array(txt)

    # Split text into lines that fit within the image width
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        line_with_word = f"{current_line} {word}"
        if draw.textbbox((0, 0), line_with_word, font=font)[2] <= width:
            current_line = line_with_word
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)

    # Draw the text line by line
    y = 0
    line_height = draw.textbbox((0, 0), "A", font=font)[3]

    for line in lines:
        if y + line_height > height:
            break  # stop drawing if the next line will be outside the image
        draw.text((0, y), line, fill="black", font=font)
        y += line_height

    return np.array(txt)


def blue_mark(x):
    x = x.copy()
    c = x[:, :, 2]
    b = cv2.blur(c, (9, 9))
    x[:, :, 2] = ((c - b) * 16.0 + b).clip(-1, 1)
    return x


def green_mark(x):
    x = x.copy()
    x[:, :, 2] = -1
    x[:, :, 0] = -1
    return x


def frame_mark(x):
    x = x.copy()
    x[:64] = -1
    x[-64:] = -1
    x[:, :8] = 1
    x[:, -8:] = 1
    return x


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


@torch.no_grad()
def duplicate_prefix_to_suffix(x, count, zero_out=False):
    if zero_out:
        return torch.cat([x, torch.zeros_like(x[:count])], dim=0)
    else:
        return torch.cat([x, x[:count]], dim=0)


def weighted_mse(a, b, weight):
    return torch.mean(weight.float() * (a.float() - b.float()) ** 2)


def clamped_linear_interpolation(x, x_min, y_min, x_max, y_max, sigma=1.0):
    x = (x - x_min) / (x_max - x_min)
    x = max(0.0, min(x, 1.0))
    x = x ** sigma
    return y_min + x * (y_max - y_min)


def expand_to_dims(x, target_dims):
    return x.view(*x.shape, *([1] * max(0, target_dims - x.dim())))


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int):
    if tensor is None:
        return None

    first_dim = tensor.shape[0]

    if first_dim == batch_size:
        return tensor

    if batch_size % first_dim != 0:
        raise ValueError(f"Cannot evenly repeat first dim {first_dim} to match batch_size {batch_size}.")

    repeat_times = batch_size // first_dim

    return tensor.repeat(repeat_times, *[1] * (tensor.dim() - 1))


def dim5(x):
    return expand_to_dims(x, 5)


def dim4(x):
    return expand_to_dims(x, 4)


def dim3(x):
    return expand_to_dims(x, 3)


def crop_or_pad_yield_mask(x, length):
    B, F, C = x.shape
    device = x.device
    dtype = x.dtype

    if F < length:
        y = torch.zeros((B, length, C), dtype=dtype, device=device)
        mask = torch.zeros((B, length), dtype=torch.bool, device=device)
        y[:, :F, :] = x
        mask[:, :F] = True
        return y, mask

    return x[:, :length, :], torch.ones((B, length), dtype=torch.bool, device=device)


def extend_dim(x, dim, minimal_length, zero_pad=False):
    original_length = int(x.shape[dim])

    if original_length >= minimal_length:
        return x

    if zero_pad:
        padding_shape = list(x.shape)
        padding_shape[dim] = minimal_length - original_length
        padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
    else:
        idx = (slice(None),) * dim + (slice(-1, None),) + (slice(None),) * (len(x.shape) - dim - 1)
        last_element = x[idx]
        padding = last_element.repeat_interleave(minimal_length - original_length, dim=dim)

    return torch.cat([x, padding], dim=dim)


def lazy_positional_encoding(t, repeats=None):
    if not isinstance(t, list):
        t = [t]

    from diffusers.models.embeddings import get_timestep_embedding

    te = torch.tensor(t)
    te = get_timestep_embedding(timesteps=te, embedding_dim=256, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1.0)

    if repeats is None:
        return te

    te = te[:, None, :].expand(-1, repeats, -1)

    return te


def state_dict_offset_merge(A, B, C=None):
    result = {}
    keys = A.keys()

    for key in keys:
        A_value = A[key]
        B_value = B[key].to(A_value)

        if C is None:
            result[key] = A_value + B_value
        else:
            C_value = C[key].to(A_value)
            result[key] = A_value + B_value - C_value

    return result


def state_dict_weighted_merge(state_dicts, weights):
    if len(state_dicts) != len(weights):
        raise ValueError("Number of state dictionaries must match number of weights")

    if not state_dicts:
        return {}

    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero")

    normalized_weights = [w / total_weight for w in weights]

    keys = state_dicts[0].keys()
    result = {}

    for key in keys:
        result[key] = state_dicts[0][key] * normalized_weights[0]

        for i in range(1, len(state_dicts)):
            state_dict_value = state_dicts[i][key].to(result[key])
            result[key] += state_dict_value * normalized_weights[i]

    return result


def group_files_by_folder(all_files):
    grouped_files = {}

    for file in all_files:
        folder_name = os.path.basename(os.path.dirname(file))
        if folder_name not in grouped_files:
            grouped_files[folder_name] = []
        grouped_files[folder_name].append(file)

    list_of_lists = list(grouped_files.values())
    return list_of_lists


def generate_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime('%y%m%d_%H%M%S')
    milliseconds = f"{int(now.microsecond / 1000):03d}"
    random_number = random.randint(0, 9999)
    return f"{timestamp}_{milliseconds}_{random_number}"


def write_PIL_image_with_png_info(image, metadata, path):
    from PIL.PngImagePlugin import PngInfo

    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, value)

    image.save(path, "PNG", pnginfo=png_info)
    return image


def torch_safe_save(content, path):
    torch.save(content, path + '_tmp')
    os.replace(path + '_tmp', path)
    return path


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save_bcthw_as_webm(x, output_filename, fps=10, quality=1.0):
    """
    将带有透明通道的视频保存为webm格式
    x: 形状为 [batch, channels, time, height, width] 的张量，channels可以是3(RGB)或4(RGBA)
    """
    import subprocess
    import tempfile
    import os
    import sys
    import shutil
    from PIL import Image
    
    # 检查ffmpeg是否可用
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("警告: ffmpeg未找到，无法生成webm格式视频。将保存为MP4格式。")
        mp4_filename = output_filename.replace(".webm", ".mp4")
        save_bcthw_as_mp4(x[:, :3] if x.shape[1] == 4 else x, mp4_filename, fps=fps, crf=int((1.0 - quality) * 51))
        return x
    
    b, c, t, h, w = x.shape
    
    # 如果输入是RGB，添加Alpha通道
    if c == 3:
        print("警告: 输入是RGB格式，没有透明通道。将创建全不透明的Alpha通道。")
        alpha_channel = torch.ones((b, 1, t, h, w), device=x.device, dtype=x.dtype)
        x = torch.cat([x, alpha_channel], dim=1)
        c = 4
    
    assert c == 4, "输入张量必须有3或4个通道 (RGB或RGBA)"
    
    # 打印Alpha通道信息
    alpha = x[:, 3:4]
    min_alpha = alpha.min().item()
    max_alpha = alpha.max().item()
    mean_alpha = alpha.mean().item()
    print(f"Alpha通道信息: 最小值={min_alpha}, 最大值={max_alpha}, 平均值={mean_alpha}")
    
    # 如果Alpha通道全是1，可能是没有正确处理透明度
    if min_alpha == 1.0 and max_alpha == 1.0:
        print("警告: Alpha通道全是1，没有透明像素。可能需要检查透明度处理流程。")
    
    # 创建临时目录存储帧
    with tempfile.TemporaryDirectory() as temp_dir:
        # 将张量转换为图像并保存为PNG序列
        x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
        x = x.detach().cpu().to(torch.uint8).numpy()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
        
        # 保存每一帧为PNG
        for i in range(t):
            frame = x[:, :, i]  # [b, c, h, w]
            
            # 处理批次维度
            per_row = b
            for p in [6, 5, 4, 3, 2]:
                if b % p == 0:
                    per_row = p
                    break
            
            # 计算行数
            rows = (b + per_row - 1) // per_row  # 向上取整
            
            # 创建足够大的画布
            frame_reshaped = np.zeros((rows * h, per_row * w, 4), dtype=np.uint8)
            
            # 填充画布
            for bi in range(b):
                row = bi // per_row
                col = bi % per_row
                # 将[c, h, w]转换为[h, w, c]
                frame_reshaped[row*h:(row+1)*h, col*w:(col+1)*w] = np.transpose(frame[bi], (1, 2, 0))
            
            # 保存为PNG，使用PIL
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            
            # 检查并打印第一帧的Alpha通道信息
            if i == 0:
                alpha_channel = frame_reshaped[:, :, 3]
                transparent_pixels = np.sum(alpha_channel < 255)
                total_pixels = alpha_channel.size
                print(f"第一帧Alpha通道: 透明像素数量={transparent_pixels}, 总像素数={total_pixels}, 比例={transparent_pixels/total_pixels:.2%}")
            
            Image.fromarray(frame_reshaped, mode='RGBA').save(frame_path)
            
            # 额外保存第一帧用于检查
            if i == 0:
                debug_path = os.path.join(os.path.dirname(output_filename), f"debug_frame_0.png")
                Image.fromarray(frame_reshaped, mode='RGBA').save(debug_path)
                print(f"已保存调试帧到 {debug_path}")
        
        # 使用ffmpeg将PNG序列转换为webm
        input_pattern = os.path.join(temp_dir, "frame_%05d.png")
        
        # 构建ffmpeg命令
        cmd = [
            ffmpeg_path,
            "-y",  # 覆盖输出文件
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libvpx-vp9",  # VP9编码器支持透明度
            "-pix_fmt", "yuva420p",  # 像素格式，支持Alpha通道
            "-b:v", f"{int(quality * 4000)}k",  # 比特率，根据quality调整
            "-auto-alt-ref", "0",
            "-metadata:s:v:0", "alpha_mode=1",  # 指示视频有Alpha通道
            output_filename
        ]
        
        try:
            # 执行ffmpeg命令
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"ffmpeg命令失败，错误码: {process.returncode}")
                print(f"stderr: {stderr.decode('utf-8', errors='ignore')}")
                # 如果ffmpeg失败，尝试保存为mp4作为备选
                mp4_filename = output_filename.replace(".webm", ".mp4")
                print(f"尝试保存为MP4: {mp4_filename}")
                # 转换回PyTorch张量，只使用RGB通道
                torch_x = torch.from_numpy(x).to(torch.float32)
                save_bcthw_as_mp4(torch_x[:, :3], mp4_filename, fps=fps, crf=int((1.0 - quality) * 51))
            else:
                print(f"成功保存透明视频到 {output_filename}")
                
        except Exception as e:
            print(f"保存透明视频失败: {str(e)}")
            # 如果ffmpeg失败，尝试保存为mp4作为备选
            mp4_filename = output_filename.replace(".webm", ".mp4")
            print(f"尝试保存为MP4: {mp4_filename}")
            # 转换回PyTorch张量，只使用RGB通道
            torch_x = torch.from_numpy(x).to(torch.float32)
            save_bcthw_as_mp4(torch_x[:, :3], mp4_filename, fps=fps, crf=int((1.0 - quality) * 51))
    
    # 转换回PyTorch张量并返回
    return torch.from_numpy(x).to(torch.float32)
