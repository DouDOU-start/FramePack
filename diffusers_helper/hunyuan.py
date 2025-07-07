import torch

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask


@torch.no_grad()
def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)

    prompt = [prompt]

    # LLAMA

    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)
    llama_attention_length = int(llama_attention_mask.sum())

    llama_outputs = text_encoder(
        input_ids=llama_input_ids,
        attention_mask=llama_attention_mask,
        output_hidden_states=True,
    )

    llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
    # llama_vec_remaining = llama_outputs.hidden_states[-3][:, llama_attention_length:]
    llama_attention_mask = llama_attention_mask[:, crop_start:llama_attention_length]

    assert torch.all(llama_attention_mask.bool())

    # CLIP

    clip_l_input_ids = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    clip_l_pooler = text_encoder_2(clip_l_input_ids.to(text_encoder_2.device), output_hidden_states=False).pooler_output

    return llama_vec, clip_l_pooler


@torch.no_grad()
def vae_decode_fake(latents, with_alpha=False):
    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],
        [0.0696, 0.0795, 0.0518],
        [0.0135, -0.0945, -0.0282],
        [0.0108, -0.0250, -0.0765],
        [-0.0209, 0.0032, 0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991, 0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [0.1166, 0.1627, 0.0962],
        [0.1165, 0.0432, 0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270, 0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [0.0249, -0.0469, -0.1703]
    ]  # From comfyui

    latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.clamp(0.0, 1.0)
    
    # 添加Alpha通道，默认为完全不透明
    if with_alpha:
        b, c, t, h, w = images.shape
        
        # 检查是否有保存的Alpha通道信息
        if hasattr(latents, '_alpha_channel') and latents._alpha_channel is not None:
            print(f"Fake解码：使用保存的Alpha通道")
            alpha_channel = latents._alpha_channel
            # 调整Alpha通道大小以匹配图像
            if alpha_channel.shape[3:] != (h, w) or alpha_channel.shape[2] != t:
                print(f"Fake解码：调整Alpha通道尺寸")
                alpha_channel = torch.nn.functional.interpolate(
                    alpha_channel.view(b, 1, -1, alpha_channel.shape[3], alpha_channel.shape[4]),
                    size=(t, h, w),
                    mode='trilinear',
                    align_corners=False
                ).view(b, 1, t, h, w)
        else:
            # 创建一个透明背景的Alpha通道
            print(f"Fake解码：创建透明背景Alpha通道")
            alpha_channel = torch.zeros((b, 1, t, h, w), device=images.device, dtype=images.dtype)
            
            # 基于RGB亮度创建Alpha通道
            # 计算RGB亮度
            luminance = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            # 使用亮度作为Alpha值
            alpha_channel = luminance.clamp(0.0, 1.0)
            
        images_rgba = torch.cat([images, alpha_channel], dim=1)
        return images_rgba
    
    return images


@torch.no_grad()
def vae_decode(latents, vae, image_mode=False, with_alpha=False):
    latents = latents / vae.config.scaling_factor

    if not image_mode:
        image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    else:
        latents = latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
        image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
        image = torch.cat(image, dim=2)
    
    # 如果需要透明通道且之前保存了Alpha通道信息
    if with_alpha:
        b, c, t, h, w = image.shape
        
        # 检查是否有保存的Alpha通道
        if hasattr(vae, '_alpha_channel') and vae._alpha_channel is not None:
            print(f"VAE解码：使用保存的Alpha通道，形状为 {vae._alpha_channel.shape}")
            
            # 如果Alpha通道与当前图像不匹配，需要调整
            alpha = vae._alpha_channel
            if alpha.shape[2] == 1 and t > 1:
                # 复制单帧Alpha到多帧
                print(f"VAE解码：将单帧Alpha扩展到 {t} 帧")
                alpha = alpha.repeat(1, 1, t, 1, 1)
            
            # 确保Alpha通道的尺寸与图像匹配
            if alpha.shape[3:] != (h, w):
                print(f"VAE解码：调整Alpha通道尺寸从 {alpha.shape[3:]} 到 {(h, w)}")
                alpha = torch.nn.functional.interpolate(
                    alpha.view(b, 1, -1, alpha.shape[3], alpha.shape[4]),
                    size=(t, h, w),
                    mode='trilinear',
                    align_corners=False
                ).view(b, 1, t, h, w)
            
            # 合并RGB和Alpha
            image = torch.cat([image, alpha], dim=1)
            print(f"VAE解码：合并后RGBA形状为 {image.shape}")
        else:
            # 如果没有保存的Alpha通道，创建全不透明的Alpha
            print("VAE解码：没有找到保存的Alpha通道，创建全不透明Alpha")
            alpha_channel = torch.ones((b, 1, t, h, w), device=image.device, dtype=image.dtype)
            image = torch.cat([image, alpha_channel], dim=1)

    return image


@torch.no_grad()
def vae_encode(image, vae, with_alpha=False):
    # 如果输入图像包含Alpha通道，分离RGB和Alpha
    alpha_channel = None
    if with_alpha and image.shape[1] == 4:
        print(f"VAE编码：输入图像有Alpha通道，形状为 {image.shape}")
        rgb = image[:, :3]
        alpha_channel = image[:, 3:4]  # 保存Alpha通道以便后续使用
        image = rgb
        print(f"VAE编码：分离后RGB形状为 {rgb.shape}, Alpha形状为 {alpha_channel.shape}")
    
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    
    # 存储Alpha通道信息，以便后续使用
    if alpha_channel is not None:
        # 将Alpha通道信息附加到模型的属性中
        vae._alpha_channel = alpha_channel.to(device=vae.device, dtype=vae.dtype)
        print(f"VAE编码：保存Alpha通道信息，形状为 {vae._alpha_channel.shape}")
    
    return latents
