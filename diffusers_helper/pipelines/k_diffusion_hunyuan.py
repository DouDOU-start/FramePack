import torch
import math

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size


def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


def generate_looping_noise(shape, generator, device, loop_strength=1.0, loop_period=None):
    """
    Generate looping noise with smooth cyclic transitions.
    
    Args:
        shape: Shape of the noise tensor (B, C, T, H, W)
        generator: Random number generator
        device: Device to create tensors on
        loop_strength: Strength of the loop effect (0=no loop, 1=full loop)
        loop_period: Period of the loop in frames (None=use full length)
    
    Returns:
        Noise tensor with smooth cyclic transitions
    """
    B, C, T, H, W = shape
    
    if loop_strength == 0.0:
        # No looping, return regular random noise
        return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    
    # Use full length as period if not specified
    if loop_period is None:
        loop_period = T
    
    # Generate base noise for one complete cycle
    base_noise = torch.randn((B, C, loop_period, H, W), generator=generator, device=device, dtype=torch.float32)
    
    if T <= loop_period:
        # For sequences shorter than or equal to one period, use seamless wrapping
        if T < loop_period:
            # Extract the needed frames with seamless wrapping
            indices = torch.arange(T, device=device) % loop_period
            tiled_noise = base_noise[:, :, indices, :, :]
        else:
            tiled_noise = base_noise
    else:
        # For longer sequences, create seamless repetition
        full_cycles = T // loop_period
        remaining_frames = T % loop_period
        
        # Create the tiled sequence
        tiled_noise = base_noise.repeat(1, 1, full_cycles, 1, 1)
        
        if remaining_frames > 0:
            # Add remaining frames with seamless wrapping
            remaining_noise = base_noise[:, :, :remaining_frames, :, :]
            tiled_noise = torch.cat([tiled_noise, remaining_noise], dim=2)
    
    # Apply cosine-based smoothing at boundaries for even smoother transitions
    if T > loop_period and loop_strength > 0:
        # Create smooth transition weights
        transition_length = min(loop_period // 4, 8)  # Smooth transition over 1/4 of period or 8 frames
        
        for cycle_start in range(loop_period, T, loop_period):
            if cycle_start + transition_length < T:
                # Create cosine transition weights
                t = torch.linspace(0, torch.pi, transition_length, device=device)
                fade_out = (torch.cos(t) + 1) / 2  # 1 -> 0
                fade_in = 1 - fade_out  # 0 -> 1
                
                # Apply smooth transition
                for i in range(transition_length):
                    frame_idx = cycle_start + i
                    if frame_idx < T:
                        weight_out = fade_out[i]
                        weight_in = fade_in[i]
                        
                        # Blend current frame with corresponding frame from previous cycle
                        prev_frame_idx = frame_idx - loop_period
                        if prev_frame_idx >= 0:
                            tiled_noise[:, :, frame_idx, :, :] = (
                                weight_out * tiled_noise[:, :, prev_frame_idx, :, :] +
                                weight_in * tiled_noise[:, :, frame_idx, :, :]
                            )
    
    # Apply loop strength blending
    if loop_strength < 1.0:
        # Mix with regular random noise
        random_noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        tiled_noise = loop_strength * tiled_noise + (1 - loop_strength) * random_noise
    
    return tiled_noise


def apply_temporal_smoothing(noise, smoothing_factor=0.1):
    """
    Apply temporal smoothing to noise to reduce flickering.
    
    Args:
        noise: Input noise tensor (B, C, T, H, W)
        smoothing_factor: Smoothing strength (0=no smoothing, 1=maximum smoothing)
    
    Returns:
        Smoothed noise tensor
    """
    if smoothing_factor == 0.0:
        return noise
    
    B, C, T, H, W = noise.shape
    
    # Apply 1D convolution along time dimension for smoothing
    # Create a simple averaging kernel
    kernel_size = 3
    kernel = torch.ones(1, 1, kernel_size, device=noise.device, dtype=noise.dtype) / kernel_size
    
    # Reshape for convolution: (B*C*H*W, 1, T)
    noise_flat = noise.permute(0, 1, 3, 4, 2).contiguous().view(-1, 1, T)
    
    # Apply padding and convolution
    noise_padded = torch.nn.functional.pad(noise_flat, (1, 1), mode='circular')
    smoothed_flat = torch.nn.functional.conv1d(noise_padded, kernel)
    
    # Reshape back to original shape
    smoothed = smoothed_flat.view(B, C, H, W, T).permute(0, 1, 4, 2, 3).contiguous()
    
    # Mix with original noise
    return smoothing_factor * smoothed + (1 - smoothing_factor) * noise


@torch.inference_mode()
def sample_hunyuan(
        transformer,
        sampler='unipc',
        initial_latent=None,
        concat_latent=None,
        strength=1.0,
        width=512,
        height=512,
        frames=16,
        real_guidance_scale=1.0,
        distilled_guidance_scale=6.0,
        guidance_rescale=0.0,
        shift=None,
        num_inference_steps=25,
        batch_size=None,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.bfloat16,
        device=None,
        negative_kwargs=None,
        callback=None,
        enable_loop=False,
        loop_strength=1.0,
        loop_period=None,
        temporal_smoothing=0.0,
        **kwargs,
):
    device = device or transformer.device

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    # Generate noise with optional looping
    noise_shape = (batch_size, 16, (frames + 3) // 4, height // 8, width // 8)
    
    if enable_loop:
        latents = generate_looping_noise(
            noise_shape, 
            generator=generator, 
            device=generator.device, 
            loop_strength=loop_strength, 
            loop_period=loop_period
        )
        
        # Apply temporal smoothing if requested
        if temporal_smoothing > 0.0:
            latents = apply_temporal_smoothing(latents, temporal_smoothing)
    else:
        latents = torch.randn(noise_shape, generator=generator, device=generator.device, dtype=torch.float32)
    
    latents = latents.to(device=device, dtype=torch.float32)

    B, C, T, H, W = latents.shape
    seq_length = T * H * W // 4

    if shift is None:
        mu = calculate_flux_mu(seq_length, exp_max=7.0)
    else:
        mu = math.log(shift)

    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)

    k_model = fm_wrapper(transformer)

    if initial_latent is not None:
        sigmas = sigmas * strength
        first_sigma = sigmas[0].to(device=device, dtype=torch.float32)
        initial_latent = initial_latent.to(device=device, dtype=torch.float32)
        latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma

    if concat_latent is not None:
        concat_latent = concat_latent.to(latents)

    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )

    if sampler == 'unipc':
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')

    return results
