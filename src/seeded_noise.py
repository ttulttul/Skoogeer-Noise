from __future__ import annotations

import logging
from typing import Dict

import torch

try:
    from .masking import blend_image_with_mask, blend_with_mask, prepare_mask_nchw
except ImportError:  # pragma: no cover - fallback for direct module loading
    from masking import blend_image_with_mask, blend_with_mask, prepare_mask_nchw

logger = logging.getLogger(__name__)

_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF
_SEED_STREAM_INCREMENT = 0x9E3779B97F4A7C15


def generate_seed(x: int) -> int:
    seed = int(x) & _SEED_MASK_64
    seed = (seed ^ (seed >> 33)) & _SEED_MASK_64
    seed = (seed * 0xFF51AFD7ED558CCD) & _SEED_MASK_64
    seed = (seed ^ (seed >> 33)) & _SEED_MASK_64
    seed = (seed * 0xC4CEB9FE1A85EC53) & _SEED_MASK_64
    seed = (seed ^ (seed >> 33)) & _SEED_MASK_64
    return seed


def next_seed_values(seed: int, *, count: int = 4) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")

    base_seed = int(seed) & _SEED_MASK_64
    derived = tuple(
        generate_seed((base_seed + (index * _SEED_STREAM_INCREMENT)) & _SEED_MASK_64)
        for index in range(1, count + 1)
    )
    logger.debug("Generated %d derived seeds from base_seed=%d: %s", count, base_seed, derived)
    return derived


def _seeded_gaussian_noise_like(tensor: torch.Tensor, *, seed: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if not tensor.is_floating_point():
        raise TypeError(f"Expected floating-point tensor, got dtype={tensor.dtype}")

    generator = torch.Generator(device="cpu").manual_seed(int(seed) & _SEED_MASK_64)
    noise = torch.randn(tensor.shape, generator=generator, device="cpu", dtype=torch.float32)
    if noise.device != tensor.device:
        noise = noise.to(device=tensor.device)
    return noise.to(dtype=tensor.dtype)


def _safe_global_std(tensor: torch.Tensor) -> float:
    std = tensor.detach().float().std(unbiased=False)
    if not torch.isfinite(std):
        return 1.0
    value = float(std)
    if value <= 1e-6:
        return 1.0
    return value


def _safe_per_sample_std(tensor: torch.Tensor) -> torch.Tensor:
    batch = int(tensor.shape[0])
    flat = tensor.detach().float().reshape(batch, -1)
    std = flat.std(dim=1, unbiased=False)
    std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    return std


def add_seeded_noise(
    tensor: torch.Tensor,
    *,
    seed: int,
    strength: float,
    scale_by_std: bool = True,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if not tensor.is_floating_point():
        raise TypeError(f"Expected floating-point tensor, got dtype={tensor.dtype}")

    strength = float(strength)
    if strength == 0.0:
        return tensor

    with torch.no_grad():
        seed_value = int(seed) & _SEED_MASK_64

        if tensor.ndim == 0:
            noise = _seeded_gaussian_noise_like(tensor, seed=seed_value)
            scale = _safe_global_std(tensor) if scale_by_std else 1.0
            logger.debug(
                "Adding seeded noise: shape=%s seed=%d strength=%.4f scale=%.6f",
                tuple(tensor.shape),
                seed_value,
                strength,
                scale,
            )
            return tensor + noise * (strength * scale)

        batch = int(tensor.shape[0])
        if batch <= 0:
            raise ValueError("Input tensor batch dimension must be non-empty.")

        if batch == 1:
            noise = _seeded_gaussian_noise_like(tensor, seed=seed_value)
            scale = _safe_global_std(tensor) if scale_by_std else 1.0
            logger.debug(
                "Adding seeded noise: shape=%s seed=%d strength=%.4f scale=%.6f",
                tuple(tensor.shape),
                seed_value,
                strength,
                scale,
            )
            return tensor + noise * (strength * scale)

        scales = _safe_per_sample_std(tensor) if scale_by_std else torch.ones(batch, device=tensor.device, dtype=torch.float32)
        noise = torch.empty_like(tensor, device=tensor.device, dtype=tensor.dtype)

        for batch_index in range(batch):
            batch_seed = (seed_value + batch_index) & _SEED_MASK_64
            sample_noise = _seeded_gaussian_noise_like(tensor[batch_index], seed=batch_seed)
            noise[batch_index] = sample_noise.to(dtype=tensor.dtype)

        scale_view = scales.view(batch, *([1] * (tensor.ndim - 1)))
        scaled_noise = noise * (strength * scale_view)
        logger.debug(
            "Adding seeded noise: shape=%s batch=%d seed=%d strength=%.4f scale_min=%.6f scale_max=%.6f",
            tuple(tensor.shape),
            batch,
            seed_value,
            strength,
            float(scales.min().item()),
            float(scales.max().item()),
        )
        return tensor + scaled_noise


class LatentNoise:
    """
    Adds seeded Gaussian noise to a ComfyUI LATENT dictionary.

    The `strength` parameter is relative to the standard deviation of the input latent samples.
    """

    CATEGORY = "latent/perturb"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "add_noise"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to receive additional Gaussian noise."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed for generating repeatable noise.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Noise strength relative to the latent's standard deviation.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise addition to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    def add_noise(self, latent, seed: int, strength: float, mask=None):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("LATENT input must be a dictionary containing a 'samples' tensor.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError(f"LATENT['samples'] must be a torch.Tensor, got {type(samples)}.")

        noised = add_seeded_noise(samples, seed=int(seed), strength=float(strength), scale_by_std=True)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(samples.shape[0]),
                height=int(samples.shape[-2]),
                width=int(samples.shape[-1]),
                device=samples.device,
            )
            noised = blend_with_mask(samples, noised, mask_nchw)

        out = latent.copy()
        out["samples"] = noised
        return (out,)


class ImageNoise:
    """
    Adds seeded Gaussian noise to a ComfyUI IMAGE tensor.

    The `strength` parameter is relative to the standard deviation of the input image tensor.
    """

    CATEGORY = "image/perturb"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_noise"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to receive additional Gaussian noise."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed for generating repeatable noise.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Noise strength relative to the image's standard deviation.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise addition to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    def add_noise(self, image: torch.Tensor, seed: int, strength: float, mask=None):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"IMAGE input must be a torch.Tensor, got {type(image)}.")

        noised = add_seeded_noise(image, seed=int(seed), strength=float(strength), scale_by_std=True)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            noised = blend_image_with_mask(image, noised, mask)
        return (noised,)


class NextSeeds:
    CATEGORY = "utils/seed"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("seed_1", "seed_2", "seed_3", "seed_4")
    FUNCTION = "next_seeds"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Base 64-bit seed. The node treats it as an unsigned 64-bit value and derives four deterministic child seeds for downstream nodes.",
                }),
            },
        }

    def next_seeds(self, seed: int):
        derived = next_seed_values(seed, count=4)
        logger.debug("NextSeeds node expanded seed=%d into %s", int(seed) & _SEED_MASK_64, derived)
        return derived


NODE_CLASS_MAPPINGS = {
    "LatentNoise": LatentNoise,
    "ImageNoise": ImageNoise,
    "NextSeeds": NextSeeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentNoise": "Latent Noise",
    "ImageNoise": "Image Noise",
    "NextSeeds": "Next Seeds",
}
