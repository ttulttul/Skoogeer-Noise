from __future__ import annotations

from typing import Dict

import torch

_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF


def _validate_latent(latent: object, *, name: str) -> Dict:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(f"{name} must be a LATENT dict containing a 'samples' tensor.")

    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise ValueError(f"{name}['samples'] must be a torch.Tensor, got {type(samples)}.")
    if samples.ndim != 4:
        raise ValueError(f"{name}['samples'] must be 4D [B, C, H, W], got shape {tuple(samples.shape)}.")
    if not samples.is_floating_point():
        raise ValueError(f"{name}['samples'] must be a floating-point tensor, got dtype={samples.dtype}.")

    return latent


def _storage_dtype(samples: torch.Tensor) -> torch.dtype:
    if samples.device.type == "cpu" and samples.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return samples.dtype


class SplitLatentPhaseMagnitude:
    """
    Converts a spatial latent into two frequency-domain latents: magnitude and phase.
    """

    CATEGORY = "latent/frequency"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("MAGNITUDE", "PHASE")
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Spatial latent to transform into frequency magnitude + phase latents."}),
            }
        }

    def split(self, latent):
        latent = _validate_latent(latent, name="latent")
        samples: torch.Tensor = latent["samples"]

        out_dtype = _storage_dtype(samples)

        with torch.no_grad():
            try:
                freq = torch.fft.fft2(samples.to(dtype=out_dtype), dim=(-2, -1))
            except RuntimeError:
                freq = torch.fft.fft2(samples.float(), dim=(-2, -1))

            magnitude = torch.abs(freq).to(dtype=out_dtype)
            phase = torch.angle(freq).to(dtype=out_dtype)

        magnitude_latent = latent.copy()
        magnitude_latent["samples"] = magnitude

        phase_latent = latent.copy()
        phase_latent["samples"] = phase

        return (magnitude_latent, phase_latent)


class CombineLatentPhaseMagnitude:
    """
    Converts magnitude + phase frequency-domain latents back into a spatial latent.
    """

    CATEGORY = "latent/frequency"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "combine"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "MAGNITUDE": ("LATENT", {"tooltip": "Magnitude latent from SplitLatentPhaseMagnitude."}),
                "PHASE": ("LATENT", {"tooltip": "Phase latent from SplitLatentPhaseMagnitude."}),
            }
        }

    def combine(self, MAGNITUDE, PHASE):
        magnitude_latent = _validate_latent(MAGNITUDE, name="MAGNITUDE")
        phase_latent = _validate_latent(PHASE, name="PHASE")

        magnitude: torch.Tensor = magnitude_latent["samples"]
        phase: torch.Tensor = phase_latent["samples"]
        if tuple(magnitude.shape) != tuple(phase.shape):
            raise ValueError(
                "MAGNITUDE['samples'] and PHASE['samples'] must have the same shape, "
                f"got {tuple(magnitude.shape)} vs {tuple(phase.shape)}."
            )
        if magnitude.device != phase.device:
            raise ValueError(
                f"MAGNITUDE and PHASE must be on the same device, got {magnitude.device} vs {phase.device}."
            )

        out_dtype = _storage_dtype(magnitude)

        with torch.no_grad():
            try:
                complex_freq = torch.polar(magnitude.to(dtype=out_dtype), phase.to(dtype=out_dtype))
                spatial_complex = torch.fft.ifft2(complex_freq, dim=(-2, -1))
            except RuntimeError:
                complex_freq = torch.polar(magnitude.float(), phase.float())
                spatial_complex = torch.fft.ifft2(complex_freq, dim=(-2, -1))

            spatial = spatial_complex.real.to(dtype=out_dtype)

        out = magnitude_latent.copy()
        out["samples"] = spatial
        return (out,)


class FrequencySelectiveStructuredNoise:
    """
    Generates frequency-domain structured noise using a reference phase latent.

    The output magnitude is taken entirely from random Gaussian noise (FFT magnitude).
    The output phase is mixed between the reference phase and noise phase using a radial frequency mask.
    """

    CATEGORY = "latent/frequency"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("MAGNITUDE", "PHASE")
    FUNCTION = "generate_fss_noise"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "PHASE": ("LATENT", {"tooltip": "Reference phase latent (from SplitLatentPhaseMagnitude)."}),
                "cutoff_radius_r": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Frequency radius below which the reference phase is preserved.",
                }),
                "sigma": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.1,
                    "tooltip": "Smoothness of the cutoff transition (higher = smoother).",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed for reproducible noise generation.",
                }),
            }
        }

    def generate_fss_noise(self, PHASE, cutoff_radius_r: int, sigma: float, seed: int):
        phase_latent_ref = _validate_latent(PHASE, name="PHASE")
        phase_ref: torch.Tensor = phase_latent_ref["samples"]

        out_dtype = _storage_dtype(phase_ref)
        batch, channels, height, width = phase_ref.shape

        cutoff_radius_r = int(cutoff_radius_r)
        sigma = float(sigma)
        seed = int(seed)

        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(seed & _SEED_MASK_64)
            spatial_noise = torch.randn(
                (batch, channels, height, width),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            if spatial_noise.device != phase_ref.device:
                spatial_noise = spatial_noise.to(device=phase_ref.device)
            spatial_noise = spatial_noise.to(dtype=out_dtype)

            try:
                complex_noise_freq = torch.fft.fft2(spatial_noise, dim=(-2, -1))
            except RuntimeError:
                complex_noise_freq = torch.fft.fft2(spatial_noise.float(), dim=(-2, -1))

            magnitude_noise = torch.abs(complex_noise_freq).to(dtype=out_dtype)
            phase_noise = torch.angle(complex_noise_freq).to(dtype=out_dtype)

            coord_dtype = torch.float32
            u = torch.arange(height, device=phase_ref.device, dtype=coord_dtype) - (height // 2)
            v = torch.arange(width, device=phase_ref.device, dtype=coord_dtype) - (width // 2)
            u_grid, v_grid = torch.meshgrid(u, v, indexing="ij")
            radius_grid = torch.sqrt(u_grid * u_grid + v_grid * v_grid)

            cutoff = float(cutoff_radius_r)
            sigma_safe = max(sigma, 1e-6)
            decay = torch.exp(-torch.square(radius_grid - cutoff) / (2.0 * sigma_safe * sigma_safe))
            mask_centered = torch.where(radius_grid <= cutoff, torch.ones_like(radius_grid), decay)
            mask = torch.fft.ifftshift(mask_centered, dim=(-2, -1)).to(dtype=out_dtype)
            mask = mask.unsqueeze(0).unsqueeze(0)

            phase_ref_out = phase_ref.to(dtype=out_dtype)
            mixed_phase = phase_ref_out * mask + phase_noise * (mask.new_tensor(1.0) - mask)

        magnitude_latent = phase_latent_ref.copy()
        magnitude_latent["samples"] = magnitude_noise

        phase_latent = phase_latent_ref.copy()
        phase_latent["samples"] = mixed_phase

        return (magnitude_latent, phase_latent)


NODE_CLASS_MAPPINGS = {
    "SplitLatentPhaseMagnitude": SplitLatentPhaseMagnitude,
    "CombineLatentPhaseMagnitude": CombineLatentPhaseMagnitude,
    "FrequencySelectiveStructuredNoise": FrequencySelectiveStructuredNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitLatentPhaseMagnitude": "Split Latent (FFT Mag/Phase)",
    "CombineLatentPhaseMagnitude": "Combine Latent (IFFT Mag/Phase)",
    "FrequencySelectiveStructuredNoise": "Frequency-Selective Structured Noise (FSS)",
}
