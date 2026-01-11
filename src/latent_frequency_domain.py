from __future__ import annotations

from typing import Dict

import torch


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
    Converts a spatial latent into a packed frequency latent.

    Output packing: channels [0:C] = magnitude, channels [C:2C] = phase (radians).
    """

    CATEGORY = "latent/frequency"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("frequency_latent",)
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Spatial latent to transform into packed frequency (magnitude+phase)."}),
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
            packed = torch.cat((magnitude, phase), dim=1)

        out = latent.copy()
        out["samples"] = packed
        return (out,)


class CombineLatentPhaseMagnitude:
    """
    Converts a packed frequency latent (magnitude+phase) back into a spatial latent.
    """

    CATEGORY = "latent/frequency"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "combine"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "frequency_latent": ("LATENT", {"tooltip": "Packed frequency latent from SplitLatentPhaseMagnitude."}),
            }
        }

    def combine(self, frequency_latent):
        frequency_latent = _validate_latent(frequency_latent, name="frequency_latent")
        packed: torch.Tensor = frequency_latent["samples"]

        total_channels = int(packed.shape[1])
        if total_channels % 2 != 0:
            raise ValueError(
                f"frequency_latent['samples'] must have an even channel count (magnitude+phase), got C={total_channels}."
            )

        out_dtype = _storage_dtype(packed)

        magnitude, phase = torch.chunk(packed, 2, dim=1)

        with torch.no_grad():
            try:
                complex_freq = torch.polar(magnitude.to(dtype=out_dtype), phase.to(dtype=out_dtype))
                spatial_complex = torch.fft.ifft2(complex_freq, dim=(-2, -1))
            except RuntimeError:
                complex_freq = torch.polar(magnitude.float(), phase.float())
                spatial_complex = torch.fft.ifft2(complex_freq, dim=(-2, -1))

            spatial = spatial_complex.real.to(dtype=out_dtype)

        out = frequency_latent.copy()
        out["samples"] = spatial
        return (out,)


NODE_CLASS_MAPPINGS = {
    "SplitLatentPhaseMagnitude": SplitLatentPhaseMagnitude,
    "CombineLatentPhaseMagnitude": CombineLatentPhaseMagnitude,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitLatentPhaseMagnitude": "Split Latent (FFT Mag/Phase)",
    "CombineLatentPhaseMagnitude": "Combine Latent (IFFT Mag/Phase)",
}

