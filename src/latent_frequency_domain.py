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


NODE_CLASS_MAPPINGS = {
    "SplitLatentPhaseMagnitude": SplitLatentPhaseMagnitude,
    "CombineLatentPhaseMagnitude": CombineLatentPhaseMagnitude,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitLatentPhaseMagnitude": "Split Latent (FFT Mag/Phase)",
    "CombineLatentPhaseMagnitude": "Combine Latent (IFFT Mag/Phase)",
}
