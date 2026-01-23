from __future__ import annotations

import logging
import math
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _validate_latent(latent: object) -> Dict:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("LATENT input must be a dictionary containing a 'samples' tensor.")
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"LATENT['samples'] must be a torch.Tensor, got {type(samples)}.")
    if samples.ndim < 4:
        raise ValueError(
            "LATENT['samples'] must have at least 4 dimensions (B, C, ..., H, W), "
            f"got {tuple(samples.shape)}."
        )
    if not samples.is_floating_point():
        raise TypeError(f"LATENT['samples'] must be floating point, got dtype={samples.dtype}.")
    if int(samples.shape[1]) == 0:
        raise ValueError("LATENT['samples'] must have at least one channel.")
    if int(samples.shape[-2]) == 0 or int(samples.shape[-1]) == 0:
        raise ValueError("LATENT['samples'] must have non-empty spatial dimensions.")
    return latent


def _flatten_samples(samples: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if samples.ndim == 4:
        return samples, ()

    batch = int(samples.shape[0])
    channels = int(samples.shape[1])
    height = int(samples.shape[-2])
    width = int(samples.shape[-1])
    extra_dims = tuple(int(dim) for dim in samples.shape[2:-2])
    extra_size = int(math.prod(extra_dims)) if extra_dims else 1
    if extra_size == 0:
        raise ValueError("LATENT['samples'] has an empty extra dimension; cannot flatten.")

    flattened = samples.reshape(batch, channels, extra_size, height, width)
    flattened = flattened.permute(0, 2, 1, 3, 4).reshape(batch * extra_size, channels, height, width)
    return flattened, extra_dims


def _normalize_minmax(images: torch.Tensor) -> torch.Tensor:
    flat = images.reshape(images.shape[0], -1)
    min_vals = flat.min(dim=1).values
    max_vals = flat.max(dim=1).values
    scale = (max_vals - min_vals).clamp_min(_EPS)
    normalized = (images - min_vals.view(-1, 1, 1)) / scale.view(-1, 1, 1)
    return normalized.clamp(0.0, 1.0)


class LatentToImage:
    CATEGORY = "latent/debug"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to render into per-channel grayscale images."}),
                "normalize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Normalize each output image to [0,1] using per-image min/max.",
                }),
                "output_layout": (["nchw", "bhwc"], {
                    "default": "nchw",
                    "tooltip": "Output tensor layout: NCHW is channel-first; BHWC matches ComfyUI IMAGE format.",
                }),
            }
        }

    def render(self, latent, normalize=False, output_layout="nchw"):
        latent = _validate_latent(latent)
        samples: torch.Tensor = latent["samples"]

        logger.debug(
            "LatentToImage input: shape=%s dtype=%s device=%s",
            tuple(samples.shape),
            samples.dtype,
            samples.device,
        )

        samples_nchw, extra_dims = _flatten_samples(samples)
        samples_nchw = samples_nchw.contiguous()
        if extra_dims:
            logger.debug(
                "LatentToImage flattened extra dims %s into batch; new shape=%s",
                extra_dims,
                tuple(samples_nchw.shape),
            )

        batch = int(samples_nchw.shape[0])
        channels = int(samples_nchw.shape[1])
        height = int(samples_nchw.shape[2])
        width = int(samples_nchw.shape[3])

        if batch == 0:
            raise ValueError("LATENT['samples'] batch dimension must be non-empty.")

        images = samples_nchw.reshape(batch * channels, height, width)
        if normalize:
            images = _normalize_minmax(images)

        if output_layout == "nchw":
            output = images.unsqueeze(1)
        elif output_layout == "bhwc":
            output = images.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown output_layout '{output_layout}', expected 'nchw' or 'bhwc'.")

        logger.debug(
            "LatentToImage output: shape=%s layout=%s normalize=%s",
            tuple(output.shape),
            output_layout,
            bool(normalize),
        )
        return (output,)


NODE_CLASS_MAPPINGS = {
    "LatentToImage": LatentToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentToImage": "Latent to Image",
}
