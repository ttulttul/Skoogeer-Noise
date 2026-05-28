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


def _validate_image_batch(image_batch: object) -> torch.Tensor:
    if not isinstance(image_batch, torch.Tensor):
        raise TypeError(f"IMAGE input must be a torch.Tensor, got {type(image_batch)}.")
    if image_batch.ndim != 4:
        raise ValueError(f"IMAGE input must be 4D (B, H, W, C), got {tuple(image_batch.shape)}.")
    if not image_batch.is_floating_point():
        raise TypeError(f"IMAGE input must be floating point, got dtype={image_batch.dtype}.")
    if int(image_batch.shape[0]) == 0:
        raise ValueError("IMAGE input must have a non-empty batch dimension.")
    if int(image_batch.shape[-1]) not in (1, 3):
        raise ValueError(
            "IMAGE input must have 1 or 3 channels in the last dimension, "
            f"got {int(image_batch.shape[-1])}."
        )
    if int(image_batch.shape[1]) == 0 or int(image_batch.shape[2]) == 0:
        raise ValueError("IMAGE input must have non-empty spatial dimensions.")
    return image_batch


def _resolve_batch_channels(total: int, batch_size: int, channels: int) -> Tuple[int, int]:
    total = int(total)
    batch_size = int(batch_size)
    channels = int(channels)
    if total <= 0:
        raise ValueError("IMAGE batch must contain at least one element.")
    if batch_size < 0 or channels < 0:
        raise ValueError("batch_size and channels must be >= 0.")
    if batch_size == 0 and channels == 0:
        raise ValueError("Provide either batch_size or channels (or both) to reconstruct the latent.")
    if batch_size == 0:
        if total % channels != 0:
            raise ValueError(f"Batch of {total} images cannot be split into {channels} channels per sample.")
        batch_size = total // channels
    if channels == 0:
        if total % batch_size != 0:
            raise ValueError(f"Batch of {total} images cannot be split into batch_size={batch_size}.")
        channels = total // batch_size
    if batch_size * channels != total:
        raise ValueError(
            f"batch_size ({batch_size}) * channels ({channels}) must equal image batch size ({total})."
        )
    if batch_size == 0 or channels == 0:
        raise ValueError("Resolved batch_size and channels must both be > 0.")
    return batch_size, channels


def _extract_grayscale(images: torch.Tensor, channel_source: str) -> torch.Tensor:
    channels = int(images.shape[-1])
    if channels == 1:
        return images[..., 0]

    channel_source = str(channel_source).lower()
    if channel_source == "r":
        return images[..., 0]
    if channel_source == "g":
        return images[..., 1]
    if channel_source == "b":
        return images[..., 2]
    if channel_source == "mean":
        return images.mean(dim=-1)
    raise ValueError(f"Unknown channel_source '{channel_source}', expected r/g/b/mean.")


def _insert_tensor_batch(batch: torch.Tensor, item: torch.Tensor, index: int, *, input_name: str) -> torch.Tensor:
    index_value = int(index)
    batch_count = int(batch.shape[0])
    item_count = int(item.shape[0])
    if item_count == 0:
        raise ValueError(f"{input_name} to insert must have a non-empty batch dimension.")
    if index_value < 0 or index_value > batch_count:
        raise ValueError(f"index must be in the inclusive range [0, {batch_count}], got {index_value}.")
    if tuple(batch.shape[1:]) != tuple(item.shape[1:]):
        raise ValueError(
            f"{input_name} shape after the batch dimension must match the target batch: "
            f"got {tuple(item.shape[1:])}, expected {tuple(batch.shape[1:])}."
        )
    if batch.dtype != item.dtype:
        raise TypeError(f"{input_name} dtype must match the target batch: got {item.dtype}, expected {batch.dtype}.")
    if batch.device != item.device:
        raise ValueError(f"{input_name} device must match the target batch: got {item.device}, expected {batch.device}.")

    before = batch[:index_value]
    after = batch[index_value:]
    return torch.cat((before, item, after), dim=0)


def _normalize_latent_mask(mask: torch.Tensor, sample_count: int, *, input_name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{input_name} noise_mask must be a torch.Tensor, got {type(mask)}.")
    if mask.ndim < 3:
        raise ValueError(f"{input_name} noise_mask must have at least 3 dimensions, got {tuple(mask.shape)}.")
    mask_count = int(mask.shape[0])
    if mask_count == sample_count:
        return mask
    if mask_count == 1:
        return mask.repeat((sample_count,) + ((1,) * (mask.ndim - 1)))
    raise ValueError(
        f"{input_name} noise_mask batch dimension must be 1 or match samples batch size {sample_count}, "
        f"got {mask_count}."
    )


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
                "output_channels": (["1", "3"], {
                    "default": "3",
                    "tooltip": "Use 3 to repeat grayscale into RGB for PreviewImage compatibility.",
                }),
            }
        }

    def render(self, latent, normalize=False, output_channels="3"):
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

        output_channels = str(output_channels)
        if output_channels not in ("1", "3"):
            raise ValueError(f"output_channels must be '1' or '3', got '{output_channels}'.")

        output = images.unsqueeze(-1)
        if output_channels == "3":
            output = output.repeat(1, 1, 1, 3)

        logger.debug(
            "LatentToImage output: shape=%s output_channels=%s normalize=%s",
            tuple(output.shape),
            output_channels,
            bool(normalize),
        )
        return (output,)


class ImageBatchToLatent:
    CATEGORY = "latent/debug"
    FUNCTION = "merge"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE", {"tooltip": "Batch of images representing latent channels."}),
                "batch_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Original latent batch size (0 to infer from channels).",
                }),
                "channels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Channel count per latent sample (0 to infer from batch_size).",
                }),
                "channel_source": (["r", "g", "b", "mean"], {
                    "default": "r",
                    "tooltip": "Which channel to use from RGB inputs (mean averages channels).",
                }),
            }
        }

    def merge(self, image_batch, batch_size=0, channels=0, channel_source="r"):
        images = _validate_image_batch(image_batch)

        logger.debug(
            "ImageBatchToLatent input: shape=%s dtype=%s device=%s",
            tuple(images.shape),
            images.dtype,
            images.device,
        )

        total = int(images.shape[0])
        height = int(images.shape[1])
        width = int(images.shape[2])
        batch_size, channels = _resolve_batch_channels(total, batch_size, channels)

        gray = _extract_grayscale(images, channel_source)
        gray = gray.contiguous()

        samples = gray.reshape(batch_size, channels, height, width)

        logger.debug(
            "ImageBatchToLatent output: shape=%s batch_size=%s channels=%s",
            tuple(samples.shape),
            batch_size,
            channels,
        )
        return ({"samples": samples},)


class ImageToBatch:
    CATEGORY = "image/batch"
    FUNCTION = "insert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE", {"tooltip": "Existing image batch to insert into."}),
                "image": ("IMAGE", {"tooltip": "Image or image batch to insert at the requested index."}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Batch index where the image is inserted. Existing items at and after this index shift right.",
                }),
            }
        }

    def insert(self, image_batch, image, index=0):
        batch = _validate_image_batch(image_batch)
        item = _validate_image_batch(image)
        output = _insert_tensor_batch(batch, item, index, input_name="IMAGE")
        logger.debug(
            "ImageToBatch inserted %d image(s) into batch_size=%d at index=%d; output_shape=%s",
            int(item.shape[0]),
            int(batch.shape[0]),
            int(index),
            tuple(output.shape),
        )
        return (output,)


class LatentToBatch:
    CATEGORY = "latent/batch"
    FUNCTION = "insert"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_batch": ("LATENT", {"tooltip": "Existing latent batch to insert into."}),
                "latent": ("LATENT", {"tooltip": "Latent or latent batch to insert at the requested index."}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Batch index where the latent is inserted. Existing items at and after this index shift right.",
                }),
            }
        }

    def insert(self, latent_batch, latent, index=0):
        batch_latent = _validate_latent(latent_batch)
        item_latent = _validate_latent(latent)
        batch_samples: torch.Tensor = batch_latent["samples"]
        item_samples: torch.Tensor = item_latent["samples"]

        output = dict(batch_latent)
        output["samples"] = _insert_tensor_batch(batch_samples, item_samples, index, input_name="LATENT")

        if "noise_mask" in batch_latent or "noise_mask" in item_latent:
            if "noise_mask" not in batch_latent or "noise_mask" not in item_latent:
                raise ValueError("Both latent_batch and latent must include noise_mask when either input includes it.")
            batch_mask = _normalize_latent_mask(
                batch_latent["noise_mask"],
                int(batch_samples.shape[0]),
                input_name="latent_batch",
            )
            item_mask = _normalize_latent_mask(
                item_latent["noise_mask"],
                int(item_samples.shape[0]),
                input_name="latent",
            )
            output["noise_mask"] = _insert_tensor_batch(batch_mask, item_mask, index, input_name="LATENT noise_mask")

        if "batch_index" in batch_latent:
            existing_indices = list(batch_latent["batch_index"])
            inserted_indices = list(item_latent.get("batch_index", range(int(item_samples.shape[0]))))
            output["batch_index"] = (
                existing_indices[: int(index)]
                + inserted_indices
                + existing_indices[int(index):]
            )

        logger.debug(
            "LatentToBatch inserted %d latent sample(s) into batch_size=%d at index=%d; output_shape=%s",
            int(item_samples.shape[0]),
            int(batch_samples.shape[0]),
            int(index),
            tuple(output["samples"].shape),
        )
        return (output,)


NODE_CLASS_MAPPINGS = {
    "LatentToImage": LatentToImage,
    "ImageBatchToLatent": ImageBatchToLatent,
    "ImageToBatch": ImageToBatch,
    "LatentToBatch": LatentToBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentToImage": "Latent to Image Batch",
    "ImageBatchToLatent": "Image Batch to Latent",
    "ImageToBatch": "ImageToBatch",
    "LatentToBatch": "LatentToBatch",
}
