from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare_mask_nchw(
    mask: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Normalize a ComfyUI `MASK` input into an NCHW `(B,1,H,W)` tensor, resized to `height x width`.

    ComfyUI convention: masks are often supplied in image space, so we resize to latent resolution.
    When downscaling, we use bicubic interpolation.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected mask as torch.Tensor, got {type(mask)}")

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[1] == 1:
            pass
        elif mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        else:
            mask = mask.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Expected mask tensor with shape (H,W), (B,H,W), or (B,1,H,W), got {tuple(mask.shape)}")

    batch_size = int(batch_size)
    if int(mask.shape[0]) != batch_size:
        if int(mask.shape[0]) == 1:
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            raise ValueError(f"Mask batch size {int(mask.shape[0])} does not match target batch size {batch_size}.")

    mask = mask.to(device=device, dtype=torch.float32)
    height = int(height)
    width = int(width)
    src_h = int(mask.shape[-2])
    src_w = int(mask.shape[-1])
    if (src_h, src_w) != (height, width):
        downscaling = src_h > height or src_w > width
        mode = "bicubic" if downscaling else "bilinear"
        mask = F.interpolate(
            mask,
            size=(height, width),
            mode=mode,
            align_corners=False,
        )

    return mask.clamp(0.0, 1.0)


def broadcast_mask_nchw(mask_nchw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a `(B,1,H,W)` mask to match `target`'s batch/spatial dimensions.

    Returns a view that is broadcastable to `target` when used in arithmetic.
    """
    if not isinstance(mask_nchw, torch.Tensor):
        raise TypeError(f"Expected mask_nchw as torch.Tensor, got {type(mask_nchw)}")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected target as torch.Tensor, got {type(target)}")
    if mask_nchw.ndim != 4 or int(mask_nchw.shape[1]) != 1:
        raise ValueError(f"Expected mask_nchw with shape (B,1,H,W), got {tuple(mask_nchw.shape)}")
    if target.ndim < 3:
        raise ValueError(f"Expected target with at least 3 dims, got {tuple(target.shape)}")
    if int(target.shape[0]) != int(mask_nchw.shape[0]):
        raise ValueError(f"Mask batch size {int(mask_nchw.shape[0])} does not match target batch size {int(target.shape[0])}.")
    if tuple(target.shape[-2:]) != tuple(mask_nchw.shape[-2:]):
        raise ValueError(
            f"Mask spatial size {tuple(mask_nchw.shape[-2:])} does not match target spatial size {tuple(target.shape[-2:])}."
        )

    if target.ndim == 3:
        return mask_nchw.squeeze(1)

    mask = mask_nchw
    for _ in range(target.ndim - 4):
        mask = mask.unsqueeze(2)
    return mask


def blend_with_mask(original: torch.Tensor, modified: torch.Tensor, mask_nchw: torch.Tensor) -> torch.Tensor:
    """
    Blend tensors as: `original * (1-mask) + modified * mask`.
    """
    if not isinstance(original, torch.Tensor) or not isinstance(modified, torch.Tensor):
        raise TypeError("blend_with_mask expects torch.Tensor inputs.")
    if tuple(original.shape) != tuple(modified.shape):
        raise ValueError(f"Shape mismatch: original={tuple(original.shape)} modified={tuple(modified.shape)}")

    mask = broadcast_mask_nchw(mask_nchw, original).to(dtype=original.dtype)
    return original * (1.0 - mask) + modified.to(dtype=original.dtype) * mask


def blend_image_with_mask(original: torch.Tensor, modified: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Blend BHWC/BTHWC image tensors with a ComfyUI `MASK` input resized to the image resolution.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected mask as torch.Tensor, got {type(mask)}")
    if not isinstance(original, torch.Tensor) or not isinstance(modified, torch.Tensor):
        raise TypeError("blend_image_with_mask expects torch.Tensor inputs.")
    if tuple(original.shape) != tuple(modified.shape):
        raise ValueError(f"Shape mismatch: original={tuple(original.shape)} modified={tuple(modified.shape)}")

    if original.ndim == 4:
        batch, height, width, _ = original.shape
        mask_nchw = prepare_mask_nchw(mask, batch_size=int(batch), height=int(height), width=int(width), device=original.device)
        original_cf = original.permute(0, 3, 1, 2)
        modified_cf = modified.permute(0, 3, 1, 2)
        blended_cf = blend_with_mask(original_cf, modified_cf, mask_nchw)
        return blended_cf.permute(0, 2, 3, 1)

    if original.ndim == 5:
        batch, frames, height, width, _ = original.shape
        mask_nchw = prepare_mask_nchw(mask, batch_size=int(batch), height=int(height), width=int(width), device=original.device)
        original_cf = original.permute(0, 4, 1, 2, 3)
        modified_cf = modified.permute(0, 4, 1, 2, 3)
        blended_cf = blend_with_mask(original_cf, modified_cf, mask_nchw)
        return blended_cf.permute(0, 2, 3, 4, 1)

    raise ValueError(f"Expected image tensor with shape (B,H,W,C) or (B,T,H,W,C), got {tuple(original.shape)}")
