from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

try:
    from .masking import blend_with_mask, prepare_mask_nchw
except ImportError:  # pragma: no cover - fallback for direct module loading
    from masking import blend_with_mask, prepare_mask_nchw

logger = logging.getLogger(__name__)

_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF
_SEED_STRIDE = 0x9E3779B97F4A7C15
_TILE_STRIDE = 0xBF58476D1CE4E5B9
_EPS = 1e-6

_LINEAR_OPS = (
    "signed_permute",
    "orthogonal_rotate",
    "householder_reflect",
    "low_rank_shear",
)

_NONLINEAR_OPS = (
    "gate_multiply",
    "gate_add",
    "quantize",
    "clip_hard",
    "clip_soft",
    "dropout_zero",
    "dropout_noise",
    "dropout_swap",
)

_SLOT_OPS = (
    "shuffle",
    "rotate_cw",
    "rotate_ccw",
    "flip_h",
    "flip_v",
)


@dataclass(frozen=True)
class _Flattened2D:
    tensor: torch.Tensor
    restore: Callable[[torch.Tensor], torch.Tensor]
    batch_size: int
    extra_dim: int


def _flatten_to_nchw(tensor: torch.Tensor) -> _Flattened2D:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    original_shape = tuple(tensor.shape)
    squeeze_channel = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
        squeeze_channel = True

    if tensor.ndim < 4:
        raise ValueError(f"Expected tensor with at least 4 dims (B,C,H,W), got {original_shape}")

    batch_size = int(tensor.shape[0])
    extra_dim = 1
    reshape_higher = tensor.ndim > 4
    if reshape_higher:
        reshape_base = tuple(tensor.shape)
        extra_dim = int(math.prod(reshape_base[2:-2])) if len(reshape_base) > 4 else 1
        tensor = tensor.reshape(reshape_base[0], reshape_base[1], -1, reshape_base[-2], reshape_base[-1])
        tensor = tensor.movedim(2, 1).reshape(-1, reshape_base[1], reshape_base[-2], reshape_base[-1])

        def restore(out: torch.Tensor) -> torch.Tensor:
            restored = out.reshape(
                original_shape[0],
                -1,
                original_shape[1],
                original_shape[-2],
                original_shape[-1],
            )
            restored = restored.movedim(2, 1).reshape(original_shape)
            return restored

        return _Flattened2D(tensor=tensor, restore=restore, batch_size=batch_size, extra_dim=extra_dim)

    def restore(out: torch.Tensor) -> torch.Tensor:
        if squeeze_channel:
            return out.squeeze(1)
        return out

    return _Flattened2D(tensor=tensor, restore=restore, batch_size=batch_size, extra_dim=extra_dim)


def _validate_latent(latent: object) -> Dict:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("LATENT input must be a dictionary containing a 'samples' tensor.")
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"LATENT['samples'] must be a torch.Tensor, got {type(samples)}.")
    if samples.ndim < 4:
        raise ValueError(f"LATENT['samples'] must have at least 4 dimensions (B,C,H,W), got {tuple(samples.shape)}.")
    if not samples.is_floating_point():
        raise TypeError(f"LATENT['samples'] must be floating point, got dtype={samples.dtype}.")
    return latent


def _seed_for_index(seed: int, index: int, *, extra: int = 0) -> int:
    return (int(seed) + int(index) * _SEED_STRIDE + int(extra) * _TILE_STRIDE) & _SEED_MASK_64


def _resolve_selection_count(channels: int, fraction: float, count: int) -> int:
    count = int(count)
    if count > 0:
        return min(channels, count)
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction <= 0.0:
        return 0
    return min(channels, max(1, int(math.ceil(channels * fraction))))


def _parse_channel_indices(raw: str, *, channels: int) -> torch.Tensor:
    raw = str(raw).strip()
    if not raw:
        return torch.empty(0, dtype=torch.long)
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    indices: List[int] = []
    seen = set()
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid channel index '{part}'.") from exc
        if value < 0:
            value += channels
        if value < 0 or value >= channels:
            raise ValueError(f"Channel index {value} out of range for {channels} channels.")
        if value not in seen:
            indices.append(value)
            seen.add(value)
    return torch.tensor(indices, dtype=torch.long)


def _channel_variance(sample: torch.Tensor) -> torch.Tensor:
    return sample.float().var(dim=(1, 2), unbiased=False)


def _channel_roughness(sample: torch.Tensor) -> torch.Tensor:
    sample_f = sample.float()
    dx = sample_f[:, :, 1:] - sample_f[:, :, :-1]
    dy = sample_f[:, 1:, :] - sample_f[:, :-1, :]
    rough = dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2))
    return rough


def _select_channel_indices(
    samples: torch.Tensor,
    *,
    mode: str,
    fraction: float,
    count: int,
    seed: int,
    indices: torch.Tensor,
    order: str,
) -> List[torch.Tensor]:
    mode = str(mode).strip().lower()
    order = str(order).strip().lower()
    if mode not in ("all", "random", "top_variance", "top_roughness", "indices"):
        raise ValueError(f"Unknown channel selection mode '{mode}'.")
    if mode == "indices":
        return [indices] * int(samples.shape[0])

    channels = int(samples.shape[1])
    if mode == "all":
        full = torch.arange(channels, dtype=torch.long, device=samples.device)
        return [full] * int(samples.shape[0])

    pick = _resolve_selection_count(channels, fraction, count)
    if pick == 0:
        empty = torch.empty(0, dtype=torch.long, device=samples.device)
        return [empty] * int(samples.shape[0])

    largest = order != "lowest"
    selected: List[torch.Tensor] = []
    for sample_index in range(int(samples.shape[0])):
        sample = samples[sample_index]
        if mode == "random":
            generator = torch.Generator(device="cpu").manual_seed(_seed_for_index(seed, sample_index))
            perm = torch.randperm(channels, generator=generator)
            chosen = perm[:pick].to(device=samples.device)
        else:
            metric = _channel_variance(sample) if mode == "top_variance" else _channel_roughness(sample)
            _, chosen = torch.topk(metric, k=pick, largest=largest)
        selected.append(chosen.to(device=samples.device, dtype=torch.long))
    return selected


def _match_channel_stats(original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
    orig_mean = original.mean(dim=(1, 2), keepdim=True)
    orig_std = original.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(_EPS)
    mod_mean = modified.mean(dim=(1, 2), keepdim=True)
    mod_std = modified.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(_EPS)
    return (modified - mod_mean) * (orig_std / mod_std) + orig_mean


def _apply_mix(original: torch.Tensor, modified: torch.Tensor, mix: float) -> torch.Tensor:
    mix = max(0.0, min(1.0, float(mix)))
    if mix <= 0.0:
        return original
    if mix >= 1.0:
        return modified
    return original + (modified - original) * mix


def _blur_scalar_map(field: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return field
    kernel = radius * 2 + 1
    return F.avg_pool2d(field, kernel_size=kernel, stride=1, padding=radius)


def _orthogonal_matrix(dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & _SEED_MASK_64)
    mat = torch.randn((dim, dim), generator=generator, device="cpu", dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag = torch.sign(torch.diag(r))
    diag[diag == 0] = 1.0
    q = q * diag
    return q


def _apply_signed_permutation(
    sample: torch.Tensor,
    indices: torch.Tensor,
    *,
    seed: int,
    sign_flip_prob: float,
    tile_size: int,
    sample_index: int,
) -> torch.Tensor:
    if indices.numel() == 0:
        return sample

    _, height, width = sample.shape
    out = sample.clone()
    sign_flip_prob = max(0.0, min(1.0, float(sign_flip_prob)))
    tile_size = int(tile_size)

    def apply_for_tile(y0: int, y1: int, x0: int, x1: int, *, tile_id: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(_seed_for_index(seed, sample_index, extra=tile_id))
        perm = torch.randperm(int(indices.numel()), generator=generator).to(device=indices.device)
        source = indices[perm]
        if sign_flip_prob > 0.0:
            flips = torch.rand(int(indices.numel()), generator=generator) < sign_flip_prob
            signs = torch.where(flips, torch.tensor(-1.0), torch.tensor(1.0))
        else:
            signs = torch.ones(int(indices.numel()))
        signs = signs.to(device=sample.device, dtype=sample.dtype).view(-1, 1, 1)
        out[indices, y0:y1, x0:x1] = sample[source, y0:y1, x0:x1] * signs

    if tile_size <= 0:
        apply_for_tile(0, height, 0, width, tile_id=0)
        return out

    tile_id = 0
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            apply_for_tile(y0, y1, x0, x1, tile_id=tile_id)
            tile_id += 1
    return out


def _apply_orthogonal_rotation(
    sample: torch.Tensor,
    indices: torch.Tensor,
    *,
    seed: int,
    block_size: int,
    sample_index: int,
) -> torch.Tensor:
    if indices.numel() == 0:
        return sample

    out = sample.clone()
    block_size = int(block_size)
    channels = int(indices.numel())
    if block_size <= 0 or block_size >= channels:
        block_size = channels

    device = sample.device
    work_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    flat = sample[indices].to(dtype=work_dtype).reshape(channels, -1)

    for start in range(0, channels, block_size):
        end = min(channels, start + block_size)
        block_len = end - start
        if block_len <= 1:
            out[indices[start:end]] = flat[start:end].reshape(block_len, *sample.shape[1:]).to(dtype=sample.dtype)
            continue
        q = _orthogonal_matrix(block_len, seed=_seed_for_index(seed, sample_index, extra=start))
        q = q.to(device=device, dtype=work_dtype)
        block = flat[start:end]
        mixed = q @ block
        out[indices[start:end]] = mixed.reshape(block_len, *sample.shape[1:]).to(dtype=sample.dtype)
    return out


def _apply_householder_reflect(
    sample: torch.Tensor,
    indices: torch.Tensor,
    *,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    if indices.numel() == 0:
        return sample

    out = sample.clone()
    device = sample.device
    work_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    flat = sample[indices].to(dtype=work_dtype).reshape(int(indices.numel()), -1)
    generator = torch.Generator(device="cpu").manual_seed(_seed_for_index(seed, sample_index))
    vec = torch.randn(int(indices.numel()), generator=generator, device="cpu", dtype=torch.float32)
    vec = vec.to(device=device, dtype=work_dtype)
    vec = vec / (vec.norm() + _EPS)
    proj = (vec[:, None] * flat).sum(dim=0)
    reflected = flat - 2.0 * vec[:, None] * proj
    out[indices] = reflected.reshape(int(indices.numel()), *sample.shape[1:]).to(dtype=sample.dtype)
    return out


def _apply_low_rank_shear(
    sample: torch.Tensor,
    indices: torch.Tensor,
    *,
    seed: int,
    alpha: float,
    sample_index: int,
) -> torch.Tensor:
    if indices.numel() == 0:
        return sample

    out = sample.clone()
    device = sample.device
    work_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    flat = sample[indices].to(dtype=work_dtype).reshape(int(indices.numel()), -1)
    generator = torch.Generator(device="cpu").manual_seed(_seed_for_index(seed, sample_index))
    u = torch.randn(int(indices.numel()), generator=generator, device="cpu", dtype=torch.float32)
    v = torch.randn(int(indices.numel()), generator=generator, device="cpu", dtype=torch.float32)
    u = u.to(device=device, dtype=work_dtype)
    v = v.to(device=device, dtype=work_dtype)
    u = u / (u.norm() + _EPS)
    v = v / (v.norm() + _EPS)
    proj = (v[:, None] * flat).sum(dim=0)
    sheared = flat + float(alpha) * u[:, None] * proj
    out[indices] = sheared.reshape(int(indices.numel()), *sample.shape[1:]).to(dtype=sample.dtype)
    return out


def _apply_gating(
    sample: torch.Tensor,
    indices: torch.Tensor,
    *,
    seed: int,
    gate_strength: float,
    beta: float,
    blur_radius: int,
    mode: str,
    sample_index: int,
) -> torch.Tensor:
    if indices.numel() == 0:
        return sample

    device = sample.device
    work_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    flat = sample[indices].to(dtype=work_dtype).reshape(int(indices.numel()), -1)
    generator = torch.Generator(device="cpu").manual_seed(_seed_for_index(seed, sample_index))
    w = torch.randn(int(indices.numel()), generator=generator, device="cpu", dtype=torch.float32)
    w = w.to(device=device, dtype=work_dtype)
    w = w / (w.norm() + _EPS)
    gate = torch.sigmoid(float(gate_strength) * (w[:, None] * flat).sum(dim=0))
    gate = gate.reshape(1, *sample.shape[1:])
    gate = _blur_scalar_map(gate, blur_radius)

    out = sample.clone()
    if mode == "gate_add":
        u = torch.randn(int(indices.numel()), generator=generator, device="cpu", dtype=torch.float32)
        u = u.to(device=device, dtype=work_dtype)
        u = u / (u.norm() + _EPS)
        updated = flat + float(beta) * u[:, None] * gate.reshape(1, -1)
    else:
        updated = flat * (1.0 + float(beta) * gate.reshape(1, -1))
    out[indices] = updated.reshape(int(indices.numel()), *sample.shape[1:]).to(dtype=sample.dtype)
    return out


def _apply_quantize(sample: torch.Tensor, indices: torch.Tensor, *, step: float) -> torch.Tensor:
    if indices.numel() == 0:
        return sample
    step = float(step)
    if step <= 0.0:
        return sample
    out = sample.clone()
    out[indices] = torch.round(out[indices] / step) * step
    return out


def _apply_clip(sample: torch.Tensor, indices: torch.Tensor, *, threshold: float, mode: str) -> torch.Tensor:
    if indices.numel() == 0:
        return sample
    threshold = float(threshold)
    if threshold <= 0.0:
        return sample
    out = sample.clone()
    if mode == "clip_soft":
        out[indices] = threshold * torch.tanh(out[indices] / threshold)
    else:
        out[indices] = out[indices].clamp(-threshold, threshold)
    return out


def _apply_dropout(
    sample: torch.Tensor,
    indices: torch.Tensor,
    *,
    seed: int,
    mode: str,
    sample_index: int,
) -> torch.Tensor:
    if indices.numel() == 0:
        return sample

    out = sample.clone()
    if mode == "dropout_zero":
        out[indices] = 0.0
        return out

    generator = torch.Generator(device="cpu").manual_seed(_seed_for_index(seed, sample_index))
    if mode == "dropout_noise":
        stats = sample[indices].float()
        mean = stats.mean(dim=(1, 2), keepdim=True)
        std = stats.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(_EPS)
        noise = torch.randn(stats.shape, generator=generator, device="cpu", dtype=torch.float32)
        noise = noise.to(device=sample.device)
        out[indices] = (mean + noise * std).to(dtype=sample.dtype)
        return out

    if mode == "dropout_swap":
        channels = int(sample.shape[0])
        replacements = torch.randint(0, channels, (int(indices.numel()),), generator=generator, device="cpu")
        replacements = replacements.to(device=sample.device, dtype=torch.long)
        out[indices] = sample[replacements]
        return out

    raise ValueError(f"Unknown dropout mode '{mode}'.")


def _apply_packed_slot_transform(
    sample: torch.Tensor,
    *,
    operation: str,
    patch_size: int,
    base_channels: int,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    slots = patch_size * patch_size
    channels = int(sample.shape[0])
    if base_channels <= 0:
        if channels % slots != 0:
            raise ValueError(
                f"Unable to infer base channels from {channels} channels and patch size {patch_size}."
            )
        base_channels = channels // slots
    if channels != base_channels * slots:
        raise ValueError(
            f"Expected {base_channels * slots} channels for base={base_channels}, patch={patch_size}, got {channels}."
        )

    packed = sample.reshape(base_channels, patch_size, patch_size, *sample.shape[1:])
    if operation == "shuffle":
        generator = torch.Generator(device="cpu").manual_seed(_seed_for_index(seed, sample_index))
        perm = torch.randperm(slots, generator=generator).to(device=sample.device)
        packed = packed.reshape(base_channels, slots, *sample.shape[1:])[:, perm]
        packed = packed.reshape(base_channels, patch_size, patch_size, *sample.shape[1:])
    elif operation == "rotate_cw":
        packed = torch.rot90(packed, k=-1, dims=(1, 2))
    elif operation == "rotate_ccw":
        packed = torch.rot90(packed, k=1, dims=(1, 2))
    elif operation == "flip_h":
        packed = torch.flip(packed, dims=(2,))
    elif operation == "flip_v":
        packed = torch.flip(packed, dims=(1,))
    else:
        raise ValueError(f"Unknown packed slot operation '{operation}'.")

    return packed.reshape(channels, *sample.shape[1:])


def _expand_mask(mask_nchw: torch.Tensor, extra_dim: int) -> torch.Tensor:
    extra_dim = int(extra_dim)
    if extra_dim <= 1:
        return mask_nchw
    return mask_nchw.repeat_interleave(extra_dim, dim=0)


class LatentChannelLinearTransform:
    """
    Applies linear channel-space transforms to a latent tensor.
    """

    CATEGORY = "latent/channel"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "transform"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to transform in channel-space."}),
                "operation": (_LINEAR_OPS, {"tooltip": "Linear channel-space transform to apply."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed for deterministic channel selection and random transforms.",
                }),
                "sign_flip_prob": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Probability of flipping channel signs when using signed_permute.",
                }),
                "tile_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "When >0 and operation is signed_permute, apply a new permutation per tile (latent pixels).",
                }),
                "block_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Channels per orthogonal block. 0 means full rotation.",
                }),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": -4.0,
                    "max": 4.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Shear strength for low_rank_shear.",
                }),
                "selection_mode": (["all", "random", "top_variance", "top_roughness", "indices"], {
                    "default": "all",
                    "tooltip": "How to choose which channels are transformed.",
                }),
                "selection_fraction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Fraction of channels to select when selection_count is 0.",
                }),
                "selection_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Exact number of channels to select (overrides selection_fraction when >0).",
                }),
                "selection_order": (["highest", "lowest"], {
                    "default": "highest",
                    "tooltip": "Whether to pick high or low variance/roughness channels.",
                }),
                "selection_indices": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated channel indices to select when selection_mode=indices.",
                }),
                "mix": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Blend factor for the transformed channels (0=off, 1=full).",
                }),
                "match_stats": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Match per-channel mean/std after the transform (stabilizes latent stats).",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask to limit the transform to masked areas."}),
            },
        }

    def transform(
        self,
        latent,
        operation: str,
        seed: int,
        sign_flip_prob: float,
        tile_size: int,
        block_size: int,
        alpha: float,
        selection_mode: str,
        selection_fraction: float,
        selection_count: int,
        selection_order: str,
        selection_indices: str,
        mix: float,
        match_stats: bool,
        mask=None,
    ):
        latent = _validate_latent(latent)
        samples = latent["samples"]
        flat = _flatten_to_nchw(samples)
        channels = int(flat.tensor.shape[1])
        indices = _parse_channel_indices(selection_indices, channels=channels)

        selected = _select_channel_indices(
            flat.tensor,
            mode=selection_mode,
            fraction=selection_fraction,
            count=selection_count,
            seed=int(seed),
            indices=indices.to(device=flat.tensor.device),
            order=selection_order,
        )

        logger.debug(
            "LatentChannelLinearTransform op=%s shape=%s selection=%s mix=%.3f match=%s",
            operation,
            tuple(samples.shape),
            selection_mode,
            float(mix),
            bool(match_stats),
        )

        out = flat.tensor.clone()
        with torch.no_grad():
            for sample_index in range(int(flat.tensor.shape[0])):
                sample = flat.tensor[sample_index]
                channel_indices = selected[sample_index]
                if channel_indices.numel() == 0:
                    continue
                if operation == "signed_permute":
                    transformed = _apply_signed_permutation(
                        sample,
                        channel_indices,
                        seed=int(seed),
                        sign_flip_prob=float(sign_flip_prob),
                        tile_size=int(tile_size),
                        sample_index=sample_index,
                    )
                elif operation == "orthogonal_rotate":
                    transformed = _apply_orthogonal_rotation(
                        sample,
                        channel_indices,
                        seed=int(seed),
                        block_size=int(block_size),
                        sample_index=sample_index,
                    )
                elif operation == "householder_reflect":
                    transformed = _apply_householder_reflect(
                        sample,
                        channel_indices,
                        seed=int(seed),
                        sample_index=sample_index,
                    )
                elif operation == "low_rank_shear":
                    transformed = _apply_low_rank_shear(
                        sample,
                        channel_indices,
                        seed=int(seed),
                        alpha=float(alpha),
                        sample_index=sample_index,
                    )
                else:
                    raise ValueError(f"Unknown linear operation '{operation}'.")

                if match_stats:
                    transformed = transformed.clone()
                    transformed[channel_indices] = _match_channel_stats(
                        sample[channel_indices],
                        transformed[channel_indices],
                    )
                transformed = _apply_mix(sample, transformed, mix)
                out[sample_index] = transformed

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=flat.batch_size,
                height=int(samples.shape[-2]),
                width=int(samples.shape[-1]),
                device=flat.tensor.device,
            )
            mask_nchw = _expand_mask(mask_nchw, flat.extra_dim)
            out = blend_with_mask(flat.tensor, out, mask_nchw)

        out_latent = latent.copy()
        out_latent["samples"] = flat.restore(out)
        return (out_latent,)


class LatentChannelNonlinearTransform:
    """
    Applies nonlinear channel-space transforms to a latent tensor.
    """

    CATEGORY = "latent/channel"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "transform"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to transform in channel-space."}),
                "operation": (_NONLINEAR_OPS, {"tooltip": "Nonlinear channel-space transform to apply."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed for deterministic gating/dropout patterns.",
                }),
                "gate_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Slope for gate sigmoid (gate_* operations).",
                }),
                "beta": ("FLOAT", {
                    "default": 1.0,
                    "min": -4.0,
                    "max": 4.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Gate strength for gate_* operations.",
                }),
                "blur_radius": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Blur radius for gate maps (latent pixels).",
                }),
                "quantize_step": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Step size for quantize operation.",
                }),
                "clip_threshold": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Clip threshold for clip_* operations.",
                }),
                "selection_mode": (["all", "random", "top_variance", "top_roughness", "indices"], {
                    "default": "all",
                    "tooltip": "How to choose which channels are transformed.",
                }),
                "selection_fraction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Fraction of channels to select when selection_count is 0.",
                }),
                "selection_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Exact number of channels to select (overrides selection_fraction when >0).",
                }),
                "selection_order": (["highest", "lowest"], {
                    "default": "highest",
                    "tooltip": "Whether to pick high or low variance/roughness channels.",
                }),
                "selection_indices": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated channel indices to select when selection_mode=indices.",
                }),
                "mix": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Blend factor for the transformed channels (0=off, 1=full).",
                }),
                "match_stats": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Match per-channel mean/std after the transform (stabilizes latent stats).",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask to limit the transform to masked areas."}),
            },
        }

    def transform(
        self,
        latent,
        operation: str,
        seed: int,
        gate_strength: float,
        beta: float,
        blur_radius: int,
        quantize_step: float,
        clip_threshold: float,
        selection_mode: str,
        selection_fraction: float,
        selection_count: int,
        selection_order: str,
        selection_indices: str,
        mix: float,
        match_stats: bool,
        mask=None,
    ):
        latent = _validate_latent(latent)
        samples = latent["samples"]
        flat = _flatten_to_nchw(samples)
        channels = int(flat.tensor.shape[1])
        indices = _parse_channel_indices(selection_indices, channels=channels)

        selected = _select_channel_indices(
            flat.tensor,
            mode=selection_mode,
            fraction=selection_fraction,
            count=selection_count,
            seed=int(seed),
            indices=indices.to(device=flat.tensor.device),
            order=selection_order,
        )

        logger.debug(
            "LatentChannelNonlinearTransform op=%s shape=%s selection=%s mix=%.3f match=%s",
            operation,
            tuple(samples.shape),
            selection_mode,
            float(mix),
            bool(match_stats),
        )

        out = flat.tensor.clone()
        with torch.no_grad():
            for sample_index in range(int(flat.tensor.shape[0])):
                sample = flat.tensor[sample_index]
                channel_indices = selected[sample_index]
                if channel_indices.numel() == 0:
                    continue
                if operation in ("gate_multiply", "gate_add"):
                    transformed = _apply_gating(
                        sample,
                        channel_indices,
                        seed=int(seed),
                        gate_strength=float(gate_strength),
                        beta=float(beta),
                        blur_radius=int(blur_radius),
                        mode=operation,
                        sample_index=sample_index,
                    )
                elif operation == "quantize":
                    transformed = _apply_quantize(sample, channel_indices, step=float(quantize_step))
                elif operation in ("clip_hard", "clip_soft"):
                    transformed = _apply_clip(
                        sample,
                        channel_indices,
                        threshold=float(clip_threshold),
                        mode=operation,
                    )
                elif operation in ("dropout_zero", "dropout_noise", "dropout_swap"):
                    transformed = _apply_dropout(
                        sample,
                        channel_indices,
                        seed=int(seed),
                        mode=operation,
                        sample_index=sample_index,
                    )
                else:
                    raise ValueError(f"Unknown nonlinear operation '{operation}'.")

                if match_stats:
                    transformed = transformed.clone()
                    transformed[channel_indices] = _match_channel_stats(
                        sample[channel_indices],
                        transformed[channel_indices],
                    )
                transformed = _apply_mix(sample, transformed, mix)
                out[sample_index] = transformed

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=flat.batch_size,
                height=int(samples.shape[-2]),
                width=int(samples.shape[-1]),
                device=flat.tensor.device,
            )
            mask_nchw = _expand_mask(mask_nchw, flat.extra_dim)
            out = blend_with_mask(flat.tensor, out, mask_nchw)

        out_latent = latent.copy()
        out_latent["samples"] = flat.restore(out)
        return (out_latent,)


class LatentChannelMerge:
    """
    Blends selected channels from a source latent into a destination latent.
    """

    CATEGORY = "latent/channel"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "destination": ("LATENT", {"tooltip": "Latent to blend into (destination)."}),
                "source": ("LATENT", {"tooltip": "Latent providing channels to blend (source)."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed for deterministic channel selection when selection_mode=random.",
                }),
                "selection_mode": (["all", "random", "top_variance", "top_roughness", "indices"], {
                    "default": "all",
                    "tooltip": "How to choose which source channels to blend.",
                }),
                "selection_fraction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Fraction of channels to select when selection_count is 0.",
                }),
                "selection_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Exact number of channels to select (overrides selection_fraction when >0).",
                }),
                "selection_order": (["highest", "lowest"], {
                    "default": "highest",
                    "tooltip": "Whether to pick high or low variance/roughness channels.",
                }),
                "selection_indices": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated channel indices to select when selection_mode=indices.",
                }),
                "blend_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -4.0,
                    "max": 4.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Blend strength for selected channels (0=none, 1=full, >1 or <0 allowed).",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask to limit the merge to masked areas."}),
            },
        }

    def merge(
        self,
        destination,
        source,
        seed: int,
        selection_mode: str,
        selection_fraction: float,
        selection_count: int,
        selection_order: str,
        selection_indices: str,
        blend_strength: float,
        mask=None,
    ):
        destination = _validate_latent(destination)
        source = _validate_latent(source)
        dest_samples = destination["samples"]
        source_samples = source["samples"]

        if dest_samples.shape != source_samples.shape:
            raise ValueError(
                "Source and destination latents must have the same shape; "
                f"got {tuple(source_samples.shape)} vs {tuple(dest_samples.shape)}."
            )
        if dest_samples.device != source_samples.device:
            raise ValueError(
                "Source and destination latents must be on the same device; "
                f"got {source_samples.device} vs {dest_samples.device}."
            )

        dest_flat = _flatten_to_nchw(dest_samples)
        source_flat = _flatten_to_nchw(source_samples)
        if dest_flat.tensor.shape != source_flat.tensor.shape:
            raise ValueError(
                "Source and destination latents must flatten to the same shape; "
                f"got {tuple(source_flat.tensor.shape)} vs {tuple(dest_flat.tensor.shape)}."
            )

        channels = int(dest_flat.tensor.shape[1])
        indices = _parse_channel_indices(selection_indices, channels=channels)
        selected = _select_channel_indices(
            source_flat.tensor,
            mode=selection_mode,
            fraction=selection_fraction,
            count=selection_count,
            seed=int(seed),
            indices=indices.to(device=source_flat.tensor.device),
            order=selection_order,
        )

        logger.debug(
            "LatentChannelMerge shape=%s selection=%s blend=%.3f",
            tuple(dest_samples.shape),
            selection_mode,
            float(blend_strength),
        )

        out = dest_flat.tensor.clone()
        blend_strength = float(blend_strength)
        with torch.no_grad():
            for sample_index in range(int(dest_flat.tensor.shape[0])):
                channel_indices = selected[sample_index]
                if channel_indices.numel() == 0:
                    continue
                dest_sample = dest_flat.tensor[sample_index]
                source_sample = source_flat.tensor[sample_index]
                blended = dest_sample[channel_indices] + (
                    source_sample[channel_indices] - dest_sample[channel_indices]
                ) * blend_strength
                out[sample_index, channel_indices] = blended

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=dest_flat.batch_size,
                height=int(dest_samples.shape[-2]),
                width=int(dest_samples.shape[-1]),
                device=dest_flat.tensor.device,
            )
            mask_nchw = _expand_mask(mask_nchw, dest_flat.extra_dim)
            out = blend_with_mask(dest_flat.tensor, out, mask_nchw)

        out_latent = destination.copy()
        out_latent["samples"] = dest_flat.restore(out)
        return (out_latent,)


class LatentPackedSlotTransform:
    """
    Applies slot-level operations to packed (space-to-depth) latents.
    """

    CATEGORY = "latent/channel"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "transform"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Packed latent to transform per space-to-depth slot."}),
                "operation": (_SLOT_OPS, {"tooltip": "Slot-space operation to apply."}),
                "patch_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Patch size used in the packed latent (P for PxP).",
                }),
                "base_channels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Base channels before packing. 0 infers from the latent.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed for deterministic slot shuffles.",
                }),
                "mix": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Blend factor for the transformed slots (0=off, 1=full).",
                }),
                "match_stats": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Match per-channel mean/std after the transform (stabilizes latent stats).",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask to limit the transform to masked areas."}),
            },
        }

    def transform(
        self,
        latent,
        operation: str,
        patch_size: int,
        base_channels: int,
        seed: int,
        mix: float,
        match_stats: bool,
        mask=None,
    ):
        latent = _validate_latent(latent)
        samples = latent["samples"]
        flat = _flatten_to_nchw(samples)

        logger.debug(
            "LatentPackedSlotTransform op=%s shape=%s patch=%d base=%d mix=%.3f match=%s",
            operation,
            tuple(samples.shape),
            int(patch_size),
            int(base_channels),
            float(mix),
            bool(match_stats),
        )

        out = flat.tensor.clone()
        with torch.no_grad():
            for sample_index in range(int(flat.tensor.shape[0])):
                sample = flat.tensor[sample_index]
                transformed = _apply_packed_slot_transform(
                    sample,
                    operation=operation,
                    patch_size=int(patch_size),
                    base_channels=int(base_channels),
                    seed=int(seed),
                    sample_index=sample_index,
                )
                if match_stats:
                    transformed = _match_channel_stats(sample, transformed)
                transformed = _apply_mix(sample, transformed, mix)
                out[sample_index] = transformed

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=flat.batch_size,
                height=int(samples.shape[-2]),
                width=int(samples.shape[-1]),
                device=flat.tensor.device,
            )
            mask_nchw = _expand_mask(mask_nchw, flat.extra_dim)
            out = blend_with_mask(flat.tensor, out, mask_nchw)

        out_latent = latent.copy()
        out_latent["samples"] = flat.restore(out)
        return (out_latent,)


NODE_CLASS_MAPPINGS = {
    "LatentChannelLinearTransform": LatentChannelLinearTransform,
    "LatentChannelNonlinearTransform": LatentChannelNonlinearTransform,
    "LatentChannelMerge": LatentChannelMerge,
    "LatentPackedSlotTransform": LatentPackedSlotTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentChannelLinearTransform": "Latent Channel Linear Transform",
    "LatentChannelNonlinearTransform": "Latent Channel Nonlinear Transform",
    "LatentChannelMerge": "Latent Channel Merge",
    "LatentPackedSlotTransform": "Latent Packed Slot Transform",
}
