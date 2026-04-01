from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_DELEGATE_OVERRIDE_KEY = "_delegate_override"
_SCOPE_OPTIONS = ("self", "cross", "both")


@dataclass
class RotorQuantAttentionConfig:
    enabled: bool = True
    keep_components: int = 3
    min_token_product: int = 65536
    attention_scope: str = "self"
    layer_start: int = -1
    layer_end: int = -1
    rotation_seed: int = 0
    max_head_dim: int = 256
    force_fp32: bool = False


def _delegate_or_original(
    original_func: Callable[..., torch.Tensor],
    delegate_override: Optional[Callable[..., torch.Tensor]],
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    if callable(delegate_override) and delegate_override is not rotorquant_attention_override:
        return delegate_override(original_func, *args, **kwargs)
    return original_func(*args, **kwargs)


def _resolve_config(config_dict: Optional[Dict[str, Any]]) -> RotorQuantAttentionConfig:
    if not isinstance(config_dict, dict):
        return RotorQuantAttentionConfig(enabled=False)

    attention_scope = str(config_dict.get("attention_scope", "self")).strip().lower()
    if attention_scope not in _SCOPE_OPTIONS:
        attention_scope = "self"

    requested_keep_components = int(config_dict.get("keep_components", 3))
    keep_components = 3
    if requested_keep_components != 3:
        logger.warning(
            "RotorQuant attention: requested keep_components=%d, but lossy modes are disabled because they produce poor image quality. Using keep_components=3.",
            requested_keep_components,
        )

    return RotorQuantAttentionConfig(
        enabled=bool(config_dict.get("enabled", True)),
        keep_components=keep_components,
        min_token_product=max(0, int(config_dict.get("min_token_product", 65536))),
        attention_scope=attention_scope,
        layer_start=int(config_dict.get("layer_start", -1)),
        layer_end=int(config_dict.get("layer_end", -1)),
        rotation_seed=int(config_dict.get("rotation_seed", 0)),
        max_head_dim=max(3, int(config_dict.get("max_head_dim", 256))),
        force_fp32=bool(config_dict.get("force_fp32", False)),
    )


def _reshape_for_attention(tensor: torch.Tensor, heads: int, skip_reshape: bool) -> torch.Tensor:
    if skip_reshape:
        if tensor.ndim != 4:
            raise ValueError(f"Expected rank-4 tensor when skip_reshape=True, got shape {tuple(tensor.shape)}")
        if int(tensor.shape[1]) != int(heads):
            raise ValueError(f"Expected {heads} heads, got tensor shape {tuple(tensor.shape)}")
        return tensor

    if tensor.ndim != 3:
        raise ValueError(f"Expected rank-3 tensor when skip_reshape=False, got shape {tuple(tensor.shape)}")

    batch, tokens, width = tensor.shape
    if width % heads != 0:
        raise ValueError(f"Embedding width {width} is not divisible by heads={heads}")

    dim_head = width // heads
    return tensor.reshape(batch, tokens, heads, dim_head).permute(0, 2, 1, 3).contiguous()


def _restore_attention_layout(tensor: torch.Tensor, skip_output_reshape: bool) -> torch.Tensor:
    if skip_output_reshape:
        return tensor
    batch, heads, tokens, dim_head = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim_head)


@lru_cache(maxsize=128)
def _rotation_bank_cpu(num_groups: int, seed: int) -> torch.Tensor:
    if num_groups <= 0:
        return torch.empty((0, 3, 3), dtype=torch.float32)

    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    axes = torch.randn((num_groups, 3), generator=generator, dtype=torch.float32)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    angles = torch.rand((num_groups,), generator=generator, dtype=torch.float32) * (2.0 * math.pi)

    x = axes[:, 0]
    y = axes[:, 1]
    z = axes[:, 2]
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    one_minus_cos = 1.0 - cos_theta

    rotation = torch.empty((num_groups, 3, 3), dtype=torch.float32)
    rotation[:, 0, 0] = cos_theta + (x * x * one_minus_cos)
    rotation[:, 0, 1] = (x * y * one_minus_cos) - (z * sin_theta)
    rotation[:, 0, 2] = (x * z * one_minus_cos) + (y * sin_theta)
    rotation[:, 1, 0] = (y * x * one_minus_cos) + (z * sin_theta)
    rotation[:, 1, 1] = cos_theta + (y * y * one_minus_cos)
    rotation[:, 1, 2] = (y * z * one_minus_cos) - (x * sin_theta)
    rotation[:, 2, 0] = (z * x * one_minus_cos) - (y * sin_theta)
    rotation[:, 2, 1] = (z * y * one_minus_cos) + (x * sin_theta)
    rotation[:, 2, 2] = cos_theta + (z * z * one_minus_cos)
    return rotation.contiguous()


def _pad_last_dim(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    pad = (-int(tensor.shape[-1])) % int(multiple)
    if pad == 0:
        return tensor, 0
    padding = tensor.new_zeros((*tensor.shape[:-1], pad))
    return torch.cat((tensor, padding), dim=-1), pad


def _project_triplets(
    tensor: torch.Tensor,
    rotations: torch.Tensor,
    keep_components: int,
) -> tuple[torch.Tensor, int, int]:
    padded, pad = _pad_last_dim(tensor, 3)
    groups = padded.shape[-1] // 3
    grouped = padded.reshape(*padded.shape[:-1], groups, 3)
    rotated = torch.einsum("...gc,gcd->...gd", grouped, rotations)
    compact = rotated[..., :keep_components].reshape(*rotated.shape[:-2], groups * keep_components)
    return compact.contiguous(), int(tensor.shape[-1]), int(pad)


def _restore_triplets(
    tensor: torch.Tensor,
    rotations: torch.Tensor,
    keep_components: int,
    original_dim: int,
) -> torch.Tensor:
    groups = rotations.shape[0]
    grouped = tensor.reshape(*tensor.shape[:-1], groups, keep_components)
    if keep_components < 3:
        expanded = grouped.new_zeros((*grouped.shape[:-1], 3))
        expanded[..., :keep_components] = grouped
    else:
        expanded = grouped
    restored = torch.einsum("...gc,gcd->...gd", expanded, rotations.transpose(-1, -2))
    return restored.reshape(*restored.shape[:-2], groups * 3)[..., :original_dim].contiguous()


def _scope_matches(attention_scope: str, query_tokens: int, key_tokens: int) -> bool:
    if attention_scope == "both":
        return True
    is_self_attention = int(query_tokens) == int(key_tokens)
    if attention_scope == "self":
        return is_self_attention
    return not is_self_attention


def _layer_matches(cfg: RotorQuantAttentionConfig, transformer_options: Optional[Dict[str, Any]]) -> bool:
    if cfg.layer_start < 0 and cfg.layer_end < 0:
        return True
    if not isinstance(transformer_options, dict):
        return False
    block_index = transformer_options.get("block_index")
    if not isinstance(block_index, int):
        return False
    if cfg.layer_start >= 0 and block_index < cfg.layer_start:
        return False
    if cfg.layer_end >= 0 and block_index > cfg.layer_end:
        return False
    return True


def _effective_rotation_seed(cfg: RotorQuantAttentionConfig, transformer_options: Optional[Dict[str, Any]]) -> int:
    block_index = -1
    if isinstance(transformer_options, dict):
        block_index = int(transformer_options.get("block_index", -1))
    if block_index < 0:
        return int(cfg.rotation_seed)
    return int(cfg.rotation_seed) + (block_index * 4099)


def rotorquant_attention_override(original_func: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
    transformer_options = kwargs.get("transformer_options")
    config_dict = transformer_options.get("rotorquant_attention") if isinstance(transformer_options, dict) else None
    delegate_override = config_dict.get(_DELEGATE_OVERRIDE_KEY) if isinstance(config_dict, dict) else None
    cfg = _resolve_config(config_dict)
    if not cfg.enabled:
        return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

    try:
        q = args[0]
        k = args[1]
        v = args[2]
        heads = int(args[3])
        skip_reshape = bool(kwargs.get("skip_reshape", False))
        skip_output_reshape = bool(kwargs.get("skip_output_reshape", False))

        q_expanded = _reshape_for_attention(q, heads=heads, skip_reshape=skip_reshape)
        k_expanded = _reshape_for_attention(k, heads=heads, skip_reshape=skip_reshape)
        v_expanded = _reshape_for_attention(v, heads=heads, skip_reshape=skip_reshape)

        if q_expanded.shape[-1] != k_expanded.shape[-1] or q_expanded.shape[-1] != v_expanded.shape[-1]:
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        batch, _, query_tokens, dim_head = q_expanded.shape
        key_tokens = int(k_expanded.shape[2])
        if dim_head <= 0 or dim_head > cfg.max_head_dim:
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)
        if (query_tokens * key_tokens) < cfg.min_token_product:
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)
        if not _scope_matches(cfg.attention_scope, query_tokens, key_tokens):
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)
        if not _layer_matches(cfg, transformer_options):
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        compute_dtype = torch.float32 if cfg.force_fp32 else q_expanded.dtype
        q_work = q_expanded.to(dtype=compute_dtype)
        k_work = k_expanded.to(dtype=compute_dtype)
        v_work = v_expanded.to(dtype=compute_dtype)

        padded_dim = dim_head + ((-dim_head) % 3)
        num_groups = padded_dim // 3
        if num_groups <= 0:
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        rotations = _rotation_bank_cpu(num_groups, _effective_rotation_seed(cfg, transformer_options)).to(
            device=q_work.device,
            dtype=compute_dtype,
        )

        q_compact, original_dim, _ = _project_triplets(q_work, rotations, cfg.keep_components)
        k_compact, _, _ = _project_triplets(k_work, rotations, cfg.keep_components)
        v_compact, _, _ = _project_triplets(v_work, rotations, cfg.keep_components)
        compact_dim = q_compact.shape[-1]
        if compact_dim <= 0 or original_dim <= 0:
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        scale_ratio = math.sqrt(float(compact_dim) / float(original_dim))
        q_compact = q_compact * scale_ratio

        out_compact = original_func(
            q_compact,
            k_compact,
            v_compact,
            heads,
            mask=kwargs.get("mask"),
            attn_precision=kwargs.get("attn_precision"),
            skip_reshape=True,
            skip_output_reshape=True,
            transformer_options=transformer_options,
        )
        if out_compact.shape != (batch, heads, query_tokens, num_groups * cfg.keep_components):
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        restored = _restore_triplets(
            out_compact,
            rotations,
            keep_components=cfg.keep_components,
            original_dim=original_dim,
        ).to(dtype=v.dtype if v.dtype.is_floating_point else q.dtype)
        return _restore_attention_layout(restored, skip_output_reshape=skip_output_reshape)
    except Exception as exc:
        logger.debug("RotorQuant attention fallback: %s", exc)
        return _delegate_or_original(original_func, delegate_override, *args, **kwargs)


class RotorQuantAttentionModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to clone and patch with RotorQuant-style attention."}),
                "keep_components": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 3,
                    "tooltip": "How many rotated coordinates to keep from each 3D rotor group. Values below 3 are currently forced back to 3 because the lossy modes degrade image quality too much.",
                }),
                "min_token_product": ("INT", {
                    "default": 65536,
                    "min": 0,
                    "max": 1073741824,
                    "tooltip": "Only patch attention calls where query_tokens * key_tokens meets this threshold.",
                }),
                "attention_scope": (_SCOPE_OPTIONS, {
                    "tooltip": "Which attention calls to patch. 'self' is usually the most useful for diffusion latents.",
                }),
                "layer_start": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4096,
                    "tooltip": "First transformer block index to patch. -1 disables the lower bound.",
                }),
                "layer_end": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4096,
                    "tooltip": "Last transformer block index to patch. -1 disables the upper bound.",
                }),
                "rotation_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed used to generate the per-triplet rotor rotations.",
                }),
                "max_head_dim": ("INT", {
                    "default": 256,
                    "min": 3,
                    "max": 4096,
                    "tooltip": "Skip heads larger than this to avoid excessive projection overhead.",
                }),
                "force_fp32": (["disable", "enable"], {
                    "tooltip": "Cast q/k/v to fp32 inside the override for extra numerical stability.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "model/patch"

    def patch_model(
        self,
        model,
        keep_components,
        min_token_product,
        attention_scope,
        layer_start,
        layer_end,
        rotation_seed,
        max_head_dim,
        force_fp32,
    ):
        patched = model.clone()

        existing_model_options = getattr(patched, "model_options", {})
        if not isinstance(existing_model_options, dict):
            existing_model_options = {}
        patched.model_options = dict(existing_model_options)

        existing_transformer_options = patched.model_options.get("transformer_options", {})
        if not isinstance(existing_transformer_options, dict):
            existing_transformer_options = {}
        transformer_options = dict(existing_transformer_options)
        patched.model_options["transformer_options"] = transformer_options

        previous_config = transformer_options.get("rotorquant_attention")
        existing_override = transformer_options.get("optimized_attention_override")
        delegate_override = None
        if callable(existing_override) and existing_override is not rotorquant_attention_override:
            delegate_override = existing_override
        elif isinstance(previous_config, dict):
            previous_delegate = previous_config.get(_DELEGATE_OVERRIDE_KEY)
            if callable(previous_delegate) and previous_delegate is not rotorquant_attention_override:
                delegate_override = previous_delegate

        normalized_keep_components = 3
        if int(keep_components) != 3:
            logger.warning(
                "RotorQuant attention patch requested keep_components=%d, but lossy modes are disabled because they produce poor image quality. Storing keep_components=3.",
                int(keep_components),
            )

        transformer_options["rotorquant_attention"] = {
            "enabled": True,
            "keep_components": normalized_keep_components,
            "min_token_product": int(min_token_product),
            "attention_scope": str(attention_scope),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "rotation_seed": int(rotation_seed),
            "max_head_dim": int(max_head_dim),
            "force_fp32": force_fp32 == "enable",
            _DELEGATE_OVERRIDE_KEY: delegate_override,
        }
        transformer_options["optimized_attention_override"] = rotorquant_attention_override
        logger.info(
            "RotorQuant attention patch applied: keep=%d min_token_product=%d scope=%s layer_start=%d layer_end=%d seed=%d max_head_dim=%d force_fp32=%s",
            int(normalized_keep_components),
            int(min_token_product),
            str(attention_scope),
            int(layer_start),
            int(layer_end),
            int(rotation_seed),
            int(max_head_dim),
            str(force_fp32 == "enable"),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "RotorQuantAttentionModelPatch": RotorQuantAttentionModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RotorQuantAttentionModelPatch": "Model (RotorQuant Attention)",
}
