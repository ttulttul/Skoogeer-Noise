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
_NORMAL_DIST = torch.distributions.Normal(0.0, 1.0)


@dataclass
class TurboQuantAttentionConfig:
    enabled: bool = True
    bits: int = 4
    qjl_dim: int = 64
    use_qjl: bool = True
    quantize_values: bool = True
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
    if callable(delegate_override) and delegate_override is not turboquant_attention_override:
        return delegate_override(original_func, *args, **kwargs)
    return original_func(*args, **kwargs)


def _resolve_config(config_dict: Optional[Dict[str, Any]]) -> TurboQuantAttentionConfig:
    if not isinstance(config_dict, dict):
        return TurboQuantAttentionConfig(enabled=False)

    attention_scope = str(config_dict.get("attention_scope", "self")).strip().lower()
    if attention_scope not in _SCOPE_OPTIONS:
        attention_scope = "self"

    return TurboQuantAttentionConfig(
        enabled=bool(config_dict.get("enabled", True)),
        bits=max(1, min(8, int(config_dict.get("bits", 4)))),
        qjl_dim=max(1, int(config_dict.get("qjl_dim", 64))),
        use_qjl=bool(config_dict.get("use_qjl", True)),
        quantize_values=bool(config_dict.get("quantize_values", True)),
        min_token_product=max(0, int(config_dict.get("min_token_product", 65536))),
        attention_scope=attention_scope,
        layer_start=int(config_dict.get("layer_start", -1)),
        layer_end=int(config_dict.get("layer_end", -1)),
        rotation_seed=int(config_dict.get("rotation_seed", 0)),
        max_head_dim=max(1, int(config_dict.get("max_head_dim", 256))),
        force_fp32=bool(config_dict.get("force_fp32", False)),
    )


def _reshape_for_attention(tensor: torch.Tensor, heads: int, skip_reshape: bool) -> torch.Tensor:
    if skip_reshape:
        if tensor.ndim != 4:
            raise ValueError(f"Expected rank-4 tensor when skip_reshape=True, got {tuple(tensor.shape)}")
        if int(tensor.shape[1]) != int(heads):
            raise ValueError(f"Expected heads={heads}, got {tuple(tensor.shape)}")
        return tensor

    if tensor.ndim != 3:
        raise ValueError(f"Expected rank-3 tensor when skip_reshape=False, got {tuple(tensor.shape)}")

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


def _scope_matches(attention_scope: str, query_tokens: int, key_tokens: int) -> bool:
    if attention_scope == "both":
        return True
    is_self_attention = int(query_tokens) == int(key_tokens)
    if attention_scope == "self":
        return is_self_attention
    return not is_self_attention


def _layer_matches(cfg: TurboQuantAttentionConfig, transformer_options: Optional[Dict[str, Any]]) -> bool:
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


def _effective_rotation_seed(cfg: TurboQuantAttentionConfig, transformer_options: Optional[Dict[str, Any]]) -> int:
    block_index = -1
    if isinstance(transformer_options, dict):
        block_index = int(transformer_options.get("block_index", -1))
    if block_index < 0:
        return int(cfg.rotation_seed)
    return int(cfg.rotation_seed) + (block_index * 7919)


@lru_cache(maxsize=128)
def _orthogonal_matrix_cpu(dim: int, seed: int) -> torch.Tensor:
    if dim <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    gaussian = torch.randn((dim, dim), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.unsqueeze(0)).contiguous()


@lru_cache(maxsize=16)
def _normal_codebook_cpu(bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    bins = 1 << int(bits)
    probabilities = torch.linspace(0, 1, steps=bins + 1, dtype=torch.float64)
    edges = _NORMAL_DIST.icdf(probabilities[1:-1]).to(dtype=torch.float32)

    lower = torch.empty((bins,), dtype=torch.float64)
    upper = torch.empty((bins,), dtype=torch.float64)
    lower[0] = float("-inf")
    lower[1:] = _NORMAL_DIST.icdf(probabilities[1:-1])
    upper[:-1] = _NORMAL_DIST.icdf(probabilities[1:-1])
    upper[-1] = float("inf")

    pdf_lower = torch.exp(_NORMAL_DIST.log_prob(lower))
    pdf_upper = torch.exp(_NORMAL_DIST.log_prob(upper))
    pdf_lower[0] = 0.0
    pdf_upper[-1] = 0.0
    mass = 1.0 / float(bins)
    reconstruction = ((pdf_lower - pdf_upper) / mass).to(dtype=torch.float32)
    return edges.contiguous(), reconstruction.contiguous()


def _standard_normal_quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    edges_cpu, reconstruction_cpu = _normal_codebook_cpu(bits)
    edges = edges_cpu.to(device=tensor.device, dtype=tensor.dtype)
    reconstruction = reconstruction_cpu.to(device=tensor.device, dtype=tensor.dtype)
    flat = tensor.reshape(-1)
    buckets = torch.bucketize(flat, edges)
    return reconstruction[buckets].reshape_as(tensor)


def _quantize_rotated_tensor(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    dim_head = int(tensor.shape[-1])
    scale = tensor.norm(dim=-1, keepdim=True) / math.sqrt(max(1, dim_head))
    scale = scale.clamp_min(1e-6)
    normalized = tensor / scale
    quantized = _standard_normal_quantize(normalized, bits=bits) * scale
    residual = tensor - quantized
    return quantized.contiguous(), residual.contiguous()


@lru_cache(maxsize=128)
def _gaussian_projection_cpu(dim: int, proj_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    projection = torch.randn((dim, proj_dim), generator=generator, dtype=torch.float32)
    projection = projection / math.sqrt(max(1, proj_dim))
    return projection.contiguous()


def _apply_attention_mask(sim: torch.Tensor, mask: Optional[torch.Tensor], heads: int) -> torch.Tensor:
    if mask is None:
        return sim
    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        return sim.masked_fill(~mask, max_neg_value)

    if mask.ndim == 2:
        mask = mask.reshape(1, 1, mask.shape[-2], mask.shape[-1])
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    return sim + mask.to(device=sim.device, dtype=sim.dtype)


def turboquant_attention_override(original_func: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
    transformer_options = kwargs.get("transformer_options")
    config_dict = transformer_options.get("turboquant_attention") if isinstance(transformer_options, dict) else None
    delegate_override = config_dict.get(_DELEGATE_OVERRIDE_KEY) if isinstance(config_dict, dict) else None
    cfg = _resolve_config(config_dict)
    if not cfg.enabled:
        return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

    try:
        q = args[0]
        k = args[1]
        v = args[2]
        heads = int(args[3])
        mask = kwargs.get("mask")
        skip_reshape = bool(kwargs.get("skip_reshape", False))
        skip_output_reshape = bool(kwargs.get("skip_output_reshape", False))

        q_expanded = _reshape_for_attention(q, heads=heads, skip_reshape=skip_reshape)
        k_expanded = _reshape_for_attention(k, heads=heads, skip_reshape=skip_reshape)
        v_expanded = _reshape_for_attention(v, heads=heads, skip_reshape=skip_reshape)

        if q_expanded.shape[-1] != k_expanded.shape[-1] or k_expanded.shape[-1] != v_expanded.shape[-1]:
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        _, _, query_tokens, dim_head = q_expanded.shape
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

        rotation = _orthogonal_matrix_cpu(dim_head, _effective_rotation_seed(cfg, transformer_options)).to(
            device=q_work.device,
            dtype=compute_dtype,
        )
        q_rot = torch.matmul(q_work, rotation)
        k_rot = torch.matmul(k_work, rotation)
        v_rot = torch.matmul(v_work, rotation) if cfg.quantize_values else v_work

        k_hat_rot, k_residual = _quantize_rotated_tensor(k_rot, bits=cfg.bits)
        if cfg.quantize_values:
            v_hat_rot, _ = _quantize_rotated_tensor(v_rot, bits=cfg.bits)
        else:
            v_hat_rot = v_rot

        scale = dim_head ** -0.5
        sim = torch.einsum("bhqd,bhkd->bhqk", q_rot.float(), k_hat_rot.float()) * scale

        if cfg.use_qjl and cfg.qjl_dim > 0:
            proj_dim = min(int(cfg.qjl_dim), dim_head)
            projection = _gaussian_projection_cpu(dim_head, proj_dim, _effective_rotation_seed(cfg, transformer_options) + 1).to(
                device=q_work.device,
                dtype=torch.float32,
            )
            q_proj = torch.matmul(q_rot.float(), projection)
            residual_proj = torch.matmul(k_residual.float(), projection).sign()
            residual_norm = k_residual.float().norm(dim=-1)
            correction = torch.einsum("bhqm,bhkm->bhqk", q_proj, residual_proj)
            correction = correction * (math.sqrt(math.pi / 2.0) / float(proj_dim))
            correction = correction * residual_norm[:, :, None, :]
            sim = sim + (correction * scale)

        sim = _apply_attention_mask(sim, mask=mask, heads=heads)
        probs = sim.softmax(dim=-1).to(dtype=v_hat_rot.dtype)
        out_rot = torch.einsum("bhqk,bhkd->bhqd", probs, v_hat_rot)
        out = torch.matmul(out_rot, rotation.transpose(0, 1)) if cfg.quantize_values else out_rot
        out = out.to(dtype=v.dtype if v.dtype.is_floating_point else q.dtype)
        return _restore_attention_layout(out, skip_output_reshape=skip_output_reshape)
    except Exception as exc:
        logger.debug("TurboQuant attention fallback: %s", exc)
        return _delegate_or_original(original_func, delegate_override, *args, **kwargs)


class TurboQuantAttentionModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to clone and patch with a TurboQuant-inspired attention approximation."}),
                "bits": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "tooltip": "Bits per rotated coordinate for the scalar quantizer.",
                }),
                "qjl_dim": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Projection width for the QJL-style residual correction on logits.",
                }),
                "use_qjl": (["enable", "disable"], {
                    "tooltip": "Enable the 1-bit residual correction term for key logits, following the TurboQuant paper's second stage.",
                }),
                "quantize_values": (["enable", "disable"], {
                    "tooltip": "Quantize values as well as keys. Disable to keep values in full precision while only approximating logits.",
                }),
                "min_token_product": ("INT", {
                    "default": 65536,
                    "min": 0,
                    "max": 1073741824,
                    "tooltip": "Only patch attention calls where query_tokens * key_tokens meets this threshold.",
                }),
                "attention_scope": (_SCOPE_OPTIONS, {
                    "tooltip": "Which attention calls to patch.",
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
                    "tooltip": "Seed used for the random orthogonal rotation and Gaussian residual projection.",
                }),
                "max_head_dim": ("INT", {
                    "default": 256,
                    "min": 1,
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
        bits,
        qjl_dim,
        use_qjl,
        quantize_values,
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

        previous_config = transformer_options.get("turboquant_attention")
        existing_override = transformer_options.get("optimized_attention_override")
        delegate_override = None
        if callable(existing_override) and existing_override is not turboquant_attention_override:
            delegate_override = existing_override
        elif isinstance(previous_config, dict):
            previous_delegate = previous_config.get(_DELEGATE_OVERRIDE_KEY)
            if callable(previous_delegate) and previous_delegate is not turboquant_attention_override:
                delegate_override = previous_delegate

        transformer_options["turboquant_attention"] = {
            "enabled": True,
            "bits": int(bits),
            "qjl_dim": int(qjl_dim),
            "use_qjl": use_qjl == "enable",
            "quantize_values": quantize_values == "enable",
            "min_token_product": int(min_token_product),
            "attention_scope": str(attention_scope),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "rotation_seed": int(rotation_seed),
            "max_head_dim": int(max_head_dim),
            "force_fp32": force_fp32 == "enable",
            _DELEGATE_OVERRIDE_KEY: delegate_override,
        }
        transformer_options["optimized_attention_override"] = turboquant_attention_override
        logger.info(
            "TurboQuant attention patch applied: bits=%d qjl_dim=%d use_qjl=%s quantize_values=%s min_token_product=%d scope=%s layer_start=%d layer_end=%d seed=%d max_head_dim=%d force_fp32=%s",
            int(bits),
            int(qjl_dim),
            str(use_qjl == "enable"),
            str(quantize_values == "enable"),
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
    "TurboQuantAttentionModelPatch": TurboQuantAttentionModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboQuantAttentionModelPatch": "Model (TurboQuant Attention)",
}
