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
    bits: int = 8
    qjl_dim: int = 64
    use_qjl: bool = True
    quantize_values: bool = False
    min_token_product: int = 65536
    max_token_product: int = 262144
    attention_scope: str = "self"
    layer_start: int = -1
    layer_end: int = -1
    rotation_seed: int = 0
    max_head_dim: int = 256
    force_fp32: bool = False
    memory_margin_mb: int = 1024
    log_every: int = 50
    log_fallbacks: bool = False


_GLOBAL_STATS: Dict[str, Any] = {
    "calls": 0,
    "applied_calls": 0,
    "fallback_calls": 0,
    "fallback_reasons": {},
}


def reset_turboquant_stats() -> None:
    _GLOBAL_STATS["calls"] = 0
    _GLOBAL_STATS["applied_calls"] = 0
    _GLOBAL_STATS["fallback_calls"] = 0
    _GLOBAL_STATS["fallback_reasons"] = {}


def get_turboquant_stats() -> Dict[str, Any]:
    return {
        "calls": int(_GLOBAL_STATS["calls"]),
        "applied_calls": int(_GLOBAL_STATS["applied_calls"]),
        "fallback_calls": int(_GLOBAL_STATS["fallback_calls"]),
        "fallback_reasons": dict(_GLOBAL_STATS["fallback_reasons"]),
    }


def _update_stats(*, applied: bool = False, reason: Optional[str] = None) -> None:
    _GLOBAL_STATS["calls"] += 1
    if applied:
        _GLOBAL_STATS["applied_calls"] += 1
    if reason is not None:
        _GLOBAL_STATS["fallback_calls"] += 1
        reasons = _GLOBAL_STATS["fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1


def _maybe_log_stats(cfg: TurboQuantAttentionConfig) -> None:
    log_every = int(cfg.log_every)
    calls = int(_GLOBAL_STATS["calls"])
    if log_every <= 0 or calls <= 0 or (calls % log_every) != 0:
        return
    logger.info(
        "TurboQuant attention stats: calls=%d applied=%d fallbacks=%d reasons=%s",
        calls,
        int(_GLOBAL_STATS["applied_calls"]),
        int(_GLOBAL_STATS["fallback_calls"]),
        dict(_GLOBAL_STATS["fallback_reasons"]),
    )


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
        bits=max(1, min(8, int(config_dict.get("bits", 8)))),
        qjl_dim=max(1, int(config_dict.get("qjl_dim", 64))),
        use_qjl=bool(config_dict.get("use_qjl", True)),
        quantize_values=bool(config_dict.get("quantize_values", False)),
        min_token_product=max(0, int(config_dict.get("min_token_product", 65536))),
        max_token_product=max(0, int(config_dict.get("max_token_product", 262144))),
        attention_scope=attention_scope,
        layer_start=int(config_dict.get("layer_start", -1)),
        layer_end=int(config_dict.get("layer_end", -1)),
        rotation_seed=int(config_dict.get("rotation_seed", 0)),
        max_head_dim=max(1, int(config_dict.get("max_head_dim", 256))),
        force_fp32=bool(config_dict.get("force_fp32", False)),
        memory_margin_mb=max(0, int(config_dict.get("memory_margin_mb", 1024))),
        log_every=max(0, int(config_dict.get("log_every", 50))),
        log_fallbacks=bool(config_dict.get("log_fallbacks", False)),
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


def _estimate_workspace_bytes(
    q_shape: torch.Size,
    k_shape: torch.Size,
    v_shape: torch.Size,
    dtype: torch.dtype,
    *,
    quantize_values: bool,
) -> int:
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    q_numel = math.prod(q_shape)
    k_numel = math.prod(k_shape)
    v_numel = math.prod(v_shape)
    total_elements = q_numel + (2 * k_numel)
    if quantize_values:
        total_elements += 2 * v_numel
    return int(total_elements * bytes_per_element)


def _workspace_budget_ok(device: torch.device, estimated_bytes: int, margin_mb: int) -> tuple[bool, Optional[int]]:
    if device.type != "cuda":
        return True, None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return True, None
    budget = max(0, int(free_bytes) - (int(margin_mb) * 1024 * 1024))
    return int(estimated_bytes) <= budget, int(free_bytes)


def turboquant_attention_override(original_func: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
    transformer_options = kwargs.get("transformer_options")
    config_dict = transformer_options.get("turboquant_attention") if isinstance(transformer_options, dict) else None
    delegate_override = config_dict.get(_DELEGATE_OVERRIDE_KEY) if isinstance(config_dict, dict) else None
    cfg = _resolve_config(config_dict)
    if not cfg.enabled:
        _update_stats(reason="disabled")
        _maybe_log_stats(cfg)
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
            _update_stats(reason="shape_mismatch")
            if cfg.log_fallbacks:
                logger.info("TurboQuant attention skip: shape mismatch q=%s k=%s v=%s", tuple(q_expanded.shape), tuple(k_expanded.shape), tuple(v_expanded.shape))
            _maybe_log_stats(cfg)
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        _, _, query_tokens, dim_head = q_expanded.shape
        key_tokens = int(k_expanded.shape[2])
        token_product = int(query_tokens * key_tokens)
        if dim_head <= 0 or dim_head > cfg.max_head_dim:
            _update_stats(reason="head_dim_out_of_range")
            if cfg.log_fallbacks:
                logger.info("TurboQuant attention skip: dim_head=%d max_head_dim=%d", dim_head, int(cfg.max_head_dim))
            _maybe_log_stats(cfg)
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)
        if token_product < cfg.min_token_product:
            _update_stats(reason="token_product_below_min")
            if cfg.log_fallbacks:
                logger.info(
                    "TurboQuant attention skip: query_tokens=%d key_tokens=%d token_product=%d min_token_product=%d",
                    query_tokens,
                    key_tokens,
                    token_product,
                    int(cfg.min_token_product),
                )
            _maybe_log_stats(cfg)
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)
        if cfg.max_token_product > 0 and token_product > cfg.max_token_product:
            _update_stats(reason="token_product_above_max")
            if cfg.log_fallbacks:
                logger.info(
                    "TurboQuant attention skip: query_tokens=%d key_tokens=%d token_product=%d max_token_product=%d",
                    query_tokens,
                    key_tokens,
                    token_product,
                    int(cfg.max_token_product),
                )
            _maybe_log_stats(cfg)
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)
        if not _scope_matches(cfg.attention_scope, query_tokens, key_tokens):
            _update_stats(reason="scope_filtered")
            if cfg.log_fallbacks:
                logger.info(
                    "TurboQuant attention skip: attention_scope=%s query_tokens=%d key_tokens=%d",
                    str(cfg.attention_scope),
                    query_tokens,
                    key_tokens,
                )
            _maybe_log_stats(cfg)
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)
        if not _layer_matches(cfg, transformer_options):
            _update_stats(reason="layer_filtered")
            if cfg.log_fallbacks:
                logger.info(
                    "TurboQuant attention skip: layer outside [%d, %d] block_index=%s",
                    int(cfg.layer_start),
                    int(cfg.layer_end),
                    None if not isinstance(transformer_options, dict) else transformer_options.get("block_index"),
                )
            _maybe_log_stats(cfg)
            return _delegate_or_original(original_func, delegate_override, *args, **kwargs)

        if cfg.use_qjl and cfg.log_fallbacks:
            logger.info("TurboQuant attention note: QJL correction is temporarily disabled in the runtime path to avoid dense-memory blowups.")

        target_dtype = torch.float32 if cfg.force_fp32 else q_expanded.dtype
        estimated_workspace_bytes = _estimate_workspace_bytes(
            q_expanded.shape,
            k_expanded.shape,
            v_expanded.shape,
            target_dtype,
            quantize_values=cfg.quantize_values,
        )
        workspace_ok, free_bytes = _workspace_budget_ok(
            q_expanded.device,
            estimated_workspace_bytes,
            cfg.memory_margin_mb,
        )
        if not workspace_ok:
            _update_stats(reason="memory_guard")
            if cfg.log_fallbacks:
                logger.info(
                    "TurboQuant attention skip: memory_guard estimated_extra=%.2f MiB free=%.2f MiB margin=%.2f MiB",
                    estimated_workspace_bytes / (1024.0 * 1024.0),
                    0.0 if free_bytes is None else free_bytes / (1024.0 * 1024.0),
                    int(cfg.memory_margin_mb),
                )
            _maybe_log_stats(cfg)
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

        k_hat_rot, _ = _quantize_rotated_tensor(k_rot, bits=cfg.bits)
        if cfg.quantize_values:
            v_hat_rot, _ = _quantize_rotated_tensor(v_rot, bits=cfg.bits)
        else:
            v_hat_rot = v_rot

        _update_stats(applied=True)
        _maybe_log_stats(cfg)
        return original_func(
            q_rot,
            k_hat_rot,
            v_hat_rot,
            heads,
            mask=mask,
            attn_precision=kwargs.get("attn_precision"),
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            transformer_options=transformer_options,
        )
    except Exception as exc:
        reason = "oom" if "out of memory" in str(exc).lower() else "exception"
        _update_stats(reason=reason)
        if cfg.log_fallbacks:
            logger.warning("TurboQuant attention fallback: %s", exc)
        else:
            logger.debug("TurboQuant attention fallback: %s", exc)
        _maybe_log_stats(cfg)
        return _delegate_or_original(original_func, delegate_override, *args, **kwargs)


class TurboQuantAttentionModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to clone and patch with a TurboQuant-inspired attention approximation."}),
                "bits": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 8,
                    "tooltip": "Bits per rotated coordinate for the scalar quantizer. Higher is safer for image quality; lower is more aggressive.",
                }),
                "qjl_dim": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Projection width for the QJL-style residual correction on logits.",
                }),
                "use_qjl": (["disable", "enable"], {
                    "tooltip": "Enable the 1-bit residual correction term for key logits. Currently forced off in the runtime path.",
                }),
                "quantize_values": (["disable", "enable"], {
                    "tooltip": "Quantize values as well as keys. Disable is safer for image quality and is the default.",
                }),
                "min_token_product": ("INT", {
                    "default": 65536,
                    "min": 0,
                    "max": 1073741824,
                    "tooltip": "Only patch attention calls where query_tokens * key_tokens meets this threshold.",
                }),
                "max_token_product": ("INT", {
                    "default": 262144,
                    "min": 0,
                    "max": 1073741824,
                    "tooltip": "Skip attention calls above this query_tokens * key_tokens threshold. Conservative default avoids the largest, most memory-sensitive layers.",
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
                "memory_margin_mb": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 65536,
                    "tooltip": "Keep this much free CUDA memory in reserve before allowing the TurboQuant workspace allocation.",
                }),
                "log_every": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000000,
                    "tooltip": "Emit a TurboQuant runtime summary every N attention calls. Set 1 for per-call summaries, 0 to disable periodic summaries.",
                }),
                "log_fallbacks": (["disable", "enable"], {
                    "tooltip": "Log individual skip/fallback reasons when TurboQuant does not activate.",
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
        max_token_product,
        attention_scope,
        layer_start,
        layer_end,
        rotation_seed,
        max_head_dim,
        force_fp32,
        memory_margin_mb,
        log_every,
        log_fallbacks,
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

        normalized_use_qjl = False
        if use_qjl == "enable":
            logger.warning(
                "TurboQuant attention patch requested use_qjl=enable, but QJL correction is temporarily disabled in the runtime path to avoid dense-memory blowups."
            )

        transformer_options["turboquant_attention"] = {
            "enabled": True,
            "bits": int(bits),
            "qjl_dim": int(qjl_dim),
            "use_qjl": normalized_use_qjl,
            "quantize_values": quantize_values == "enable",
            "min_token_product": int(min_token_product),
            "max_token_product": int(max_token_product),
            "attention_scope": str(attention_scope),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "rotation_seed": int(rotation_seed),
            "max_head_dim": int(max_head_dim),
            "force_fp32": force_fp32 == "enable",
            "memory_margin_mb": int(memory_margin_mb),
            "log_every": int(log_every),
            "log_fallbacks": log_fallbacks == "enable",
            _DELEGATE_OVERRIDE_KEY: delegate_override,
        }
        transformer_options["optimized_attention_override"] = turboquant_attention_override
        logger.info(
            "TurboQuant attention patch applied: bits=%d qjl_dim=%d use_qjl=%s quantize_values=%s min_token_product=%d max_token_product=%d scope=%s layer_start=%d layer_end=%d seed=%d max_head_dim=%d force_fp32=%s memory_margin_mb=%d log_every=%d log_fallbacks=%s",
            int(bits),
            int(qjl_dim),
            str(normalized_use_qjl),
            str(quantize_values == "enable"),
            int(min_token_product),
            int(max_token_product),
            str(attention_scope),
            int(layer_start),
            int(layer_end),
            int(rotation_seed),
            int(max_head_dim),
            str(force_fp32 == "enable"),
            int(memory_margin_mb),
            int(log_every),
            str(log_fallbacks == "enable"),
        )
        if int(bits) <= 4 or quantize_values == "enable":
            logger.warning(
                "TurboQuant attention patch warning: aggressive settings (bits=%d quantize_values=%s) can significantly degrade diffusion image quality and may still be slower than baseline.",
                int(bits),
                str(quantize_values),
            )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "TurboQuantAttentionModelPatch": TurboQuantAttentionModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboQuantAttentionModelPatch": "Model (TurboQuant Attention)",
}
