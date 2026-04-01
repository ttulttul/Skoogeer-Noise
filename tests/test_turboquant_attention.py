import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.turboquant_attention as tqa  # noqa: E402


def exact_attention(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    if skip_reshape:
        q_heads = q
        k_heads = k
        v_heads = v
    else:
        batch, q_tokens, width = q.shape
        dim_head = width // heads
        q_heads = q.reshape(batch, q_tokens, heads, dim_head).permute(0, 2, 1, 3).contiguous()
        k_heads = k.reshape(batch, k.shape[1], heads, dim_head).permute(0, 2, 1, 3).contiguous()
        v_heads = v.reshape(batch, v.shape[1], heads, dim_head).permute(0, 2, 1, 3).contiguous()

    scale = q_heads.shape[-1] ** -0.5
    sim = torch.einsum("bhqd,bhkd->bhqk", q_heads.float(), k_heads.float()) * scale
    if mask is not None:
        sim = sim + mask
    probs = sim.softmax(dim=-1).to(dtype=v_heads.dtype)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v_heads)
    if skip_output_reshape:
        return out
    batch, out_heads, q_tokens, dim_head = out.shape
    return out.permute(0, 2, 1, 3).reshape(batch, q_tokens, out_heads * dim_head)


def test_turboquant_attention_quantized_path_preserves_shape_and_finiteness():
    torch.manual_seed(3)
    q = torch.randn((2, 4, 17, 12), dtype=torch.float32)
    k = torch.randn((2, 4, 17, 12), dtype=torch.float32)
    v = torch.randn((2, 4, 17, 12), dtype=torch.float32)
    transformer_options = {
        "block_index": 5,
        "turboquant_attention": {
            "enabled": True,
            "bits": 4,
            "qjl_dim": 8,
            "use_qjl": True,
            "quantize_values": True,
            "min_token_product": 1,
            "attention_scope": "both",
            "rotation_seed": 17,
            "max_head_dim": 256,
            "force_fp32": False,
        },
    }

    out = tqa.turboquant_attention_override(
        exact_attention,
        q,
        k,
        v,
        4,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=transformer_options,
    )

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_turboquant_attention_uses_delegate_override_when_gated_off():
    called = {"delegate": 0}

    def delegate_override(original_func, *args, **kwargs):
        called["delegate"] += 1
        return torch.full_like(args[2], 5.0)

    q = torch.randn((1, 2, 9, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 9, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 9, 8), dtype=torch.float32)
    transformer_options = {
        "turboquant_attention": {
            "enabled": True,
            "bits": 4,
            "qjl_dim": 8,
            "use_qjl": True,
            "quantize_values": True,
            "min_token_product": 1,
            "attention_scope": "cross",
            "rotation_seed": 0,
            "max_head_dim": 256,
            "force_fp32": False,
            tqa._DELEGATE_OVERRIDE_KEY: delegate_override,
        },
    }

    out = tqa.turboquant_attention_override(
        exact_attention,
        q,
        k,
        v,
        2,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=transformer_options,
    )

    assert called["delegate"] == 1
    assert torch.all(out == 5.0)


def test_turboquant_model_patch_clones_and_installs_override_without_mutating_source():
    def previous_override(original_func, *args, **kwargs):
        return original_func(*args, **kwargs)

    class FakeModel:
        def __init__(self, model_options=None):
            self.model_options = {} if model_options is None else model_options

        def clone(self):
            return FakeModel(self.model_options)

    source = FakeModel(
        {
            "transformer_options": {
                "optimized_attention_override": previous_override,
                "existing_key": "keep-me",
            }
        }
    )

    node = tqa.TurboQuantAttentionModelPatch()
    (patched,) = node.patch_model(
        model=source,
        bits=3,
        qjl_dim=32,
        use_qjl="enable",
        quantize_values="disable",
        min_token_product=12345,
        attention_scope="self",
        layer_start=2,
        layer_end=8,
        rotation_seed=77,
        max_head_dim=192,
        force_fp32="enable",
    )

    assert patched is not source
    assert source.model_options["transformer_options"]["optimized_attention_override"] is previous_override
    assert "turboquant_attention" not in source.model_options["transformer_options"]

    patched_transformer_options = patched.model_options["transformer_options"]
    assert patched_transformer_options["existing_key"] == "keep-me"
    assert patched_transformer_options["optimized_attention_override"] is tqa.turboquant_attention_override
    assert patched_transformer_options["turboquant_attention"]["bits"] == 3
    assert patched_transformer_options["turboquant_attention"]["qjl_dim"] == 32
    assert patched_transformer_options["turboquant_attention"]["use_qjl"] is True
    assert patched_transformer_options["turboquant_attention"]["quantize_values"] is False
    assert patched_transformer_options["turboquant_attention"]["min_token_product"] == 12345
    assert patched_transformer_options["turboquant_attention"]["attention_scope"] == "self"
    assert patched_transformer_options["turboquant_attention"]["layer_start"] == 2
    assert patched_transformer_options["turboquant_attention"]["layer_end"] == 8
    assert patched_transformer_options["turboquant_attention"]["rotation_seed"] == 77
    assert patched_transformer_options["turboquant_attention"]["max_head_dim"] == 192
    assert patched_transformer_options["turboquant_attention"]["force_fp32"] is True
    assert patched_transformer_options["turboquant_attention"][tqa._DELEGATE_OVERRIDE_KEY] is previous_override


def test_turboquant_attention_respects_min_token_product_gate():
    q = torch.randn((1, 2, 4, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 4, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 4, 8), dtype=torch.float32)
    baseline = exact_attention(q, k, v, 2, skip_reshape=True, skip_output_reshape=True)

    transformer_options = {
        "turboquant_attention": {
            "enabled": True,
            "bits": 4,
            "qjl_dim": 8,
            "use_qjl": True,
            "quantize_values": True,
            "min_token_product": 10_000,
            "attention_scope": "both",
            "rotation_seed": 0,
            "max_head_dim": 256,
            "force_fp32": False,
        },
    }

    out = tqa.turboquant_attention_override(
        exact_attention,
        q,
        k,
        v,
        2,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=transformer_options,
    )

    assert torch.allclose(out, baseline, atol=1e-6, rtol=1e-6)
