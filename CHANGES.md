# Changelog

## 1.2.25 - 2026-04-01
- Changed `Model (TurboQuant Attention)` defaults to conservative values for diffusion use: `bits=8`, `quantize_values=disable`, `use_qjl=disable`, `max_token_product=262144`, and `memory_margin_mb=1024`.
- Added an explicit warning log when users choose aggressive TurboQuant settings that are likely to hurt image quality or still be slower than baseline.

## 1.2.24 - 2026-04-01
- Reworked `Model (TurboQuant Attention)` to preserve ComfyUI's original optimized attention kernel by passing transformed `q/k/v` back into `original_func(...)` instead of materializing dense logits in Python.
- Added `max_token_product` and `memory_margin_mb` guards so oversized or memory-risky attention calls skip cleanly before OOM.
- Temporarily disabled the QJL correction path at runtime because it reintroduced dense-memory blowups on large diffusion layers.

## 1.2.23 - 2026-04-01
- Added TurboQuant runtime instrumentation with applied/fallback counters, per-reason skip tracking, periodic summary logging, and optional per-fallback logging.
- Added `log_every` and `log_fallbacks` controls to `Model (TurboQuant Attention)` and test coverage for the new stats behavior.

## 1.2.22 - 2026-04-01
- Added `Model (TurboQuant Attention)`, a second `MODEL` patch node that applies a TurboQuant-inspired attention override with random orthogonal rotation, scalar quantization, and optional QJL-style residual correction on logits.
- Locked `Model (RotorQuant Attention)` back to `keep_components=3` because the lossy `1`/`2` component variants produced poor image quality in practice.
- Added test coverage for the TurboQuant model patcher and updated RotorQuant tests for the exact-only behavior.

## 1.2.21 - 2026-04-01
- Added `Model (RotorQuant Attention)`, a ComfyUI-style `MODEL` patch node that injects an experimental RotorQuant-inspired attention override through `transformer_options`.
- The override uses deterministic 3D rotor blocks with optional per-triplet rank reduction (`keep_components=1..3`) and falls back cleanly to any previously installed attention override when its gating conditions are not met.
- Added tests covering exact full-rank behavior, compressed-path shape preservation, delegate fallback, and non-mutating model clone patching.

## 1.2.19 - 2026-03-01
- Fixed intermittent bypass-LoRA CPU/GPU mismatches by synchronizing adapter tensors to the active UNet input device at runtime in `KSampler (LoRA Sigma Inverse)`.
- Added test coverage for adapter-device synchronization behavior.

## 1.2.18 - 2026-03-01
- Added `min_lora_step` and `max_lora_step` to `KSampler (LoRA Sigma Inverse)` to gate LoRA application by step index.
- Added `-1` unbounded semantics for step gates, so each side defaults to always enabled when set to `-1`.

## 1.2.16 - 2026-03-01
- Added `min_lora_strength` to `KSampler (LoRA Sigma Inverse)` so LoRA strength can interpolate from a non-zero (or negative) start value to `max_lora_strength` across the sigma schedule.

## 1.2.14 - 2026-03-01
- Optimized `KSampler (LoRA Sigma Inverse)` to prefer one-time bypass LoRA injection with runtime multiplier updates, avoiding per-step LoRA repatching for adapter-based LoRAs.
- Added bypass-path test coverage and retained hook-scheduling fallback for non-bypass-compatible LoRA patch types.

## 1.2.13 - 2026-03-01
- Added `KSampler (LoRA Sigma Inverse)`, a KSampler-style node with embedded model-only LoRA loading and per-step sigma-inverse LoRA strength scheduling.
- Added tests covering sigma-inverse schedule math and scheduled hook application behavior.

## 1.1.0 - 2025-12-23
- Added `Latent Noise` and `Image Noise` nodes.

## 1.0.0 - 2025-12-23
- Initial release.
- Extracted Mesh Drag nodes and Latent Channel Stats Preview from `ComfyUI-FlowMatching-Upscaler`.
