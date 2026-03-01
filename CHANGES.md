# Changelog

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
