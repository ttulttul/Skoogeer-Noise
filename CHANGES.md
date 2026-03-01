# Changelog

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
