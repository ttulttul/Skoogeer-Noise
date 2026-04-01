# Learnings

- Flux.2 latents are 2x2 patchified (128 channels); unpatchify and patchify helpers/nodes let us work at 32 channels with doubled spatial resolution.
- Channel-space transforms can be applied in latent space using linear/nonlinear ops with optional channel selection heuristics and per-channel stat matching.
- Latent Channel Merge blends selected channels from a source latent into a destination latent with the same selection modes as other channel ops.
- README now has a structured node reference with a table of contents, grouped sections, and example visuals.
- Latent to Image Batch now emits standard ComfyUI BHWC images with optional RGB repetition, and Image Batch to Latent reconstructs latents from channel batches.
- Batch-aware noise now offsets seeds per sample (`seed + batch index`) so batched outputs stay deterministic and aligned with single-sample runs.
- ComfyUI hook keyframes can be generated from a sampler's sigma schedule to modulate LoRA strength during a single continuous KSampler pass, avoiding multi-pass step chunking.
- Hook-keyframe LoRA scheduling still repatches weights when keyframes change; bypass injection + dynamic adapter multipliers avoids that repatching cost for adapter-only LoRAs.
- Sigma-scheduled LoRA strength is more flexible when modeled as interpolation between explicit `min_lora_strength` and `max_lora_strength`, not just a fixed zero-to-max ramp.
- Sigma-scheduled LoRA control is more practical with optional step-window gating (`min_lora_step`/`max_lora_step`), where `-1` cleanly means unbounded on that side.
- In bypass LoRA mode, adapter tensors may need runtime device re-sync to match active UNet execution device and avoid intermittent CPU/GPU mismatch errors.
- RotorQuant targets LLM KV-cache compression, so the practical ComfyUI adaptation is a model patch that injects a rotor-style attention override into `transformer_options`, not a custom sampler or a direct port of the KV quantizer.
- For image diffusion, RotorQuant-style rank reduction (`keep_components < 3`) degrades quality badly enough that it is better treated as unsupported; the TurboQuant-style quantization path is the more appropriate place for lossy attention experiments.
- Attention-patch experiments need explicit runtime instrumentation; without per-reason skip counters and periodic summaries, it is too easy to mistake silent gating/fallback for a working approximation.
