# Learnings

- Flux.2 latents are 2x2 patchified (128 channels); unpatchify and patchify helpers/nodes let us work at 32 channels with doubled spatial resolution.
- Channel-space transforms can be applied in latent space using linear/nonlinear ops with optional channel selection heuristics and per-channel stat matching.
- Latent Channel Merge blends selected channels from a source latent into a destination latent with the same selection modes as other channel ops.
- README now has a structured node reference with a table of contents, grouped sections, and example visuals.
- Latent to Image Batch now emits standard ComfyUI BHWC images with optional RGB repetition, and Image Batch to Latent reconstructs latents from channel batches.
- Batch-aware noise now offsets seeds per sample (`seed + batch index`) so batched outputs stay deterministic and aligned with single-sample runs.
