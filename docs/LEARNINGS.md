# Learnings

- Flux.2 latents are 2x2 patchified (128 channels); unpatchify and patchify helpers/nodes let us work at 32 channels with doubled spatial resolution.
- Channel-space transforms can be applied in latent space using linear/nonlinear ops with optional channel selection heuristics and per-channel stat matching.
- README now has a structured node reference with a table of contents, grouped sections, and example visuals.
- Latent-to-image conversion now emits standard ComfyUI BHWC images, with an option to repeat grayscale into RGB for PreviewImage compatibility.
