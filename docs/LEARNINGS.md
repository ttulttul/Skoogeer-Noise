# Learnings

- Flux.2 latents are 2x2 patchified (128 channels); unpatchify and patchify helpers/nodes let us work at 32 channels with doubled spatial resolution.
- Channel-space transforms can be applied in latent space using linear/nonlinear ops with optional channel selection heuristics and per-channel stat matching.
