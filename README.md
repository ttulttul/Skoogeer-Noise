# Skoogeer-Noise

A small ComfyUI custom node pack containing noise/latent diagnostic helpers and
mesh-drag spatial perturbation nodes.

This pack was extracted from `ComfyUI-FlowMatching-Upscaler` so the perturbation
and debug nodes can be installed independently.

It also includes the latent/image/conditioning noise + filtering nodes that were
previously shipped in `ComfyUI-QwenRectifiedFlowInverter`.

## Nodes

- `Latent Mesh Drag` (`latent/perturb`)
- `Image Mesh Drag` (`image/perturb`)
- `Latent Noise` (`latent/perturb`)
- `Image Noise` (`image/perturb`)
- `Latent Channel Stats Preview` (`latent/debug`)
- `Latent Gaussian Blur` (`Latent/Filter`)
- `Latent Frequency Split` (`Latent/Filter`)
- `Add Latent Noise (Seeded)` (`Latent/Noise`)
- `Add Image Noise (Seeded)` (`Image/Noise`)
- `Latent Perlin Fractal Noise` (`Latent/Noise`)
- `Image Perlin Fractal Noise` (`Image/Noise`)
- `Latent Simplex Noise` (`Latent/Noise`)
- `Image Simplex Noise` (`Image/Noise`)
- `Latent Worley Noise` (`Latent/Noise`)
- `Image Worley Noise` (`Image/Noise`)
- `Latent Reaction-Diffusion` (`Latent/Noise`)
- `Image Reaction-Diffusion` (`Image/Noise`)
- `Latent Fractal Brownian Motion` (`Latent/Noise`)
- `Image Fractal Brownian Motion` (`Image/Noise`)
- `Latent Swirl Noise` (`Latent/Noise`)
- `Image Swirl Noise` (`Image/Noise`)
- `Forward Diffusion (Add Scheduled Noise)` (`Latent/Noise`)
- `Conditioning (Add Noise)` (`conditioning/noise`)
- `Conditioning (Gaussian Blur)` (`conditioning/filter`)
- `Conditioning (Frequency Split)` (`conditioning/filter`)
- `Conditioning (Frequency Merge)` (`conditioning/filter`)
- `Conditioning (Scale)` (`conditioning/filter`)

## Installation

Clone into your ComfyUI `custom_nodes/` directory and restart ComfyUI.

## Development

Run tests with `pytest`.

## Noise

`Latent Noise` and `Image Noise` add seeded Gaussian noise to `LATENT` / `IMAGE` tensors.
Noise strength is relative to the standard deviation of the input tensor.

### Node Parameters: Latent Noise

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `latent` | LATENT | – | Latent to receive additional noise. |
| `seed` | INT | `0` | Seed for generating repeatable noise. |
| `strength` | FLOAT | `1.0` | Noise strength relative to the latent std. |

### Node Parameters: Image Noise

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `image` | IMAGE | – | Image to receive additional noise. |
| `seed` | INT | `0` | Seed for generating repeatable noise. |
| `strength` | FLOAT | `1.0` | Noise strength relative to the image std. |

## Latent Channel Stats Preview

`Latent Channel Stats Preview` renders a quick bar chart of per-channel mean and
standard deviation for a `LATENT` tensor.

### Node Parameters: Latent Channel Stats Preview

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `latent` | LATENT | – | Latent to analyze. |
| `channel_limit` | INT | `16` | Number of channels to display. |
| `height` | INT | `256` | Output image height (chart picks the closest supported layout). |

## Mesh Drag

`Latent Mesh Drag` applies a cloth-like deformation directly to a `LATENT` by
randomly dragging a subset of vertices on a coarse mesh and smoothly
interpolating the displacement across the latent.

Drag distances are specified in **latent pixels** (multiply by ~8 for image-space pixels with SD-style VAEs).

### Node Parameters: Latent Mesh Drag

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `latent` | LATENT | – | Latent to warp. |
| `seed` | INT | `0` | Controls vertex selection and drag vectors. |
| `points` | INT | `12` | Number of mesh vertices to drag. |
| `drag_min` | FLOAT | `0.0` | Minimum drag distance (latent pixels). |
| `drag_max` | FLOAT | `4.0` | Maximum drag distance (latent pixels). |
| `displacement_interpolation` | enum | `bicubic` | Interpolation used to expand the mesh drags into a displacement field (`bspline` is smoother). |
| `spline_passes` | INT | `2` | B-spline smoothing passes (only used when `displacement_interpolation = bspline`). |
| `sampling_interpolation` | enum | `bilinear` | Interpolation used while sampling the source latent during the warp. |

---

`Image Mesh Drag` applies the same deformation in image space and accepts ComfyUI `IMAGE` tensors.

Drag distances are specified in **image pixels**.

### Node Parameters: Image Mesh Drag

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `image` | IMAGE | – | Image to warp. |
| `seed` | INT | `0` | Controls vertex selection and drag vectors. |
| `points` | INT | `12` | Number of mesh vertices to drag. |
| `drag_min` | FLOAT | `0.0` | Minimum drag distance (image pixels). |
| `drag_max` | FLOAT | `32.0` | Maximum drag distance (image pixels). |
| `displacement_interpolation` | enum | `bicubic` | Interpolation used to expand the mesh drags into a displacement field (`bspline` is smoother). |
| `spline_passes` | INT | `2` | B-spline smoothing passes (only used when `displacement_interpolation = bspline`). |
| `sampling_interpolation` | enum | `bilinear` | Interpolation used while sampling the source image during the warp. |
