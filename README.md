# Skoogeer-Noise

A small ComfyUI custom node pack containing noise/latent diagnostic helpers and
mesh-drag spatial perturbation nodes.

This pack was extracted from `ComfyUI-FlowMatching-Upscaler` so the perturbation
and debug nodes can be installed independently.

## Nodes

- `Latent Mesh Drag` (`latent/perturb`)
- `Image Mesh Drag` (`image/perturb`)
- `Latent Channel Stats Preview` (`latent/debug`)

## Installation

Clone into your ComfyUI `custom_nodes/` directory and restart ComfyUI.

## Development

Run tests with `pytest`.

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
