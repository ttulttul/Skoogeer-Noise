from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SEED_STRIDE = 0x9E3779B97F4A7C15  # 64-bit golden ratio constant
_STROKE_SEED_OFFSET = 0xD1B54A32D192ED03  # Separate stream for stroke params.
_DEFAULT_VERTEX_SPACING = 16  # Latent pixels between control vertices.
_DEFAULT_IMAGE_VERTEX_SPACING = _DEFAULT_VERTEX_SPACING * 8  # Approx SD-style VAE scale factor.
_DISPLACEMENT_INTERPOLATION_MODES = ("bilinear", "bicubic", "bspline", "nearest")
_SAMPLING_INTERPOLATION_MODES = ("bilinear", "bicubic", "nearest")


@dataclass(frozen=True)
class _Flattened2D:
    tensor: torch.Tensor
    restore: Callable[[torch.Tensor], torch.Tensor]
    batch_size: int
    extra_dim: int


def _flatten_to_nchw(tensor: torch.Tensor) -> _Flattened2D:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    original_shape = tuple(tensor.shape)
    squeeze_channel = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
        squeeze_channel = True

    if tensor.ndim < 4:
        raise ValueError(f"Expected tensor with at least 3 dims (B,H,W) or 4 dims (B,C,H,W), got {original_shape}")

    batch_size = int(tensor.shape[0])
    extra_dim = 1
    reshape_higher = tensor.ndim > 4
    if reshape_higher:
        reshape_base = tuple(tensor.shape)
        extra_dim = int(math.prod(reshape_base[2:-2])) if len(reshape_base) > 4 else 1
        tensor = tensor.reshape(reshape_base[0], reshape_base[1], -1, reshape_base[-2], reshape_base[-1])
        tensor = tensor.movedim(2, 1).reshape(-1, reshape_base[1], reshape_base[-2], reshape_base[-1])

        def restore(out: torch.Tensor) -> torch.Tensor:
            restored = out.reshape(
                original_shape[0],
                -1,
                original_shape[1],
                original_shape[-2],
                original_shape[-1],
            )
            restored = restored.movedim(2, 1).reshape(original_shape)
            return restored

        return _Flattened2D(tensor=tensor, restore=restore, batch_size=batch_size, extra_dim=extra_dim)

    def restore(out: torch.Tensor) -> torch.Tensor:
        if squeeze_channel:
            return out.squeeze(1)
        return out

    return _Flattened2D(tensor=tensor, restore=restore, batch_size=batch_size, extra_dim=extra_dim)


def _compute_vertex_grid_size(height: int, width: int, points: int, *, spacing: int = _DEFAULT_VERTEX_SPACING) -> Tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid spatial dimensions {height}x{width}.")
    if spacing <= 0:
        raise ValueError(f"Vertex spacing must be positive, got {spacing}.")

    base_h = max(4, int(math.ceil((height - 1) / spacing)) + 1)
    base_w = max(4, int(math.ceil((width - 1) / spacing)) + 1)

    grid_h = base_h
    grid_w = base_w
    if points <= 0:
        return grid_h, grid_w

    while grid_h * grid_w < points:
        if grid_h <= grid_w:
            grid_h += 1
        else:
            grid_w += 1

    return grid_h, grid_w


def _seed_for_batch(seed: int, batch_index: int) -> int:
    return (int(seed) + int(batch_index) * _SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF


def _seed_for_stroke(seed: int) -> int:
    return (int(seed) + _STROKE_SEED_OFFSET) & 0xFFFFFFFFFFFFFFFF


def _stroke_params(*, seed: int, direction: float, height: int, width: int) -> Tuple[float, float, float, float]:
    """
    Returns (center_x, center_y, dir_x, dir_y) in pixel coordinates.

    Direction uses degrees with 0=up, 90=right, 180=down, 270=left.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)

    width = int(width)
    height = int(height)
    if width <= 1:
        center_x = 0.0
    else:
        center_x = float(
            torch.empty((), dtype=torch.float64).uniform_(0.0, float(width - 1), generator=generator).item()
        )
    if height <= 1:
        center_y = 0.0
    else:
        center_y = float(
            torch.empty((), dtype=torch.float64).uniform_(0.0, float(height - 1), generator=generator).item()
        )

    direction = float(direction)
    if direction >= 0.0:
        angle = math.radians(direction % 360.0)
    else:
        angle = float(torch.empty((), dtype=torch.float64).uniform_(0.0, float(2.0 * math.pi), generator=generator).item())

    dir_x = float(math.sin(angle))
    dir_y = float(-math.cos(angle))
    return center_x, center_y, dir_x, dir_y


def _stroke_mask(
    *,
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    dir_x: float,
    dir_y: float,
    stroke_width: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns a (1,1,H,W) mask that is 1 on the stroke center line and falls to 0 at +/- stroke_width/2.
    """
    height = int(height)
    width = int(width)
    stroke_width = float(stroke_width)
    half_width = max(stroke_width * 0.5, 1e-6)

    ys = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, width)

    dx = xs - float(center_x)
    dy = ys - float(center_y)
    dist = torch.abs(dx * float(dir_y) - dy * float(dir_x))

    t = (half_width - dist) / half_width
    t = torch.clamp(t, 0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)
    return mask.unsqueeze(0).unsqueeze(0)


def _make_control_displacement(
    *,
    grid_h: int,
    grid_w: int,
    points: int,
    drag_min: float,
    drag_max: float,
    seed: int,
    direction: float = -1.0,
    stroke_width: float = -1.0,
    stroke_params: Tuple[float, float, float, float] | None = None,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    disp = torch.zeros((2, grid_h, grid_w), dtype=torch.float32)
    if points <= 0 or drag_max <= 0.0:
        return disp

    total_vertices = grid_h * grid_w
    if points > total_vertices:
        raise ValueError(f"Requested {points} mesh points but only {total_vertices} vertices available.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    stroke_width = float(stroke_width)
    if stroke_width > 0.0 and stroke_params is not None and height is not None and width is not None:
        center_x, center_y, dir_x, dir_y = stroke_params
        vertex_indices = torch.arange(total_vertices, dtype=torch.long)
        ys_all = (vertex_indices // grid_w).to(torch.float32)
        xs_all = (vertex_indices % grid_w).to(torch.float32)

        if grid_w > 1 and width > 1:
            x_coords = xs_all / float(grid_w - 1) * float(width - 1)
        else:
            x_coords = torch.zeros_like(xs_all)
        if grid_h > 1 and height > 1:
            y_coords = ys_all / float(grid_h - 1) * float(height - 1)
        else:
            y_coords = torch.zeros_like(ys_all)

        dist = torch.abs((x_coords - float(center_x)) * float(dir_y) - (y_coords - float(center_y)) * float(dir_x))
        half_width = stroke_width * 0.5
        eligible = torch.nonzero(dist <= half_width, as_tuple=False).flatten()
        if eligible.numel() < points:
            eligible = dist.argsort()[:points]

        perm = torch.randperm(int(eligible.numel()), generator=generator)
        chosen = eligible[perm[:points]]
    else:
        chosen = torch.randperm(total_vertices, generator=generator)[:points]
    ys = (chosen // grid_w).to(torch.long)
    xs = (chosen % grid_w).to(torch.long)

    magnitudes = torch.empty((points,), dtype=torch.float32).uniform_(float(drag_min), float(drag_max), generator=generator)
    direction = float(direction)
    if direction >= 0.0:
        angle = math.radians(direction % 360.0)
        dx = magnitudes * float(math.sin(angle))
        dy = magnitudes * float(-math.cos(angle))
    else:
        angles = torch.empty((points,), dtype=torch.float32).uniform_(0.0, float(2.0 * math.pi), generator=generator)
        dx = magnitudes * torch.cos(angles)
        dy = magnitudes * torch.sin(angles)

    disp[0, ys, xs] = dx
    disp[1, ys, xs] = dy
    return disp


def _base_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.unsqueeze(0)

def _apply_bspline_smoothing(displacement: torch.Tensor, *, passes: int = 1) -> torch.Tensor:
    if passes <= 0:
        return displacement
    if displacement.ndim != 4:
        raise ValueError(f"Expected displacement tensor with shape (B,2,H,W), got {tuple(displacement.shape)}")

    channels = int(displacement.shape[1])
    if channels != 2:
        raise ValueError(f"Expected displacement to have 2 channels (dx,dy), got {channels}")

    device = displacement.device
    dtype = displacement.dtype

    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype) / 16.0
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    weight = kernel_2d.expand(channels, 1, 5, 5).contiguous()

    smoothed = displacement
    for _ in range(int(passes)):
        smoothed = F.conv2d(smoothed, weight, padding=2, groups=channels)
    return smoothed


def _upsample_displacement(
    control: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
    spline_passes: int,
) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode not in _DISPLACEMENT_INTERPOLATION_MODES:
        raise ValueError(f"Unsupported displacement_interpolation '{mode}'. Supported: {_DISPLACEMENT_INTERPOLATION_MODES}")

    if mode == "bspline":
        base = F.interpolate(
            control,
            size=(height, width),
            mode="bicubic",
            align_corners=True,
        )
        return _apply_bspline_smoothing(base, passes=int(spline_passes))

    align_corners = True if mode in ("bilinear", "bicubic") else None
    return F.interpolate(
        control,
        size=(height, width),
        mode=mode,
        align_corners=align_corners,
    )


def mesh_drag_warp(
    tensor: torch.Tensor,
    *,
    points: int,
    drag_min: float,
    drag_max: float,
    seed: int,
    direction: float = -1.0,
    stroke_width: float = -1.0,
    padding_mode: str = "border",
    vertex_spacing: int = _DEFAULT_VERTEX_SPACING,
    displacement_interpolation: str = "bicubic",
    spline_passes: int = 2,
    sampling_interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Warp a latent-like tensor by perturbing random vertices on a coarse mesh and interpolating
    the displacement across the image.

    Drag amounts are specified in pixels of the input tensor's spatial dimensions.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    if points < 0:
        raise ValueError(f"points must be >= 0, got {points}")
    if drag_min < 0.0 or drag_max < 0.0:
        raise ValueError("drag_min and drag_max must be >= 0.0")
    if drag_min > drag_max:
        raise ValueError(f"drag_min ({drag_min}) must be <= drag_max ({drag_max})")

    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    if height <= 1 or width <= 1:
        logger.debug("Mesh drag warp skipped for degenerate tensor size %sx%s.", height, width)
        return tensor
    if points == 0 or drag_max <= 0.0:
        return tensor
    sampling_interpolation = str(sampling_interpolation).strip().lower()
    if sampling_interpolation not in _SAMPLING_INTERPOLATION_MODES:
        raise ValueError(f"Unsupported sampling_interpolation '{sampling_interpolation}'. Supported: {_SAMPLING_INTERPOLATION_MODES}")

    flattened = _flatten_to_nchw(tensor)
    nchw = flattened.tensor
    device = nchw.device

    grid_h, grid_w = _compute_vertex_grid_size(height, width, points, spacing=int(vertex_spacing))
    stroke_width = float(stroke_width)
    stroke_params = None
    if stroke_width > 0.0:
        stroke_params = [
            _stroke_params(
                seed=_seed_for_stroke(_seed_for_batch(seed, batch_index)),
                direction=direction,
                height=height,
                width=width,
            )
            for batch_index in range(flattened.batch_size)
        ]
    control = torch.stack(
        [
            _make_control_displacement(
                grid_h=grid_h,
                grid_w=grid_w,
                points=points,
                drag_min=drag_min,
                drag_max=drag_max,
                seed=_seed_for_batch(seed, batch_index),
                direction=direction,
                stroke_width=stroke_width,
                stroke_params=stroke_params[batch_index] if stroke_params is not None else None,
                height=height,
                width=width,
            )
            for batch_index in range(flattened.batch_size)
        ],
        dim=0,
    ).to(device=device)

    with torch.no_grad():
        disp = _upsample_displacement(
            control,
            height=height,
            width=width,
            mode=displacement_interpolation,
            spline_passes=int(spline_passes),
        )
        if stroke_params is not None:
            masks = torch.cat(
                [
                    _stroke_mask(
                        height=height,
                        width=width,
                        center_x=params[0],
                        center_y=params[1],
                        dir_x=params[2],
                        dir_y=params[3],
                        stroke_width=stroke_width,
                        device=device,
                    )
                    for params in stroke_params
                ],
                dim=0,
            )
            disp = disp * masks
        if drag_max > 0.0:
            disp = disp.clamp(min=-float(drag_max), max=float(drag_max))

        if flattened.extra_dim != 1:
            disp = disp.repeat_interleave(flattened.extra_dim, dim=0)

        if disp.shape[0] != nchw.shape[0]:
            raise RuntimeError(
                f"Internal shape mismatch: got displacement batch {disp.shape[0]} but tensor batch {nchw.shape[0]}"
            )

        disp_norm = disp.permute(0, 2, 3, 1).to(dtype=torch.float32)
        disp_norm[..., 0] *= 2.0 / float(width - 1)
        disp_norm[..., 1] *= 2.0 / float(height - 1)

        grid = _base_grid(height, width, device=device) - disp_norm
        warped = F.grid_sample(
            nchw,
            grid,
            mode=sampling_interpolation,
            padding_mode=padding_mode,
            align_corners=True,
        )
        return flattened.restore(warped)


def mesh_drag_warp_image(
    image: torch.Tensor,
    *,
    points: int,
    drag_min: float,
    drag_max: float,
    seed: int,
    direction: float = -1.0,
    stroke_width: float = -1.0,
    padding_mode: str = "border",
    vertex_spacing: int = _DEFAULT_IMAGE_VERTEX_SPACING,
    displacement_interpolation: str = "bicubic",
    spline_passes: int = 2,
    sampling_interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Warp an image tensor using the same mesh drag logic.

    Accepts ComfyUI's `IMAGE` format (B,H,W,C) as well as NCHW inputs.
    Drag amounts are in image pixels.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        warped = mesh_drag_warp_image(
            image.unsqueeze(0),
            points=points,
            drag_min=drag_min,
            drag_max=drag_max,
            seed=seed,
            direction=direction,
            stroke_width=stroke_width,
            padding_mode=padding_mode,
            vertex_spacing=vertex_spacing,
            displacement_interpolation=displacement_interpolation,
            spline_passes=spline_passes,
            sampling_interpolation=sampling_interpolation,
        )
        return warped.squeeze(0)

    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with shape (B,H,W,C) or (B,C,H,W), got {tuple(image.shape)}")

    if image.shape[-1] in (1, 3, 4):
        nchw = image.permute(0, 3, 1, 2)
        warped_nchw = mesh_drag_warp(
            nchw,
            points=points,
            drag_min=drag_min,
            drag_max=drag_max,
            seed=seed,
            direction=direction,
            stroke_width=stroke_width,
            padding_mode=padding_mode,
            vertex_spacing=vertex_spacing,
            displacement_interpolation=displacement_interpolation,
            spline_passes=spline_passes,
            sampling_interpolation=sampling_interpolation,
        )
        return warped_nchw.permute(0, 2, 3, 1)

    if image.shape[1] in (1, 3, 4):
        return mesh_drag_warp(
            image,
            points=points,
            drag_min=drag_min,
            drag_max=drag_max,
            seed=seed,
            direction=direction,
            stroke_width=stroke_width,
            padding_mode=padding_mode,
            vertex_spacing=vertex_spacing,
            displacement_interpolation=displacement_interpolation,
            spline_passes=spline_passes,
            sampling_interpolation=sampling_interpolation,
        )

    raise ValueError(
        "Unable to infer channel axis for image tensor. Expected channels-last (B,H,W,C) "
        "or channels-first (B,C,H,W) with C in {1,3,4}, got shape "
        f"{tuple(image.shape)}."
    )


class LatentMeshDrag:
    """
    Applies a cloth-like mesh warp to a latent tensor by randomly dragging a subset of control vertices.
    """

    CATEGORY = "latent/perturb"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "drag"

    _DISPLACEMENT_INTERPOLATION_OPTIONS: Tuple[str, ...] = _DISPLACEMENT_INTERPOLATION_MODES
    _SAMPLING_INTERPOLATION_OPTIONS: Tuple[str, ...] = _SAMPLING_INTERPOLATION_MODES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to warp spatially using a random mesh drag."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed controlling which mesh points are dragged and by how much.",
                }),
                "points": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 2048,
                    "tooltip": "Number of mesh vertices to randomly drag.",
                }),
                "drag_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 128.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Minimum drag distance (latent pixels).",
                }),
                "drag_max": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 128.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Maximum drag distance (latent pixels).",
                }),
                "direction": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 360.0,
                    "step": 1.0,
                    "round": 1.0,
                    "tooltip": "Drag direction in degrees (0=up, 90=right, 180=down, 270=left). Set to -1 to allow random directions.",
                }),
                "stroke_width": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 4096.0,
                    "step": 1.0,
                    "round": 0.1,
                    "tooltip": "When >0, limits the warp to a narrow strip aligned with the direction (like a brush stroke). "
                               "Units are latent pixels. Set to -1 to disable.",
                }),
            },
            "optional": {
                "displacement_interpolation": (cls._DISPLACEMENT_INTERPOLATION_OPTIONS, {
                    "default": "bicubic",
                    "tooltip": "How to interpolate the sparse mesh drags into a full displacement field. "
                               "Use 'bspline' for smoother, more organic warps.",
                }),
                "spline_passes": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Only used when displacement_interpolation = 'bspline'. More passes = smoother warp.",
                }),
                "sampling_interpolation": (cls._SAMPLING_INTERPOLATION_OPTIONS, {
                    "default": "bilinear",
                    "tooltip": "How to sample the source tensor when applying the warp (bilinear/bicubic/nearest).",
                }),
            },
        }

    def drag(
        self,
        latent,
        seed: int,
        points: int,
        drag_min: float,
        drag_max: float,
        direction: float = -1.0,
        stroke_width: float = -1.0,
        displacement_interpolation: str = "bicubic",
        spline_passes: int = 2,
        sampling_interpolation: str = "bilinear",
    ):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("LATENT input must be a dictionary containing a 'samples' tensor.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError(f"LATENT['samples'] must be a torch.Tensor, got {type(samples)}.")

        logger.debug(
            "Applying LatentMeshDrag: samples=%s points=%d drag=[%.3f, %.3f] seed=%d disp=%s spline=%d sample=%s",
            tuple(samples.shape),
            points,
            drag_min,
            drag_max,
            seed,
            displacement_interpolation,
            spline_passes,
            sampling_interpolation,
        )

        warped = mesh_drag_warp(
            samples,
            points=int(points),
            drag_min=float(drag_min),
            drag_max=float(drag_max),
            seed=int(seed),
            direction=float(direction),
            stroke_width=float(stroke_width),
            displacement_interpolation=str(displacement_interpolation),
            spline_passes=int(spline_passes),
            sampling_interpolation=str(sampling_interpolation),
        )

        output = latent.copy()
        output["samples"] = warped

        noise_mask = output.get("noise_mask")
        if isinstance(noise_mask, torch.Tensor):
            output["noise_mask"] = mesh_drag_warp(
                noise_mask,
                points=int(points),
                drag_min=float(drag_min),
                drag_max=float(drag_max),
                seed=int(seed),
                direction=float(direction),
                stroke_width=float(stroke_width),
                displacement_interpolation=str(displacement_interpolation),
                spline_passes=int(spline_passes),
                sampling_interpolation=str(sampling_interpolation),
            )
        return (output,)


class ImageMeshDrag:
    """
    Applies a cloth-like mesh warp to an image tensor by randomly dragging a subset of control vertices.

    This node operates directly in image space (pixel units) and accepts ComfyUI's BHWC `IMAGE` tensors.
    """

    CATEGORY = "image/perturb"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "drag"

    _DISPLACEMENT_INTERPOLATION_OPTIONS: Tuple[str, ...] = _DISPLACEMENT_INTERPOLATION_MODES
    _SAMPLING_INTERPOLATION_OPTIONS: Tuple[str, ...] = _SAMPLING_INTERPOLATION_MODES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to warp spatially using a random mesh drag."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed controlling which mesh points are dragged and by how much.",
                }),
                "points": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Number of mesh vertices to randomly drag.",
                }),
                "drag_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 1.0,
                    "round": 0.01,
                    "tooltip": "Minimum drag distance (image pixels).",
                }),
                "drag_max": ("FLOAT", {
                    "default": 32.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 1.0,
                    "round": 0.01,
                    "tooltip": "Maximum drag distance (image pixels).",
                }),
                "direction": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 360.0,
                    "step": 1.0,
                    "round": 1.0,
                    "tooltip": "Drag direction in degrees (0=up, 90=right, 180=down, 270=left). Set to -1 to allow random directions.",
                }),
                "stroke_width": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 16384.0,
                    "step": 1.0,
                    "round": 0.1,
                    "tooltip": "When >0, limits the warp to a narrow strip aligned with the direction (like a brush stroke). "
                               "Units are image pixels. Set to -1 to disable.",
                }),
            },
            "optional": {
                "displacement_interpolation": (cls._DISPLACEMENT_INTERPOLATION_OPTIONS, {
                    "default": "bicubic",
                    "tooltip": "How to interpolate the sparse mesh drags into a full displacement field. "
                               "Use 'bspline' for smoother, more organic warps.",
                }),
                "spline_passes": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Only used when displacement_interpolation = 'bspline'. More passes = smoother warp.",
                }),
                "sampling_interpolation": (cls._SAMPLING_INTERPOLATION_OPTIONS, {
                    "default": "bilinear",
                    "tooltip": "How to sample the source image when applying the warp (bilinear/bicubic/nearest).",
                }),
            },
        }

    def drag(
        self,
        image,
        seed: int,
        points: int,
        drag_min: float,
        drag_max: float,
        direction: float = -1.0,
        stroke_width: float = -1.0,
        displacement_interpolation: str = "bicubic",
        spline_passes: int = 2,
        sampling_interpolation: str = "bilinear",
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"IMAGE input must be a torch.Tensor, got {type(image)}.")

        logger.debug(
            "Applying ImageMeshDrag: image=%s points=%d drag=[%.3f, %.3f] seed=%d disp=%s spline=%d sample=%s",
            tuple(image.shape),
            points,
            drag_min,
            drag_max,
            seed,
            displacement_interpolation,
            spline_passes,
            sampling_interpolation,
        )

        warped = mesh_drag_warp_image(
            image,
            points=int(points),
            drag_min=float(drag_min),
            drag_max=float(drag_max),
            seed=int(seed),
            direction=float(direction),
            stroke_width=float(stroke_width),
            displacement_interpolation=str(displacement_interpolation),
            spline_passes=int(spline_passes),
            sampling_interpolation=str(sampling_interpolation),
        )
        return (warped,)


NODE_CLASS_MAPPINGS = {
    "LatentMeshDrag": LatentMeshDrag,
    "ImageMeshDrag": ImageMeshDrag,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentMeshDrag": "Latent Mesh Drag",
    "ImageMeshDrag": "Image Mesh Drag",
}
