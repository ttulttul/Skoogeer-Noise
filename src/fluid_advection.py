from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from .masking import blend_image_with_mask, blend_with_mask, prepare_mask_nchw
except ImportError:  # pragma: no cover - fallback for direct module loading
    from masking import blend_image_with_mask, blend_with_mask, prepare_mask_nchw

logger = logging.getLogger(__name__)

_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF
_WRAP_MODES: Tuple[str, ...] = ("clamp", "wrap", "mirror")


@dataclass(frozen=True)
class _FlattenedLatent:
    tensor: torch.Tensor
    restore: Callable[[torch.Tensor], torch.Tensor]


def _flatten_latent_channels(tensor: torch.Tensor) -> _FlattenedLatent:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if tensor.ndim < 3:
        raise ValueError(f"Expected tensor with at least 3 dims, got {tuple(tensor.shape)}")

    if tensor.ndim == 3:
        tensor_nchw = tensor.unsqueeze(1)

        def restore(out: torch.Tensor) -> torch.Tensor:
            return out.squeeze(1)

        return _FlattenedLatent(tensor=tensor_nchw, restore=restore)

    if tensor.ndim == 4:

        def restore(out: torch.Tensor) -> torch.Tensor:
            return out

        return _FlattenedLatent(tensor=tensor, restore=restore)

    batch = int(tensor.shape[0])
    channels = int(tensor.shape[1])
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    extra_shape = tuple(int(dim) for dim in tensor.shape[2:-2])
    extra_dim = int(math.prod(extra_shape)) if extra_shape else 1

    flattened = tensor.reshape(batch, channels * extra_dim, height, width)

    def restore(out: torch.Tensor) -> torch.Tensor:
        return out.reshape(batch, channels, *extra_shape, height, width)

    return _FlattenedLatent(tensor=flattened, restore=restore)


def _pad_mode_for_wrap(wrap_mode: str) -> str:
    wrap_mode = str(wrap_mode).strip().lower()
    if wrap_mode == "wrap":
        return "circular"
    if wrap_mode == "mirror":
        return "reflect"
    return "replicate"


def _base_pixel_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    ys = torch.arange(int(height), device=device, dtype=torch.float32)
    xs = torch.arange(int(width), device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.unsqueeze(0)


def _advect_nchw(field: torch.Tensor, velocity: torch.Tensor, *, dt: float, wrap_mode: str, base_grid: torch.Tensor) -> torch.Tensor:
    if field.ndim != 4:
        raise ValueError(f"Expected field shape (B,C,H,W), got {tuple(field.shape)}")
    if velocity.ndim != 4 or int(velocity.shape[1]) != 2:
        raise ValueError(f"Expected velocity shape (B,2,H,W), got {tuple(velocity.shape)}")

    batch, _, height, width = field.shape
    if tuple(velocity.shape[0:1] + velocity.shape[2:]) != (batch, height, width):
        raise ValueError(f"Velocity shape {tuple(velocity.shape)} does not match field {tuple(field.shape)}")

    wrap_mode = str(wrap_mode).strip().lower()
    if wrap_mode not in _WRAP_MODES:
        raise ValueError(f"Unsupported wrap_mode '{wrap_mode}'. Supported: {_WRAP_MODES}")

    dt = float(dt)
    if dt == 0.0:
        return field

    vel_hw2 = velocity.permute(0, 2, 3, 1).to(dtype=torch.float32)
    prev = base_grid - vel_hw2 * dt

    if wrap_mode == "wrap":
        x = prev[..., 0]
        y = prev[..., 1]

        x0_floor = torch.floor(x)
        y0_floor = torch.floor(y)
        x1_floor = x0_floor + 1.0
        y1_floor = y0_floor + 1.0

        wx = (x - x0_floor).clamp(0.0, 1.0).unsqueeze(1)
        wy = (y - y0_floor).clamp(0.0, 1.0).unsqueeze(1)

        x0 = torch.remainder(x0_floor.to(torch.int64), int(width))
        x1 = torch.remainder(x1_floor.to(torch.int64), int(width))
        y0 = torch.remainder(y0_floor.to(torch.int64), int(height))
        y1 = torch.remainder(y1_floor.to(torch.int64), int(height))

        field_flat = field.reshape(int(batch), int(field.shape[1]), -1)

        def gather(x_idx: torch.Tensor, y_idx: torch.Tensor) -> torch.Tensor:
            lin = (y_idx * int(width) + x_idx).view(int(batch), 1, -1)
            lin = lin.expand(-1, int(field.shape[1]), -1)
            gathered = field_flat.gather(dim=2, index=lin)
            return gathered.view(int(batch), int(field.shape[1]), int(height), int(width))

        v00 = gather(x0, y0)
        v10 = gather(x1, y0)
        v01 = gather(x0, y1)
        v11 = gather(x1, y1)

        v0 = v00 * (1.0 - wx) + v10 * wx
        v1 = v01 * (1.0 - wx) + v11 * wx
        return v0 * (1.0 - wy) + v1 * wy

    padding_mode = "border"
    if wrap_mode == "mirror":
        padding_mode = "reflection"

    if width > 1:
        x_norm = prev[..., 0] * (2.0 / float(width - 1)) - 1.0
    else:
        x_norm = torch.zeros((batch, height, width), device=field.device, dtype=torch.float32)
    if height > 1:
        y_norm = prev[..., 1] * (2.0 / float(height - 1)) - 1.0
    else:
        y_norm = torch.zeros((batch, height, width), device=field.device, dtype=torch.float32)
    grid_norm = torch.stack((x_norm, y_norm), dim=-1)

    return F.grid_sample(
        field,
        grid_norm,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def _diffuse_velocity(velocity: torch.Tensor, *, diffusion: float, wrap_mode: str) -> torch.Tensor:
    diffusion = float(diffusion)
    if diffusion <= 0.0:
        return velocity

    _, channels, height, width = velocity.shape
    if channels != 2:
        raise ValueError(f"Expected velocity to have 2 channels, got {channels}")

    pad_mode = _pad_mode_for_wrap(wrap_mode)
    if pad_mode == "reflect" and (height < 2 or width < 2):
        pad_mode = "replicate"

    kernel = torch.tensor(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        device=velocity.device,
        dtype=velocity.dtype,
    ) / 16.0
    weight = kernel.view(1, 1, 3, 3).repeat(2, 1, 1, 1)
    padded = F.pad(velocity, (1, 1, 1, 1), mode=pad_mode)
    blurred = F.conv2d(padded, weight, padding=0, groups=2)
    return torch.lerp(velocity, blurred, diffusion)


def _vorticity_force(velocity: torch.Tensor, *, wrap_mode: str) -> torch.Tensor:
    if velocity.ndim != 4 or int(velocity.shape[1]) != 2:
        raise ValueError(f"Expected velocity shape (B,2,H,W), got {tuple(velocity.shape)}")

    pad_mode = _pad_mode_for_wrap(wrap_mode)
    height = int(velocity.shape[-2])
    width = int(velocity.shape[-1])
    if pad_mode == "reflect" and (height < 2 or width < 2):
        pad_mode = "replicate"

    v_x = velocity[:, 0:1]
    v_y = velocity[:, 1:2]

    v_y_pad = F.pad(v_y, (1, 1, 0, 0), mode=pad_mode)
    dv_y_dx = (v_y_pad[..., 2:] - v_y_pad[..., :-2]) * 0.5

    v_x_pad = F.pad(v_x, (0, 0, 1, 1), mode=pad_mode)
    dv_x_dy = (v_x_pad[..., 2:, :] - v_x_pad[..., :-2, :]) * 0.5

    curl = dv_y_dx - dv_x_dy
    abs_curl = curl.abs()

    abs_pad_x = F.pad(abs_curl, (1, 1, 0, 0), mode=pad_mode)
    grad_x = (abs_pad_x[..., 2:] - abs_pad_x[..., :-2]) * 0.5

    abs_pad_y = F.pad(abs_curl, (0, 0, 1, 1), mode=pad_mode)
    grad_y = (abs_pad_y[..., 2:, :] - abs_pad_y[..., :-2, :]) * 0.5

    norm = torch.sqrt(grad_x * grad_x + grad_y * grad_y)
    norm = torch.where(norm > 1e-6, norm, torch.ones_like(norm))
    n_x = grad_x / norm
    n_y = grad_y / norm

    force_x = -n_y * curl
    force_y = n_x * curl
    return torch.cat((force_x, force_y), dim=1)


def _inject_forces(
    velocity: torch.Tensor,
    *,
    generator: torch.Generator,
    force_count: int,
    strength: float,
    radius: float,
    swirl_strength: float,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    batch = int(velocity.shape[0])
    height = int(velocity.shape[-2])
    width = int(velocity.shape[-1])

    strength = float(strength)
    swirl_strength = float(swirl_strength)
    radius = float(radius)
    if force_count <= 0 or (strength == 0.0 and swirl_strength == 0.0) or radius <= 0.0:
        return velocity

    radius_px = max(radius * float(min(height, width)), 1.0)
    radius2 = float(radius_px * radius_px)
    inv_radius2 = 1.0 / max(radius2, 1e-12)

    for _ in range(int(force_count)):
        x0 = torch.rand((batch,), generator=generator, dtype=torch.float32) * float(max(width - 1, 1))
        y0 = torch.rand((batch,), generator=generator, dtype=torch.float32) * float(max(height - 1, 1))
        angles = torch.rand((batch,), generator=generator, dtype=torch.float32) * float(2.0 * math.pi)
        dir_x = torch.cos(angles)
        dir_y = torch.sin(angles)

        x0 = x0.to(device=velocity.device).view(batch, 1, 1)
        y0 = y0.to(device=velocity.device).view(batch, 1, 1)
        dir_x = dir_x.to(device=velocity.device).view(batch, 1, 1)
        dir_y = dir_y.to(device=velocity.device).view(batch, 1, 1)

        rx = grid_x - x0
        ry = grid_y - y0
        d2 = rx * rx + ry * ry

        mask = d2 < radius2
        falloff = torch.exp(-d2 * inv_radius2) * mask

        inv_d = torch.rsqrt(torch.clamp(d2, min=1e-12))
        tan_x = -ry * inv_d
        tan_y = rx * inv_d

        dv_x = (dir_x * strength + tan_x * swirl_strength) * falloff
        dv_y = (dir_y * strength + tan_y * swirl_strength) * falloff
        velocity[:, 0] += dv_x
        velocity[:, 1] += dv_y

    return velocity


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h = torch.remainder(h, 1.0)
    s = torch.clamp(s, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)

    c = v * s
    hp = h * 6.0
    x = c * (1.0 - torch.abs(torch.remainder(hp, 2.0) - 1.0))
    m = v - c

    z = torch.zeros_like(h)
    conds = [
        (hp >= 0.0) & (hp < 1.0),
        (hp >= 1.0) & (hp < 2.0),
        (hp >= 2.0) & (hp < 3.0),
        (hp >= 3.0) & (hp < 4.0),
        (hp >= 4.0) & (hp < 5.0),
        (hp >= 5.0) & (hp <= 6.0),
    ]

    r = torch.where(conds[0], c, z)
    g = torch.where(conds[0], x, z)
    b = torch.where(conds[0], z, z)

    r = torch.where(conds[1], x, r)
    g = torch.where(conds[1], c, g)
    b = torch.where(conds[1], z, b)

    r = torch.where(conds[2], z, r)
    g = torch.where(conds[2], c, g)
    b = torch.where(conds[2], x, b)

    r = torch.where(conds[3], z, r)
    g = torch.where(conds[3], x, g)
    b = torch.where(conds[3], c, b)

    r = torch.where(conds[4], x, r)
    g = torch.where(conds[4], z, g)
    b = torch.where(conds[4], c, b)

    r = torch.where(conds[5], c, r)
    g = torch.where(conds[5], z, g)
    b = torch.where(conds[5], x, b)

    rgb = torch.stack((r + m, g + m, b + m), dim=-1)
    return torch.clamp(rgb, 0.0, 1.0)


def _velocity_preview_image(velocity: torch.Tensor) -> torch.Tensor:
    if velocity.ndim != 4 or int(velocity.shape[1]) != 2:
        raise ValueError(f"Expected velocity shape (B,2,H,W), got {tuple(velocity.shape)}")

    vx = velocity[:, 0]
    vy = velocity[:, 1]
    mag = torch.sqrt(vx * vx + vy * vy)
    mag_max = mag.amax(dim=(1, 2), keepdim=True)
    mag_max = torch.where(mag_max > 1e-6, mag_max, torch.ones_like(mag_max))

    hue = torch.atan2(vy, vx) / float(2.0 * math.pi)
    hue = torch.remainder(hue, 1.0)
    sat = torch.ones_like(hue)
    val = torch.clamp(mag / mag_max, 0.0, 1.0)

    rgb = _hsv_to_rgb(hue, sat, val)
    return rgb.to(dtype=torch.float32)


def _simulate_fluid_advection(
    field: torch.Tensor,
    *,
    steps: int,
    dt: float,
    resolution_scale: float,
    force_count: int,
    force_strength: float,
    force_radius: float,
    swirl_strength: float,
    velocity_damping: float,
    diffusion: float,
    vorticity: float,
    seed: int,
    wrap_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if field.ndim != 4:
        raise ValueError(f"Expected field shape (B,C,H,W), got {tuple(field.shape)}")
    if not field.is_floating_point():
        raise TypeError(f"Expected floating-point field tensor, got dtype={field.dtype}")

    steps = int(steps)
    if steps <= 0:
        preview = torch.zeros((int(field.shape[0]), int(field.shape[-2]), int(field.shape[-1]), 3), device=field.device, dtype=torch.float32)
        return field, preview

    wrap_mode = str(wrap_mode).strip().lower()
    if wrap_mode not in _WRAP_MODES:
        raise ValueError(f"Unsupported wrap_mode '{wrap_mode}'. Supported: {_WRAP_MODES}")

    force_count = int(force_count)
    force_strength = float(force_strength)
    force_radius = float(force_radius)
    swirl_strength = float(swirl_strength)
    diffusion = float(diffusion)
    vorticity = float(vorticity)

    if force_count <= 0 or (force_strength == 0.0 and swirl_strength == 0.0) or force_radius <= 0.0:
        preview = torch.zeros((int(field.shape[0]), int(field.shape[-2]), int(field.shape[-1]), 3), device=field.device, dtype=torch.float32)
        return field, preview

    velocity_damping = float(velocity_damping)
    velocity_damping = max(0.0, min(velocity_damping, 1.0))

    batch, channels, height, width = field.shape
    resolution_scale = float(resolution_scale)
    resolution_scale = max(0.01, min(resolution_scale, 1.0))
    sim_h = max(4, int(round(float(height) * resolution_scale)))
    sim_w = max(4, int(round(float(width) * resolution_scale)))

    generator = torch.Generator(device="cpu").manual_seed(int(seed) & _SEED_MASK_64)

    with torch.no_grad():
        if (sim_h, sim_w) != (int(height), int(width)):
            field_sim = F.interpolate(field.to(dtype=torch.float32), size=(sim_h, sim_w), mode="bilinear", align_corners=False)
        else:
            field_sim = field.to(dtype=torch.float32)

        velocity = torch.zeros((int(batch), 2, sim_h, sim_w), device=field.device, dtype=torch.float32)
        base_grid = _base_pixel_grid(sim_h, sim_w, device=field.device)
        grid_x = base_grid[..., 0]
        grid_y = base_grid[..., 1]

        for step in range(int(steps)):
            velocity = _inject_forces(
                velocity,
                generator=generator,
                force_count=force_count,
                strength=force_strength,
                radius=force_radius,
                swirl_strength=swirl_strength,
                grid_x=grid_x,
                grid_y=grid_y,
            )

            if diffusion > 0.0:
                velocity = _diffuse_velocity(velocity, diffusion=diffusion, wrap_mode=wrap_mode)

            if vorticity > 0.0:
                velocity = velocity + _vorticity_force(velocity, wrap_mode=wrap_mode) * vorticity

            velocity = _advect_nchw(velocity, velocity, dt=float(dt), wrap_mode=wrap_mode, base_grid=base_grid)
            velocity = velocity * velocity_damping
            field_sim = _advect_nchw(field_sim, velocity, dt=float(dt), wrap_mode=wrap_mode, base_grid=base_grid)

            if not torch.isfinite(field_sim).all():
                logger.warning("Fluid simulation produced non-finite values at step %d; clamping to finite range.", step)
                field_sim = torch.nan_to_num(field_sim, nan=0.0, posinf=0.0, neginf=0.0)

        if (sim_h, sim_w) != (int(height), int(width)):
            field_out = F.interpolate(field_sim, size=(int(height), int(width)), mode="bilinear", align_corners=False)
            velocity_out = F.interpolate(velocity, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        else:
            field_out = field_sim
            velocity_out = velocity

        preview = _velocity_preview_image(velocity_out)
        return field_out.to(dtype=field.dtype), preview


class FluidLatentAdvection:
    """
    Treats a latent's channels as a dye field and advects them through a random, viscous velocity field.

    See `docs/fluid-simulation.md` for the underlying algorithm design.
    """

    CATEGORY = "latent/perturb"
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "velocity_preview")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to distort by fluid advection (treats channels as dye)."}),
                "steps": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 256,
                    "tooltip": "Number of simulation steps.",
                }),
                "dt": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Time step size used during semi-Lagrangian advection.",
                }),
                "resolution_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.25,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Internal simulation resolution scale (lower = faster).",
                }),
                "force_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 64,
                    "tooltip": "Number of random force injections ('sticks') per step.",
                }),
                "force_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 64.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Linear force magnitude added to velocity (in latent pixels / step).",
                }),
                "force_radius": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Normalized radius of influence (0.1 = 10% of min(width,height)).",
                }),
                "swirl_strength": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 64.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Rotational component added around each stick position.",
                }),
                "velocity_damping": ("FLOAT", {
                    "default": 0.98,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": 0.001,
                    "tooltip": "Velocity decay applied each step (lower = more viscous / quickly settles).",
                }),
                "diffusion": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Optional velocity diffusion (simple Gaussian smoothing).",
                }),
                "vorticity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Vorticity confinement strength (adds swirling detail).",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed controlling the random stick injections.",
                }),
                "wrap_mode": (list(_WRAP_MODES), {"default": "clamp", "tooltip": "How advection samples outside the bounds."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask to limit the effect to masked areas (mask is resized to latent resolution)."}),
            },
        }

    def run(
        self,
        latent,
        steps: int,
        dt: float,
        resolution_scale: float,
        force_count: int,
        force_strength: float,
        force_radius: float,
        swirl_strength: float,
        velocity_damping: float,
        diffusion: float,
        vorticity: float,
        seed: int,
        wrap_mode: str,
        mask=None,
    ):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("LATENT input must be a dictionary containing a 'samples' tensor.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError(f"LATENT['samples'] must be a torch.Tensor, got {type(samples)}.")
        if samples.ndim < 3:
            raise ValueError(f"LATENT['samples'] must have at least 3 dims, got {tuple(samples.shape)}.")

        flattened = _flatten_latent_channels(samples)
        field_out, preview = _simulate_fluid_advection(
            flattened.tensor,
            steps=int(steps),
            dt=float(dt),
            resolution_scale=float(resolution_scale),
            force_count=int(force_count),
            force_strength=float(force_strength),
            force_radius=float(force_radius),
            swirl_strength=float(swirl_strength),
            velocity_damping=float(velocity_damping),
            diffusion=float(diffusion),
            vorticity=float(vorticity),
            seed=int(seed),
            wrap_mode=str(wrap_mode),
        )
        out_samples = flattened.restore(field_out)

        out_latent = latent.copy()
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")

            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(samples.shape[0]),
                height=int(samples.shape[-2]),
                width=int(samples.shape[-1]),
                device=samples.device,
            )
            out_samples = blend_with_mask(samples, out_samples, mask_nchw)

            preview_cf = preview.permute(0, 3, 1, 2)
            masked_preview_cf = preview_cf * mask_nchw
            preview = masked_preview_cf.permute(0, 2, 3, 1)

        out_latent["samples"] = out_samples

        if "noise_mask" in out_latent and isinstance(out_latent["noise_mask"], torch.Tensor):
            noise_mask = out_latent["noise_mask"]
            flattened_mask = _flatten_latent_channels(noise_mask)
            advected_mask, _ = _simulate_fluid_advection(
                flattened_mask.tensor.to(device=samples.device),
                steps=int(steps),
                dt=float(dt),
                resolution_scale=float(resolution_scale),
                force_count=int(force_count),
                force_strength=float(force_strength),
                force_radius=float(force_radius),
                swirl_strength=float(swirl_strength),
                velocity_damping=float(velocity_damping),
                diffusion=float(diffusion),
                vorticity=float(vorticity),
                seed=int(seed),
                wrap_mode=str(wrap_mode),
            )
            advected_mask = flattened_mask.restore(advected_mask)
            if mask is not None:
                mask_nchw = prepare_mask_nchw(
                    mask,
                    batch_size=int(noise_mask.shape[0]),
                    height=int(noise_mask.shape[-2]),
                    width=int(noise_mask.shape[-1]),
                    device=samples.device,
                )
                advected_mask = blend_with_mask(noise_mask, advected_mask, mask_nchw)
            out_latent["noise_mask"] = advected_mask.clamp(0.0, 1.0).to(dtype=noise_mask.dtype)

        return (out_latent, preview)


class FluidImageAdvection:
    """
    Distorts an `IMAGE` tensor by advecting it through a random, viscous velocity field.

    See `docs/fluid-simulation.md` for the underlying algorithm design.
    """

    CATEGORY = "image/perturb"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "velocity_preview")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to distort by fluid advection."}),
                "steps": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 256,
                    "tooltip": "Number of simulation steps.",
                }),
                "dt": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Time step size used during semi-Lagrangian advection.",
                }),
                "resolution_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.25,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Internal simulation resolution scale (lower = faster).",
                }),
                "force_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 64,
                    "tooltip": "Number of random force injections ('sticks') per step.",
                }),
                "force_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 256.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Linear force magnitude added to velocity (in pixels / step).",
                }),
                "force_radius": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Normalized radius of influence (0.1 = 10% of min(width,height)).",
                }),
                "swirl_strength": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 256.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Rotational component added around each stick position.",
                }),
                "velocity_damping": ("FLOAT", {
                    "default": 0.98,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": 0.001,
                    "tooltip": "Velocity decay applied each step (lower = more viscous / quickly settles).",
                }),
                "diffusion": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Optional velocity diffusion (simple Gaussian smoothing).",
                }),
                "vorticity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Vorticity confinement strength (adds swirling detail).",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "Seed controlling the random stick injections.",
                }),
                "wrap_mode": (list(_WRAP_MODES), {"default": "clamp", "tooltip": "How advection samples outside the bounds."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask to limit the effect to masked areas (mask is resized to image resolution)."}),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        steps: int,
        dt: float,
        resolution_scale: float,
        force_count: int,
        force_strength: float,
        force_radius: float,
        swirl_strength: float,
        velocity_damping: float,
        diffusion: float,
        vorticity: float,
        seed: int,
        wrap_mode: str,
        mask=None,
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"IMAGE input must be a torch.Tensor, got {type(image)}.")

        if image.ndim == 3 and int(image.shape[-1]) in (1, 3, 4):
            (out_image, out_preview) = self.run(
                image.unsqueeze(0),
                steps=steps,
                dt=dt,
                resolution_scale=resolution_scale,
                force_count=force_count,
                force_strength=force_strength,
                force_radius=force_radius,
                swirl_strength=swirl_strength,
                velocity_damping=velocity_damping,
                diffusion=diffusion,
                vorticity=vorticity,
                seed=seed,
                wrap_mode=wrap_mode,
                mask=mask,
            )
            return (out_image.squeeze(0), out_preview.squeeze(0))

        if image.ndim != 4 or int(image.shape[-1]) not in (1, 3, 4):
            raise ValueError(f"Expected IMAGE tensor with shape (B,H,W,C), got {tuple(image.shape)}")

        image_cf = image.permute(0, 3, 1, 2)
        out_cf, preview = _simulate_fluid_advection(
            image_cf,
            steps=int(steps),
            dt=float(dt),
            resolution_scale=float(resolution_scale),
            force_count=int(force_count),
            force_strength=float(force_strength),
            force_radius=float(force_radius),
            swirl_strength=float(swirl_strength),
            velocity_damping=float(velocity_damping),
            diffusion=float(diffusion),
            vorticity=float(vorticity),
            seed=int(seed),
            wrap_mode=str(wrap_mode),
        )
        out = out_cf.permute(0, 2, 3, 1)
        if mask is not None:
            out = blend_image_with_mask(image, out, mask)
            preview_cf = preview.permute(0, 3, 1, 2)
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(image.shape[0]),
                height=int(image.shape[1]),
                width=int(image.shape[2]),
                device=image.device,
            )
            preview_cf = preview_cf * mask_nchw
            preview = preview_cf.permute(0, 2, 3, 1)

        return (out, preview)


NODE_CLASS_MAPPINGS = {
    "FluidLatentAdvection": FluidLatentAdvection,
    "FluidImageAdvection": FluidImageAdvection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluidLatentAdvection": "Fluid Latent Advection",
    "FluidImageAdvection": "Fluid Image Advection",
}
