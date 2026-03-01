from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from .masking import blend_image_with_mask, blend_with_mask, prepare_mask_nchw
except ImportError:  # pragma: no cover - fallback for direct module loading
    from masking import blend_image_with_mask, blend_with_mask, prepare_mask_nchw

try:
    import torchvision.transforms.functional as TF  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without torchvision
    TF = None

try:
    import comfy.model_management as comfy_model_management  # type: ignore[import-not-found]
    import comfy.lora as comfy_lora  # type: ignore[import-not-found]
    import comfy.sample as comfy_sample  # type: ignore[import-not-found]
    import comfy.samplers as comfy_samplers  # type: ignore[import-not-found]
    import comfy.hooks as comfy_hooks  # type: ignore[import-not-found]
    import comfy.weight_adapter as comfy_weight_adapter  # type: ignore[import-not-found]
    import comfy.utils as comfy_utils  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - comfy is only available inside ComfyUI
    comfy_model_management = None
    comfy_lora = None
    comfy_sample = None
    comfy_samplers = None
    comfy_hooks = None
    comfy_weight_adapter = None
    comfy_utils = None

try:
    import folder_paths  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - folder_paths exists inside ComfyUI
    folder_paths = None

try:
    import latent_preview  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - latent_preview exists inside ComfyUI
    latent_preview = None

logger = logging.getLogger(__name__)

_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF


def _get_device() -> torch.device:
    if comfy_model_management is not None:
        return comfy_model_management.get_torch_device()
    return torch.device("cpu")


class _ProgressBar:
    def __init__(self, total: int):
        self.total = int(total)
        self.current = 0
        self._bar = None
        if comfy_utils is not None:
            try:
                self._bar = comfy_utils.ProgressBar(self.total)
            except Exception:  # pragma: no cover - defensive
                self._bar = None

    def update(self, amount: int) -> None:
        amount = int(amount)
        self.current += amount
        if self._bar is None:
            return
        if hasattr(self._bar, "update"):
            self._bar.update(amount)
        elif hasattr(self._bar, "update_absolute"):
            self._bar.update_absolute(self.current, total=self.total)

    def update_absolute(self, value: int, *, total: Optional[int] = None) -> None:
        if total is not None:
            self.total = int(total)
        value = int(value)
        delta = value - self.current
        self.current = value
        if self._bar is None:
            return
        if hasattr(self._bar, "update_absolute"):
            self._bar.update_absolute(self.current, total=self.total)
        elif hasattr(self._bar, "update") and delta:
            self._bar.update(delta)


def _progress_bar(total: int) -> _ProgressBar:
    return _ProgressBar(total)


def _seeded_gaussian_noise_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    seed = int(seed) & _SEED_MASK_64
    if tensor.ndim == 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((), generator=generator, device="cpu", dtype=torch.float32)
        if noise.device != tensor.device:
            noise = noise.to(device=tensor.device)
        return noise.to(dtype=tensor.dtype)

    batch = int(tensor.shape[0])
    if batch <= 0:
        raise ValueError("Input tensor batch dimension must be non-empty.")

    if batch == 1:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(tensor.shape, generator=generator, device="cpu", dtype=torch.float32)
        if noise.device != tensor.device:
            noise = noise.to(device=tensor.device)
        return noise.to(dtype=tensor.dtype)

    noise = torch.empty_like(tensor, device=tensor.device, dtype=tensor.dtype)
    for batch_index in range(batch):
        batch_seed = (seed + batch_index) & _SEED_MASK_64
        generator = torch.Generator(device="cpu").manual_seed(batch_seed)
        sample_noise = torch.randn(tensor[batch_index].shape, generator=generator, device="cpu", dtype=torch.float32)
        if sample_noise.device != tensor.device:
            sample_noise = sample_noise.to(device=tensor.device)
        noise[batch_index] = sample_noise.to(dtype=tensor.dtype)

    logger.debug("Generated seeded noise: shape=%s batch=%d seed=%d", tuple(tensor.shape), batch, seed)
    return noise


def _per_sample_std(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        std = tensor.detach().float().std(unbiased=False)
        if not torch.isfinite(std) or std <= 1e-6:
            return std.new_zeros(())
        return std

    batch = int(tensor.shape[0])
    if batch <= 0:
        raise ValueError("Input tensor batch dimension must be non-empty.")

    flat = tensor.detach().float().reshape(batch, -1)
    std = flat.std(dim=1, unbiased=False)
    std = torch.where(torch.isfinite(std), std, torch.zeros_like(std))
    std = torch.where(std > 1e-6, std, torch.zeros_like(std))
    return std


def _gaussian_kernel1d(kernel_size, sigma, device, dtype):
    radius = kernel_size // 2
    positions = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    sigma = max(float(sigma), 1e-6)
    kernel = torch.exp(-(positions**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_fallback(tensor, kernel_size, sigma):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)

    ky, kx = kernel_size
    sy, sx = sigma

    if ky <= 1 and kx <= 1:
        return tensor

    device = tensor.device
    dtype = tensor.dtype

    kernel_y = _gaussian_kernel1d(ky, sy, device, dtype)
    kernel_x = _gaussian_kernel1d(kx, sx, device, dtype)

    kernel_2d = torch.outer(kernel_y, kernel_x)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, ky, kx)
    channels = tensor.shape[1]
    kernel = kernel.expand(channels, 1, ky, kx)

    padding = (ky // 2, kx // 2)
    return F.conv2d(tensor, kernel, padding=padding, groups=channels)


if TF is None:

    class _FunctionalFallback:
        @staticmethod
        def gaussian_blur(tensor, kernel_size, sigma):
            return _gaussian_blur_fallback(tensor, kernel_size, sigma)

    TF = _FunctionalFallback()


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    return a + t * (b - a)


def _generate_permutation(seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(256, generator=generator)
    return torch.cat([perm, perm]).to(device)


def _coordinate_grid(size, device, dtype):
    axes = [torch.linspace(0.0, 1.0, steps=s, device=device, dtype=dtype) for s in size]
    return torch.meshgrid(*axes, indexing="ij")


def _perlin_2d(x, y, perm, dtype):
    gradients = torch.tensor(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
        dtype=dtype,
        device=x.device,
    )

    x0 = torch.floor(x)
    y0 = torch.floor(y)

    xi = torch.remainder(x0.to(torch.int64), 256)
    yi = torch.remainder(y0.to(torch.int64), 256)

    xf = x - x0
    yf = y - y0

    xi1 = torch.remainder(xi + 1, 256)
    yi1 = torch.remainder(yi + 1, 256)

    perm_xi = perm[xi]
    perm_xi1 = perm[xi1]

    aa = perm[perm_xi + yi]
    ab = perm[perm_xi + yi1]
    ba = perm[perm_xi1 + yi]
    bb = perm[perm_xi1 + yi1]

    grad_count = gradients.shape[0]

    g_aa = gradients[torch.remainder(aa, grad_count)]
    g_ab = gradients[torch.remainder(ab, grad_count)]
    g_ba = gradients[torch.remainder(ba, grad_count)]
    g_bb = gradients[torch.remainder(bb, grad_count)]

    xf_1 = xf - 1.0
    yf_1 = yf - 1.0

    dot_aa = g_aa[..., 0] * xf + g_aa[..., 1] * yf
    dot_ba = g_ba[..., 0] * xf_1 + g_ba[..., 1] * yf
    dot_ab = g_ab[..., 0] * xf + g_ab[..., 1] * yf_1
    dot_bb = g_bb[..., 0] * xf_1 + g_bb[..., 1] * yf_1

    u = _fade(xf)
    v = _fade(yf)

    x1 = _lerp(dot_aa, dot_ba, u)
    x2 = _lerp(dot_ab, dot_bb, u)
    return _lerp(x1, x2, v)


def _perlin_3d(x, y, z, perm, dtype):
    gradients = torch.tensor(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
        ],
        dtype=dtype,
        device=x.device,
    )

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    z0 = torch.floor(z)

    xi = torch.remainder(x0.to(torch.int64), 256)
    yi = torch.remainder(y0.to(torch.int64), 256)
    zi = torch.remainder(z0.to(torch.int64), 256)

    xf = x - x0
    yf = y - y0
    zf = z - z0

    xi1 = torch.remainder(xi + 1, 256)
    yi1 = torch.remainder(yi + 1, 256)
    zi1 = torch.remainder(zi + 1, 256)

    perm_xi = perm[xi]
    perm_xi1 = perm[xi1]

    perm_xy = perm[perm_xi + yi]
    perm_xy1 = perm[perm_xi + yi1]
    perm_x1y = perm[perm_xi1 + yi]
    perm_x1y1 = perm[perm_xi1 + yi1]

    aaa = perm[perm_xy + zi]
    aab = perm[perm_xy + zi1]
    aba = perm[perm_xy1 + zi]
    abb = perm[perm_xy1 + zi1]
    baa = perm[perm_x1y + zi]
    bab = perm[perm_x1y + zi1]
    bba = perm[perm_x1y1 + zi]
    bbb = perm[perm_x1y1 + zi1]

    grad_count = gradients.shape[0]

    g_aaa = gradients[torch.remainder(aaa, grad_count)]
    g_aab = gradients[torch.remainder(aab, grad_count)]
    g_aba = gradients[torch.remainder(aba, grad_count)]
    g_abb = gradients[torch.remainder(abb, grad_count)]
    g_baa = gradients[torch.remainder(baa, grad_count)]
    g_bab = gradients[torch.remainder(bab, grad_count)]
    g_bba = gradients[torch.remainder(bba, grad_count)]
    g_bbb = gradients[torch.remainder(bbb, grad_count)]

    xf_1 = xf - 1.0
    yf_1 = yf - 1.0
    zf_1 = zf - 1.0

    dot_aaa = g_aaa[..., 0] * xf + g_aaa[..., 1] * yf + g_aaa[..., 2] * zf
    dot_baa = g_baa[..., 0] * xf_1 + g_baa[..., 1] * yf + g_baa[..., 2] * zf
    dot_aba = g_aba[..., 0] * xf + g_aba[..., 1] * yf_1 + g_aba[..., 2] * zf
    dot_bba = g_bba[..., 0] * xf_1 + g_bba[..., 1] * yf_1 + g_bba[..., 2] * zf
    dot_aab = g_aab[..., 0] * xf + g_aab[..., 1] * yf + g_aab[..., 2] * zf_1
    dot_bab = g_bab[..., 0] * xf_1 + g_bab[..., 1] * yf + g_bab[..., 2] * zf_1
    dot_abb = g_abb[..., 0] * xf + g_abb[..., 1] * yf_1 + g_abb[..., 2] * zf_1
    dot_bbb = g_bbb[..., 0] * xf_1 + g_bbb[..., 1] * yf_1 + g_bbb[..., 2] * zf_1

    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    x1 = _lerp(dot_aaa, dot_baa, u)
    x2 = _lerp(dot_aba, dot_bba, u)
    y1 = _lerp(x1, x2, v)

    x3 = _lerp(dot_aab, dot_bab, u)
    x4 = _lerp(dot_abb, dot_bbb, u)
    y2 = _lerp(x3, x4, v)

    return _lerp(y1, y2, w)


def _fractal_perlin(
    size,
    seed,
    frequency,
    octaves,
    persistence,
    lacunarity,
    device,
    *,
    progress=None,
    progress_state=None,
    progress_total=None,
):
    if len(size) not in (2, 3):
        raise ValueError("Perlin noise supports 2D or 3D shapes")

    noise_dtype = torch.float32
    coords = _coordinate_grid(size, device, noise_dtype)

    total = torch.zeros(size, dtype=noise_dtype, device=device)
    amplitude = 1.0
    max_amplitude = 0.0
    freq = frequency

    progress_cursor = 0
    if progress_state is not None:
        progress_cursor = int(progress_state.get("current", 0))

    for octave in range(octaves):
        perm = _generate_permutation(seed + octave, device)

        scaled_coords = [coord * freq for coord in coords]

        if len(size) == 2:
            octave_noise = _perlin_2d(scaled_coords[1], scaled_coords[0], perm, noise_dtype)
        else:
            octave_noise = _perlin_3d(scaled_coords[2], scaled_coords[1], scaled_coords[0], perm, noise_dtype)

        total = total + octave_noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

        progress_cursor += 1
        if progress is not None:
            effective_total = progress_total if progress_total is not None else progress_cursor
            if hasattr(progress, "update_absolute"):
                progress.update_absolute(progress_cursor, total=effective_total)
            else:  # pragma: no cover
                progress.update(1)

    if progress_state is not None:
        progress_state["current"] = progress_cursor

    if max_amplitude > 0:
        total = total / max_amplitude

    return total


def _simplex_noise_2d(x, y, perm, dtype):
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0

    s = (x + y) * F2
    i = torch.floor(x + s)
    j = torch.floor(y + s)

    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0

    cond = x0 > y0
    i1 = torch.where(cond, torch.ones_like(x0), torch.zeros_like(x0))
    j1 = torch.where(cond, torch.zeros_like(y0), torch.ones_like(y0))

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ii = torch.remainder(i.to(torch.int64), 256)
    jj = torch.remainder(j.to(torch.int64), 256)

    i1_int = i1.to(torch.int64)
    j1_int = j1.to(torch.int64)

    gi0 = perm[ii + perm[jj]] % 12
    gi1 = perm[ii + i1_int + perm[jj + j1_int]] % 12
    gi2 = perm[ii + 1 + perm[jj + 1]] % 12

    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    gradients = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [inv_sqrt2, inv_sqrt2],
            [-inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2],
            [-inv_sqrt2, -inv_sqrt2],
        ],
        dtype=dtype,
        device=x.device,
    )

    def dot(g, xx, yy):
        return g[..., 0] * xx + g[..., 1] * yy

    t0 = 0.5 - x0 * x0 - y0 * y0
    t1 = 0.5 - x1 * x1 - y1 * y1
    t2 = 0.5 - x2 * x2 - y2 * y2

    n0 = torch.zeros_like(x0)
    n1 = torch.zeros_like(x0)
    n2 = torch.zeros_like(x0)

    mask0 = t0 > 0
    mask1 = t1 > 0
    mask2 = t2 > 0

    if mask0.any():
        t0_sq = t0[mask0] * t0[mask0]
        t0_4 = t0_sq * t0_sq
        g = gradients[gi0[mask0]]
        n0[mask0] = t0_4 * dot(g, x0[mask0], y0[mask0])

    if mask1.any():
        t1_sq = t1[mask1] * t1[mask1]
        t1_4 = t1_sq * t1_sq
        g = gradients[gi1[mask1]]
        n1[mask1] = t1_4 * dot(g, x1[mask1], y1[mask1])

    if mask2.any():
        t2_sq = t2[mask2] * t2[mask2]
        t2_4 = t2_sq * t2_sq
        g = gradients[gi2[mask2]]
        n2[mask2] = t2_4 * dot(g, x2[mask2], y2[mask2])

    return 70.0 * (n0 + n1 + n2)


def _fractal_simplex(
    size,
    seed,
    frequency,
    octaves,
    persistence,
    lacunarity,
    device,
    *,
    progress=None,
    progress_state=None,
    progress_total=None,
):
    if len(size) != 2:
        raise ValueError("Simplex noise is implemented for 2D shapes")

    dtype = torch.float32
    coords = _coordinate_grid(size, device, dtype)

    total = torch.zeros(size, dtype=dtype, device=device)
    amplitude = 1.0
    max_amplitude = 0.0
    freq = frequency

    progress_cursor = 0
    if progress_state is not None:
        progress_cursor = int(progress_state.get("current", 0))

    for octave in range(octaves):
        perm = _generate_permutation(seed + octave, device)
        x = coords[1] * freq
        y = coords[0] * freq
        octave_noise = _simplex_noise_2d(x, y, perm, dtype)
        total = total + octave_noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

        progress_cursor += 1
        if progress is not None:
            effective_total = progress_total if progress_total is not None else progress_cursor
            if hasattr(progress, "update_absolute"):
                progress.update_absolute(progress_cursor, total=effective_total)
            else:  # pragma: no cover
                progress.update(1)

    if progress_state is not None:
        progress_state["current"] = progress_cursor

    if max_amplitude > 0:
        total = total / max_amplitude

    return total


def _worley_noise(
    size,
    seed,
    feature_points,
    metric,
    jitter,
    device,
    *,
    chunk_capacity=None,
    progress=None,
    progress_offset=0,
    progress_total=None,
):
    if len(size) != 2:
        raise ValueError("Worley noise expects a 2D shape")

    height, width = size
    dtype = torch.float32
    feature_points = max(1, int(feature_points))
    cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
    points = torch.rand((feature_points, 2), generator=cpu_gen, dtype=dtype)

    if jitter > 0.0:
        jitter_gen = torch.rand((feature_points, 2), generator=cpu_gen, dtype=dtype) - 0.5
        points = torch.clamp(points + jitter_gen * jitter, 0.0, 1.0)

    points = points.to(device=device)

    ys = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y = ys.view(height, 1).expand(height, width)
    grid_x = xs.view(1, width).expand(height, width)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)

    if chunk_capacity is None:
        chunk_capacity = _worley_chunk_capacity(height, width)
    else:
        chunk_capacity = max(1, int(chunk_capacity))
    chunk_size = _worley_chunk_size(feature_points, chunk_capacity)
    total_chunks = max(1, math.ceil(feature_points / chunk_size))
    effective_total = progress_total if progress_total is not None else progress_offset + total_chunks

    min_dist = None

    for chunk_index, start in enumerate(range(0, feature_points, chunk_size)):
        chunk = points[start : start + chunk_size]
        diff = grid - chunk.view(1, 1, -1, 2)

        if metric == "manhattan":
            distances = diff.abs().sum(dim=-1)
        elif metric == "chebyshev":
            distances = diff.abs().max(dim=-1).values
        else:
            distances = torch.sqrt((diff * diff).sum(dim=-1))

        chunk_min = distances.min(dim=-1).values
        if min_dist is None:
            min_dist = chunk_min
        else:
            min_dist = torch.minimum(min_dist, chunk_min)

        if progress is not None:
            absolute = progress_offset + chunk_index + 1
            if hasattr(progress, "update_absolute"):
                progress.update_absolute(absolute, total=effective_total)
            else:  # pragma: no cover - backwards-compatible progress bars
                progress.update(1)

    if min_dist is None:
        min_dist = torch.zeros((height, width), dtype=dtype, device=device)

    max_dist = torch.max(min_dist)
    if max_dist > 0:
        min_dist = min_dist / max_dist

    return 1.0 - min_dist


_WORLEY_MAX_CHUNK_ELEMENTS = 4_194_304


def _worley_chunk_capacity(height, width):
    pixels = max(1, height * width)
    return max(1, _WORLEY_MAX_CHUNK_ELEMENTS // pixels)


def _worley_chunk_size(feature_points, chunk_capacity):
    feature_points = max(1, int(feature_points))
    chunk_capacity = max(1, int(chunk_capacity))
    return max(1, min(feature_points, chunk_capacity))


def _worley_progress_steps(size, base_points, octaves, lacunarity):
    height, width = size
    chunk_capacity = _worley_chunk_capacity(height, width)
    total = 0
    points = float(base_points)
    for _ in range(octaves):
        octave_points = max(1, int(round(points)))
        chunk_size = _worley_chunk_size(octave_points, chunk_capacity)
        total += math.ceil(octave_points / chunk_size)
        points *= lacunarity
    return max(1, total)


def _fractal_worley(
    size,
    seed,
    base_points,
    octaves,
    persistence,
    lacunarity,
    metric,
    jitter,
    device,
    *,
    progress=None,
    progress_state=None,
    progress_total=None,
):
    dtype = torch.float32
    total = torch.zeros(size, dtype=dtype, device=device)
    amplitude = 1.0
    max_amplitude = 0.0
    points = float(base_points)

    height, width = size
    chunk_capacity = _worley_chunk_capacity(height, width)
    progress_cursor = 0
    if progress_state is not None:
        progress_cursor = int(progress_state.get("current", 0))

    for octave in range(octaves):
        octave_points = max(1, int(round(points)))
        chunk_size = _worley_chunk_size(octave_points, chunk_capacity)
        chunk_count = max(1, math.ceil(octave_points / chunk_size))
        effective_total = progress_total if progress_total is not None else progress_cursor + chunk_count

        noise = _worley_noise(
            size,
            seed + octave,
            octave_points,
            metric,
            jitter,
            device,
            chunk_capacity=chunk_capacity,
            progress=progress,
            progress_offset=progress_cursor,
            progress_total=effective_total,
        )
        total = total + noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        points *= lacunarity

        progress_cursor += chunk_count

    if progress_state is not None:
        progress_state["current"] = progress_cursor

    if max_amplitude > 0:
        total = total / max_amplitude

    return total * 2.0 - 1.0


def _gray_scott_pattern(size, seed, steps, feed, kill, diff_u, diff_v, dt, device):
    if len(size) != 2:
        raise ValueError("Gray-Scott pattern expects a 2D shape")

    height, width = size
    dtype = torch.float32

    U = torch.ones((1, 1, height, width), device=device, dtype=dtype)
    V = torch.zeros((1, 1, height, width), device=device, dtype=dtype)

    cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((height, width), generator=cpu_gen, dtype=dtype)
    noise = noise.to(device=device)

    U = U - 0.5 * noise.unsqueeze(0).unsqueeze(0)
    V = V + 0.5 * noise.unsqueeze(0).unsqueeze(0)

    square = max(2, min(height, width) // 10)
    start_y = height // 2 - square // 2
    start_x = width // 2 - square // 2
    V[:, :, start_y : start_y + square, start_x : start_x + square] = 1.0
    U[:, :, start_y : start_y + square, start_x : start_x + square] = 0.0

    lap_kernel = (
        torch.tensor(
            [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
            dtype=dtype,
            device=device,
        )
        .view(1, 1, 3, 3)
    )

    for _ in range(steps):
        U_pad = torch.nn.functional.pad(U, (1, 1, 1, 1), mode="reflect")
        V_pad = torch.nn.functional.pad(V, (1, 1, 1, 1), mode="reflect")
        lap_u = torch.nn.functional.conv2d(U_pad, lap_kernel)
        lap_v = torch.nn.functional.conv2d(V_pad, lap_kernel)

        reaction = U * V * V
        U = U + (diff_u * lap_u - reaction + feed * (1.0 - U)) * dt
        V = V + (diff_v * lap_v + reaction - (feed + kill) * V) * dt

        U = torch.clamp(U, 0.0, 1.0)
        V = torch.clamp(V, 0.0, 1.0)

    pattern = V.squeeze(0).squeeze(0)
    return _normalize_noise_tensor(pattern)


def _count_noise_calls(batch, channels, frames, channel_mode, temporal_mode):
    if channel_mode == "shared":
        if frames > 1 and temporal_mode == "animated":
            per_sample = frames
        else:
            per_sample = 1
    else:
        per_sample = channels
        if frames > 1 and temporal_mode == "animated":
            per_sample *= frames
    return batch * per_sample


def _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator):
    if strength == 0.0:
        return latent_tensor

    with torch.no_grad():
        device = latent_tensor.device
        is_video = latent_tensor.dim() == 5
        if is_video:
            batch, channels, frames, height, width = latent_tensor.shape
        else:
            batch, channels, height, width = latent_tensor.shape
            frames = 1

        output = latent_tensor.clone()

        for batch_index in range(batch):
            sample = latent_tensor[batch_index]
            sample_std = torch.std(sample.float())
            scale = sample_std.item() * strength if sample_std > 1e-6 else strength

            if channel_mode == "shared":
                if is_video:
                    if temporal_mode == "locked":
                        base_noise = generator((height, width), seed + batch_index)
                        base_noise = _normalize_noise_tensor(base_noise)
                        noise = base_noise.unsqueeze(0).unsqueeze(0).expand(channels, frames, height, width).clone()
                    else:
                        noise = torch.zeros((channels, frames, height, width), dtype=torch.float32, device=device)
                        for frame_index in range(frames):
                            frame_seed = seed + batch_index + frame_index * 131
                            frame_noise = generator((height, width), frame_seed)
                            frame_plane = _normalize_noise_tensor(frame_noise).unsqueeze(0).expand(channels, height, width)
                            noise[:, frame_index] = frame_plane.clone()
                else:
                    base_noise = generator((height, width), seed + batch_index)
                    base_noise = _normalize_noise_tensor(base_noise)
                    noise = base_noise.unsqueeze(0).expand(channels, height, width).clone()
            else:
                if is_video:
                    noise = torch.zeros((channels, frames, height, width), dtype=torch.float32, device=device)
                    for channel_index in range(channels):
                        channel_seed = seed + batch_index * 997 + channel_index * 1013
                        if temporal_mode == "locked":
                            channel_noise = generator((height, width), channel_seed)
                            normalized = _normalize_noise_tensor(channel_noise)
                            noise[channel_index] = normalized.unsqueeze(0).expand(frames, height, width).clone()
                        else:
                            for frame_index in range(frames):
                                frame_seed = channel_seed + frame_index * 131
                                frame_noise = generator((height, width), frame_seed)
                                noise[channel_index, frame_index] = _normalize_noise_tensor(frame_noise)
                else:
                    noise = torch.zeros((channels, height, width), dtype=torch.float32, device=device)
                    for channel_index in range(channels):
                        channel_seed = seed + batch_index * 997 + channel_index * 1013
                        channel_noise = generator((height, width), channel_seed)
                        noise[channel_index] = _normalize_noise_tensor(channel_noise)

            scaled_noise = (noise * scale).to(sample.dtype)
            output[batch_index] = sample + scaled_noise

    return output


def _apply_image_2d_noise(image_tensor, seed, strength, channel_mode, temporal_mode, generator):
    if strength == 0.0:
        return image_tensor

    with torch.no_grad():
        device = image_tensor.device
        is_video = image_tensor.dim() == 5

        if is_video:
            batch, frames, height, width, channels = image_tensor.shape
        else:
            batch, height, width, channels = image_tensor.shape
            frames = 1

        output = image_tensor.clone()

        for batch_index in range(batch):
            sample = image_tensor[batch_index]
            sample_std = torch.std(sample.float())
            scale = sample_std.item() * strength if sample_std > 1e-6 else strength

            if channel_mode == "shared":
                if is_video:
                    if temporal_mode == "locked":
                        base_noise = generator((height, width), seed + batch_index)
                        base_noise = _normalize_noise_tensor(base_noise)
                        noise = base_noise.unsqueeze(0).unsqueeze(-1).expand(frames, height, width, channels).clone()
                    else:
                        noise = torch.zeros((frames, height, width, channels), dtype=torch.float32, device=device)
                        for frame_index in range(frames):
                            frame_seed = seed + batch_index + frame_index * 131
                            frame_noise = generator((height, width), frame_seed)
                            frame_plane = _normalize_noise_tensor(frame_noise).unsqueeze(-1).expand(height, width, channels)
                            noise[frame_index] = frame_plane.clone()
                else:
                    base_noise = generator((height, width), seed + batch_index)
                    base_noise = _normalize_noise_tensor(base_noise)
                    noise = base_noise.unsqueeze(-1).expand(height, width, channels).clone()
            else:
                if is_video:
                    noise = torch.zeros((frames, height, width, channels), dtype=torch.float32, device=device)
                    for channel_index in range(channels):
                        channel_seed = seed + batch_index * 997 + channel_index * 1013
                        if temporal_mode == "locked":
                            channel_noise = generator((height, width), channel_seed)
                            normalized = _normalize_noise_tensor(channel_noise)
                            noise[:, :, :, channel_index] = normalized.unsqueeze(0).unsqueeze(-1).expand(frames, height, width)
                        else:
                            for frame_index in range(frames):
                                frame_seed = channel_seed + frame_index * 131
                                frame_noise = generator((height, width), frame_seed)
                                noise[frame_index, :, :, channel_index] = _normalize_noise_tensor(frame_noise)
                else:
                    noise = torch.zeros((height, width, channels), dtype=torch.float32, device=device)
                    for channel_index in range(channels):
                        channel_seed = seed + batch_index * 997 + channel_index * 1013
                        channel_noise = generator((height, width), channel_seed)
                        noise[:, :, channel_index] = _normalize_noise_tensor(channel_noise)

            scaled_noise = (noise * scale).to(sample.dtype)
            output[batch_index] = sample + scaled_noise

    return output


def _normalize_noise_tensor(noise):
    mean = torch.mean(noise)
    std = torch.std(noise)
    if std > 1e-6:
        return (noise - mean) / std
    return noise - mean


def _normalized_meshgrid(height, width, device, dtype):
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    return torch.meshgrid(ys, xs, indexing="ij")


def _swirl_grid(base_y, base_x, vortices):
    total_dx = torch.zeros_like(base_x)
    total_dy = torch.zeros_like(base_y)

    for center_x, center_y, strength, radius, direction in vortices:
        radius = max(radius, 1e-3)
        dx = base_x - center_x
        dy = base_y - center_y
        r = torch.sqrt(dx * dx + dy * dy)
        falloff = torch.exp(-torch.square(r / radius))
        angle = direction * strength * falloff
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        rotated_dx = dx * cos_a - dy * sin_a
        rotated_dy = dx * sin_a + dy * cos_a
        total_dx = total_dx + (rotated_dx - dx)
        total_dy = total_dy + (rotated_dy - dy)

    x_new = base_x + total_dx
    y_new = base_y + total_dy
    grid = torch.stack((x_new, y_new), dim=-1)
    return torch.clamp(grid, -1.0, 1.0)

_FLUX_BASE_CHANNELS = 32
_FLUX_PATCH_SIZE = (2, 2)


def _parse_patch_size(patch_size):
    if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
        raise ValueError("patch_size must be a tuple/list of length 2.")
    try:
        pi = int(patch_size[0])
        pj = int(patch_size[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("patch_size values must be integers.") from exc
    if pi <= 0 or pj <= 0:
        raise ValueError("patch_size values must be positive.")
    return pi, pj


def _flux_unpatchify(z, patch_size=(2, 2)):
    """(B, 128, H, W) -> (B, 32, H*2, W*2)"""
    if z.dim() != 4:
        raise ValueError("Flux unpatchify expects a 4D tensor shaped (B, C, H, W).")
    pi, pj = _parse_patch_size(patch_size)
    expected_channels = _FLUX_BASE_CHANNELS * pi * pj
    channels = z.shape[1]
    if channels != expected_channels:
        raise ValueError(
            f"Flux unpatchify expects {expected_channels} channels (got {channels})."
        )
    return rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=pi, pj=pj)


def _flux_patchify(z, patch_size=(2, 2)):
    """(B, 32, H, W) -> (B, 128, H//2, W//2)"""
    if z.dim() != 4:
        raise ValueError("Flux patchify expects a 4D tensor shaped (B, C, H, W).")
    pi, pj = _parse_patch_size(patch_size)
    channels = z.shape[1]
    if channels != _FLUX_BASE_CHANNELS:
        raise ValueError(
            f"Flux patchify expects {_FLUX_BASE_CHANNELS} channels (got {channels})."
        )
    height, width = z.shape[2], z.shape[3]
    if height % pi != 0 or width % pj != 0:
        raise ValueError(
            "Flux patchify expects height/width divisible by the patch size "
            f"(got height={height}, width={width}, patch={pi}x{pj})."
        )
    return rearrange(z, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=pi, pj=pj)


def _apply_flux_unpatchify(samples: torch.Tensor, patch_size=_FLUX_PATCH_SIZE) -> torch.Tensor:
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for latent samples, got {type(samples)}.")

    if samples.dim() == 4:
        logger.debug("Flux unpatchify input shape=%s", tuple(samples.shape))
        result = _flux_unpatchify(samples, patch_size=patch_size)
        logger.debug("Flux unpatchify output shape=%s", tuple(result.shape))
        return result

    if samples.dim() == 5:
        batch, channels, frames, height, width = samples.shape
        logger.debug("Flux unpatchify video input shape=%s", tuple(samples.shape))
        working = samples.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        result = _flux_unpatchify(working, patch_size=patch_size)
        _, new_channels, new_height, new_width = result.shape
        result = result.reshape(batch, frames, new_channels, new_height, new_width).permute(0, 2, 1, 3, 4)
        logger.debug("Flux unpatchify video output shape=%s", tuple(result.shape))
        return result

    raise ValueError("Flux unpatchify expects latent samples with 4 or 5 dimensions.")


def _apply_flux_patchify(samples: torch.Tensor, patch_size=_FLUX_PATCH_SIZE) -> torch.Tensor:
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for latent samples, got {type(samples)}.")

    if samples.dim() == 4:
        logger.debug("Flux patchify input shape=%s", tuple(samples.shape))
        result = _flux_patchify(samples, patch_size=patch_size)
        logger.debug("Flux patchify output shape=%s", tuple(result.shape))
        return result

    if samples.dim() == 5:
        batch, channels, frames, height, width = samples.shape
        logger.debug("Flux patchify video input shape=%s", tuple(samples.shape))
        working = samples.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        result = _flux_patchify(working, patch_size=patch_size)
        _, new_channels, new_height, new_width = result.shape
        result = result.reshape(batch, frames, new_channels, new_height, new_width).permute(0, 2, 1, 3, 4)
        logger.debug("Flux patchify video output shape=%s", tuple(result.shape))
        return result

    raise ValueError("Flux patchify expects latent samples with 4 or 5 dimensions.")


class UnpatchifyFlux2Latent:
    """Converts Flux.2 patchified latents to their unpatchified 2x spatial form."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Flux.2 latent to unpatchify from 2x2 patch format."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "unpatchify"
    CATEGORY = "Latent/Flux"

    def unpatchify(self, latent):
        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"LATENT samples must be a torch.Tensor, got {type(samples)}.")

        device = _get_device()
        working = samples.clone().to(device)
        result = _apply_flux_unpatchify(working, patch_size=_FLUX_PATCH_SIZE)

        out = latent.copy()
        out["samples"] = result.cpu()
        return (out,)


class PatchifyFlux2Latent:
    """Converts unpatchified Flux.2 latents back to the 2x2 patchified format."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Unpatchified Flux.2 latent to patchify back to 2x2 format."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "patchify"
    CATEGORY = "Latent/Flux"

    def patchify(self, latent):
        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"LATENT samples must be a torch.Tensor, got {type(samples)}.")

        device = _get_device()
        working = samples.clone().to(device)
        result = _apply_flux_patchify(working, patch_size=_FLUX_PATCH_SIZE)

        out = latent.copy()
        out["samples"] = result.cpu()
        return (out,)


class LatentGaussianBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent tensor to blur."}),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Standard deviation for the Gaussian kernel.",
                }),
                "blur_mode": (["Spatial Only", "Spatial and Channel"], {
                    "default": "Spatial Only",
                    "tooltip": "Choose whether to blur across spatial dimensions only or include channels.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the blur to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blur_latent"
    CATEGORY = "Latent/Filter"

    def blur_latent(self, latent, sigma, blur_mode, mask=None):
        if sigma == 0.0:
            return (latent,)

        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)
        is_5d = latent_tensor.dim() == 5

        if is_5d:
            b, c, t, h, w = latent_tensor.shape
            latent_4d = latent_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        else:
            latent_4d = latent_tensor

        if blur_mode == "Spatial Only":
            kernel_size = int(sigma * 6) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred_4d = TF.gaussian_blur(latent_4d, kernel_size=kernel_size, sigma=sigma)

        else:
            channels = latent_4d.shape[1]
            latent_5d_for_conv = latent_4d.unsqueeze(1)

            k_size_spatial = int(sigma * 6) + 1
            if k_size_spatial % 2 == 0:
                k_size_spatial += 1
            k_size_channel = min(channels, k_size_spatial)
            if k_size_channel % 2 == 0:
                k_size_channel -= 1
            if k_size_channel < 1:
                k_size_channel = 1

            ax_c = torch.linspace(-(k_size_channel - 1) / 2.0, (k_size_channel - 1) / 2.0, k_size_channel, device=device)
            ax_s = torch.linspace(-(k_size_spatial - 1) / 2.0, (k_size_spatial - 1) / 2.0, k_size_spatial, device=device)
            gauss_c = torch.exp(-0.5 * torch.square(ax_c / sigma))
            gauss_s = torch.exp(-0.5 * torch.square(ax_s / sigma))

            gauss_c = gauss_c.view(k_size_channel, 1, 1)
            gauss_h = gauss_s.view(1, k_size_spatial, 1)
            gauss_w = gauss_s.view(1, 1, k_size_spatial)
            kernel_3d = gauss_c * gauss_h * gauss_w

            kernel_3d /= torch.sum(kernel_3d)
            kernel_3d = kernel_3d.view(1, 1, k_size_channel, k_size_spatial, k_size_spatial)

            padding = (k_size_channel // 2, k_size_spatial // 2, k_size_spatial // 2)
            blurred_5d = F.conv3d(latent_5d_for_conv, kernel_3d, padding=padding)
            blurred_4d = blurred_5d.squeeze(1)

        if is_5d:
            blurred_tensor = blurred_4d.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        else:
            blurred_tensor = blurred_4d

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            blurred_tensor = blend_with_mask(latent_tensor, blurred_tensor, mask_nchw)

        out = latent.copy()
        out["samples"] = blurred_tensor.cpu()
        return (out,)


class LatentFrequencySplit:
    """Splits a latent into low- and high-frequency bands via Gaussian smoothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to decompose into low/high frequency bands."}),
                "sigma": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Radius of the Gaussian used for the low-pass. Higher values move detail into the high band.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the split to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("low_pass", "high_pass")
    FUNCTION = "split"
    CATEGORY = "Latent/Filter"

    def split(self, latent, sigma, mask=None):
        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)

        if sigma <= 0.0:
            low_samples = latent_tensor
            high_samples = torch.zeros_like(latent_tensor)
        else:
            low_samples = self._gaussian_blur(latent_tensor, sigma)
            high_samples = latent_tensor - low_samples

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            low_samples = blend_with_mask(latent_tensor, low_samples, mask_nchw)
            high_samples = blend_with_mask(torch.zeros_like(high_samples), high_samples, mask_nchw)

        low_latent = latent.copy()
        low_latent["samples"] = low_samples.cpu()

        high_latent = latent.copy()
        high_latent["samples"] = high_samples.cpu()

        return (low_latent, high_latent)

    @staticmethod
    def _gaussian_blur(latent_tensor, sigma):
        tensor = latent_tensor
        is_5d = tensor.dim() == 5

        if is_5d:
            b, c, t, h, w = tensor.shape
            tensor_4d = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        else:
            tensor_4d = tensor

        kernel_size = int(sigma * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred_4d = TF.gaussian_blur(tensor_4d, kernel_size=kernel_size, sigma=sigma)

        if is_5d:
            blurred = blurred_4d.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        else:
            blurred = blurred_4d

        return blurred


class LatentFrequencyMerge:
    """Recombines low/high latent bands back into a single latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_pass": ("LATENT", {"tooltip": "Low-frequency latent band produced by Latent Frequency Split."}),
                "high_pass": ("LATENT", {"tooltip": "High-frequency latent band to merge back in."}),
                "low_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Multiplier for the low-pass band before merging.",
                }),
                "high_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Multiplier for the high-pass band before merging.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the merge adjustments to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "merge"
    CATEGORY = "Latent/Filter"

    def merge(self, low_pass, high_pass, low_gain, high_gain, mask=None):
        device = _get_device()
        low_tensor = low_pass["samples"].clone().to(device)
        high_tensor = high_pass["samples"].clone().to(device)

        if low_tensor.shape != high_tensor.shape:
            raise ValueError(
                "Low-pass and high-pass latent shapes must match, got "
                f"{tuple(low_tensor.shape)} vs {tuple(high_tensor.shape)}"
            )

        if low_tensor.dtype != high_tensor.dtype:
            high_tensor = high_tensor.to(dtype=low_tensor.dtype)

        base = low_tensor + high_tensor
        merged = low_tensor * float(low_gain) + high_tensor * float(high_gain)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(base.shape[0]),
                height=int(base.shape[-2]),
                width=int(base.shape[-1]),
                device=device,
            )
            merged = blend_with_mask(base, merged, mask_nchw)

        out = low_pass.copy()
        out["samples"] = merged.cpu()
        return (out,)


class LatentAddNoise:
    """
    Adds a configurable amount of seeded random noise to a latent tensor.
    The strength is relative to the standard deviation of the input latent.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to receive additional Gaussian noise."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for generating repeatable noise.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Strength of the noise. 1.0 adds noise with the same standard deviation as the latent.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise addition to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_noise"
    CATEGORY = "Latent/Noise"

    def add_noise(self, latent, seed, strength, mask=None):
        if strength == 0.0:
            return (latent,)

        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)
        if not latent_tensor.is_floating_point():
            raise RuntimeError(f"LATENT samples must be floating point, got dtype={latent_tensor.dtype}.")

        noise = _seeded_gaussian_noise_like(latent_tensor, seed)
        latent_std = _per_sample_std(latent_tensor)
        strength_value = float(strength)
        scale = latent_std * strength_value
        scale_view = scale.view(int(latent_tensor.shape[0]), *([1] * (latent_tensor.ndim - 1)))
        scaled_noise = noise * scale_view
        logger.debug(
            "LatentAddNoise: batch=%d seed=%d strength=%.4f std_min=%.6f std_max=%.6f",
            int(latent_tensor.shape[0]),
            int(seed),
            strength_value,
            float(latent_std.min().item()),
            float(latent_std.max().item()),
        )

        noised_latent = latent_tensor + scaled_noise

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            noised_latent = blend_with_mask(latent_tensor, noised_latent, mask_nchw)

        out = latent.copy()
        out["samples"] = noised_latent.cpu()

        return (out,)


class ImageAddNoise:
    """Adds seeded Gaussian noise to image tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to receive additional Gaussian noise."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for generating repeatable noise.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Strength of the noise. 1.0 adds noise with the same standard deviation as the image.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise addition to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_noise"
    CATEGORY = "Image/Noise"

    def add_noise(self, image, seed, strength, mask=None):
        if strength == 0.0:
            return (image,)

        device = _get_device()
        image_tensor = image.clone().to(device)
        if not image_tensor.is_floating_point():
            raise RuntimeError(f"IMAGE input must be floating point, got dtype={image_tensor.dtype}.")

        noise = _seeded_gaussian_noise_like(image_tensor, seed)
        image_std = _per_sample_std(image_tensor)
        strength_value = float(strength)
        scale = image_std * strength_value
        scale_view = scale.view(int(image_tensor.shape[0]), *([1] * (image_tensor.ndim - 1)))
        scaled_noise = noise * scale_view
        logger.debug(
            "ImageAddNoise: batch=%d seed=%d strength=%.4f std_min=%.6f std_max=%.6f",
            int(image_tensor.shape[0]),
            int(seed),
            strength_value,
            float(image_std.min().item()),
            float(image_std.max().item()),
        )

        noised_image = image_tensor + scaled_noise

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            noised_image = blend_image_with_mask(image_tensor, noised_image, mask)

        return (noised_image.cpu(),)


class LatentPerlinFractalNoise:
    """Adds smooth fractal Perlin noise to latent tensors for structured variation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent that will be perturbed with fractal Perlin noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed controlling the procedural noise pattern."}),
                "frequency": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.01,
                    "max": 64.0,
                    "step": 0.01,
                    "tooltip": "Base lattice frequency. Higher values produce finer details.",
                }),
                "octaves": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 12,
                    "tooltip": "Number of noise layers to accumulate.",
                }),
                "persistence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Amplitude multiplier between octaves.",
                }),
                "lacunarity": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 6.0,
                    "step": 0.1,
                    "tooltip": "Frequency multiplier between octaves.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Scales the normalized noise relative to the latent's standard deviation.",
                }),
                "channel_mode": (["shared", "per_channel"], {
                    "default": "shared",
                    "tooltip": "Shared noise across channels or unique noise per channel.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_perlin_noise"
    CATEGORY = "Latent/Noise"

    def add_perlin_noise(self, latent, seed, frequency, octaves, persistence, lacunarity, strength, channel_mode, mask=None):
        if strength == 0.0:
            return (latent,)

        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)
        is_video = latent_tensor.dim() == 5

        if is_video:
            dims = (latent_tensor.shape[2], latent_tensor.shape[3], latent_tensor.shape[4])
        else:
            dims = (latent_tensor.shape[-2], latent_tensor.shape[-1])

        batch = latent_tensor.shape[0]
        channels = latent_tensor.shape[1]

        output = latent_tensor.clone()

        for batch_index in range(batch):
            sample = latent_tensor[batch_index]
            sample_std = torch.std(sample.float())

            if channel_mode == "shared":
                noise = _fractal_perlin(dims, seed + batch_index, frequency, octaves, persistence, lacunarity, device)
                noise = _normalize_noise_tensor(noise)

                if is_video:
                    noise = noise.unsqueeze(0).expand(channels, dims[0], dims[1], dims[2])
                else:
                    noise = noise.unsqueeze(0).expand(channels, dims[-2], dims[-1])
            else:
                if is_video:
                    noise = torch.zeros((channels, dims[0], dims[1], dims[2]), dtype=torch.float32, device=device)
                else:
                    noise = torch.zeros((channels, dims[-2], dims[-1]), dtype=torch.float32, device=device)

                for channel_index in range(channels):
                    channel_seed = seed + batch_index * 997 + channel_index * 1013
                    channel_noise = _fractal_perlin(dims, channel_seed, frequency, octaves, persistence, lacunarity, device)
                    noise[channel_index] = _normalize_noise_tensor(channel_noise)

            scale = sample_std * strength
            scaled_noise = (noise * scale).to(sample.dtype)
            output[batch_index] = sample + scaled_noise

        out = latent.copy()
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            output = blend_with_mask(latent_tensor, output, mask_nchw)
        out["samples"] = output.cpu()

        return (out,)


class ImagePerlinFractalNoise:
    """Adds smooth fractal Perlin noise to images for structured variation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image that will be perturbed with fractal Perlin noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed controlling the procedural noise pattern."}),
                "frequency": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.01,
                    "max": 64.0,
                    "step": 0.01,
                    "tooltip": "Base lattice frequency. Higher values produce finer details.",
                }),
                "octaves": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 12,
                    "tooltip": "Number of noise layers to accumulate.",
                }),
                "persistence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Amplitude multiplier between octaves.",
                }),
                "lacunarity": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 6.0,
                    "step": 0.1,
                    "tooltip": "Frequency multiplier between octaves.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Scale of the normalized noise relative to the image's standard deviation.",
                }),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Reuse a single noise field for all channels or reseed per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reruns the same pattern per frame; animated reseeds each frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_perlin_noise"
    CATEGORY = "Image/Noise"

    def add_perlin_noise(self, image, seed, frequency, octaves, persistence, lacunarity, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        image_tensor = image.clone().to(device)

        generator = lambda size, noise_seed: _fractal_perlin(size, noise_seed, frequency, octaves, persistence, lacunarity, device)
        output = _apply_image_2d_noise(image_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            output = blend_image_with_mask(image_tensor, output, mask)

        return (output.cpu(),)


class LatentSimplexNoise:
    """Applies layered simplex noise for organic latent perturbations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to perturb with simplex noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed controlling the simplex lattice offsets."}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "tooltip": "Base lattice frequency for the simplex grid."}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 12, "tooltip": "Number of simplex layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier applied between octaves."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Frequency multiplier applied between octaves."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized simplex noise relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Reuse a single noise field for all channels or reseed per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reruns the same pattern per frame; animated reseeds each frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_simplex_noise"
    CATEGORY = "Latent/Noise"

    def add_simplex_noise(self, latent, seed, frequency, octaves, persistence, lacunarity, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)

        generator = lambda size, noise_seed: _fractal_simplex(size, noise_seed, frequency, octaves, persistence, lacunarity, device)
        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            output = blend_with_mask(latent_tensor, output, mask_nchw)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class ImageSimplexNoise:
    """Applies layered simplex noise for organic image perturbations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to perturb with simplex noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed controlling the simplex lattice offsets."}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "tooltip": "Base lattice frequency for the simplex grid."}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 12, "tooltip": "Number of simplex layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier applied between octaves."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Frequency multiplier applied between octaves."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized simplex noise relative to the image's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Reuse a single noise field for all channels or reseed per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reruns the same pattern per frame; animated reseeds each frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_simplex_noise"
    CATEGORY = "Image/Noise"

    def add_simplex_noise(self, image, seed, frequency, octaves, persistence, lacunarity, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        image_tensor = image.clone().to(device)

        generator = lambda size, noise_seed: _fractal_simplex(size, noise_seed, frequency, octaves, persistence, lacunarity, device)
        output = _apply_image_2d_noise(image_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            output = blend_image_with_mask(image_tensor, output, mask)

        return (output.cpu(),)


class LatentWorleyNoise:
    """Generates cellular Worley noise for cracked or biological textures."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to perturb with Worley (cellular) noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for the feature point distribution."}),
                "feature_points": ("INT", {"default": 16, "min": 1, "max": 4096, "tooltip": "Base number of feature points scattered across the plane."}),
                "octaves": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Number of cellular layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between Worley octaves."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Multiplier for the feature point count between octaves."}),
                "distance_metric": (["euclidean", "manhattan", "chebyshev"], {"default": "euclidean", "tooltip": "Distance metric used when measuring feature proximity."}),
                "jitter": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How far feature points can drift inside each cell."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized Worley noise relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Shared noise for all channels or reseeded per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses a single noise pattern per frame; animated reseeds each frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_worley_noise"
    CATEGORY = "Latent/Noise"

    def add_worley_noise(self, latent, seed, feature_points, octaves, persistence, lacunarity, distance_metric, jitter, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)

        base_points = float(max(1, feature_points))
        progress_bar = None
        progress_state = None
        progress_total = None

        if strength != 0.0:
            if latent_tensor.dim() == 5:
                batch, channels, frames, height, width = latent_tensor.shape
            else:
                batch, channels, height, width = latent_tensor.shape
                frames = 1
            steps_per_call = _worley_progress_steps((height, width), base_points, octaves, lacunarity)
            total_calls = _count_noise_calls(batch, channels, frames, channel_mode, temporal_mode)
            progress_total = max(1, steps_per_call * max(1, total_calls))
            progress_bar = _progress_bar(progress_total)
            progress_state = {"current": 0}

        def generator(size, noise_seed):
            return _fractal_worley(
                size,
                noise_seed,
                base_points,
                octaves,
                persistence,
                lacunarity,
                distance_metric,
                jitter,
                device,
                progress=progress_bar,
                progress_state=progress_state,
                progress_total=progress_total,
            )

        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            output = blend_with_mask(latent_tensor, output, mask_nchw)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class ImageWorleyNoise:
    """Generates cellular Worley noise for cracked or biological image textures."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to perturb with Worley (cellular) noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for the feature point distribution."}),
                "feature_points": ("INT", {"default": 16, "min": 1, "max": 4096, "tooltip": "Base number of feature points scattered across the plane."}),
                "octaves": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Number of cellular layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between Worley octaves."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Multiplier for the feature point count between octaves."}),
                "distance_metric": (["euclidean", "manhattan", "chebyshev"], {"default": "euclidean", "tooltip": "Distance metric used when measuring feature proximity."}),
                "jitter": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How far feature points can drift inside each cell."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized Worley noise relative to the image's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Shared noise for all channels or reseeded per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses a single noise pattern per frame; animated reseeds each frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_worley_noise"
    CATEGORY = "Image/Noise"

    def add_worley_noise(self, image, seed, feature_points, octaves, persistence, lacunarity, distance_metric, jitter, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        image_tensor = image.clone().to(device)

        base_points = float(max(1, feature_points))
        progress_bar = None
        progress_state = None
        progress_total = None

        if strength != 0.0:
            if image_tensor.dim() == 5:
                batch, frames, height, width, channels = image_tensor.shape
            else:
                batch, height, width, channels = image_tensor.shape
                frames = 1
            steps_per_call = _worley_progress_steps((height, width), base_points, octaves, lacunarity)
            total_calls = _count_noise_calls(batch, channels, frames, channel_mode, temporal_mode)
            progress_total = max(1, steps_per_call * max(1, total_calls))
            progress_bar = _progress_bar(progress_total)
            progress_state = {"current": 0}

        def generator(size, noise_seed):
            return _fractal_worley(
                size,
                noise_seed,
                base_points,
                octaves,
                persistence,
                lacunarity,
                distance_metric,
                jitter,
                device,
                progress=progress_bar,
                progress_state=progress_state,
                progress_total=progress_total,
            )

        output = _apply_image_2d_noise(image_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            output = blend_image_with_mask(image_tensor, output, mask)

        return (output.cpu(),)


class LatentReactionDiffusion:
    """Runs a Gray-Scott reaction-diffusion simulation and injects the resulting pattern."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent that will receive reaction-diffusion patterns."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for the initial chemical concentrations."}),
                "iterations": ("INT", {"default": 200, "min": 1, "max": 2000, "tooltip": "Number of Gray-Scott simulation steps."}),
                "feed_rate": ("FLOAT", {"default": 0.036, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Feed rate (F) controlling how quickly chemical U is replenished."}),
                "kill_rate": ("FLOAT", {"default": 0.065, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Kill rate (K) regulating removal of chemical V."}),
                "diffusion_u": ("FLOAT", {"default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Diffusion rate for chemical U."}),
                "diffusion_v": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Diffusion rate for chemical V."}),
                "time_step": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01, "tooltip": "Simulation time step used during integration."}),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized pattern relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Reuse one simulation for all channels or rerun per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses the same pattern for every frame; animated reruns the simulation per frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the pattern injection to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_reaction_diffusion"
    CATEGORY = "Latent/Noise"

    def add_reaction_diffusion(self, latent, seed, iterations, feed_rate, kill_rate, diffusion_u, diffusion_v, time_step, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)

        generator = lambda size, noise_seed: _gray_scott_pattern(size, noise_seed, iterations, feed_rate, kill_rate, diffusion_u, diffusion_v, time_step, device)
        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            output = blend_with_mask(latent_tensor, output, mask_nchw)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class ImageReactionDiffusion:
    """Runs a Gray-Scott reaction-diffusion simulation and injects the resulting pattern into an image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image that will receive reaction-diffusion patterns."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for the initial chemical concentrations."}),
                "iterations": ("INT", {"default": 200, "min": 1, "max": 2000, "tooltip": "Number of Gray-Scott simulation steps."}),
                "feed_rate": ("FLOAT", {"default": 0.036, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Feed rate (F) controlling how quickly chemical U is replenished."}),
                "kill_rate": ("FLOAT", {"default": 0.065, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Kill rate (K) regulating removal of chemical V."}),
                "diffusion_u": ("FLOAT", {"default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Diffusion rate for chemical U."}),
                "diffusion_v": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Diffusion rate for chemical V."}),
                "time_step": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01, "tooltip": "Simulation time step used during integration."}),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized pattern relative to the image's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Reuse one simulation for all channels or rerun per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses the same pattern for every frame; animated reruns the simulation per frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the pattern injection to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_reaction_diffusion"
    CATEGORY = "Image/Noise"

    def add_reaction_diffusion(self, image, seed, iterations, feed_rate, kill_rate, diffusion_u, diffusion_v, time_step, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        image_tensor = image.clone().to(device)

        generator = lambda size, noise_seed: _gray_scott_pattern(size, noise_seed, iterations, feed_rate, kill_rate, diffusion_u, diffusion_v, time_step, device)
        output = _apply_image_2d_noise(image_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            output = blend_image_with_mask(image_tensor, output, mask)

        return (output.cpu(),)


class LatentFractalBrownianMotion:
    """Builds fractal Brownian motion from a selectable base noise and injects it into the latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to enrich with fractal Brownian motion."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for the base noise generator."}),
                "base_noise": (["simplex", "perlin", "worley"], {"default": "simplex", "tooltip": "Noise primitive accumulated by the fBm stack."}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "tooltip": "Fundamental frequency for simplex/perlin bases (acts as a multiplier for Worley)."}),
                "feature_points": ("INT", {"default": 16, "min": 1, "max": 4096, "tooltip": "Base feature point count (used when base noise is Worley)."}),
                "octaves": ("INT", {"default": 5, "min": 1, "max": 12, "tooltip": "Number of fBm layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between fBm layers."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Frequency multiplier between fBm layers."}),
                "distance_metric": (["euclidean", "manhattan", "chebyshev"], {"default": "euclidean", "tooltip": "Distance metric used when the base noise is Worley."}),
                "jitter": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Feature jitter amount for Worley base noise."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized fBm field relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Shared fBm field per sample or reseeded per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses the same fBm per frame; animated reseeds each frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_fbm_noise"
    CATEGORY = "Latent/Noise"

    def add_fbm_noise(self, latent, seed, base_noise, frequency, feature_points, octaves, persistence, lacunarity, distance_metric, jitter, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)

        base_points = float(max(1, feature_points)) * max(frequency, 0.01)
        progress_bar = None
        progress_state = None
        progress_total = None

        if base_noise == "worley" and strength != 0.0:
            if latent_tensor.dim() == 5:
                batch, channels, frames, height, width = latent_tensor.shape
            else:
                batch, channels, height, width = latent_tensor.shape
                frames = 1
            steps_per_call = _worley_progress_steps((height, width), base_points, octaves, lacunarity)
            total_calls = _count_noise_calls(batch, channels, frames, channel_mode, temporal_mode)
            progress_total = max(1, steps_per_call * max(1, total_calls))
            progress_bar = _progress_bar(progress_total)
            progress_state = {"current": 0}

        def generator(size, noise_seed):
            if base_noise == "perlin":
                return _fractal_perlin(size, noise_seed, frequency, octaves, persistence, lacunarity, device)
            if base_noise == "worley":
                return _fractal_worley(
                    size,
                    noise_seed,
                    base_points,
                    octaves,
                    persistence,
                    lacunarity,
                    distance_metric,
                    jitter,
                    device,
                    progress=progress_bar,
                    progress_state=progress_state,
                    progress_total=progress_total,
                )
            return _fractal_simplex(size, noise_seed, frequency, octaves, persistence, lacunarity, device)

        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            output = blend_with_mask(latent_tensor, output, mask_nchw)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class ImageFractalBrownianMotion:
    """Builds fractal Brownian motion from a selectable base noise and injects it into the image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to enrich with fractal Brownian motion."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for the base noise generator."}),
                "base_noise": (["simplex", "perlin", "worley"], {"default": "simplex", "tooltip": "Noise primitive accumulated by the fBm stack."}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "tooltip": "Fundamental frequency for simplex/perlin bases (acts as a multiplier for Worley)."}),
                "feature_points": ("INT", {"default": 16, "min": 1, "max": 4096, "tooltip": "Base feature point count (used when base noise is Worley)."}),
                "octaves": ("INT", {"default": 5, "min": 1, "max": 12, "tooltip": "Number of fBm layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between fBm layers."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Frequency multiplier between fBm layers."}),
                "distance_metric": (["euclidean", "manhattan", "chebyshev"], {"default": "euclidean", "tooltip": "Distance metric used when the base noise is Worley."}),
                "jitter": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Feature jitter amount for Worley base noise."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized fBm field relative to the image's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Shared fBm field per sample or reseeded per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses the same fBm per frame; animated reseeds each frame."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noise injection to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_fbm_noise"
    CATEGORY = "Image/Noise"

    def add_fbm_noise(self, image, seed, base_noise, frequency, feature_points, octaves, persistence, lacunarity, distance_metric, jitter, strength, channel_mode, temporal_mode, mask=None):
        device = _get_device()
        image_tensor = image.clone().to(device)

        base_points = float(max(1, feature_points)) * max(frequency, 0.01)
        progress_bar = None
        progress_state = None
        progress_total = None

        if base_noise == "worley" and strength != 0.0:
            if image_tensor.dim() == 5:
                batch, frames, height, width, channels = image_tensor.shape
            else:
                batch, height, width, channels = image_tensor.shape
                frames = 1
            steps_per_call = _worley_progress_steps((height, width), base_points, octaves, lacunarity)
            total_calls = _count_noise_calls(batch, channels, frames, channel_mode, temporal_mode)
            progress_total = max(1, steps_per_call * max(1, total_calls))
            progress_bar = _progress_bar(progress_total)
            progress_state = {"current": 0}

        def generator(size, noise_seed):
            if base_noise == "perlin":
                return _fractal_perlin(size, noise_seed, frequency, octaves, persistence, lacunarity, device)
            if base_noise == "worley":
                return _fractal_worley(
                    size,
                    noise_seed,
                    base_points,
                    octaves,
                    persistence,
                    lacunarity,
                    distance_metric,
                    jitter,
                    device,
                    progress=progress_bar,
                    progress_state=progress_state,
                    progress_total=progress_total,
                )
            return _fractal_simplex(size, noise_seed, frequency, octaves, persistence, lacunarity, device)

        use_parallel = (
            strength != 0.0
            and channel_mode == "shared"
            and base_noise in ("simplex", "perlin")
            and image_tensor.dim() == 4
            and int(image_tensor.shape[0]) > 1
        )

        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor

            batch, height, width, channels = image_tensor.shape
            cpu_count = os.cpu_count() or 1
            max_workers = min(int(batch), int(cpu_count))

            def make_noise(batch_index: int) -> torch.Tensor:
                noise_seed = int(seed) + int(batch_index)
                base_field = generator((height, width), noise_seed)
                return _normalize_noise_tensor(base_field)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                base_fields = list(executor.map(make_noise, range(int(batch))))

            base_fields_tensor = torch.stack(base_fields, dim=0)
            noise = base_fields_tensor.unsqueeze(-1).expand(batch, height, width, channels)

            sample_std = image_tensor.float().reshape(int(batch), -1).std(dim=1)
            strength_value = float(strength)
            scale = torch.where(
                sample_std > 1e-6,
                sample_std * strength_value,
                torch.full_like(sample_std, strength_value),
            )

            scaled_noise = (noise * scale.view(int(batch), 1, 1, 1)).to(dtype=image_tensor.dtype)
            output = image_tensor + scaled_noise
        else:
            output = _apply_image_2d_noise(image_tensor, seed, strength, channel_mode, temporal_mode, generator)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            output = blend_image_with_mask(image_tensor, output, mask)

        return (output.cpu(),)


class LatentSwirlNoise:
    """Swirls latent pixels around randomized centers for vortex-like perturbations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to deform with vortex-style warps."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for vortex placement and direction randomness."}),
                "vortices": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of independent vortex centres to spawn per latent.",
                }),
                "channel_mode": (["global", "per_channel"], {
                    "default": "global",
                    "tooltip": "Use a shared swirl grid for all channels or generate unique grids per affected channel.",
                }),
                "channel_fraction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fraction of channels to swirl. The subset is randomly selected per sample.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 6.28,
                    "step": 0.01,
                    "tooltip": "Peak swirl rotation in radians near the vortex center.",
                }),
                "radius": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Normalized radius controlling how far the vortex influence extends.",
                }),
                "center_spread": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How far the vortex origin drifts from the latent center.",
                }),
                "direction_bias": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Bias toward counter-clockwise (1) or clockwise (-1) swirl.",
                }),
                "mix": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend between original latent (0) and fully swirled result (1).",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the swirl to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_swirl_noise"
    CATEGORY = "Latent/Noise"

    def add_swirl_noise(self, latent, seed, vortices, channel_mode, channel_fraction, strength, radius, center_spread, direction_bias, mix, mask=None):
        if strength == 0.0 or mix == 0.0:
            return (latent,)

        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)
        is_video = latent_tensor.dim() == 5

        if is_video:
            batch, channels, frames, height, width = latent_tensor.shape
            working = latent_tensor.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        else:
            batch, channels, height, width = latent_tensor.shape
            frames = 1
            working = latent_tensor

        base_dtype = torch.float32
        base_y, base_x = _normalized_meshgrid(height, width, device, base_dtype)
        result = torch.empty_like(working)

        center_spread = max(0.0, min(center_spread, 1.0))
        mix = max(0.0, min(mix, 1.0))
        bias_threshold = max(0.0, min(1.0, (direction_bias + 1.0) * 0.5))
        vortex_count = max(1, int(vortices))
        channel_fraction = max(0.0, min(1.0, channel_fraction))

        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)

        total = working.shape[0]

        for idx in range(total):
            if channel_fraction <= 0.0:
                result[idx] = working[idx]
                continue

            channel_count = working.shape[1]
            selected_count = min(channel_count, max(1, math.ceil(channel_count * channel_fraction)))

            channel_indices = torch.randperm(channel_count, generator=cpu_generator)[:selected_count]

            sample = working[idx : idx + 1]
            sample_result = sample.clone()

            if channel_mode == "global":
                rand_vals = torch.rand((vortex_count, 3), generator=cpu_generator)
                vortex_params = []

                for vortex_idx in range(vortex_count):
                    offsets = (rand_vals[vortex_idx, :2] * 2.0 - 1.0) * center_spread
                    center_y = offsets[0].item()
                    center_x = offsets[1].item()
                    direction = 1.0 if rand_vals[vortex_idx, 2].item() < bias_threshold else -1.0
                    vortex_params.append((center_x, center_y, float(strength), float(radius), direction))

                grid = _swirl_grid(base_y, base_x, vortex_params).unsqueeze(0)
                warped = F.grid_sample(
                    sample.to(base_dtype),
                    grid,
                    mode="bilinear",
                    padding_mode="reflection",
                    align_corners=True,
                ).to(sample.dtype)

                if mix >= 1.0:
                    sample_result[:, channel_indices] = warped[:, channel_indices]
                else:
                    delta = warped[:, channel_indices] - sample[:, channel_indices]
                    sample_result[:, channel_indices] = sample[:, channel_indices] + delta * mix

            else:  # per_channel
                for channel_idx in channel_indices.tolist():
                    rand_vals = torch.rand((vortex_count, 3), generator=cpu_generator)
                    vortex_params = []

                    for vortex_idx in range(vortex_count):
                        offsets = (rand_vals[vortex_idx, :2] * 2.0 - 1.0) * center_spread
                        center_y = offsets[0].item()
                        center_x = offsets[1].item()
                        direction = 1.0 if rand_vals[vortex_idx, 2].item() < bias_threshold else -1.0
                        vortex_params.append((center_x, center_y, float(strength), float(radius), direction))

                    grid = _swirl_grid(base_y, base_x, vortex_params).unsqueeze(0)

                    channel_slice = sample[:, channel_idx : channel_idx + 1].to(base_dtype)
                    warped_channel = F.grid_sample(
                        channel_slice,
                        grid,
                        mode="bilinear",
                        padding_mode="reflection",
                        align_corners=True,
                    ).to(sample.dtype)

                    if mix >= 1.0:
                        sample_result[:, channel_idx : channel_idx + 1] = warped_channel
                    else:
                        delta = warped_channel - sample[:, channel_idx : channel_idx + 1]
                        sample_result[:, channel_idx : channel_idx + 1] = sample[:, channel_idx : channel_idx + 1] + delta * mix

            result[idx] = sample_result[0]

        if is_video:
            result = result.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            result = blend_with_mask(latent_tensor, result, mask_nchw)

        out = latent.copy()
        out["samples"] = result.cpu()

        return (out,)


class ImageSwirlNoise:
    """Swirls image pixels around randomized centers for vortex-like perturbations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to deform with vortex-style warps."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for vortex placement and direction randomness."}),
                "vortices": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of independent vortex centres to spawn per image.",
                }),
                "channel_mode": (["global", "per_channel"], {
                    "default": "global",
                    "tooltip": "Use a shared swirl grid for all channels or generate unique grids per affected channel.",
                }),
                "channel_fraction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fraction of channels to swirl. The subset is randomly selected per sample.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 6.28,
                    "step": 0.01,
                    "tooltip": "Peak swirl rotation in radians near the vortex center.",
                }),
                "radius": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Normalized radius controlling how far the vortex influence extends.",
                }),
                "center_spread": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How far the vortex origin drifts from the image center.",
                }),
                "direction_bias": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Bias toward counter-clockwise (1) or clockwise (-1) swirl.",
                }),
                "mix": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend between original image (0) and fully swirled result (1).",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the swirl to masked areas. "
                                          "The mask is resized to the image resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_swirl_noise"
    CATEGORY = "Image/Noise"

    def add_swirl_noise(self, image, seed, vortices, channel_mode, channel_fraction, strength, radius, center_spread, direction_bias, mix, mask=None):
        if strength == 0.0 or mix == 0.0:
            return (image,)

        device = _get_device()
        image_tensor = image.clone().to(device)
        is_video = image_tensor.dim() == 5

        if is_video:
            batch, frames, height, width, channels = image_tensor.shape
            working = image_tensor.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
        else:
            batch, height, width, channels = image_tensor.shape
            frames = 1
            working = image_tensor.permute(0, 3, 1, 2)

        base_dtype = torch.float32
        base_y, base_x = _normalized_meshgrid(height, width, device, base_dtype)
        result = torch.empty_like(working)

        center_spread = max(0.0, min(center_spread, 1.0))
        mix = max(0.0, min(mix, 1.0))
        bias_threshold = max(0.0, min(1.0, (direction_bias + 1.0) * 0.5))
        vortex_count = max(1, int(vortices))
        channel_fraction = max(0.0, min(1.0, channel_fraction))

        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)

        total = working.shape[0]

        for idx in range(total):
            if channel_fraction <= 0.0:
                result[idx] = working[idx]
                continue

            channel_count = working.shape[1]
            selected_count = min(channel_count, max(1, math.ceil(channel_count * channel_fraction)))

            channel_indices = torch.randperm(channel_count, generator=cpu_generator)[:selected_count]

            sample = working[idx : idx + 1]
            sample_result = sample.clone()

            if channel_mode == "global":
                rand_vals = torch.rand((vortex_count, 3), generator=cpu_generator)
                vortex_params = []

                for vortex_idx in range(vortex_count):
                    offsets = (rand_vals[vortex_idx, :2] * 2.0 - 1.0) * center_spread
                    center_y = offsets[0].item()
                    center_x = offsets[1].item()
                    direction = 1.0 if rand_vals[vortex_idx, 2].item() < bias_threshold else -1.0
                    vortex_params.append((center_x, center_y, float(strength), float(radius), direction))

                grid = _swirl_grid(base_y, base_x, vortex_params).unsqueeze(0)
                warped = F.grid_sample(
                    sample.to(base_dtype),
                    grid,
                    mode="bilinear",
                    padding_mode="reflection",
                    align_corners=True,
                ).to(sample.dtype)

                if mix >= 1.0:
                    sample_result[:, channel_indices] = warped[:, channel_indices]
                else:
                    delta = warped[:, channel_indices] - sample[:, channel_indices]
                    sample_result[:, channel_indices] = sample[:, channel_indices] + delta * mix

            else:  # per_channel
                for channel_idx in channel_indices.tolist():
                    rand_vals = torch.rand((vortex_count, 3), generator=cpu_generator)
                    vortex_params = []

                    for vortex_idx in range(vortex_count):
                        offsets = (rand_vals[vortex_idx, :2] * 2.0 - 1.0) * center_spread
                        center_y = offsets[0].item()
                        center_x = offsets[1].item()
                        direction = 1.0 if rand_vals[vortex_idx, 2].item() < bias_threshold else -1.0
                        vortex_params.append((center_x, center_y, float(strength), float(radius), direction))

                    grid = _swirl_grid(base_y, base_x, vortex_params).unsqueeze(0)

                    channel_slice = sample[:, channel_idx : channel_idx + 1].to(base_dtype)
                    warped_channel = F.grid_sample(
                        channel_slice,
                        grid,
                        mode="bilinear",
                        padding_mode="reflection",
                        align_corners=True,
                    ).to(sample.dtype)

                    if mix >= 1.0:
                        sample_result[:, channel_idx : channel_idx + 1] = warped_channel
                    else:
                        delta = warped_channel - sample[:, channel_idx : channel_idx + 1]
                        sample_result[:, channel_idx : channel_idx + 1] = sample[:, channel_idx : channel_idx + 1] + delta * mix

            result[idx] = sample_result[0]

        if is_video:
            result = result.reshape(batch, frames, channels, height, width).permute(0, 1, 3, 4, 2)
        else:
            result = result.permute(0, 2, 3, 1)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            result = blend_image_with_mask(image_tensor, result, mask)

        return (result.cpu(),)


class LatentForwardDiffusion:
    """
    Applies the 'natural' forward diffusion process to a clean latent.
    This produces a statistically 'perfect' noisy latent that samplers expect.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model that defines the forward noise schedule."}),
                "latent": ("LATENT", {"tooltip": "Clean latent to push forward along the schedule."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed for the forward diffusion noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of steps in the sampler's schedule."}),
                "noise_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The point in the schedule to noise to. Must match the KSampler's effective start step.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask (often image-sized) to limit the noising to masked areas. "
                                          "The mask is resized to latent resolution (bicubic when downscaling)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_scheduled_noise"
    CATEGORY = "Latent/Noise"

    def add_scheduled_noise(self, model, latent, seed, steps, noise_strength, mask=None):
        if noise_strength == 0.0:
            return (latent,)

        device = _get_device()
        latent_tensor = latent["samples"].clone().to(device)
        if not latent_tensor.is_floating_point():
            raise RuntimeError(f"LATENT samples must be floating point, got dtype={latent_tensor.dtype}.")

        sigmas = None
        if comfy_samplers is not None and model is not None:
            sampler = comfy_samplers.KSampler(model, steps=steps, device=device)
            sigmas = getattr(sampler, "sigmas", None)

        if sigmas is None:
            sigmas = torch.linspace(1.0, 0.0, steps, device=device)

        start_step = steps - int(steps * noise_strength)
        if start_step >= len(sigmas):
            start_step = len(sigmas) - 1

        sigma = sigmas[start_step].to(device)

        noise = _seeded_gaussian_noise_like(latent_tensor, seed)
        logger.debug(
            "LatentForwardDiffusion: batch=%d seed=%d sigma=%.6f",
            int(latent_tensor.shape[0]),
            int(seed),
            float(sigma),
        )

        noised_latent = latent_tensor + noise * sigma

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
            mask_nchw = prepare_mask_nchw(
                mask,
                batch_size=int(latent_tensor.shape[0]),
                height=int(latent_tensor.shape[-2]),
                width=int(latent_tensor.shape[-1]),
                device=device,
            )
            noised_latent = blend_with_mask(latent_tensor, noised_latent, mask_nchw)

        out = latent.copy()
        out["samples"] = noised_latent.cpu()
        return (out,)


class KSamplerLoraSigmaInverse:
    """
    KSampler variant with an embedded model-only LoRA loader whose strength follows:
    strength[i] = min_lora_strength + (max_lora_strength - min_lora_strength) * (1 - sigma[i] / max_sigma)
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        sampler_names: Sequence[str] = ("euler",)
        scheduler_names: Sequence[str] = ("normal",)
        if comfy_samplers is not None:
            sampler_names = comfy_samplers.KSampler.SAMPLERS
            scheduler_names = comfy_samplers.KSampler.SCHEDULERS

        lora_names: Sequence[str] = []
        if folder_paths is not None:
            lora_names = folder_paths.get_filename_list("loras")

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of denoising steps."}),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Classifier-Free Guidance scale.",
                }),
                "sampler_name": (sampler_names, {"tooltip": "Sampling algorithm."}),
                "scheduler": (scheduler_names, {"tooltip": "Sigma scheduler."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "lora_name": (lora_names, {"tooltip": "LoRA file to load and schedule over sampling."}),
                "min_lora_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Starting LoRA strength at the beginning of sampling (highest sigma).",
                }),
                "max_lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Target LoRA strength at the end of sampling when sigma reaches its minimum.",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How much of the schedule to run.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    @staticmethod
    def _compute_inverse_sigma_strengths(
        sigmas: torch.Tensor,
        max_lora_strength: float,
        min_lora_strength: float = 0.0,
    ) -> torch.Tensor:
        sigma_tensor = torch.as_tensor(sigmas, dtype=torch.float32).flatten()
        if sigma_tensor.numel() == 0:
            return sigma_tensor

        sigma_max = float(torch.max(sigma_tensor).item())
        if not math.isfinite(sigma_max) or sigma_max <= 0.0:
            return torch.full_like(sigma_tensor, float(min_lora_strength))

        normalized = torch.clamp(sigma_tensor / sigma_max, min=0.0, max=1.0)
        strength_range = float(max_lora_strength) - float(min_lora_strength)
        strengths = float(min_lora_strength) + (1.0 - normalized) * strength_range
        strengths = torch.where(torch.isfinite(strengths), strengths, torch.zeros_like(strengths))
        return strengths

    @staticmethod
    def _sigma_to_percent(model_sampling: Any, sigma: float, iterations: int = 32) -> float:
        target_sigma = float(sigma)
        if not math.isfinite(target_sigma):
            return 0.0

        low = 0.0
        high = 1.0
        sigma_low = float(model_sampling.percent_to_sigma(low))
        sigma_high = float(model_sampling.percent_to_sigma(high))
        decreasing = sigma_low >= sigma_high

        if decreasing:
            if target_sigma >= sigma_low:
                return 0.0
            if target_sigma <= sigma_high:
                return 1.0
        else:
            if target_sigma <= sigma_low:
                return 0.0
            if target_sigma >= sigma_high:
                return 1.0

        for _ in range(max(1, int(iterations))):
            mid = 0.5 * (low + high)
            sigma_mid = float(model_sampling.percent_to_sigma(mid))

            if not math.isfinite(sigma_mid):
                break
            if math.isclose(sigma_mid, target_sigma, rel_tol=1e-5, abs_tol=1e-6):
                return max(0.0, min(1.0, mid))

            if decreasing:
                if sigma_mid > target_sigma:
                    low = mid
                else:
                    high = mid
            else:
                if sigma_mid < target_sigma:
                    low = mid
                else:
                    high = mid

        return max(0.0, min(1.0, 0.5 * (low + high)))

    @classmethod
    def _build_percent_strength_schedule(
        cls,
        model_sampling: Any,
        sigmas: torch.Tensor,
        max_lora_strength: float,
        min_lora_strength: float = 0.0,
    ) -> List[Tuple[float, float]]:
        sigma_tensor = torch.as_tensor(sigmas, dtype=torch.float32).flatten()
        strengths = cls._compute_inverse_sigma_strengths(
            sigma_tensor,
            max_lora_strength=max_lora_strength,
            min_lora_strength=min_lora_strength,
        )

        schedule: List[Tuple[float, float]] = []
        previous_percent = -1.0
        for sigma_value, strength_value in zip(sigma_tensor.tolist(), strengths.tolist()):
            percent = cls._sigma_to_percent(model_sampling, float(sigma_value))
            if percent < previous_percent:
                percent = previous_percent
            previous_percent = percent
            schedule.append((float(percent), float(strength_value)))
        return schedule

    def _resolve_lora_path(self, lora_name: str) -> str:
        if folder_paths is None:
            raise RuntimeError("folder_paths is unavailable. This node only works inside ComfyUI.")

        if hasattr(folder_paths, "get_full_path_or_raise"):
            return folder_paths.get_full_path_or_raise("loras", lora_name)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise FileNotFoundError(f"Unable to resolve LoRA path for '{lora_name}'.")
        return lora_path

    def _load_lora(self, lora_name: str) -> Dict[str, Any]:
        if comfy_utils is None:
            raise RuntimeError("comfy.utils is unavailable. This node only works inside ComfyUI.")

        lora_path = self._resolve_lora_path(lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy_utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
            logger.info("Loaded LoRA '%s' for scheduled sampling.", os.path.basename(lora_path))

        return lora

    def _build_scheduled_model_hook(
        self,
        model: Any,
        lora: Dict[str, Any],
        sigmas: torch.Tensor,
        max_lora_strength: float,
        min_lora_strength: float,
    ):
        if comfy_hooks is None:
            raise RuntimeError("comfy.hooks is unavailable. This node only works inside ComfyUI.")

        model_sampling = model.get_model_object("model_sampling")
        schedule = self._build_percent_strength_schedule(
            model_sampling,
            sigmas,
            max_lora_strength=max_lora_strength,
            min_lora_strength=min_lora_strength,
        )

        hooks = comfy_hooks.create_hook_lora(lora=lora, strength_model=1.0, strength_clip=0.0)
        keyframes = comfy_hooks.HookKeyframeGroup()
        for percent, strength in schedule:
            keyframes.add(
                comfy_hooks.HookKeyframe(
                    strength=float(strength),
                    start_percent=float(percent),
                    guarantee_steps=1,
                )
            )
        hooks.set_keyframes_on_hooks(hook_kf=keyframes)
        return hooks, schedule

    @staticmethod
    def _strength_for_sigma(
        sigma_value: float,
        max_sigma: float,
        max_lora_strength: float,
        min_lora_strength: float = 0.0,
    ) -> float:
        sigma = float(sigma_value)
        sigma_max = float(max_sigma)
        if not math.isfinite(sigma) or not math.isfinite(sigma_max) or sigma_max <= 0.0:
            return float(min_lora_strength)

        normalized = max(0.0, min(1.0, sigma / sigma_max))
        strength_range = float(max_lora_strength) - float(min_lora_strength)
        return float(min_lora_strength) + strength_range * (1.0 - normalized)

    def _load_unet_lora_patches(self, model: Any, lora: Dict[str, Any]) -> Dict[Any, Any]:
        if comfy_lora is None:
            raise RuntimeError("comfy.lora is unavailable. This node only works inside ComfyUI.")

        key_map: Dict[str, Any] = comfy_lora.model_lora_keys_unet(model.model, {})
        return comfy_lora.load_lora(lora, key_map, log_missing=False)

    def _split_lora_patches_for_bypass(self, loaded_patches: Dict[Any, Any]) -> Tuple[Dict[str, Any], Dict[Any, Any]]:
        adapter_patches: Dict[str, Any] = {}
        unsupported_patches: Dict[Any, Any] = {}

        for key, patch in loaded_patches.items():
            if not isinstance(key, str):
                unsupported_patches[key] = patch
                continue
            if comfy_weight_adapter is None:
                unsupported_patches[key] = patch
                continue
            if isinstance(patch, comfy_weight_adapter.WeightAdapterBase):
                adapter_patches[key] = patch
            else:
                unsupported_patches[key] = patch

        return adapter_patches, unsupported_patches

    def _install_bypass_strength_wrapper(
        self,
        model_with_lora: Any,
        adapters: Sequence[Any],
        max_sigma: float,
        max_lora_strength: float,
        min_lora_strength: float,
    ) -> None:
        previous_wrapper = model_with_lora.model_options.get("model_function_wrapper")
        state = {"last_strength": None}

        def _update_strength_from_timestep(timestep: Any) -> None:
            sigma_tensor = torch.as_tensor(timestep, dtype=torch.float32).flatten()
            sigma_value = float(sigma_tensor[0].item()) if sigma_tensor.numel() > 0 else 0.0
            strength = self._strength_for_sigma(
                sigma_value,
                max_sigma,
                max_lora_strength,
                min_lora_strength=min_lora_strength,
            )

            last_strength = state["last_strength"]
            if last_strength is not None and math.isclose(last_strength, strength, rel_tol=1e-6, abs_tol=1e-6):
                return

            for adapter in adapters:
                adapter.multiplier = strength
            state["last_strength"] = strength

        def _wrapper(model_function, call_args):
            _update_strength_from_timestep(call_args.get("timestep", 0.0))

            if previous_wrapper is not None:
                return previous_wrapper(model_function, call_args)

            return model_function(
                call_args["input"],
                call_args["timestep"],
                **call_args["c"],
            )

        model_with_lora.set_model_unet_function_wrapper(_wrapper)

    def _build_bypass_lora_model(
        self,
        model: Any,
        adapter_patches: Dict[str, Any],
        max_sigma: float,
        max_lora_strength: float,
        min_lora_strength: float,
    ) -> Tuple[Any, int, int]:
        if comfy_weight_adapter is None:
            raise RuntimeError("comfy.weight_adapter is unavailable. This node only works inside ComfyUI.")

        model_with_lora = model.clone()
        manager = comfy_weight_adapter.BypassInjectionManager()
        model_state_keys = set(model_with_lora.model.state_dict().keys())

        missing_keys = 0
        for key, adapter in adapter_patches.items():
            if key in model_state_keys:
                manager.add_adapter(key, adapter, strength=float(min_lora_strength))
            else:
                missing_keys += 1

        injections = manager.create_injections(model_with_lora.model)
        if manager.get_hook_count() <= 0:
            raise RuntimeError("Bypass LoRA mode did not produce any injectable hooks for the selected model/LoRA.")

        model_with_lora.set_injections("skoogeer_lora_sigma_inverse", injections)
        active_adapters = [hook.adapter for hook in manager.hooks]
        self._install_bypass_strength_wrapper(
            model_with_lora=model_with_lora,
            adapters=active_adapters,
            max_sigma=max_sigma,
            max_lora_strength=max_lora_strength,
            min_lora_strength=min_lora_strength,
        )

        return model_with_lora, len(active_adapters), missing_keys

    def _sample_like_ksampler(
        self,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Dict[str, Any],
        denoise: float,
        sigmas: torch.Tensor,
    ) -> Tuple[Dict[str, Any]]:
        if comfy_sample is None:
            raise RuntimeError("comfy.sample is unavailable. This node only works inside ComfyUI.")

        latent_tensor = latent_image["samples"]
        latent_tensor = comfy_sample.fix_empty_latent_channels(
            model,
            latent_tensor,
            latent_image.get("downscale_ratio_spacial", None),
        )

        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise = comfy_sample.prepare_noise(latent_tensor, seed, batch_inds)

        noise_mask = latent_image.get("noise_mask", None)
        callback = None
        if latent_preview is not None:
            try:
                callback = latent_preview.prepare_callback(model, steps)
            except Exception:  # pragma: no cover - preview callback is best effort
                callback = None

        disable_pbar = False
        if comfy_utils is not None and hasattr(comfy_utils, "PROGRESS_BAR_ENABLED"):
            disable_pbar = not bool(comfy_utils.PROGRESS_BAR_ENABLED)

        samples = comfy_sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_tensor,
            denoise=denoise,
            noise_mask=noise_mask,
            sigmas=sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return (out,)

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        lora_name,
        min_lora_strength,
        max_lora_strength,
        denoise=1.0,
    ):
        if comfy_samplers is None:
            raise RuntimeError("comfy.samplers is unavailable. This node only works inside ComfyUI.")

        ksampler = comfy_samplers.KSampler(
            model,
            steps=steps,
            device=model.load_device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            model_options=model.model_options,
        )
        sigmas = getattr(ksampler, "sigmas", None)
        if sigmas is None:
            sigmas = torch.linspace(1.0, 0.0, steps + 1, device=model.load_device)

        if float(min_lora_strength) == 0.0 and float(max_lora_strength) == 0.0:
            return self._sample_like_ksampler(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                sigmas=sigmas,
            )

        lora = self._load_lora(lora_name)
        sigmas_tensor = torch.as_tensor(sigmas, dtype=torch.float32)
        max_sigma = float(torch.max(sigmas_tensor).item()) if sigmas_tensor.numel() > 0 else 0.0

        loaded_patches = None
        adapter_patches: Dict[str, Any] = {}
        unsupported_patches: Dict[Any, Any] = {}

        if comfy_lora is not None and comfy_weight_adapter is not None:
            try:
                loaded_patches = self._load_unet_lora_patches(model, lora)
                adapter_patches, unsupported_patches = self._split_lora_patches_for_bypass(loaded_patches)
            except Exception as exc:
                logger.warning("Bypass LoRA setup failed, falling back to hook scheduling: %s", exc)
                loaded_patches = None

        can_use_bypass = (
            loaded_patches is not None
            and len(adapter_patches) > 0
            and len(unsupported_patches) == 0
        )

        if can_use_bypass:
            model_with_lora, adapter_count, missing_keys = self._build_bypass_lora_model(
                model=model,
                adapter_patches=adapter_patches,
                max_sigma=max_sigma,
                max_lora_strength=max_lora_strength,
                min_lora_strength=min_lora_strength,
            )
            logger.info(
                "KSamplerLoraSigmaInverse: using bypass LoRA path (adapters=%d missing=%d min=%.4f max=%.4f lora=%s).",
                int(adapter_count),
                int(missing_keys),
                float(min_lora_strength),
                float(max_lora_strength),
                lora_name,
            )
            return self._sample_like_ksampler(
                model=model_with_lora,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                sigmas=sigmas,
            )

        if comfy_hooks is None:
            raise RuntimeError(
                "Unable to apply scheduled LoRA: bypass mode was unavailable and comfy.hooks fallback is unavailable."
            )

        if loaded_patches is not None and len(unsupported_patches) > 0:
            logger.warning(
                "KSamplerLoraSigmaInverse: %d non-bypass LoRA patches detected; falling back to hook scheduling.",
                int(len(unsupported_patches)),
            )

        hooks, schedule = self._build_scheduled_model_hook(
            model=model,
            lora=lora,
            sigmas=sigmas,
            max_lora_strength=max_lora_strength,
            min_lora_strength=min_lora_strength,
        )
        logger.info(
            "KSamplerLoraSigmaInverse: using hook fallback path (steps=%d min=%.4f max=%.4f lora=%s).",
            int(len(schedule)),
            float(min_lora_strength),
            float(max_lora_strength),
            lora_name,
        )

        cache: Dict[Tuple[Any, Any], Any] = {}
        positive_with_hooks = comfy_hooks.set_hooks_for_conditioning(positive, hooks, append_hooks=True, cache=cache)
        negative_with_hooks = comfy_hooks.set_hooks_for_conditioning(negative, hooks, append_hooks=True, cache=cache)

        return self._sample_like_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_with_hooks,
            negative=negative_with_hooks,
            latent_image=latent_image,
            denoise=denoise,
            sigmas=sigmas,
        )


class ConditioningAddNoise:
    """Adds seeded Gaussian noise to conditioning embeddings and pooled outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list to perturb with Gaussian noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Seed that drives the conditioning noise."}),
                "strength": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Noise strength relative to each tensor's standard deviation.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "add_noise"
    CATEGORY = "conditioning/noise"

    def add_noise(self, conditioning, seed, strength):
        if strength == 0.0:
            return (conditioning,)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor):
                result.append([embedding, metadata])
                continue

            embedding_std = torch.std(embedding)
            noise = torch.randn(embedding.shape, generator=generator, device=embedding.device, dtype=embedding.dtype)
            noised_embedding = embedding + noise * embedding_std * strength

            new_metadata = dict(metadata)
            pooled_output = new_metadata.get("pooled_output")

            if isinstance(pooled_output, torch.Tensor):
                pooled_std = torch.std(pooled_output)
                pooled_noise = torch.randn(
                    pooled_output.shape,
                    generator=generator,
                    device=pooled_output.device,
                    dtype=pooled_output.dtype,
                )
                new_metadata["pooled_output"] = pooled_output + pooled_noise * pooled_std * strength

            result.append([noised_embedding, new_metadata])

        return (result,)


class ConditioningGaussianBlur:
    """Applies Gaussian smoothing along the token dimension of conditioning embeddings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list whose token dimension will be blurred."}),
                "sigma": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Standard deviation of the blur kernel along the token axis.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "blur"
    CATEGORY = "conditioning/filter"

    def blur(self, conditioning, sigma):
        if sigma <= 0.0:
            return (conditioning,)

        result = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor) or embedding.dim() < 2:
                result.append([embedding, metadata])
                continue

            needs_batch_dim = embedding.dim() == 2
            embedding_tensor = embedding.unsqueeze(0) if needs_batch_dim else embedding

            conv_dtype = self._select_conv_dtype(embedding_tensor)
            blur_kernel, padding = self._build_kernel(embedding_tensor.device, conv_dtype, sigma)

            original_shape = embedding_tensor.shape
            tokens = original_shape[-2]
            features = original_shape[-1]

            batch = math.prod(original_shape[:-2]) if len(original_shape) > 2 else original_shape[0]
            reshaped = embedding_tensor.reshape(batch, tokens, features).permute(0, 2, 1)

            if reshaped.dtype != conv_dtype:
                reshaped = reshaped.to(conv_dtype)

            kernel = blur_kernel.repeat(features, 1, 1)
            blurred = F.conv1d(reshaped, kernel, padding=padding, groups=features)
            blurred = blurred.permute(0, 2, 1).reshape(original_shape)

            if blurred.dtype != embedding_tensor.dtype:
                blurred = blurred.to(embedding_tensor.dtype)

            if needs_batch_dim:
                blurred = blurred.squeeze(0)

            result.append([blurred, metadata])

        return (result,)

    @staticmethod
    def _build_kernel(device, dtype, sigma):
        kernel_size = int(sigma * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        half = kernel_size // 2
        positions = torch.arange(kernel_size, device=device, dtype=dtype) - half
        kernel = torch.exp(-0.5 * (positions / max(sigma, 1e-6)) ** 2)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size)
        return kernel, half

    @staticmethod
    def _select_conv_dtype(tensor):
        if tensor.device.type == "cpu" and tensor.dtype not in (torch.float32, torch.float64):
            return torch.float32
        return tensor.dtype


class ConditioningFrequencySplit:
    """Separates conditioning embeddings into low/high bands via Gaussian smoothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list to separate into low/high bands."}),
                "sigma": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Cutoff for the Gaussian low-pass applied along the token axis.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("low_pass", "high_pass")
    FUNCTION = "split"
    CATEGORY = "conditioning/filter"

    def split(self, conditioning, sigma):
        if sigma <= 0.0:
            low_list = [self._clone_entry(item) for item in conditioning]
            high_list = [self._zero_entry(item) for item in conditioning]
            return (low_list, high_list)

        low_list = []
        high_list = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor) or embedding.dim() < 2:
                low_list.append([embedding, dict(metadata)])
                high_list.append(self._zero_entry([embedding, metadata]))
                continue

            needs_batch_dim = embedding.dim() == 2
            embedding_tensor = embedding.unsqueeze(0) if needs_batch_dim else embedding

            conv_dtype = ConditioningGaussianBlur._select_conv_dtype(embedding_tensor)
            kernel, padding = ConditioningGaussianBlur._build_kernel(embedding_tensor.device, conv_dtype, sigma)

            original_shape = embedding_tensor.shape
            tokens = original_shape[-2]
            features = original_shape[-1]

            batch = math.prod(original_shape[:-2]) if len(original_shape) > 2 else original_shape[0]
            reshaped = embedding_tensor.reshape(batch, tokens, features).permute(0, 2, 1)

            if reshaped.dtype != conv_dtype:
                reshaped = reshaped.to(conv_dtype)

            kernel = kernel.repeat(features, 1, 1)
            blurred = F.conv1d(reshaped, kernel, padding=padding, groups=features)
            blurred = blurred.permute(0, 2, 1).reshape(original_shape)

            if blurred.dtype != embedding_tensor.dtype:
                blurred = blurred.to(embedding_tensor.dtype)

            if needs_batch_dim:
                blurred = blurred.squeeze(0)

            high_embedding = embedding - blurred

            low_meta = dict(metadata)
            high_meta = dict(metadata)

            pooled_output = low_meta.get("pooled_output")
            if isinstance(pooled_output, torch.Tensor):
                high_meta["pooled_output"] = torch.zeros_like(pooled_output)

            low_list.append([blurred, low_meta])
            high_list.append([high_embedding, high_meta])

        return (low_list, high_list)

    @staticmethod
    def _clone_entry(entry):
        embedding, metadata = entry
        return [embedding, dict(metadata)]

    @staticmethod
    def _zero_entry(entry):
        embedding, metadata = entry
        if isinstance(embedding, torch.Tensor):
            zero_embedding = torch.zeros_like(embedding)
        else:
            zero_embedding = embedding

        meta_copy = dict(metadata)
        pooled_output = meta_copy.get("pooled_output")
        if isinstance(pooled_output, torch.Tensor):
            meta_copy["pooled_output"] = torch.zeros_like(pooled_output)
        return [zero_embedding, meta_copy]


class ConditioningFrequencyMerge:
    """Recombines low/high conditioning bands back into a single conditioning list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_pass": ("CONDITIONING", {"tooltip": "Low-frequency conditioning list produced by the split node."}),
                "high_pass": ("CONDITIONING", {"tooltip": "High-frequency conditioning list to recombine."}),
                "low_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Multiplier for the low-pass band before merging.",
                }),
                "high_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Multiplier for the high-pass band before merging.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "merge"
    CATEGORY = "conditioning/filter"

    def merge(self, low_pass, high_pass, low_gain, high_gain):
        if len(low_pass) != len(high_pass):
            raise ValueError("Low-pass and high-pass conditioning lists must have the same length")

        merged = []

        for low_item, high_item in zip(low_pass, high_pass):
            low_embedding, low_meta = low_item
            high_embedding, high_meta = high_item

            if not isinstance(low_embedding, torch.Tensor) or not isinstance(high_embedding, torch.Tensor):
                merged.append([low_embedding, dict(low_meta)])
                continue

            combined_embedding = low_embedding * low_gain + high_embedding * high_gain

            new_metadata = dict(low_meta)
            low_pooled = low_meta.get("pooled_output")
            high_pooled = high_meta.get("pooled_output") if isinstance(high_meta, dict) else None

            if isinstance(low_pooled, torch.Tensor) or isinstance(high_pooled, torch.Tensor):
                low_pooled_tensor = low_pooled if isinstance(low_pooled, torch.Tensor) else torch.zeros_like(high_pooled)
                high_pooled_tensor = high_pooled if isinstance(high_pooled, torch.Tensor) else torch.zeros_like(low_pooled_tensor)
                new_metadata["pooled_output"] = low_pooled_tensor * low_gain + high_pooled_tensor * high_gain

            merged.append([combined_embedding, new_metadata])

        return (merged,)


class ConditioningScale:
    """Scales conditioning embeddings (and pooled outputs) to amplify or mute prompt influence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list to scale."}),
                "factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Multiplier applied to embeddings. 0.0 mutes, 1.0 keeps original strength.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "scale"
    CATEGORY = "conditioning/filter"

    def scale(self, conditioning, factor):
        if factor == 1.0:
            return (conditioning,)

        result = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor):
                result.append([embedding, metadata])
                continue

            scaled_embedding = embedding * factor
            new_metadata = dict(metadata)

            pooled_output = new_metadata.get("pooled_output")
            if isinstance(pooled_output, torch.Tensor):
                new_metadata["pooled_output"] = pooled_output * factor

            result.append([scaled_embedding, new_metadata])

        return (result,)


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "UnpatchifyFlux2Latent": UnpatchifyFlux2Latent,
    "PatchifyFlux2Latent": PatchifyFlux2Latent,
    "LatentGaussianBlur": LatentGaussianBlur,
    "LatentFrequencySplit": LatentFrequencySplit,
    "LatentFrequencyMerge": LatentFrequencyMerge,
    "LatentAddNoise": LatentAddNoise,
    "LatentPerlinFractalNoise": LatentPerlinFractalNoise,
    "LatentSimplexNoise": LatentSimplexNoise,
    "LatentWorleyNoise": LatentWorleyNoise,
    "LatentReactionDiffusion": LatentReactionDiffusion,
    "LatentFractalBrownianMotion": LatentFractalBrownianMotion,
    "LatentSwirlNoise": LatentSwirlNoise,
    "ImageAddNoise": ImageAddNoise,
    "ImagePerlinFractalNoise": ImagePerlinFractalNoise,
    "ImageSimplexNoise": ImageSimplexNoise,
    "ImageWorleyNoise": ImageWorleyNoise,
    "ImageReactionDiffusion": ImageReactionDiffusion,
    "ImageFractalBrownianMotion": ImageFractalBrownianMotion,
    "ImageSwirlNoise": ImageSwirlNoise,
    "LatentForwardDiffusion": LatentForwardDiffusion,
    "KSamplerLoraSigmaInverse": KSamplerLoraSigmaInverse,
    "ConditioningAddNoise": ConditioningAddNoise,
    "ConditioningGaussianBlur": ConditioningGaussianBlur,
    "ConditioningFrequencySplit": ConditioningFrequencySplit,
    "ConditioningFrequencyMerge": ConditioningFrequencyMerge,
    "ConditioningScale": ConditioningScale,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "UnpatchifyFlux2Latent": "Unpatchify Flux.2 Latent",
    "PatchifyFlux2Latent": "Patchify Flux.2 Latent",
    "LatentGaussianBlur": "Latent Gaussian Blur",
    "LatentFrequencySplit": "Latent Frequency Split",
    "LatentFrequencyMerge": "Latent Frequency Merge",
    "LatentAddNoise": "Add Latent Noise (Seeded)",
    "LatentPerlinFractalNoise": "Latent Perlin Fractal Noise",
    "LatentSimplexNoise": "Latent Simplex Noise",
    "LatentWorleyNoise": "Latent Worley Noise",
    "LatentReactionDiffusion": "Latent Reaction-Diffusion",
    "LatentFractalBrownianMotion": "Latent Fractal Brownian Motion",
    "LatentSwirlNoise": "Latent Swirl Noise",
    "ImageAddNoise": "Add Image Noise (Seeded)",
    "ImagePerlinFractalNoise": "Image Perlin Fractal Noise",
    "ImageSimplexNoise": "Image Simplex Noise",
    "ImageWorleyNoise": "Image Worley Noise",
    "ImageReactionDiffusion": "Image Reaction-Diffusion",
    "ImageFractalBrownianMotion": "Image Fractal Brownian Motion",
    "ImageSwirlNoise": "Image Swirl Noise",
    "LatentForwardDiffusion": "Forward Diffusion (Add Scheduled Noise)",
    "KSamplerLoraSigmaInverse": "KSampler (LoRA Sigma Inverse)",
    "ConditioningAddNoise": "Conditioning (Add Noise)",
    "ConditioningGaussianBlur": "Conditioning (Gaussian Blur)",
    "ConditioningFrequencySplit": "Conditioning (Frequency Split)",
    "ConditioningFrequencyMerge": "Conditioning (Frequency Merge)",
    "ConditioningScale": "Conditioning (Scale)",
}
