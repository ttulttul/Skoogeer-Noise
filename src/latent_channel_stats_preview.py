from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_BACKGROUND_COLOR = np.array([0.08, 0.08, 0.08], dtype=np.float32)
_GRIDLINE_COLOR = np.array([0.25, 0.25, 0.25], dtype=np.float32)
_MEAN_BAR_COLOR = np.array([0.2, 0.65, 0.95], dtype=np.float32)
_STD_BAR_COLOR = np.array([0.95, 0.6, 0.2], dtype=np.float32)
_TEXT_COLOR = np.array([0.85, 0.85, 0.85], dtype=np.float32)

_FONT_BITMAPS = {
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    "3": [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "5": [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
    "6": [
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    "7": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    "9": [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100",
    ],
    "-": [
        "00000",
        "00000",
        "00100",
        "00100",
        "00100",
        "00000",
        "00000",
    ],
    ".": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00100",
        "00100",
    ],
    " ": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10101",
        "10001",
        "10001",
        "10001",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "S": [
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "D": [
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ],
}
_FONT_HEIGHT = len(next(iter(_FONT_BITMAPS.values())))
_FONT_WIDTH = len(next(iter(_FONT_BITMAPS.values()))[0])


def _measure_text(text: str) -> int:
    """Compute pixel width of text using the fixed bitmap font."""
    if not text:
        return 0
    width = 0
    for char in text:
        glyph = _FONT_BITMAPS.get(char.upper(), _FONT_BITMAPS[" "])
        width += len(glyph[0]) + 1
    return max(0, width - 1)


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int, color: np.ndarray) -> None:
    """Draw text onto the canvas in-place using the bitmap font."""
    cursor = x
    for char in text:
        glyph = _FONT_BITMAPS.get(char.upper(), _FONT_BITMAPS[" "])
        glyph_height = len(glyph)
        glyph_width = len(glyph[0])
        for gy in range(glyph_height):
            cy = y + gy
            if cy < 0 or cy >= canvas.shape[0]:
                continue
            row = glyph[gy]
            for gx in range(glyph_width):
                if row[gx] != "1":
                    continue
                cx = cursor + gx
                if cx < 0 or cx >= canvas.shape[1]:
                    continue
                canvas[cy, cx, :] = color
        cursor += glyph_width + 1


def _reduce_channel_statistics(tensor: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute per-channel mean and standard deviation across all non-channel dimensions.
    Returns None when statistics cannot be computed (e.g., invalid tensor shape).
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.ndim < 2:
        return None

    with torch.no_grad():
        data = tensor.detach()
        if data.shape[1] == 0:
            return None

        reduce_dims = tuple(dim for dim in range(data.ndim) if dim != 1)
        channel_means = data.float().mean(dim=reduce_dims)
        channel_stds = data.float().std(dim=reduce_dims, unbiased=False)
    return channel_means, channel_stds


def _render_channel_stats_image(
    channel_means: torch.Tensor,
    channel_stds: torch.Tensor,
    *,
    limit: int = 16,
    height: int = 256,
) -> Tuple[torch.Tensor, dict]:
    """
    Render a simple bar chart visualizing per-channel mean (top) and std (bottom).

    Returns:
        image (torch.Tensor): Float tensor with shape (1, H, W, 3) in [0, 1].
        layout (dict): Metadata describing bar placement and section bounds.
    """
    if channel_means.numel() == 0 or channel_stds.numel() == 0:
        raise ValueError("Channel statistics must be non-empty to render.")

    display_channels = max(1, min(int(limit), channel_means.numel()))
    means_np = channel_means.detach().cpu().float().numpy()
    stds_np = channel_stds.detach().cpu().float().numpy()

    display_means = means_np[:display_channels]
    display_stds = np.clip(stds_np[:display_channels], a_min=0.0, a_max=None)

    margin_left = 52
    margin_right = 18
    margin_y = 16
    label_margin_bottom = _FONT_HEIGHT + 10
    bar_spacing = 6
    bar_width = 16
    section_count = 2
    min_section_height = 28

    if height < 72:
        height = 72

    available = height - label_margin_bottom - margin_y * (section_count + 1)
    if available < section_count * min_section_height:
        section_height = min_section_height
    else:
        section_height = max(min_section_height, available // section_count)

    actual_height = margin_y * (section_count + 1) + section_height * section_count + label_margin_bottom

    width = margin_left + margin_right + display_channels * bar_width + max(0, (display_channels - 1) * bar_spacing)
    canvas = np.broadcast_to(_BACKGROUND_COLOR, (actual_height, width, 3)).copy()

    # draw outer border
    canvas[0, :, :] = _GRIDLINE_COLOR
    canvas[-1, :, :] = _GRIDLINE_COLOR
    canvas[:, 0, :] = _GRIDLINE_COLOR
    canvas[:, -1, :] = _GRIDLINE_COLOR

    top_start = margin_y
    bottom_start = margin_y * 2 + section_height
    x_start = margin_left
    x_end = width - margin_right
    channel_label_y = bottom_start + section_height + 4

    # Baselines for sections
    canvas[top_start - 1, x_start:x_end, :] = _GRIDLINE_COLOR
    canvas[top_start + section_height - 1, x_start:x_end, :] = _GRIDLINE_COLOR
    canvas[bottom_start - 1, x_start:x_end, :] = _GRIDLINE_COLOR
    canvas[bottom_start + section_height - 1, x_start:x_end, :] = _GRIDLINE_COLOR

    mean_min = float(np.min(display_means))
    mean_max = float(np.max(display_means))
    if math.isclose(mean_min, mean_max):
        mean_min -= 0.5
        mean_max += 0.5
    mean_range = max(mean_max - mean_min, 1e-6)

    def _clamp_unit(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _section_value_to_y(norm_value: float, section_start: int) -> int:
        clamped = _clamp_unit(norm_value)
        return section_start + section_height - 1 - int(round(clamped * (section_height - 1)))

    def _draw_y_tick(y_pos: int):
        tick_x0 = max(1, margin_left - 8)
        tick_x1 = max(tick_x0 + 1, margin_left - 2)
        if 0 <= y_pos < canvas.shape[0]:
            canvas[y_pos, tick_x0:tick_x1, :] = _GRIDLINE_COLOR

    def _clamp_text_y(y_pos: int) -> int:
        max_y = canvas.shape[0] - _FONT_HEIGHT - 1
        return max(1, min(max_y, y_pos))

    mean_max_y = _section_value_to_y(1.0, top_start)
    mean_min_y = _section_value_to_y(0.0, top_start)

    if mean_min <= 0.0 <= mean_max:
        zero_norm = (0.0 - mean_min) / mean_range
        zero_y = _section_value_to_y(zero_norm, top_start)
        canvas[zero_y, x_start:x_end, :] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    else:
        zero_y = None

    std_max = float(np.max(display_stds))
    if math.isclose(std_max, 0.0):
        std_max = 1.0

    bar_layout = []
    for idx in range(display_channels):
        x0 = x_start + idx * (bar_width + bar_spacing)
        x1 = x0 + bar_width
        x1 = min(x1, x_end)

        # mean bar (top section)
        mean_val = float(display_means[idx])
        mean_norm = np.clip((mean_val - mean_min) / mean_range, 0.0, 1.0)
        mean_pixels = max(1, int(round(mean_norm * (section_height - 2))))
        mean_bottom = top_start + section_height - 2
        mean_top = mean_bottom - mean_pixels
        mean_top = max(top_start, mean_top)
        canvas[mean_top:mean_bottom, x0:x1, :] = _MEAN_BAR_COLOR

        # std bar (bottom section)
        std_val = float(display_stds[idx])
        std_norm = 0.0 if std_max <= 0.0 else np.clip(std_val / std_max, 0.0, 1.0)
        std_pixels = 0 if std_max <= 0.0 else max(1, int(round(std_norm * (section_height - 2))))
        std_bottom = bottom_start + section_height - 2
        std_top = std_bottom - std_pixels
        std_top = max(bottom_start, std_top)
        if std_pixels > 0:
            canvas[std_top:std_bottom, x0:x1, :] = _STD_BAR_COLOR

        bar_layout.append(
            {
                "channel": idx,
                "x_range": (x0, x1),
            }
        )

        x_center = (x0 + x1 - 1) // 2

        tick_y_start = bottom_start + section_height - 1
        tick_y_end = min(canvas.shape[0], tick_y_start + 6)
        xa = max(0, x_center - 1)
        xb = min(canvas.shape[1], x_center + 2)
        canvas[tick_y_start:tick_y_end, xa:xb, :] = _GRIDLINE_COLOR

        channel_text = str(idx)
        text_width = _measure_text(channel_text)
        text_x = x_center - text_width // 2
        text_y = channel_label_y
        _draw_text(canvas, channel_text, text_x, text_y, _TEXT_COLOR)

    if channel_means.numel() > display_channels:
        overflow_indicator_width = min(bar_width, x_end - (x_end - bar_width))
        canvas[top_start - 1:top_start + 1, x_end - overflow_indicator_width:x_end, :] = np.array(
            [0.8, 0.2, 0.2], dtype=np.float32
        )
        canvas[
            bottom_start + section_height - 1:bottom_start + section_height + 1,
            x_end - overflow_indicator_width:x_end,
            :
        ] = np.array([0.8, 0.2, 0.2], dtype=np.float32)

    _draw_y_tick(mean_max_y)
    _draw_y_tick(mean_min_y)
    mean_max_label = f"{mean_max:.2f}"
    mean_min_label = f"{mean_min:.2f}"
    _draw_text(canvas, mean_max_label, 4, _clamp_text_y(mean_max_y - _FONT_HEIGHT // 2), _TEXT_COLOR)
    _draw_text(canvas, mean_min_label, 4, _clamp_text_y(mean_min_y - _FONT_HEIGHT // 2), _TEXT_COLOR)

    std_max_y = _section_value_to_y(1.0, bottom_start)
    std_min_y = _section_value_to_y(0.0, bottom_start)
    _draw_y_tick(std_max_y)
    _draw_y_tick(std_min_y)
    std_max_label = f"{std_max:.2f}"
    std_min_label = f"{0.0:.2f}"
    _draw_text(canvas, std_max_label, 4, _clamp_text_y(std_max_y - _FONT_HEIGHT // 2), _TEXT_COLOR)
    _draw_text(canvas, std_min_label, 4, _clamp_text_y(std_min_y - _FONT_HEIGHT // 2), _TEXT_COLOR)

    _draw_text(
        canvas,
        "MEAN",
        4,
        _clamp_text_y(top_start + section_height // 2 - _FONT_HEIGHT // 2),
        _MEAN_BAR_COLOR,
    )
    _draw_text(
        canvas,
        "STD",
        4,
        _clamp_text_y(bottom_start + section_height // 2 - _FONT_HEIGHT // 2),
        _STD_BAR_COLOR,
    )

    image = torch.from_numpy(canvas.astype(np.float32)).unsqueeze(0)
    layout = {
        "bar_ranges": bar_layout,
        "mean_section": (top_start, section_height),
        "std_section": (bottom_start, section_height),
        "width": width,
        "height": actual_height,
        "channel_label_y": channel_label_y,
    }
    return image, layout


class LatentChannelStatsPreview:
    CATEGORY = "latent/debug"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "channel_limit": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                }),
                "height": ("INT", {
                    "default": 256,
                    "min": 72,
                    "max": 1024,
                    "step": 4,
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Mark as dynamic so previews refresh reliably.
        return float("nan")

    def render(self, latent, channel_limit=16, height=256):
        stats = _reduce_channel_statistics(latent["samples"])
        if stats is None:
            raise ValueError("Latent tensor must expose a channel dimension to compute statistics.")

        means, stds = stats
        image, _ = _render_channel_stats_image(
            means,
            stds,
            limit=max(1, int(channel_limit)),
            height=max(72, int(height)),
        )
        return (image,)


NODE_CLASS_MAPPINGS = {
    "LatentChannelStatsPreview": LatentChannelStatsPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentChannelStatsPreview": "Latent Channel Stats Preview",
}
