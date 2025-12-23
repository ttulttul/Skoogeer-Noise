import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.latent_channel_stats_preview as stats  # noqa: E402
from src.latent_channel_stats_preview import LatentChannelStatsPreview  # noqa: E402


def test_render_channel_stats_image_monotonic():
    means = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    stds = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    image, layout = stats._render_channel_stats_image(means, stds, limit=3, height=128)

    assert tuple(image.shape[-1:]) == (3,)
    assert image.shape[0] == 1
    assert torch.all(image >= 0.0)
    assert torch.all(image <= 1.0)

    mean_color = torch.from_numpy(stats._MEAN_BAR_COLOR)
    std_color = torch.from_numpy(stats._STD_BAR_COLOR)
    background = torch.from_numpy(stats._BACKGROUND_COLOR)

    mean_start, mean_height = layout["mean_section"]
    std_start, std_height = layout["std_section"]
    channel_label_y = layout["channel_label_y"]

    mean_heights = []
    std_heights = []
    for bar in layout["bar_ranges"]:
        x0, x1 = bar["x_range"]
        x_center = (x0 + x1 - 1) // 2

        mean_column = image[0, mean_start : mean_start + mean_height, x_center, :]
        std_column = image[0, std_start : std_start + std_height, x_center, :]

        mean_mask = torch.linalg.norm(mean_column - mean_color, dim=-1) < 1e-3
        std_mask = torch.linalg.norm(std_column - std_color, dim=-1) < 1e-3

        mean_heights.append(int(mean_mask.sum().item()))
        std_heights.append(int(std_mask.sum().item()))

        label_slice = image[
            0,
            channel_label_y : channel_label_y + stats._FONT_HEIGHT,
            max(0, x_center - 2) : min(image.shape[2], x_center + 3),
            :,
        ]
        background_diff = torch.abs(label_slice - background.view(1, 1, -1)).max()
        assert background_diff.item() > 1e-3

    assert mean_heights[1] > mean_heights[0]
    assert mean_heights[2] > mean_heights[1]
    assert std_heights[1] > std_heights[0]
    assert std_heights[2] > std_heights[1]

    mean_label_region = image[0, mean_start : mean_start + mean_height, 4 : 4 + stats._FONT_WIDTH, :]
    std_label_region = image[0, std_start : std_start + std_height, 4 : 4 + stats._FONT_WIDTH, :]
    assert torch.abs(mean_label_region - background.view(1, 1, -1)).max().item() > 1e-3
    assert torch.abs(std_label_region - background.view(1, 1, -1)).max().item() > 1e-3


def test_latent_channel_stats_preview_outputs_image():
    samples = torch.stack(
        [
            torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32).reshape(1, 4, 4),
            torch.linspace(0.0, 0.5, steps=16, dtype=torch.float32).reshape(1, 4, 4),
            torch.linspace(-0.5, 0.5, steps=16, dtype=torch.float32).reshape(1, 4, 4),
            torch.linspace(0.25, 0.75, steps=16, dtype=torch.float32).reshape(1, 4, 4),
        ],
        dim=1,
    )
    latent = {"samples": samples}

    preview_node = LatentChannelStatsPreview()
    (image_tensor,) = preview_node.render(latent=latent, channel_limit=4, height=128)

    assert image_tensor.shape[0] == 1
    assert image_tensor.shape[1] > 0
    assert image_tensor.shape[2] > 0
    assert image_tensor.shape[3] == 3
    assert image_tensor.dtype == torch.float32
    assert torch.isfinite(image_tensor).all()
    assert torch.all(image_tensor >= 0.0)
    assert torch.all(image_tensor <= 1.0)


def test_latent_channel_stats_preview_rejects_invalid_tensor():
    latent = {"samples": torch.ones((8,), dtype=torch.float32)}
    node = LatentChannelStatsPreview()
    try:
        node.render(latent)
    except ValueError as exc:
        assert "channel dimension" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid latent tensor shape.")
