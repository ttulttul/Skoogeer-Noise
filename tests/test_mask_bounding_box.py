"""Tests for the Bounding Box mask node."""

import pathlib
import sys
from collections import deque

import numpy as np
import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mask_bounding_box import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    BoundingBox,
    MaskBoundingBox,
    crop_mask_to_box,
    dilate_for_gap,
    label_components,
    mask_bounding_boxes,
)


def _box(mask, **kwargs):
    boxes, _ = mask_bounding_boxes(mask, **kwargs)
    assert len(boxes) == 1
    return boxes[0]


def _scene(height=100, width=120):
    """A 40x30 main subject plus a bright 2x2 speck and a 5x5 clump far away."""
    mask = torch.zeros(height, width)
    mask[30:60, 40:80] = 1.0
    mask[2:4, 2:4] = 1.0
    mask[90:95, 110:115] = 1.0
    return mask


def _bfs_component_count(binary: np.ndarray) -> int:
    seen = np.zeros_like(binary, dtype=bool)
    count = 0
    height, width = binary.shape
    for start in zip(*np.nonzero(binary)):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    r, c = row + d_row, col + d_col
                    if 0 <= r < height and 0 <= c < width and binary[r, c] and not seen[r, c]:
                        seen[r, c] = True
                        queue.append((r, c))
    return count


def test_node_contract():
    assert NODE_CLASS_MAPPINGS == {"MaskBoundingBox": MaskBoundingBox}
    assert NODE_DISPLAY_NAME_MAPPINGS == {"MaskBoundingBox": "Bounding Box"}
    assert MaskBoundingBox.RETURN_TYPES == ("INT", "INT", "INT", "INT", "MASK", "MASK")
    assert MaskBoundingBox.RETURN_NAMES == ("x", "y", "width", "height", "filtered_mask", "bounded_mask")
    assert MaskBoundingBox.OUTPUT_IS_LIST == (True, True, True, True, False, True)
    inputs = MaskBoundingBox.INPUT_TYPES()
    assert set(inputs["required"]) == {"mask", "threshold", "noise_filter"}
    assert set(inputs["optional"]) == {"merge_distance", "secondary_blob_ratio", "min_clump_ratio", "padding"}


def test_plain_box_covers_every_pixel_above_threshold():
    assert _box(_scene()) == BoundingBox(x=2, y=2, width=118 - 5, height=95 - 2)


def test_threshold_is_strict():
    mask = torch.zeros(10, 10)
    mask[2, 3] = 0.5
    mask[6, 7] = 0.51

    assert _box(mask, threshold=0.5) == BoundingBox(7, 6, 1, 1)
    assert _box(mask, threshold=0.49) == BoundingBox(3, 2, 5, 5)


def test_main_blob_ignores_stray_specks_and_clumps():
    box = _box(_scene(), noise_filter="main_blob", merge_distance=8)

    assert box == BoundingBox(x=40, y=30, width=40, height=30)


def test_main_blob_prefers_the_blob_with_most_mask_value_over_a_dense_speck():
    mask = torch.zeros(60, 60)
    mask[10:40, 10:40] = 0.6  # large but dim
    mask[50:53, 50:53] = 1.0  # small but bright

    assert _box(mask, noise_filter="main_blob", merge_distance=0, threshold=0.5) == BoundingBox(10, 10, 30, 30)


def test_merge_distance_keeps_a_fragmented_subject_together():
    mask = torch.zeros(40, 80)
    mask[10:30, 10:30] = 1.0
    mask[10:30, 34:50] = 1.0  # 4-pixel gap from the first piece; smaller than it on its own

    assert _box(mask, noise_filter="main_blob", merge_distance=3) == BoundingBox(10, 10, 20, 20)
    assert _box(mask, noise_filter="main_blob", merge_distance=4) == BoundingBox(10, 10, 40, 20)


@pytest.mark.parametrize("gap", [0, 1, 2, 3, 4, 7, 10])
def test_dilate_for_gap_bridges_exactly_the_requested_gap(gap):
    for spacing in (gap, gap + 1):
        row = torch.zeros(1, 1, 40, dtype=torch.bool)
        row[0, 0, 5] = True
        row[0, 0, 5 + spacing + 1] = True
        grown = dilate_for_gap(row, gap)[0].numpy()
        _, count = label_components(grown)
        assert count == (1 if spacing <= gap else 2), (gap, spacing)


def test_dilate_for_gap_measures_diagonal_gaps_like_straight_ones():
    grid = torch.zeros(1, 20, 20, dtype=torch.bool)
    grid[0, 5, 5] = True
    grid[0, 9, 9] = True  # three empty pixels between them along the diagonal

    assert label_components(dilate_for_gap(grid, 3)[0].numpy())[1] == 1
    assert label_components(dilate_for_gap(grid, 2)[0].numpy())[1] == 2


def test_secondary_blob_ratio_keeps_comparable_blobs():
    mask = torch.zeros(50, 100)
    mask[10:30, 10:30] = 1.0  # score 400
    mask[10:25, 70:90] = 1.0  # score 300
    mask[45:47, 0:2] = 1.0  # score 4

    assert _box(mask, noise_filter="main_blob", merge_distance=2) == BoundingBox(10, 10, 20, 20)
    assert _box(
        mask, noise_filter="main_blob", merge_distance=2, secondary_blob_ratio=0.5
    ) == BoundingBox(10, 10, 80, 20)
    assert _box(
        mask, noise_filter="main_blob", merge_distance=2, secondary_blob_ratio=0.0
    ) == BoundingBox(0, 10, 90, 37)


def test_min_clump_ratio_stops_scattered_specks_chaining_into_the_main_blob():
    mask = torch.zeros(200, 200)
    mask[::10, ::10] = 1.0  # a grid of single-pixel specks every 10 pixels across the whole mask
    mask[82:118, 82:118] = 1.0  # placed so no speck touches its edge

    filtered = _box(mask, noise_filter="main_blob", merge_distance=12)
    chained = _box(mask, noise_filter="main_blob", merge_distance=12, min_clump_ratio=0.0)

    assert filtered == BoundingBox(82, 82, 36, 36)
    assert chained == BoundingBox(0, 0, 191, 191)


def test_min_clump_ratio_is_relative_so_all_small_dots_still_count():
    mask = torch.zeros(60, 60)
    mask[10:30:3, 10:40:3] = 1.0  # a dithered patch made only of single pixels
    mask[55, 2] = 1.0  # one lone dot far away

    assert _box(mask, noise_filter="main_blob", merge_distance=4) == BoundingBox(10, 10, 28, 19)


def test_padding_grows_the_box_and_clamps_to_edges():
    mask = torch.zeros(20, 30)
    mask[1:5, 20:28] = 1.0

    assert _box(mask, padding=3) == BoundingBox(17, 0, 13, 8)


def test_empty_mask_returns_empty_box_and_warns(caplog):
    with caplog.at_level("WARNING", logger="src.mask_bounding_box"):
        box = _box(torch.zeros(8, 8), noise_filter="main_blob")

    assert box == BoundingBox(0, 0, 0, 0)
    assert box.is_empty
    assert "no pixels above threshold" in caplog.text


def test_node_returns_lists_per_batch_item_and_filtered_mask():
    batch = torch.zeros(2, 100, 120)
    batch[0] = _scene()
    batch[1, 5:15, 6:26] = 0.8

    x, y, width, height, filtered, _ = MaskBoundingBox().find_box(
        batch, threshold=0.5, noise_filter="main_blob", merge_distance=8
    )

    assert (x, y, width, height) == ([40, 6], [30, 5], [40, 20], [30, 10])
    assert filtered.shape == (2, 100, 120)
    assert float(filtered[0, 3, 3]) == 0.0  # stray speck removed
    assert float(filtered[0, 40, 50]) == 1.0
    assert torch.allclose(filtered[1], batch[1])  # original values kept


def test_bounded_mask_is_the_input_cropped_to_each_box():
    batch = torch.zeros(2, 100, 120)
    batch[0] = _scene()
    batch[0, 45, 60] = 0.3  # below threshold, but inside the box, so it stays in the crop
    batch[1, 5:15, 6:26] = 0.8

    x, y, width, height, _, bounded = MaskBoundingBox().find_box(
        batch, threshold=0.5, noise_filter="main_blob", merge_distance=8, padding=2
    )

    assert [tuple(item.shape) for item in bounded] == [(1, 34, 44), (1, 14, 24)]
    for index, item in enumerate(bounded):
        expected = batch[index, y[index] : y[index] + height[index], x[index] : x[index] + width[index]]
        assert torch.equal(item[0], expected)
    assert float(bounded[0][0, 45 - y[0], 60 - x[0]]) == pytest.approx(0.3)


def test_bounded_mask_crop_does_not_alias_the_input():
    mask = torch.zeros(10, 10)
    mask[2:5, 3:7] = 1.0

    crop = crop_mask_to_box(mask, BoundingBox(3, 2, 4, 3))
    crop.zero_()

    assert float(mask.sum()) == 12.0


def test_bounded_mask_for_empty_box_is_a_single_zero_pixel():
    *_, bounded = MaskBoundingBox().find_box(torch.zeros(1, 8, 8), threshold=0.5, noise_filter="none")

    assert len(bounded) == 1
    assert bounded[0].shape == (1, 1, 1)
    assert float(bounded[0].sum()) == 0.0


@pytest.mark.parametrize("shape", [(30, 40), (1, 30, 40), (1, 1, 30, 40)])
def test_accepts_common_mask_shapes(shape):
    mask = torch.zeros(shape)
    mask[..., 10:20, 5:15] = 1.0

    assert _box(mask) == BoundingBox(5, 10, 10, 10)


def test_label_components_uses_8_connectivity():
    binary = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 0, 1],
        ],
        dtype=bool,
    )

    labels, count = label_components(binary)

    assert count == 4
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 3] == labels[1, 3]
    assert len({labels[0, 0], labels[0, 3], labels[3, 0], labels[3, 3]}) == 4


@pytest.mark.parametrize("seed", range(8))
def test_label_components_matches_flood_fill_on_random_masks(seed):
    rng = np.random.default_rng(seed)
    binary = rng.random((37, 53)) < (0.3 + 0.05 * seed)

    labels, count = label_components(binary)

    assert count == _bfs_component_count(binary)
    assert ((labels > 0) == binary).all()
    for label in range(1, count + 1):
        assert _bfs_component_count(labels == label) == 1


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="noise_filter"):
        mask_bounding_boxes(torch.zeros(4, 4), noise_filter="biggest")
    with pytest.raises(ValueError, match="secondary_blob_ratio"):
        mask_bounding_boxes(torch.zeros(4, 4), secondary_blob_ratio=1.5)
    with pytest.raises(ValueError, match="min_clump_ratio"):
        mask_bounding_boxes(torch.zeros(4, 4), min_clump_ratio=-0.1)
    with pytest.raises(ValueError, match="shape"):
        mask_bounding_boxes(torch.zeros(1, 3, 4, 4))
