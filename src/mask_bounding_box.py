"""Bounding Box: find the rectangle around a mask's active pixels, optionally ignoring stray clumps."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

FILTER_NONE = "none"
FILTER_MAIN_BLOB = "main_blob"
FILTER_MODES = (FILTER_NONE, FILTER_MAIN_BLOB)

MAX_MERGE_DISTANCE = 1024
MAX_PADDING = 16384


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned box in mask pixels. ``width == height == 0`` means nothing was found."""

    x: int
    y: int
    width: int
    height: int

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0


def _masks_bhw(mask: torch.Tensor) -> torch.Tensor:
    """Return ``mask`` as a float32 ``(B,H,W)`` tensor on its original device."""
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"MASK input must be a torch.Tensor, got {type(mask)}.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and int(mask.shape[1]) == 1:
        mask = mask[:, 0]
    elif mask.ndim != 3:
        raise ValueError(f"Expected mask tensor with shape (H,W), (B,H,W), or (B,1,H,W), got {tuple(mask.shape)}.")
    return mask.to(dtype=torch.float32)


def _window_sum_1d(values: torch.Tensor, before: int, after: int, dim: int) -> torch.Tensor:
    """Sum ``values`` over the window ``[i - before, i + after]`` along ``dim``, clipped at the edges."""
    length = int(values.shape[dim])
    shape = list(values.shape)
    shape[dim] = 1
    cumulative = torch.cat([torch.zeros(shape, dtype=values.dtype, device=values.device), values.cumsum(dim)], dim)
    positions = torch.arange(length, device=values.device)
    upper = (positions + after + 1).clamp(max=length)
    lower = (positions - before).clamp(min=0)
    return cumulative.index_select(dim, upper) - cumulative.index_select(dim, lower)


def dilate_for_gap(active: torch.Tensor, gap: int) -> torch.Tensor:
    """Grow a boolean ``(B,H,W)`` mask so that clumps separated by up to ``gap`` empty pixels touch.

    The growth uses a square ``(gap+1) x (gap+1)`` window, so the gap is measured along rows,
    columns, and diagonals alike. Window sums come from running totals, so the cost does not
    depend on ``gap``.
    """
    gap = int(gap)
    if gap <= 0:
        return active
    before = gap // 2
    after = gap - before
    counts = active.to(dtype=torch.int64)
    counts = _window_sum_1d(counts, before, after, dim=-1)
    counts = _window_sum_1d(counts, before, after, dim=-2)
    return counts > 0


def label_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label 8-connected regions of a 2-D boolean array.

    Returns ``(labels, count)`` where background is ``0`` and regions are numbered ``1..count``
    in row-major order of their first pixel. Works on horizontal runs with union-find, so the
    Python-level work scales with the number of runs rather than the number of pixels.
    """
    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"label_components expects a 2-D array, got shape {binary.shape}.")
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if not binary.any():
        return labels, 0

    edges = np.diff(np.pad(binary, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    run_rows, run_starts = np.nonzero(edges == 1)
    _, run_ends = np.nonzero(edges == -1)  # exclusive; row-major order keeps them paired with starts
    run_count = len(run_rows)
    row_offsets = np.searchsorted(run_rows, np.arange(height + 1))

    parent = list(range(run_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    starts = run_starts.tolist()
    ends = run_ends.tolist()
    offsets = row_offsets.tolist()
    for row in range(1, height):
        above, above_end = offsets[row - 1], offsets[row]
        current, current_end = offsets[row], offsets[row + 1]
        while above < above_end and current < current_end:
            # 8-connectivity: runs touch when they overlap or meet diagonally.
            if starts[above] <= ends[current] and starts[current] <= ends[above]:
                root_above, root_current = find(above), find(current)
                if root_above != root_current:
                    if root_above < root_current:
                        parent[root_current] = root_above
                    else:
                        parent[root_above] = root_current
            if ends[above] < ends[current]:
                above += 1
            else:
                current += 1

    roots = np.fromiter((find(index) for index in range(run_count)), dtype=np.int64, count=run_count)
    _, run_labels = np.unique(roots, return_inverse=True)
    run_labels = run_labels.astype(np.int32) + 1
    # np.nonzero(binary) visits pixels in the same row-major order as the runs.
    labels[binary] = np.repeat(run_labels, run_ends - run_starts)
    return labels, int(run_labels.max())


def _keep_by_score(
    values: np.ndarray,
    active: np.ndarray,
    regions: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, int, int]:
    """Keep active pixels whose region of ``regions`` scores at least ``ratio`` of the best region.

    A region's score is the sum of the mask values of its active pixels.
    """
    labels, count = label_components(regions)
    scores = np.bincount(labels[active], weights=values[active].astype(np.float64), minlength=count + 1)
    scores[0] = 0.0
    keep = scores >= float(scores.max()) * float(ratio) * (1.0 - 1e-9)
    keep[0] = False
    return active & keep[labels], count, int(keep.sum())


def select_main_blob(
    values: np.ndarray,
    active: np.ndarray,
    *,
    merge_distance: int,
    secondary_blob_ratio: float,
    min_clump_ratio: float,
) -> np.ndarray:
    """Keep the active pixels that belong to the blob with the most mask value.

    1. Split the active pixels into clumps (8-connected pieces) and drop crumbs scoring less
       than ``min_clump_ratio`` of the biggest clump, so scattered specks cannot chain together.
    2. Group the remaining clumps that lie within ``merge_distance`` empty pixels of each other
       into blobs, so a subject broken into pieces counts as one.
    3. Keep the blob with the highest score (total mask value), plus any blob scoring at least
       ``secondary_blob_ratio`` of it.
    """
    if not active.any():
        return active
    survivors, clump_count, kept_clumps = _keep_by_score(values, active, active, min_clump_ratio)
    grouping = survivors
    if merge_distance > 0:
        grouping = dilate_for_gap(torch.from_numpy(survivors)[None], merge_distance)[0].numpy()
    kept, blob_count, kept_blobs = _keep_by_score(values, survivors, grouping, secondary_blob_ratio)
    logger.debug(
        "Bounding Box kept %d/%d clumps, grouped into %d blobs, and kept %d blob(s).",
        kept_clumps,
        clump_count,
        blob_count,
        kept_blobs,
    )
    return kept


def bounding_box_of(active: np.ndarray, *, padding: int = 0) -> BoundingBox:
    """Return the box around the ``True`` pixels of a 2-D array, grown by ``padding`` and clamped."""
    rows = np.flatnonzero(active.any(axis=1))
    if rows.size == 0:
        return BoundingBox(0, 0, 0, 0)
    cols = np.flatnonzero(active.any(axis=0))
    height, width = active.shape
    top = max(int(rows[0]) - padding, 0)
    left = max(int(cols[0]) - padding, 0)
    bottom = min(int(rows[-1]) + padding, height - 1)
    right = min(int(cols[-1]) + padding, width - 1)
    return BoundingBox(left, top, right - left + 1, bottom - top + 1)


def mask_bounding_boxes(
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    noise_filter: str = FILTER_NONE,
    merge_distance: int = 16,
    secondary_blob_ratio: float = 1.0,
    min_clump_ratio: float = 0.01,
    padding: int = 0,
) -> Tuple[List[BoundingBox], torch.Tensor]:
    """Compute one bounding box per mask in the batch, plus the mask of the pixels that were bounded."""
    if noise_filter not in FILTER_MODES:
        raise ValueError(f"noise_filter must be one of {FILTER_MODES}, got {noise_filter!r}.")
    if merge_distance < 0 or padding < 0:
        raise ValueError(f"merge_distance and padding must be >= 0, got {merge_distance} and {padding}.")
    for name, ratio in (("secondary_blob_ratio", secondary_blob_ratio), ("min_clump_ratio", min_clump_ratio)):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {ratio}.")

    masks = _masks_bhw(mask)
    values_all = masks.detach().cpu().numpy()
    active_np = values_all > float(threshold)

    boxes: List[BoundingBox] = []
    kept_layers: List[np.ndarray] = []
    for index in range(values_all.shape[0]):
        active = active_np[index]
        if noise_filter == FILTER_MAIN_BLOB:
            active = select_main_blob(
                values_all[index],
                active,
                merge_distance=merge_distance,
                secondary_blob_ratio=secondary_blob_ratio,
                min_clump_ratio=min_clump_ratio,
            )
        box = bounding_box_of(active, padding=padding)
        if box.is_empty:
            logger.warning(
                "Bounding Box: mask %d has no pixels above threshold %.3f; returning an empty box (0, 0, 0, 0).",
                index,
                threshold,
            )
        boxes.append(box)
        kept_layers.append(active)

    kept = torch.from_numpy(np.stack(kept_layers)).to(device=masks.device)
    filtered = torch.where(kept, masks, torch.zeros_like(masks))
    return boxes, filtered


class MaskBoundingBox:
    """Bounding box of a mask's above-threshold pixels, with optional removal of stray clumps."""

    CATEGORY = "Mask/Analysis"
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "MASK")
    RETURN_NAMES = ("x", "y", "width", "height", "filtered_mask")
    OUTPUT_IS_LIST = (True, True, True, True, False)
    FUNCTION = "find_box"
    DESCRIPTION = (
        "Finds the box around every mask pixel above threshold: x and y of the upper-left corner, "
        "width, and height. With noise_filter = main_blob it ignores stray specks and clumps and "
        "bounds only the blob holding the most mask value."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to measure. Batches give one box per mask."}),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Pixels with a value strictly greater than this count as part of the mask.",
                }),
                "noise_filter": (list(FILTER_MODES), {
                    "default": FILTER_NONE,
                    "tooltip": (
                        "none bounds every pixel above threshold. main_blob groups nearby pixels into blobs "
                        "and bounds only the blob with the most total mask value, ignoring stray clumps."
                    ),
                }),
            },
            "optional": {
                "merge_distance": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": MAX_MERGE_DISTANCE,
                    "tooltip": (
                        "main_blob only. Clumps separated by up to this many empty pixels count as one blob, "
                        "so a subject broken into pieces stays together. 0 means pixels must touch."
                    ),
                }),
                "secondary_blob_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "main_blob only. Also keep other blobs whose total mask value is at least this fraction "
                        "of the main blob's. 1.0 keeps only the main blob; 0.25 also keeps blobs a quarter its size."
                    ),
                }),
                "min_clump_ratio": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": (
                        "main_blob only. Before grouping, drop connected clumps whose total mask value is below "
                        "this fraction of the biggest clump's, so scattered specks cannot chain into the main blob. "
                        "0 keeps every clump."
                    ),
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_PADDING,
                    "tooltip": "Grow the box by this many pixels on every side, clamped to the mask edges.",
                }),
            },
        }

    def find_box(
        self,
        mask: torch.Tensor,
        threshold: float,
        noise_filter: str,
        merge_distance: int = 16,
        secondary_blob_ratio: float = 1.0,
        min_clump_ratio: float = 0.01,
        padding: int = 0,
    ):
        boxes, filtered = mask_bounding_boxes(
            mask,
            threshold=float(threshold),
            noise_filter=str(noise_filter),
            merge_distance=int(merge_distance),
            secondary_blob_ratio=float(secondary_blob_ratio),
            min_clump_ratio=float(min_clump_ratio),
            padding=int(padding),
        )
        logger.info(
            "Bounding Box (%s, threshold=%.3f) found %s.",
            noise_filter,
            threshold,
            ", ".join(f"({box.x}, {box.y}, {box.width}x{box.height})" for box in boxes),
        )
        return (
            [box.x for box in boxes],
            [box.y for box in boxes],
            [box.width for box in boxes],
            [box.height for box in boxes],
            filtered,
        )


NODE_CLASS_MAPPINGS = {
    "MaskBoundingBox": MaskBoundingBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBoundingBox": "Bounding Box",
}
