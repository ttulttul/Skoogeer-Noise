"""Float Range: emit an evenly spaced list of floats for list-mapped downstream nodes."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_RANGE_ITEMS = 10000
"""Upper bound on emitted items so a tiny step cannot flood the queue."""

_ROUND_DIGITS = 10
"""Rounding applied to each value to remove float noise such as 0.30000000000000004."""

_STOP_TOLERANCE = 1e-9
"""Relative tolerance for deciding that the last step lands exactly on ``stop``."""

_WIDGET_LIMIT = 1.0e9


def _step_count(start: float, stop: float, step: float, include_stop: bool) -> int:
    """Return how many values the range holds, counting ``start`` itself."""
    if step == 0.0 or not math.isfinite(step):
        raise ValueError(f"Float Range step must be a finite non-zero number, got {step}.")
    if not (math.isfinite(start) and math.isfinite(stop)):
        raise ValueError(f"Float Range start and stop must be finite, got start={start} stop={stop}.")
    if start == stop:
        return 1 if include_stop else 0
    if (stop - start) * step < 0:
        raise ValueError(
            f"Float Range step={step} moves away from stop: start={start} stop={stop}. "
            "Use a negative step to count down."
        )
    exact_steps = (stop - start) / step
    whole_steps = math.floor(exact_steps + _STOP_TOLERANCE)
    lands_on_stop = abs(exact_steps - round(exact_steps)) <= _STOP_TOLERANCE * max(1.0, abs(exact_steps))
    if lands_on_stop and not include_stop:
        return whole_steps
    return whole_steps + 1


def float_range(
    start: float,
    stop: float,
    step: float,
    *,
    include_stop: bool = True,
    repeat_each: int = 1,
    repeat_list: int = 1,
) -> List[float]:
    """Build the evenly spaced values, optionally repeating each value or the whole list.

    ``repeat_each`` turns ``[a, b]`` into ``[a, a, b, b]`` and ``repeat_list`` turns it
    into ``[a, b, a, b]``. Together they let two list inputs line up as a full grid,
    because ComfyUI pairs list inputs by position rather than forming every combination.
    """
    if repeat_each < 1 or repeat_list < 1:
        raise ValueError(
            f"Float Range repeats must be at least 1, got repeat_each={repeat_each} repeat_list={repeat_list}."
        )
    count = _step_count(float(start), float(stop), float(step), include_stop)
    total = count * repeat_each * repeat_list
    if total > MAX_RANGE_ITEMS:
        raise ValueError(
            f"Float Range would emit {total} values, more than the {MAX_RANGE_ITEMS} limit. "
            "Use a larger step or fewer repeats."
        )
    base = [round(float(start) + index * float(step), _ROUND_DIGITS) for index in range(count)]
    expanded = [value for value in base for _ in range(repeat_each)]
    return expanded * repeat_list


class FloatRange:
    """List-valued FLOAT source, like Python's ``range`` for floats."""

    CATEGORY = "utils/list"
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("values", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "build_range"
    DESCRIPTION = (
        "Emits evenly spaced floats from start toward stop as a list, so downstream nodes "
        "such as KSampler run once per value (for example a denoise sweep)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        """Declare the range bounds, step, and repeat controls."""
        float_widget = {"min": -_WIDGET_LIMIT, "max": _WIDGET_LIMIT, "step": 0.01, "round": False}
        return {
            "required": {
                "start": ("FLOAT", {**float_widget, "default": 0.0, "tooltip": "First value in the list."}),
                "stop": ("FLOAT", {**float_widget, "default": 1.0, "tooltip": "Value the list counts toward."}),
                "step": (
                    "FLOAT",
                    {
                        **float_widget,
                        "default": 0.1,
                        "tooltip": "Spacing between values. Use a negative step to count down from start to stop.",
                    },
                ),
                "include_stop": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Include stop when a step lands exactly on it (0.3 to 0.9 by 0.1 gives 7 values).",
                    },
                ),
            },
            "optional": {
                "repeat_each": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_RANGE_ITEMS,
                        "tooltip": "Repeat every value this many times in a row: [a, b] becomes [a, a, b, b].",
                    },
                ),
                "repeat_list": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_RANGE_ITEMS,
                        "tooltip": "Repeat the whole list this many times: [a, b] becomes [a, b, a, b].",
                    },
                ),
            },
        }

    def build_range(
        self,
        start: float,
        stop: float,
        step: float,
        include_stop: bool,
        repeat_each: int = 1,
        repeat_list: int = 1,
    ) -> Tuple[List[float], int]:
        """Return the float list and its length."""
        values = float_range(
            start,
            stop,
            step,
            include_stop=bool(include_stop),
            repeat_each=int(repeat_each),
            repeat_list=int(repeat_list),
        )
        logger.info(
            "Float Range emitted %d values start=%s stop=%s step=%s include_stop=%s repeat_each=%d repeat_list=%d.",
            len(values),
            start,
            stop,
            step,
            include_stop,
            repeat_each,
            repeat_list,
        )
        return values, len(values)


NODE_CLASS_MAPPINGS = {
    "FloatRange": FloatRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatRange": "Float Range",
}
