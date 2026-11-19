"""Random Float: emit one or more seeded random floats between ``min`` and ``max``."""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_RANDOM_FLOATS = 10000
"""Upper bound on emitted items so a large count cannot flood the queue."""

_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF

_WIDGET_LIMIT = 1.0e9


def random_floats(seed: int, minimum: float, maximum: float, count: int) -> List[float]:
    """Return ``count`` uniform random floats between ``minimum`` and ``maximum``.

    Values come from a local ``random.Random`` seeded with ``seed``, so the same inputs
    always give the same list and the first values do not change when ``count`` grows.
    ``minimum`` may be larger than ``maximum``; values still fall between the two.
    """
    minimum = float(minimum)
    maximum = float(maximum)
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError(f"Random Float min and max must be finite, got min={minimum} max={maximum}.")
    count = int(count)
    if count < 1 or count > MAX_RANDOM_FLOATS:
        raise ValueError(f"Random Float count must be between 1 and {MAX_RANDOM_FLOATS}, got {count}.")
    rng = random.Random(int(seed) & _SEED_MASK_64)
    return [rng.uniform(minimum, maximum) for _ in range(count)]


class RandomFloat:
    """Seeded random FLOAT source that emits one value or a list of values."""

    CATEGORY = "utils/random"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("values",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    DESCRIPTION = (
        "Emits seeded random floats between min and max. With count 1 it acts as a single "
        "FLOAT; with a larger count, downstream nodes run once per value."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        """Declare the seed, bounds, and count."""
        float_widget = {"min": -_WIDGET_LIMIT, "max": _WIDGET_LIMIT, "step": 0.01, "round": False}
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _SEED_MASK_64,
                        "tooltip": "Seed for the random values. The same seed always gives the same values.",
                    },
                ),
                "min": ("FLOAT", {**float_widget, "default": 0.0, "tooltip": "Lower bound of the random values."}),
                "max": ("FLOAT", {**float_widget, "default": 1.0, "tooltip": "Upper bound of the random values."}),
                "count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_RANDOM_FLOATS,
                        "tooltip": "How many values to emit. Above 1, the output is a list and downstream nodes run once per value.",
                    },
                ),
            },
        }

    def generate(self, seed: int, min: float, max: float, count: int) -> Tuple[List[float]]:  # noqa: A002
        """Return the random float list."""
        values = random_floats(seed, min, max, count)
        logger.info("Random Float emitted %d values seed=%s min=%s max=%s.", len(values), seed, min, max)
        return (values,)


NODE_CLASS_MAPPINGS = {
    "RandomFloat": RandomFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomFloat": "Random Float",
}
