"""Tests for the Float Range list node."""

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.float_range import MAX_RANGE_ITEMS, FloatRange, float_range  # noqa: E402


def test_float_range_declares_list_valued_float_output() -> None:
    """The values socket is a FLOAT list and count is a single INT."""
    assert FloatRange.RETURN_TYPES == ("FLOAT", "INT")
    assert FloatRange.OUTPUT_IS_LIST == (True, False)
    inputs = FloatRange.INPUT_TYPES()
    assert set(inputs["required"]) == {"start", "stop", "step", "include_stop"}
    assert set(inputs["optional"]) == {"repeat_each", "repeat_list"}


def test_float_range_includes_stop_without_float_noise() -> None:
    """0.3 to 0.9 by 0.1 gives exactly seven clean values."""
    values, count = FloatRange().build_range(0.3, 0.9, 0.1, True)

    assert values == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert count == 7


def test_float_range_can_exclude_stop() -> None:
    """With include_stop off it behaves like Python's half-open range."""
    assert float_range(0.0, 1.0, 0.25, include_stop=False) == [0.0, 0.25, 0.5, 0.75]


def test_float_range_stops_before_an_overshooting_step() -> None:
    """A step that does not land on stop never goes past it."""
    assert float_range(0.0, 1.0, 0.3) == [0.0, 0.3, 0.6, 0.9]
    assert float_range(0.0, 1.0, 0.3, include_stop=False) == [0.0, 0.3, 0.6, 0.9]


def test_float_range_counts_down_with_negative_step() -> None:
    """A negative step walks from a larger start to a smaller stop."""
    assert float_range(1.0, 0.5, -0.25) == [1.0, 0.75, 0.5]


def test_float_range_single_value_when_start_equals_stop() -> None:
    """Equal bounds give one value, or none when stop is excluded."""
    assert float_range(0.5, 0.5, 0.1) == [0.5]
    assert float_range(0.5, 0.5, 0.1, include_stop=False) == []


def test_float_range_repeats_build_a_grid() -> None:
    """repeat_each and repeat_list produce the two halves of a cross product."""
    assert float_range(0.0, 1.0, 1.0, repeat_each=2) == [0.0, 0.0, 1.0, 1.0]
    assert float_range(0.0, 1.0, 1.0, repeat_list=2) == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [(0.0, 1.0, 0.0), (0.0, 1.0, -0.1), (1.0, 0.0, 0.1), (0.0, float("inf"), 0.1)],
)
def test_float_range_rejects_steps_that_never_reach_stop(start: float, stop: float, step: float) -> None:
    """Zero, wrong-direction, and non-finite inputs raise instead of looping forever."""
    with pytest.raises(ValueError):
        float_range(start, stop, step)


def test_float_range_rejects_oversized_lists() -> None:
    """A tiny step that would emit too many values raises."""
    with pytest.raises(ValueError, match="limit"):
        float_range(0.0, 1.0, 1.0 / (MAX_RANGE_ITEMS * 2))


def test_float_range_rejects_non_positive_repeats() -> None:
    """Repeats below one are invalid."""
    with pytest.raises(ValueError):
        float_range(0.0, 1.0, 0.5, repeat_each=0)
