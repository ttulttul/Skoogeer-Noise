"""Tests for the Random Float node."""

import math
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.random_float import MAX_RANDOM_FLOATS, RandomFloat, random_floats  # noqa: E402


def test_random_float_declares_list_valued_float_output() -> None:
    """The output is a FLOAT list and the inputs are seed, min, max, and count."""
    assert RandomFloat.RETURN_TYPES == ("FLOAT",)
    assert RandomFloat.OUTPUT_IS_LIST == (True,)
    assert list(RandomFloat.INPUT_TYPES()["required"]) == ["seed", "min", "max", "count"]


def test_random_float_count_one_emits_a_single_value_in_range() -> None:
    """count=1 gives one value, which ComfyUI treats as a single FLOAT."""
    (values,) = RandomFloat().generate(seed=42, min=-2.0, max=3.0, count=1)

    assert len(values) == 1
    assert -2.0 <= values[0] <= 3.0


def test_random_float_count_emits_list_in_range() -> None:
    """A larger count gives that many values, all within the bounds and not all equal."""
    values = random_floats(7, 0.25, 0.75, 500)

    assert len(values) == 500
    assert all(0.25 <= value <= 0.75 for value in values)
    assert len(set(values)) > 1


def test_random_float_is_deterministic_per_seed() -> None:
    """The same seed repeats the values and a different seed changes them."""
    assert random_floats(123, 0.0, 1.0, 8) == random_floats(123, 0.0, 1.0, 8)
    assert random_floats(123, 0.0, 1.0, 8) != random_floats(124, 0.0, 1.0, 8)


def test_random_float_longer_count_extends_the_same_stream() -> None:
    """Raising count keeps the earlier values unchanged."""
    assert random_floats(9, 0.0, 1.0, 10)[:3] == random_floats(9, 0.0, 1.0, 3)


def test_random_float_accepts_reversed_and_equal_bounds() -> None:
    """min above max still gives values between them; equal bounds give a constant."""
    assert all(1.0 <= value <= 5.0 for value in random_floats(1, 5.0, 1.0, 100))
    assert random_floats(1, 0.5, 0.5, 3) == [0.5, 0.5, 0.5]


def test_random_float_accepts_full_64_bit_seed() -> None:
    """The largest ComfyUI seed works."""
    assert len(random_floats(2**64 - 1, 0.0, 1.0, 2)) == 2


@pytest.mark.parametrize("count", [0, -1, MAX_RANDOM_FLOATS + 1])
def test_random_float_rejects_out_of_range_count(count: int) -> None:
    """count must be between 1 and the limit."""
    with pytest.raises(ValueError, match="count"):
        random_floats(0, 0.0, 1.0, count)


@pytest.mark.parametrize("bounds", [(math.inf, 1.0), (0.0, math.nan)])
def test_random_float_rejects_non_finite_bounds(bounds: tuple) -> None:
    """Infinite or NaN bounds raise an error."""
    with pytest.raises(ValueError, match="finite"):
        random_floats(0, bounds[0], bounds[1], 1)
