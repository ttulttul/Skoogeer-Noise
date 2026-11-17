"""Tests for the Image Compare (Side by Side) node."""

import os
import pathlib
import shutil
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make ComfyUI's v3 API importable when a checkout is available, so the node class is built.
_COMFYUI_PATH = pathlib.Path(os.environ.get("COMFYUI_PATH", PROJECT_ROOT.parent / "ComfyUI"))
if (_COMFYUI_PATH / "comfy_api").is_dir() and str(_COMFYUI_PATH) not in sys.path:
    sys.path.append(str(_COMFYUI_PATH))

from src import image_compare  # noqa: E402
from src.image_compare import (  # noqa: E402
    BUILTIN_FONT,
    compare_image_batches,
    default_font_choice,
    discover_fonts,
    load_font,
)

_DARK = 0.0


def _solid(batch: int, height: int, width: int, value: float) -> torch.Tensor:
    return torch.full((batch, height, width, 3), value, dtype=torch.float32)


def _find_system_font() -> pathlib.Path:
    for directory in image_compare.system_font_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in (".ttf", ".otf"):
                return path
    pytest.skip("No .ttf/.otf font installed on this machine.")


def test_unlabelled_horizontal_layout_places_images_with_gap() -> None:
    """Without labels the output is the two images plus the gap, nothing else."""
    out = compare_image_batches(_solid(1, 32, 40, 1.0), _solid(1, 32, 40, 0.5), "", "", gap=6)

    assert out.shape == (1, 32, 40 + 6 + 40, 3)
    assert torch.allclose(out[0, :, :40], torch.ones(32, 40, 3))
    assert torch.allclose(out[0, :, 40:46], torch.full((32, 6, 3), _DARK))
    assert torch.allclose(out[0, :, 46:], torch.full((32, 40, 3), 0.5), atol=1 / 255)


def test_vertical_layout_stacks_images_with_border() -> None:
    """Vertical layout stacks the images and border pads every side."""
    out = compare_image_batches(
        _solid(1, 20, 30, 1.0), _solid(1, 20, 30, 1.0), "", "", layout="vertical", gap=4, border=3
    )

    assert out.shape == (1, 3 + 20 + 4 + 20 + 3, 3 + 30 + 3, 3)


def test_labels_add_a_panel_with_text_below_each_image() -> None:
    """A labelled comparison is taller than the images and the panel contains text pixels."""
    out = compare_image_batches(_solid(1, 64, 64, 0.5), _solid(1, 64, 64, 0.5), "A", "B", font_size=20, gap=0)

    assert out.shape[1] > 64 and out.shape[2] == 128
    label_panel = out[0, 64:]
    assert label_panel.max() > 0.9  # white text on the dark theme


def test_label_position_top_puts_image_at_the_bottom() -> None:
    """With labels on top, the image rows sit after the panel."""
    out = compare_image_batches(
        _solid(1, 64, 64, 0.5), _solid(1, 64, 64, 0.5), "A", "B", font_size=20, label_position="top", gap=0
    )

    assert torch.allclose(out[0, -64:, :64], torch.full((64, 64, 3), 0.5), atol=1 / 255)


def test_label_position_none_hides_labels() -> None:
    """label_position none produces the same size as empty labels."""
    out = compare_image_batches(_solid(1, 32, 32, 1.0), _solid(1, 32, 32, 1.0), "A", "B", label_position="none", gap=0)

    assert out.shape == (1, 32, 64, 3)


def test_light_theme_uses_white_background() -> None:
    """The gap is painted with the theme background."""
    out = compare_image_batches(_solid(1, 8, 8, 0.0), _solid(1, 8, 8, 0.0), "", "", gap=2, theme="light")

    assert torch.allclose(out[0, :, 8:10], torch.ones(8, 2, 3))


@pytest.mark.parametrize(
    ("resize", "expected_width"),
    [("fit", 40 + 80), ("stretch", 40 + 40), ("none", 40 + 160)],
)
def test_resize_modes_size_image_b_against_image_a(resize: str, expected_width: int) -> None:
    """fit keeps B's aspect at A's height, stretch copies A's size, none leaves B alone."""
    out = compare_image_batches(_solid(1, 40, 40, 1.0), _solid(1, 80, 160, 1.0), "", "", resize=resize, gap=0)

    assert out.shape[2] == expected_width
    assert out.shape[1] == (80 if resize == "none" else 40)


def test_singleton_batch_is_paired_with_every_image() -> None:
    """A single image on one side is compared against each image in the other batch."""
    out = compare_image_batches(_solid(1, 16, 16, 1.0), _solid(3, 16, 16, 0.0), "", "", gap=0)

    assert out.shape[0] == 3


def test_mismatched_batches_raise() -> None:
    """Two batches of different sizes greater than one cannot be paired."""
    with pytest.raises(ValueError, match="cannot pair"):
        compare_image_batches(_solid(2, 8, 8, 1.0), _solid(3, 8, 8, 1.0), "", "")


def test_grayscale_and_rgba_inputs_are_accepted() -> None:
    """One-channel and four-channel images are converted to RGB."""
    gray = torch.ones((1, 8, 8, 1))
    rgba = torch.ones((1, 8, 8, 4))

    out = compare_image_batches(gray, rgba, "", "", gap=0)

    assert out.shape == (1, 8, 16, 3)


def test_unknown_font_falls_back_to_builtin() -> None:
    """A font name missing on this machine (for example from another machine's workflow) still renders."""
    font = load_font("Definitely Not A Real Font", 18)

    assert font is not None


def test_discover_fonts_reads_family_and_style(tmp_path: pathlib.Path) -> None:
    """Fonts are named by family and style, and duplicates keep both entries."""
    source = _find_system_font()
    shutil.copy(source, tmp_path / f"first{source.suffix}")
    shutil.copy(source, tmp_path / f"second{source.suffix}")

    fonts = discover_fonts([tmp_path])

    assert len(fonts) == 2
    name = next(iter(fonts))
    assert not name.endswith(source.suffix)
    assert any("(" in key for key in fonts)  # the duplicate is disambiguated by file name


def test_discover_fonts_ignores_missing_and_non_font_files(tmp_path: pathlib.Path) -> None:
    """Missing directories and non-font files are skipped."""
    (tmp_path / "notes.txt").write_text("not a font")
    (tmp_path / "broken.ttf").write_bytes(b"not a font either")

    assert discover_fonts([tmp_path, tmp_path / "missing"]) == {}


def test_default_font_prefers_common_sans_then_builtin() -> None:
    """The default is a familiar sans-serif when installed, else the built-in font."""
    assert default_font_choice([BUILTIN_FONT, "Arial Regular", "Zapfino Regular"]) == "Arial Regular"
    assert default_font_choice([BUILTIN_FONT, "Zapfino Regular"]) == BUILTIN_FONT


def test_v3_node_schema_and_execute() -> None:
    """The node is a ComfyUI v3 node with a font dropdown and returns an IMAGE."""
    pytest.importorskip("comfy_api.latest")
    node = image_compare.ImageCompareSideBySide

    node.GET_SCHEMA()
    required = node.INPUT_TYPES()["required"]
    assert list(required) == [
        "image_a",
        "image_b",
        "label_a",
        "label_b",
        "font",
        "font_size",
        "label_position",
        "layout",
        "resize",
        "gap",
        "border",
        "theme",
    ]
    font_options = required["font"][1]["options"] if required["font"][0] == "COMBO" else required["font"][0]
    assert font_options[0] == BUILTIN_FONT
    assert list(node.RETURN_TYPES) == ["IMAGE"]
    assert image_compare.NODE_CLASS_MAPPINGS == {"SkoogeerImageCompareSideBySide": node}

    output = node.execute(
        image_a=_solid(1, 16, 16, 1.0),
        image_b=_solid(1, 16, 16, 0.0),
        label_a="A",
        label_b="B",
        font=BUILTIN_FONT,
        font_size=10,
        label_position="bottom",
        layout="horizontal",
        resize="fit",
        gap=0,
        border=0,
        theme="dark",
    )
    assert output.args[0].shape[0] == 1 and output.args[0].shape[2] == 32
