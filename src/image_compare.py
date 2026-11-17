"""Image Compare (Side by Side): place two images next to each other with text labels.

Modelled on Comfyroll's ``CR Simple Image Compare``, but written against ComfyUI's v3
node API (``comfy_api.latest``) and drawing labels with any font installed on the
machine running ComfyUI instead of a bundled font folder.
"""

from __future__ import annotations

import functools
import logging
import os
import pathlib
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

BUILTIN_FONT = "Pillow default"
"""Font choice that needs no system fonts, for minimal containers."""

LAYOUTS = ("horizontal", "vertical")
LABEL_POSITIONS = ("bottom", "top", "none")
RESIZE_MODES = ("fit", "stretch", "none")
THEMES = ("dark", "light")

_THEME_COLORS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    # theme -> (background, text)
    "dark": ((0, 0, 0), (255, 255, 255)),
    "light": ((255, 255, 255), (0, 0, 0)),
}

_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")
_MAX_FACES_PER_COLLECTION = 64

_PREFERRED_DEFAULT_FONTS = (
    "Helvetica Neue Regular",
    "Helvetica Regular",
    "Arial Regular",
    "Segoe UI Regular",
    "DejaVu Sans Book",
    "Liberation Sans Regular",
    "Noto Sans Regular",
    "Roboto Regular",
)


# --------------------------------------------------------------------------------------
# Font discovery
# --------------------------------------------------------------------------------------


def system_font_dirs() -> List[pathlib.Path]:
    """Return the usual font directories for the current OS, whether or not they exist."""
    home = pathlib.Path.home()
    dirs: List[pathlib.Path] = []
    if sys.platform == "darwin":
        dirs += [
            pathlib.Path("/System/Library/Fonts"),
            pathlib.Path("/Library/Fonts"),
            home / "Library" / "Fonts",
        ]
    elif sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(pathlib.Path(windir) / "Fonts")
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            dirs.append(pathlib.Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
    else:
        data_home = pathlib.Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
        dirs += [
            pathlib.Path("/usr/share/fonts"),
            pathlib.Path("/usr/local/share/fonts"),
            home / ".fonts",
            data_home / "fonts",
        ]
    extra = os.environ.get("SKOOGEER_FONT_DIRS")
    if extra:
        dirs += [pathlib.Path(part) for part in extra.split(os.pathsep) if part]
    return dirs


def _iter_font_files(directories: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
    seen: set = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for root, _dirs, files in os.walk(directory, followlinks=True):
            for name in files:
                if not name.lower().endswith(_FONT_EXTENSIONS):
                    continue
                path = pathlib.Path(root) / name
                try:
                    key = path.resolve()
                except OSError:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                yield path


def _font_faces(path: pathlib.Path) -> Iterable[Tuple[int, str]]:
    """Yield ``(face_index, "Family Style")`` for every face Pillow can open in ``path``."""
    is_collection = path.suffix.lower() in (".ttc", ".otc")
    max_faces = _MAX_FACES_PER_COLLECTION if is_collection else 1
    for index in range(max_faces):
        try:
            font = ImageFont.truetype(str(path), size=12, index=index)
        except (OSError, ValueError):
            break
        family, style = font.getname()
        family = (family or path.stem).strip()
        style = (style or "").strip()
        if family.startswith("."):
            # macOS private system UI faces (".SF NS", ...) are not meant for direct use.
            continue
        yield index, f"{family} {style}".strip() if style else family


def discover_fonts(directories: Optional[Sequence[pathlib.Path]] = None) -> Dict[str, Tuple[str, int]]:
    """Map a readable font name to ``(file path, face index)`` for every usable font.

    Names come from the font's own family and style (``"Arial Bold"``). When two files
    report the same name, the later one gets its file name appended so both stay selectable.
    """
    fonts: Dict[str, Tuple[str, int]] = {}
    for path in _iter_font_files(directories if directories is not None else system_font_dirs()):
        for index, name in _font_faces(path):
            if name in fonts:
                name = f"{name} ({path.name})" if index == 0 else f"{name} ({path.name} #{index})"
                if name in fonts:
                    continue
            fonts[name] = (str(path), index)
    return dict(sorted(fonts.items(), key=lambda item: item[0].casefold()))


@functools.lru_cache(maxsize=1)
def available_fonts() -> Dict[str, Tuple[str, int]]:
    """Discover system fonts once per process."""
    fonts = discover_fonts()
    logger.info("Image Compare found %d system font faces.", len(fonts))
    return fonts


def font_choices() -> List[str]:
    """Dropdown values: the built-in font first, then system fonts by name."""
    return [BUILTIN_FONT, *available_fonts().keys()]


def default_font_choice(choices: Sequence[str]) -> str:
    """Pick a common sans-serif font when one is installed, else the built-in font."""
    available = set(choices)
    for name in _PREFERRED_DEFAULT_FONTS:
        if name in available:
            return name
    return BUILTIN_FONT


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a dropdown font at ``size`` pixels, falling back to the built-in font."""
    entry = available_fonts().get(name) if name != BUILTIN_FONT else None
    if entry is not None:
        path, index = entry
        try:
            return ImageFont.truetype(path, size=size, index=index)
        except OSError:
            logger.warning("Image Compare could not load font %r from %s; using the built-in font.", name, path)
    elif name != BUILTIN_FONT:
        logger.warning("Image Compare font %r is not installed on this machine; using the built-in font.", name)
    return ImageFont.load_default(size=size)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert one ``(H, W, C)`` float image in ``0..1`` to an RGB PIL image."""
    array = image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    array = (array * 255.0 + 0.5).astype(np.uint8)
    channels = array.shape[-1]
    if channels == 1:
        array = np.repeat(array, 3, axis=-1)
    elif channels == 2:
        array = np.repeat(array[..., :1], 3, axis=-1)
    elif channels > 3:
        array = array[..., :3]
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)


def _resize_second(first: Image.Image, second: Image.Image, layout: str, resize: str) -> Image.Image:
    """Resize ``second`` relative to ``first`` according to ``resize``."""
    if resize == "none" or second.size == first.size:
        return second
    if resize == "stretch":
        return second.resize(first.size, Image.Resampling.LANCZOS)
    # fit: match the shared edge, keep aspect ratio
    if layout == "horizontal":
        height = first.height
        width = max(1, round(second.width * height / second.height))
    else:
        width = first.width
        height = max(1, round(second.height * width / second.width))
    return second.resize((width, height), Image.Resampling.LANCZOS)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    return right - left


def _wrap_label(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    """Word-wrap each line of ``text`` so it fits in ``max_width`` pixels where possible."""
    lines: List[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if _text_width(draw, candidate, font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _line_height(font) -> int:
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except AttributeError:  # pragma: no cover - bitmap fonts without metrics
        _l, top, _r, bottom = font.getbbox("Ag")
        return bottom - top


def _label_panel(
    width: int,
    height: int,
    font,
    lines: List[str],
    background: Tuple[int, int, int],
    color: Tuple[int, int, int],
) -> Image.Image:
    panel = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(panel)
    line_height = _line_height(font)
    spacing = max(1, line_height // 5)
    block_height = len(lines) * line_height + (len(lines) - 1) * spacing
    y = (height - block_height) // 2
    for line in lines:
        draw.text(((width - _text_width(draw, line, font)) // 2, y), line, font=font, fill=color)
        y += line_height + spacing
    return panel


def _auto_font_size(images: Sequence[Image.Image]) -> int:
    return max(12, round(0.05 * min(min(image.size) for image in images)))


def compose_comparison(
    image_a: Image.Image,
    image_b: Image.Image,
    label_a: str,
    label_b: str,
    *,
    font_name: str = BUILTIN_FONT,
    font_size: int = 0,
    layout: str = "horizontal",
    label_position: str = "bottom",
    resize: str = "fit",
    gap: int = 8,
    border: int = 0,
    theme: str = "dark",
) -> Image.Image:
    """Build one comparison image from two PIL images."""
    if layout not in LAYOUTS:
        raise ValueError(f"Image Compare layout must be one of {LAYOUTS}, got {layout!r}.")
    if label_position not in LABEL_POSITIONS:
        raise ValueError(f"Image Compare label_position must be one of {LABEL_POSITIONS}, got {label_position!r}.")
    if resize not in RESIZE_MODES:
        raise ValueError(f"Image Compare resize must be one of {RESIZE_MODES}, got {resize!r}.")
    if theme not in _THEME_COLORS:
        raise ValueError(f"Image Compare theme must be one of {THEMES}, got {theme!r}.")
    background, text_color = _THEME_COLORS[theme]

    image_b = _resize_second(image_a, image_b, layout, resize)
    images = (image_a, image_b)
    labels = (label_a or "", label_b or "")

    # Each image gets a label panel of the same height so the two halves line up.
    show_labels = label_position != "none" and any(label.strip() for label in labels)
    columns = []
    if show_labels:
        size = int(font_size) if font_size and font_size > 0 else _auto_font_size(images)
        font = load_font(font_name, size)
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        padding = max(4, size // 2)
        wrapped = [
            _wrap_label(measure, label, font, max(1, image.width - 2 * padding)) for image, label in zip(images, labels)
        ]
        line_height = _line_height(font)
        spacing = max(1, line_height // 5)
        max_lines = max(len(lines) for lines in wrapped)
        panel_height = max_lines * line_height + (max_lines - 1) * spacing + 2 * padding
        for image, lines in zip(images, wrapped):
            panel = _label_panel(image.width, panel_height, font, lines, background, text_color)
            column = Image.new("RGB", (image.width, image.height + panel_height), background)
            if label_position == "top":
                column.paste(panel, (0, 0))
                column.paste(image, (0, panel_height))
            else:
                column.paste(image, (0, 0))
                column.paste(panel, (0, image.height))
            columns.append(column)
    else:
        columns = list(images)

    gap = max(0, int(gap))
    border = max(0, int(border))
    first, second = columns
    if layout == "horizontal":
        inner_w = first.width + gap + second.width
        inner_h = max(first.height, second.height)
        offsets = ((0, (inner_h - first.height) // 2), (first.width + gap, (inner_h - second.height) // 2))
    else:
        inner_w = max(first.width, second.width)
        inner_h = first.height + gap + second.height
        offsets = (((inner_w - first.width) // 2, 0), ((inner_w - second.width) // 2, first.height + gap))

    canvas = Image.new("RGB", (inner_w + 2 * border, inner_h + 2 * border), background)
    for column, (x, y) in zip(columns, offsets):
        canvas.paste(column, (x + border, y + border))
    return canvas


def compare_image_batches(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    label_a: str,
    label_b: str,
    **options,
) -> torch.Tensor:
    """Compare two ComfyUI ``IMAGE`` batches pair by pair.

    A batch of one is paired with every image in the other batch. Batches of different
    sizes larger than one are an error, since there is no obvious pairing.
    """
    if image_a.ndim == 3:
        image_a = image_a.unsqueeze(0)
    if image_b.ndim == 3:
        image_b = image_b.unsqueeze(0)
    if image_a.ndim != 4 or image_b.ndim != 4:
        raise ValueError(
            f"Image Compare expects IMAGE tensors shaped (B,H,W,C); got {tuple(image_a.shape)} and {tuple(image_b.shape)}."
        )
    count_a, count_b = image_a.shape[0], image_b.shape[0]
    if count_a != count_b and 1 not in (count_a, count_b):
        raise ValueError(
            f"Image Compare cannot pair a batch of {count_a} with a batch of {count_b}. "
            "Use equal batch sizes, or a single image on one side."
        )
    count = max(count_a, count_b)
    outputs = []
    for index in range(count):
        pil_a = _tensor_to_pil(image_a[min(index, count_a - 1)])
        pil_b = _tensor_to_pil(image_b[min(index, count_b - 1)])
        outputs.append(_pil_to_tensor(compose_comparison(pil_a, pil_b, label_a, label_b, **options)))
    result = torch.stack(outputs, dim=0)
    logger.info(
        "Image Compare built %d comparison image(s) of %dx%d from batches of %d and %d.",
        count,
        result.shape[2],
        result.shape[1],
        count_a,
        count_b,
    )
    return result


# --------------------------------------------------------------------------------------
# ComfyUI v3 node
# --------------------------------------------------------------------------------------

try:
    from comfy_api.latest import io
except ImportError:  # pragma: no cover - only when ComfyUI is not importable (plain unit tests)
    io = None  # type: ignore[assignment]
    NODE_CLASS_MAPPINGS: Dict[str, type] = {}
    NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}
else:

    class ImageCompareSideBySide(io.ComfyNode):
        """Lay two images out side by side (or stacked) with a text label under each."""

        @classmethod
        def define_schema(cls) -> io.Schema:
            fonts = font_choices()
            return io.Schema(
                node_id="SkoogeerImageCompareSideBySide",
                display_name="Image Compare (Side by Side)",
                category="image/compare",
                search_aliases=["image compare", "side by side", "before after", "label images", "comparison"],
                description=(
                    "Places two images side by side or stacked, with a text label for each, "
                    "drawn in any font installed on the machine running ComfyUI."
                ),
                inputs=[
                    io.Image.Input("image_a", tooltip="First (left or top) image. Batches are compared pair by pair."),
                    io.Image.Input("image_b", tooltip="Second (right or bottom) image."),
                    io.String.Input("label_a", default="A", multiline=True, tooltip="Label for image_a. Empty for none."),
                    io.String.Input("label_b", default="B", multiline=True, tooltip="Label for image_b. Empty for none."),
                    io.Combo.Input(
                        "font",
                        options=fonts,
                        default=default_font_choice(fonts),
                        tooltip=(
                            "Fonts installed on the ComfyUI server. Add folders with the SKOOGEER_FONT_DIRS "
                            "environment variable. 'Pillow default' needs no installed fonts."
                        ),
                    ),
                    io.Int.Input(
                        "font_size",
                        default=0,
                        min=0,
                        max=1024,
                        tooltip="Label size in pixels. 0 picks 5% of the smaller image's shorter edge.",
                    ),
                    io.Combo.Input(
                        "label_position",
                        options=list(LABEL_POSITIONS),
                        default="bottom",
                        tooltip="Put labels above or below each image, or hide them.",
                    ),
                    io.Combo.Input(
                        "layout",
                        options=list(LAYOUTS),
                        default="horizontal",
                        tooltip="horizontal puts the images side by side; vertical stacks them.",
                    ),
                    io.Combo.Input(
                        "resize",
                        options=list(RESIZE_MODES),
                        default="fit",
                        tooltip=(
                            "How image_b is sized against image_a. fit matches the shared edge and keeps "
                            "the aspect ratio; stretch matches both dimensions; none leaves it alone."
                        ),
                    ),
                    io.Int.Input("gap", default=8, min=0, max=1024, tooltip="Space between the two images in pixels."),
                    io.Int.Input("border", default=0, min=0, max=1024, tooltip="Margin around the whole image in pixels."),
                    io.Combo.Input(
                        "theme",
                        options=list(THEMES),
                        default="dark",
                        tooltip="dark is white text on black; light is black text on white.",
                    ),
                ],
                outputs=[io.Image.Output("image")],
            )

        @classmethod
        def execute(
            cls,
            image_a: torch.Tensor,
            image_b: torch.Tensor,
            label_a: str,
            label_b: str,
            font: str,
            font_size: int,
            label_position: str,
            layout: str,
            resize: str,
            gap: int,
            border: int,
            theme: str,
        ) -> io.NodeOutput:
            result = compare_image_batches(
                image_a,
                image_b,
                label_a,
                label_b,
                font_name=font,
                font_size=font_size,
                layout=layout,
                label_position=label_position,
                resize=resize,
                gap=gap,
                border=border,
                theme=theme,
            )
            return io.NodeOutput(result)

    NODE_CLASS_MAPPINGS = {
        "SkoogeerImageCompareSideBySide": ImageCompareSideBySide,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "SkoogeerImageCompareSideBySide": "Image Compare (Side by Side)",
    }
