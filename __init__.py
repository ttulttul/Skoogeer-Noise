from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Any, Dict


def _load_module_from_path(module_name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load module '{module_name}' from '{path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __package__:
    from .src.latent_channel_stats_preview import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as STATS_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as STATS_DISPLAY_NAME_MAPPINGS,
    )
    from .src.latent_mesh_drag import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as DRAG_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as DRAG_DISPLAY_NAME_MAPPINGS,
    )
    from .src.seeded_noise import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as NOISE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as NOISE_DISPLAY_NAME_MAPPINGS,
    )
    from .src.fluid_advection import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as FLUID_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as FLUID_DISPLAY_NAME_MAPPINGS,
    )
    from .src.latent_frequency_domain import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as FREQ_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as FREQ_DISPLAY_NAME_MAPPINGS,
    )
    from .src.latent_channel_space_ops import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as CHANNEL_OP_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as CHANNEL_OP_DISPLAY_NAME_MAPPINGS,
    )
    from .src.qwen_noise_nodes import (  # type: ignore[attr-defined] # noqa: F401
        NODE_CLASS_MAPPINGS as QWEN_NOISE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as QWEN_NOISE_DISPLAY_NAME_MAPPINGS,
    )
else:  # pragma: no cover - direct execution fallback
    _ROOT_DIR = pathlib.Path(__file__).resolve().parent
    _load_module_from_path("masking", _ROOT_DIR / "src" / "masking.py")
    stats_module = _load_module_from_path(
        "latent_channel_stats_preview", _ROOT_DIR / "src" / "latent_channel_stats_preview.py"
    )
    drag_module = _load_module_from_path("latent_mesh_drag", _ROOT_DIR / "src" / "latent_mesh_drag.py")
    noise_module = _load_module_from_path("seeded_noise", _ROOT_DIR / "src" / "seeded_noise.py")
    fluid_module = _load_module_from_path("fluid_advection", _ROOT_DIR / "src" / "fluid_advection.py")
    freq_module = _load_module_from_path(
        "latent_frequency_domain", _ROOT_DIR / "src" / "latent_frequency_domain.py"
    )
    channel_op_module = _load_module_from_path(
        "latent_channel_space_ops", _ROOT_DIR / "src" / "latent_channel_space_ops.py"
    )
    qwen_noise_module = _load_module_from_path("qwen_noise_nodes", _ROOT_DIR / "src" / "qwen_noise_nodes.py")

    STATS_NODE_CLASS_MAPPINGS = getattr(stats_module, "NODE_CLASS_MAPPINGS")
    STATS_DISPLAY_NAME_MAPPINGS = getattr(stats_module, "NODE_DISPLAY_NAME_MAPPINGS")
    DRAG_NODE_CLASS_MAPPINGS = getattr(drag_module, "NODE_CLASS_MAPPINGS")
    DRAG_DISPLAY_NAME_MAPPINGS = getattr(drag_module, "NODE_DISPLAY_NAME_MAPPINGS")
    NOISE_NODE_CLASS_MAPPINGS = getattr(noise_module, "NODE_CLASS_MAPPINGS")
    NOISE_DISPLAY_NAME_MAPPINGS = getattr(noise_module, "NODE_DISPLAY_NAME_MAPPINGS")
    FLUID_NODE_CLASS_MAPPINGS = getattr(fluid_module, "NODE_CLASS_MAPPINGS")
    FLUID_DISPLAY_NAME_MAPPINGS = getattr(fluid_module, "NODE_DISPLAY_NAME_MAPPINGS")
    FREQ_NODE_CLASS_MAPPINGS = getattr(freq_module, "NODE_CLASS_MAPPINGS")
    FREQ_DISPLAY_NAME_MAPPINGS = getattr(freq_module, "NODE_DISPLAY_NAME_MAPPINGS")
    CHANNEL_OP_NODE_CLASS_MAPPINGS = getattr(channel_op_module, "NODE_CLASS_MAPPINGS")
    CHANNEL_OP_DISPLAY_NAME_MAPPINGS = getattr(channel_op_module, "NODE_DISPLAY_NAME_MAPPINGS")
    QWEN_NOISE_NODE_CLASS_MAPPINGS = getattr(qwen_noise_module, "NODE_CLASS_MAPPINGS")
    QWEN_NOISE_DISPLAY_NAME_MAPPINGS = getattr(qwen_noise_module, "NODE_DISPLAY_NAME_MAPPINGS")

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    **DRAG_NODE_CLASS_MAPPINGS,
    **CHANNEL_OP_NODE_CLASS_MAPPINGS,
    **FLUID_NODE_CLASS_MAPPINGS,
    **FREQ_NODE_CLASS_MAPPINGS,
    **NOISE_NODE_CLASS_MAPPINGS,
    **QWEN_NOISE_NODE_CLASS_MAPPINGS,
    **STATS_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    **DRAG_DISPLAY_NAME_MAPPINGS,
    **CHANNEL_OP_DISPLAY_NAME_MAPPINGS,
    **FLUID_DISPLAY_NAME_MAPPINGS,
    **FREQ_DISPLAY_NAME_MAPPINGS,
    **NOISE_DISPLAY_NAME_MAPPINGS,
    **QWEN_NOISE_DISPLAY_NAME_MAPPINGS,
    **STATS_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
