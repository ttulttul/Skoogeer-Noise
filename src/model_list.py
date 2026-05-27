from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_MAX_MODELS = 50


class ModelsList:
    CATEGORY = "model/batch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "models_list"
    DESCRIPTION = "Combines multiple MODEL inputs into a list for downstream list-mapped nodes such as KSampler."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        required = {
            "model_1": ("MODEL", {"tooltip": "First model patcher object to include in the output list."}),
            "model_2": ("MODEL", {"tooltip": "Second model patcher object to include in the output list."}),
        }
        optional = {
            f"model_{index}": (
                "MODEL",
                {"tooltip": f"Optional model patcher object #{index} to append to the output list."},
            )
            for index in range(3, _MAX_MODELS + 1)
        }
        return {"required": required, "optional": optional}

    def models_list(self, model_1: Any, model_2: Any, **optional_models: Any):
        models = [model_1, model_2]
        for index in range(3, _MAX_MODELS + 1):
            key = f"model_{index}"
            if key in optional_models and optional_models[key] is not None:
                models.append(optional_models[key])

        if not models:
            raise ValueError("Models List requires at least one MODEL input.")

        logger.info("ModelsList emitted %d MODEL objects as a list output.", len(models))
        return (models,)


NODE_CLASS_MAPPINGS = {
    "ModelsList": ModelsList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelsList": "Models List",
}
