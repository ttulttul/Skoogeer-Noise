from __future__ import annotations

import itertools
import logging
import random
import re
from typing import Dict, List, Sequence

import yaml

logger = logging.getLogger(__name__)

MustacheVariablesDict = Dict[str, List[str]]
MustacheVariableList = List[Dict[str, str]]

_SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF
_DEFAULT_VARIABLES_YAML = (
    "haircolor:\n"
    "  - brown\n"
    "  - blonde\n"
    "leglength:\n"
    "  - short\n"
    "  - long\n"
    "  - weird\n"
)
_DEFAULT_TEMPLATE = "The man has {{haircolor}} hair and {{leglength}} legs."
_MUSTACHE_VARIABLE_PATTERN = re.compile(r"{{\s*([^{}]+?)\s*}}")
_SAMPLING_METHODS = ("sequential", "random")
_REORDER_MODES = ("shuffle", "reverse")


def _coerce_yaml_scalar_to_string(value, *, variable_name: str) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError(
            f"Mustache variable '{variable_name}' values must be scalars or lists of scalars, got {type(value).__name__}."
        )
    if value is None:
        raise ValueError(f"Mustache variable '{variable_name}' contains a null value, which cannot be rendered into text.")
    return str(value)


def parse_mustache_variables_yaml(yaml_text: str) -> MustacheVariablesDict:
    text = str(yaml_text or "").strip()
    if not text:
        logger.debug("Mustache variables YAML input was empty; returning an empty variable set.")
        return {}

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse Mustache variables YAML: {exc}") from exc

    if parsed is None:
        logger.debug("Mustache variables YAML parsed to None; returning an empty variable set.")
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Mustache variables YAML must parse to a mapping, got {type(parsed).__name__}.")

    variables: MustacheVariablesDict = {}
    for raw_key, raw_values in parsed.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("Mustache variable names must be non-empty.")

        if isinstance(raw_values, list):
            values = [_coerce_yaml_scalar_to_string(item, variable_name=key) for item in raw_values]
        else:
            values = [_coerce_yaml_scalar_to_string(raw_values, variable_name=key)]

        if not values:
            raise ValueError(f"Mustache variable '{key}' must contain at least one value.")

        variables[key] = values

    logger.debug("Parsed %d mustache variables from YAML: %s", len(variables), tuple(variables.keys()))
    return variables


def extract_template_variables(template: str) -> List[str]:
    ordered_variables: List[str] = []
    seen = set()
    for match in _MUSTACHE_VARIABLE_PATTERN.finditer(template):
        name = match.group(1).strip()
        if name and name not in seen:
            ordered_variables.append(name)
            seen.add(name)
    return ordered_variables


def _validate_sampling_method(sampling_method: str) -> str:
    method = str(sampling_method).strip().lower()
    if method not in _SAMPLING_METHODS:
        raise ValueError(f"sampling_method must be one of {_SAMPLING_METHODS}, got '{sampling_method}'.")
    return method


def _validate_reorder_mode(mode: str) -> str:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in _REORDER_MODES:
        raise ValueError(f"mode must be one of {_REORDER_MODES}, got '{mode}'.")
    return normalized_mode


def _normalize_limit(limit: int) -> int | None:
    limit_value = int(limit)
    if limit_value < -1:
        raise ValueError(f"limit must be >= -1, got {limit_value}.")
    return None if limit_value == -1 else limit_value


def _count_permutations(value_sets: Sequence[Sequence[str]]) -> int:
    total = 1
    for values in value_sets:
        total *= len(values)
    return total


def _render_template_with_mapping(template_text: str, mapping: Dict[str, str]) -> str:
    return _MUSTACHE_VARIABLE_PATTERN.sub(lambda match: mapping[match.group(1).strip()], template_text)


def _variable_setting_for_index(
    ordered_keys: Sequence[str],
    value_sets: Sequence[Sequence[str]],
    permutation_index: int,
) -> Dict[str, str]:
    index = permutation_index
    chosen_values = [None] * len(value_sets)

    for position in range(len(value_sets) - 1, -1, -1):
        values = value_sets[position]
        value_index = index % len(values)
        index //= len(values)
        chosen_values[position] = values[value_index]

    return dict(zip(ordered_keys, chosen_values, strict=True))


def sample_mustache_variable_list(
    variables: MustacheVariablesDict,
    *,
    sampling_mode: str = "sequential",
    seed: int = 0,
    limit: int = -1,
) -> MustacheVariableList:
    normalized_sampling_mode = _validate_sampling_method(sampling_mode)
    normalized_limit = _normalize_limit(limit)
    rng = random.Random(int(seed) & _SEED_MASK_64)

    ordered_keys = list(variables.keys())
    if normalized_sampling_mode == "random":
        rng.shuffle(ordered_keys)

    if not ordered_keys:
        sample_count = 1 if normalized_limit is None else min(normalized_limit, 1)
        result = [{}] if sample_count == 1 else []
        logger.debug(
            "Sampled empty mustache variable set into %d variable settings with sampling_mode=%s seed=%d limit=%s.",
            len(result),
            normalized_sampling_mode,
            int(seed) & _SEED_MASK_64,
            "unbounded" if normalized_limit is None else normalized_limit,
        )
        return result

    value_sets: List[List[str]] = []
    for key in ordered_keys:
        values = list(variables[key])
        if normalized_sampling_mode == "random":
            values = rng.sample(values, k=len(values))
        value_sets.append(values)

    total_permutations = _count_permutations(value_sets)
    sample_count = total_permutations if normalized_limit is None else min(normalized_limit, total_permutations)

    sampled_variables: MustacheVariableList = []
    if normalized_sampling_mode == "random":
        sampled_indices = rng.sample(range(total_permutations), sample_count)
        sampled_variables = [
            _variable_setting_for_index(ordered_keys, value_sets, permutation_index)
            for permutation_index in sampled_indices
        ]
    else:
        for combination in itertools.islice(itertools.product(*value_sets), sample_count):
            sampled_variables.append(dict(zip(ordered_keys, combination, strict=True)))

    logger.debug(
        "Sampled %d/%d mustache variable settings using %s mode with seed=%d limit=%s.",
        len(sampled_variables),
        total_permutations,
        normalized_sampling_mode,
        int(seed) & _SEED_MASK_64,
        "unbounded" if normalized_limit is None else normalized_limit,
    )
    return sampled_variables


def render_mustache_template_list(
    template: str,
    variable_list: MustacheVariableList,
) -> List[str]:
    template_text = str(template)
    referenced_variables = extract_template_variables(template_text)

    if not variable_list:
        logger.debug("Mustache template received an empty variable list; returning no outputs.")
        return []

    if not referenced_variables:
        logger.debug(
            "Mustache template contains no variables; repeating the raw template for %d variable settings.",
            len(variable_list),
        )
        return [template_text for _ in variable_list]

    rendered_outputs: List[str] = []
    for setting_index, variables in enumerate(variable_list):
        if not isinstance(variables, dict):
            raise ValueError(
                f"MUSTACHE_VARIABLE_LIST items must be dictionaries, got {type(variables).__name__} at index {setting_index}."
            )
        missing = [name for name in referenced_variables if name not in variables]
        if missing:
            raise ValueError(
                f"Mustache template references undefined variables in entry {setting_index}: {', '.join(sorted(missing))}"
            )
        rendered_outputs.append(_render_template_with_mapping(template_text, variables))

    logger.debug(
        "Rendered mustache template for %d variable settings into %d strings.",
        len(variable_list),
        len(rendered_outputs),
    )
    return rendered_outputs


class MustacheVariables:
    CATEGORY = "text/template"
    RETURN_TYPES = ("MUSTACHE_VARIABLES",)
    RETURN_NAMES = ("variables",)
    FUNCTION = "parse_variables"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "yaml_text": ("STRING", {
                    "default": _DEFAULT_VARIABLES_YAML,
                    "multiline": True,
                    "tooltip": (
                        "YAML mapping of variable names to values. Each key becomes a mustache variable and each "
                        "value should usually be a list of render options. Scalar values are accepted as shorthand "
                        "for a single-item list."
                    ),
                }),
            },
        }

    def parse_variables(self, yaml_text: str):
        variables = parse_mustache_variables_yaml(yaml_text)
        logger.debug("MustacheVariables node produced %d variable groups.", len(variables))
        return (variables,)


class MustacheVariableSampler:
    CATEGORY = "text/template"
    RETURN_TYPES = ("MUSTACHE_VARIABLE_LIST",)
    RETURN_NAMES = ("variables",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "variables": ("MUSTACHE_VARIABLES", {
                    "tooltip": "Mustache variables mapping to expand into concrete variable settings.",
                }),
                "sampling_mode": (_SAMPLING_METHODS, {
                    "default": "sequential",
                    "tooltip": (
                        "How to generate concrete variable settings. 'sequential' walks the Cartesian product in "
                        "stable order. 'random' randomizes key order, value order, and the sampled permutation order."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": (
                        "64-bit seed used when sampling_mode is random. The same seed produces the same sampled "
                        "variable-setting order."
                    ),
                }),
                "limit": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": (
                        "Maximum number of concrete variable settings to emit. Use -1 to emit the full Cartesian product."
                    ),
                }),
            },
        }

    def sample(self, variables: MustacheVariablesDict, sampling_mode: str, seed: int, limit: int):
        if not isinstance(variables, dict):
            raise ValueError(f"MUSTACHE_VARIABLES input must be a dictionary, got {type(variables).__name__}.")

        sampled_variables = sample_mustache_variable_list(
            variables,
            sampling_mode=str(sampling_mode),
            seed=int(seed),
            limit=int(limit),
        )
        logger.debug(
            "MustacheVariableSampler node emitted %d concrete variable settings with sampling_mode=%s seed=%d limit=%d.",
            len(sampled_variables),
            str(sampling_mode),
            int(seed) & _SEED_MASK_64,
            int(limit),
        )
        return (sampled_variables,)


class MustacheTemplate:
    CATEGORY = "text/template"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "variables": ("MUSTACHE_VARIABLE_LIST", {
                    "tooltip": (
                        "Concrete mustache variable settings generated by Mustache Variable Sampler. One rendered "
                        "prompt is produced for each list entry."
                    ),
                }),
                "template": ("STRING", {
                    "default": _DEFAULT_TEMPLATE,
                    "multiline": True,
                    "tooltip": (
                        "Template text containing placeholders like {{haircolor}}. The node renders the template once "
                        "for each variable-setting entry."
                    ),
                }),
            },
        }

    def render(self, variables: MustacheVariableList, template: str):
        if not isinstance(variables, list):
            raise ValueError(f"MUSTACHE_VARIABLE_LIST input must be a list, got {type(variables).__name__}.")

        rendered_outputs = render_mustache_template_list(template, variables)
        logger.debug("MustacheTemplate node rendered %d outputs.", len(rendered_outputs))
        return (rendered_outputs,)


class JoinTextList:
    CATEGORY = "text/debug"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "count")
    INPUT_IS_LIST = (True,)
    FUNCTION = "join_text"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "text": ("STRING", {
                    "tooltip": (
                        "List of text values to join into a single previewable string. Connect a list-valued "
                        "STRING output such as Mustache Template here."
                    ),
                }),
            },
        }

    def join_text(self, text):
        joined = "\n".join(str(item) for item in text)
        count = len(text)
        logger.debug("JoinTextList node joined %d text items into one preview string.", count)
        return (joined, count)


class ReorderList:
    CATEGORY = "utils/list"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("items",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "reorder"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "items": ("*", {
                    "tooltip": "List-valued input to reorder. The node preserves the item type and returns a reordered list.",
                }),
                "mode": (_REORDER_MODES, {
                    "default": "shuffle",
                    "tooltip": "Reordering strategy. 'shuffle' applies a seeded random permutation. 'reverse' flips the list order.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": _SEED_MASK_64,
                    "tooltip": "64-bit seed used when mode is shuffle. The same seed yields the same permutation.",
                }),
            },
        }

    def reorder(self, items, mode, seed):
        normalized_mode = _validate_reorder_mode(mode[0] if isinstance(mode, list) and mode else mode)
        seed_value = int(seed[0] if isinstance(seed, list) and seed else seed) & _SEED_MASK_64
        values = list(items)

        if normalized_mode == "reverse":
            reordered = list(reversed(values))
        else:
            rng = random.Random(seed_value)
            reordered = rng.sample(values, k=len(values))

        logger.debug(
            "ReorderList node reordered %d items using mode=%s seed=%d.",
            len(reordered),
            normalized_mode,
            seed_value,
        )
        return (reordered,)


NODE_CLASS_MAPPINGS = {
    "JoinTextList": JoinTextList,
    "MustacheVariables": MustacheVariables,
    "MustacheVariableSampler": MustacheVariableSampler,
    "MustacheTemplate": MustacheTemplate,
    "ReorderList": ReorderList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoinTextList": "Join Text List",
    "MustacheVariables": "Mustache Variables",
    "MustacheVariableSampler": "Mustache Variable Sampler",
    "MustacheTemplate": "Mustache Template",
    "ReorderList": "Reorder List",
}
