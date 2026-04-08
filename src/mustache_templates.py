from __future__ import annotations

import itertools
import logging
import random
import re
from typing import Dict, List, Sequence

import yaml

logger = logging.getLogger(__name__)

MustacheVariablesDict = Dict[str, List[str]]

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


def _normalize_limit(limit: int) -> int | None:
    limit_value = int(limit)
    if limit_value < 0:
        raise ValueError(f"limit must be >= 0, got {limit_value}.")
    return None if limit_value == 0 else limit_value


def _count_permutations(value_sets: Sequence[Sequence[str]]) -> int:
    total = 1
    for values in value_sets:
        total *= len(values)
    return total


def _render_template_with_mapping(template_text: str, mapping: Dict[str, str]) -> str:
    return _MUSTACHE_VARIABLE_PATTERN.sub(lambda match: mapping[match.group(1).strip()], template_text)


def _render_permutation_index(
    template_text: str,
    referenced_variables: Sequence[str],
    value_sets: Sequence[Sequence[str]],
    permutation_index: int,
) -> str:
    index = permutation_index
    chosen_values = [None] * len(value_sets)

    for position in range(len(value_sets) - 1, -1, -1):
        values = value_sets[position]
        value_index = index % len(values)
        index //= len(values)
        chosen_values[position] = values[value_index]

    mapping = dict(zip(referenced_variables, chosen_values, strict=True))
    return _render_template_with_mapping(template_text, mapping)


def render_mustache_permutations(
    template: str,
    variables: MustacheVariablesDict,
    *,
    limit: int = 0,
    sampling_method: str = "sequential",
) -> List[str]:
    template_text = str(template)
    referenced_variables = extract_template_variables(template_text)
    normalized_limit = _normalize_limit(limit)
    normalized_sampling_method = _validate_sampling_method(sampling_method)

    if not referenced_variables:
        logger.debug("Mustache template contains no variables; returning the raw template as a single output.")
        return [template_text]

    missing = [name for name in referenced_variables if name not in variables]
    if missing:
        raise ValueError(
            "Mustache template references undefined variables: "
            + ", ".join(sorted(missing))
        )

    value_sets: Sequence[Sequence[str]] = [variables[name] for name in referenced_variables]
    total_permutations = _count_permutations(value_sets)
    sample_count = total_permutations if normalized_limit is None else min(normalized_limit, total_permutations)

    rendered_outputs: List[str] = []
    if normalized_sampling_method == "sequential":
        for combination in itertools.islice(itertools.product(*value_sets), sample_count):
            mapping = dict(zip(referenced_variables, combination, strict=True))
            rendered_outputs.append(_render_template_with_mapping(template_text, mapping))
    else:
        sampled_indices = random.sample(range(total_permutations), sample_count)
        rendered_outputs = [
            _render_permutation_index(template_text, referenced_variables, value_sets, permutation_index)
            for permutation_index in sampled_indices
        ]

    logger.debug(
        "Rendered mustache template with %d referenced variables into %d/%d permutations using %s sampling (limit=%s).",
        len(referenced_variables),
        len(rendered_outputs),
        total_permutations,
        normalized_sampling_method,
        "unbounded" if normalized_limit is None else normalized_limit,
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
                "variables": ("MUSTACHE_VARIABLES", {
                    "tooltip": (
                        "Mustache variables mapping generated by the Mustache Variables node. Each variable maps to "
                        "one or more possible string values."
                    ),
                }),
                "template": ("STRING", {
                    "default": _DEFAULT_TEMPLATE,
                    "multiline": True,
                    "tooltip": (
                        "Template text containing placeholders like {{haircolor}}. The node renders every permutation "
                        "of the referenced variable values and outputs the results as a STRING list."
                    ),
                }),
                "limit": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": (
                        "Maximum number of rendered prompts to output. Use 0 to disable the cap. This protects "
                        "against large Cartesian products."
                    ),
                }),
                "sampling_method": (_SAMPLING_METHODS, {
                    "default": "sequential",
                    "tooltip": (
                        "How to choose permutations when a limit is active. 'sequential' takes the first prompts in "
                        "Cartesian-product order, while 'random' samples unique permutations without first materializing "
                        "the full permutation list."
                    ),
                }),
            },
        }

    def render(self, variables: MustacheVariablesDict, template: str, limit: int, sampling_method: str):
        if not isinstance(variables, dict):
            raise ValueError(f"MUSTACHE_VARIABLES input must be a dictionary, got {type(variables).__name__}.")

        rendered_outputs = render_mustache_permutations(
            template,
            variables,
            limit=int(limit),
            sampling_method=str(sampling_method),
        )
        logger.debug("MustacheTemplate node rendered %d outputs.", len(rendered_outputs))
        return (rendered_outputs,)


NODE_CLASS_MAPPINGS = {
    "MustacheVariables": MustacheVariables,
    "MustacheTemplate": MustacheTemplate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MustacheVariables": "Mustache Variables",
    "MustacheTemplate": "Mustache Template",
}
