from __future__ import annotations

import itertools
import logging
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


def render_mustache_permutations(template: str, variables: MustacheVariablesDict) -> List[str]:
    template_text = str(template)
    referenced_variables = extract_template_variables(template_text)

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
    rendered_outputs: List[str] = []

    for combination in itertools.product(*value_sets):
        mapping = dict(zip(referenced_variables, combination, strict=True))
        rendered_outputs.append(
            _MUSTACHE_VARIABLE_PATTERN.sub(lambda match: mapping[match.group(1).strip()], template_text)
        )

    logger.debug(
        "Rendered mustache template with %d referenced variables into %d permutations.",
        len(referenced_variables),
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
            },
        }

    def render(self, variables: MustacheVariablesDict, template: str):
        if not isinstance(variables, dict):
            raise ValueError(f"MUSTACHE_VARIABLES input must be a dictionary, got {type(variables).__name__}.")

        rendered_outputs = render_mustache_permutations(template, variables)
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
