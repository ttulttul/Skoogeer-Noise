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
_MAX_UNBOUNDED_VARIABLE_SETTINGS = 100_000
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _coerce_yaml_scalar_to_string(value, *, variable_name: str) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError(
            f"Mustache variable '{variable_name}' values must be scalars or lists of scalars, got {type(value).__name__}."
        )
    if value is None:
        raise ValueError(f"Mustache variable '{variable_name}' contains a null value, which cannot be rendered into text.")
    return str(value)


def _merge_parsed_mustache_variables(parsed, *, variables: MustacheVariablesDict) -> None:
    if parsed is None:
        return

    if isinstance(parsed, list):
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(
                    "Mustache variables YAML lists must contain mappings, "
                    f"got {type(item).__name__} at list index {index}."
                )
            _merge_parsed_mustache_variables(item, variables=variables)
        return

    if not isinstance(parsed, dict):
        raise ValueError(
            "Mustache variables YAML must parse to a mapping or a list of mappings, "
            f"got {type(parsed).__name__}."
        )

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

        if key not in variables:
            variables[key] = list(values)
        else:
            variables[key].extend(values)


def parse_mustache_variables_yaml(yaml_text: str) -> MustacheVariablesDict:
    text = str(yaml_text or "").strip()
    if not text:
        logger.debug("Mustache variables YAML input was empty; returning an empty variable set.")
        return {}

    try:
        parsed = yaml.load(text, Loader=_YAML_SAFE_LOADER)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse Mustache variables YAML: {exc}") from exc

    if parsed is None:
        logger.debug("Mustache variables YAML parsed to None; returning an empty variable set.")
        return {}

    variables: MustacheVariablesDict = {}
    _merge_parsed_mustache_variables(parsed, variables=variables)

    logger.debug("Parsed %d mustache variables from YAML: %s", len(variables), tuple(variables.keys()))
    return variables


def parse_mustache_variables_inputs(yaml_inputs: Sequence[str] | str) -> MustacheVariablesDict:
    if isinstance(yaml_inputs, str):
        inputs = [yaml_inputs]
    else:
        inputs = [str(item) for item in yaml_inputs]

    merged_variables: MustacheVariablesDict = {}
    for input_index, yaml_text in enumerate(inputs):
        parsed_variables = parse_mustache_variables_yaml(yaml_text)
        for key, values in parsed_variables.items():
            if key not in merged_variables:
                merged_variables[key] = list(values)
            else:
                merged_variables[key].extend(values)
        logger.debug(
            "Merged mustache variables input %d with %d keys into aggregate key count %d.",
            input_index,
            len(parsed_variables),
            len(merged_variables),
        )

    return merged_variables


def _flatten_mustache_variable_list_input(variables, flattened: MustacheVariableList) -> None:
    if isinstance(variables, dict):
        flattened.append(variables)
        return

    if isinstance(variables, list):
        for item in variables:
            _flatten_mustache_variable_list_input(item, flattened)
        return

    raise ValueError(
        "MUSTACHE_VARIABLE_LIST input must be a list of dictionaries containing string values."
    )


def _normalize_mustache_variable_list_input(variables) -> MustacheVariableList | None:
    if variables is None:
        return None

    flattened: MustacheVariableList = []
    _flatten_mustache_variable_list_input(variables, flattened)
    return flattened


def merge_mustache_variable_lists(
    variables_1,
    variables_2,
) -> MustacheVariableList:
    left = _normalize_mustache_variable_list_input(variables_1)
    right = _normalize_mustache_variable_list_input(variables_2)

    left = [] if left is None else left
    right = [] if right is None else right

    if not left:
        return [dict(item) for item in right]
    if not right:
        return [dict(item) for item in left]

    if len(left) == len(right):
        pairs = zip(left, right, strict=True)
    elif len(left) == 1:
        pairs = ((left[0], item) for item in right)
    elif len(right) == 1:
        pairs = ((item, right[0]) for item in left)
    else:
        raise ValueError(
            "Cannot merge MUSTACHE_VARIABLE_LIST inputs with different lengths unless one side has exactly one entry, "
            f"got {len(left)} and {len(right)}."
        )

    merged: MustacheVariableList = []
    for left_item, right_item in pairs:
        merged.append({**left_item, **right_item})

    logger.debug(
        "Merged MUSTACHE_VARIABLE_LIST inputs of lengths %d and %d into %d entries.",
        len(left),
        len(right),
        len(merged),
    )
    return merged


def render_mustache_yaml_inputs(
    yaml_inputs: Sequence[str] | str,
    variables: MustacheVariableList | None,
) -> List[str]:
    if isinstance(yaml_inputs, str):
        inputs = [yaml_inputs]
    else:
        inputs = [str(item) for item in yaml_inputs]

    if variables is None:
        return inputs

    rendered_inputs: List[str] = []
    for yaml_text in inputs:
        rendered_inputs.extend(render_mustache_template_list(yaml_text, variables))
    return rendered_inputs


class _FormatMapView:
    __slots__ = ("variables", "field_name_to_variable")

    def __init__(self, variables: Dict[str, str], field_name_to_variable: Dict[str, str]):
        self.variables = variables
        self.field_name_to_variable = field_name_to_variable

    def __getitem__(self, field_name: str) -> str:
        return self.variables[self.field_name_to_variable.get(field_name, field_name)]


def _escape_format_literal(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _can_use_direct_format_field(variable_name: str) -> bool:
    return variable_name.isidentifier()


def _compile_mustache_template(template_text: str) -> tuple[str, List[str], Dict[str, str]]:
    format_parts: List[str] = []
    referenced_variables: List[str] = []
    field_name_to_variable: Dict[str, str] = {}
    field_name_by_variable: Dict[str, str] = {}
    last_index = 0

    for match in _MUSTACHE_VARIABLE_PATTERN.finditer(template_text):
        format_parts.append(_escape_format_literal(template_text[last_index:match.start()]))
        variable_name = match.group(1).strip()
        if variable_name not in field_name_by_variable:
            if _can_use_direct_format_field(variable_name):
                field_name = variable_name
            else:
                field_name = f"mustache_{len(field_name_to_variable)}"
                field_name_to_variable[field_name] = variable_name
            field_name_by_variable[variable_name] = field_name
            referenced_variables.append(variable_name)
        format_parts.append("{")
        format_parts.append(field_name_by_variable[variable_name])
        format_parts.append("}")
        last_index = match.end()

    format_parts.append(_escape_format_literal(template_text[last_index:]))
    return "".join(format_parts), referenced_variables, field_name_to_variable


def extract_template_variables(template: str) -> List[str]:
    _, referenced_variables, _ = _compile_mustache_template(str(template))
    return referenced_variables


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


def _resolve_sample_count(total_permutations: int, normalized_limit: int | None, sampling_mode: str) -> int:
    if normalized_limit is None:
        if total_permutations > _MAX_UNBOUNDED_VARIABLE_SETTINGS:
            raise ValueError(
                "Mustache Variable Sampler would emit "
                f"{total_permutations} variable settings in {sampling_mode} mode with no limit. "
                "Set a finite limit when the permutation space is large."
            )
        return total_permutations

    return min(normalized_limit, total_permutations)

def _render_compiled_template(
    compiled_format: str,
    field_name_to_variable: Dict[str, str],
    mapping: Dict[str, str],
) -> str:
    if not field_name_to_variable:
        return compiled_format.format_map(mapping)
    return compiled_format.format_map(_FormatMapView(mapping, field_name_to_variable))


def _decode_text_separator(separator: str) -> str:
    return (
        str(separator)
        .replace("\\r", "\r")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )


def _unwrap_singleton_list(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


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


def _sample_unique_indices(total_count: int, sample_count: int, rng: random.Random) -> List[int]:
    if sample_count < 0:
        raise ValueError(f"sample_count must be >= 0, got {sample_count}.")
    if sample_count > total_count:
        raise ValueError(f"sample_count must be <= total_count, got {sample_count} > {total_count}.")
    if sample_count == 0:
        return []

    # Floyd's algorithm samples k distinct integers from [0, n) in O(k) time and memory
    # without relying on range lengths that have to fit into Py_ssize_t.
    selected: dict[int, int] = {}
    sampled_indices: List[int] = []
    for candidate in range(total_count - sample_count, total_count):
        draw = rng.randrange(candidate + 1)
        chosen = selected.get(draw, draw)
        selected[draw] = selected.get(candidate, candidate)
        sampled_indices.append(chosen)

    rng.shuffle(sampled_indices)
    return sampled_indices


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
    sample_count = _resolve_sample_count(total_permutations, normalized_limit, normalized_sampling_mode)

    sampled_variables: MustacheVariableList = []
    if normalized_sampling_mode == "random":
        sampled_indices = _sample_unique_indices(total_permutations, sample_count, rng)
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
    compiled_format, referenced_variables, field_name_to_variable = _compile_mustache_template(template_text)

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
        try:
            rendered_outputs.append(
                _render_compiled_template(compiled_format, field_name_to_variable, variables)
            )
        except KeyError:
            missing = [name for name in referenced_variables if name not in variables]
            raise ValueError(
                f"Mustache template references undefined variables in entry {setting_index}: {', '.join(sorted(missing))}"
            )

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
    INPUT_IS_LIST = True
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
                        "for a single-item list. If the optional variables input is connected, this YAML is treated "
                        "as a mustache template and rendered once per variable-setting entry before parsing."
                    ),
                }),
            },
            "optional": {
                "variables": ("MUSTACHE_VARIABLE_LIST", {
                    "tooltip": (
                        "Optional concrete variable settings used to render templated YAML before parsing. This lets "
                        "you chain one mustache-variable stage into another."
                    ),
                }),
            },
        }

    def parse_variables(self, yaml_text, variables=None):
        variable_list = _normalize_mustache_variable_list_input(variables)
        rendered_yaml_inputs = render_mustache_yaml_inputs(yaml_text, variable_list)
        variables = parse_mustache_variables_inputs(rendered_yaml_inputs)
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
                "separator": ("STRING", {
                    "default": "\\n",
                    "multiline": True,
                    "tooltip": (
                        "Separator inserted between each text item. Escape sequences like \\n, \\r, and \\t are "
                        "decoded, so values like \\n===\\n work as expected."
                    ),
                }),
            },
        }

    def join_text(self, text, separator):
        decoded_separator = _decode_text_separator(_unwrap_singleton_list(separator))
        joined = decoded_separator.join(str(item) for item in text)
        count = len(text)
        logger.debug(
            "JoinTextList node joined %d text items into one preview string using separator length %d.",
            count,
            len(decoded_separator),
        )
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


class ConcatenateLists:
    CATEGORY = "utils/list"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("items",)
    INPUT_IS_LIST = (True, True)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "concatenate"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "items_1": ("*", {
                    "tooltip": "First list-valued input to concatenate.",
                }),
                "items_2": ("*", {
                    "tooltip": "Second list-valued input to concatenate after items_1.",
                }),
            },
        }

    def concatenate(self, items_1, items_2):
        combined = list(items_1) + list(items_2)
        logger.debug(
            "ConcatenateLists node concatenated %d and %d items into %d items.",
            len(items_1),
            len(items_2),
            len(combined),
        )
        return (combined,)


class MergeMustacheVariableLists:
    CATEGORY = "text/template"
    RETURN_TYPES = ("MUSTACHE_VARIABLE_LIST",)
    RETURN_NAMES = ("items",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {
            "required": {
                "items_1": ("MUSTACHE_VARIABLE_LIST", {
                    "tooltip": "First mustache variable-setting list to merge.",
                }),
                "items_2": ("MUSTACHE_VARIABLE_LIST", {
                    "tooltip": "Second mustache variable-setting list to merge into items_1.",
                }),
            },
        }

    def merge(self, items_1, items_2):
        return (merge_mustache_variable_lists(items_1, items_2),)


NODE_CLASS_MAPPINGS = {
    "ConcatenateLists": ConcatenateLists,
    "JoinTextList": JoinTextList,
    "MergeMustacheVariableLists": MergeMustacheVariableLists,
    "MustacheVariables": MustacheVariables,
    "MustacheVariableSampler": MustacheVariableSampler,
    "MustacheTemplate": MustacheTemplate,
    "ReorderList": ReorderList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatenateLists": "Concatenate Lists",
    "JoinTextList": "Join Text List",
    "MergeMustacheVariableLists": "Merge Mustache Variable Lists",
    "MustacheVariables": "Mustache Variables",
    "MustacheVariableSampler": "Mustache Variable Sampler",
    "MustacheTemplate": "Mustache Template",
    "ReorderList": "Reorder List",
}
