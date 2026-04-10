from __future__ import annotations

import heapq
import itertools
import logging
import random
import re
from bisect import bisect_left
from typing import Dict, List, Sequence

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

logger = logging.getLogger(__name__)

MustacheVariablesDict = Dict[str, List[str]]
MustacheVariableList = List[Dict[str, str]]
CompiledMustacheTemplate = tuple[str, List[str], Dict[str, str], Dict[str, str]]

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
_VARIABLE_WEIGHTS_KEY = "__mustache_weights__"
_VARIABLE_TEMPLATE_SPECS_KEY = "__mustache_template_specs__"
_WEIGHTED_VALUE_PATTERN = re.compile(r"^(.*?):([0-9]*\.?[0-9]+)\s*$")
_WEIGHT_TOLERANCE = 1e-6
_RESERVED_VARIABLE_KEYS = {_VARIABLE_WEIGHTS_KEY, _VARIABLE_TEMPLATE_SPECS_KEY}
_TEMPLATE_INSTANCE_SETTINGS = {"repeat", "randomize"}


def _coerce_yaml_scalar_to_string(value, *, variable_name: str) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError(
            f"Mustache variable '{variable_name}' values must be scalars or lists of scalars, got {type(value).__name__}."
        )
    if value is None:
        raise ValueError(f"Mustache variable '{variable_name}' contains a null value, which cannot be rendered into text.")
    return str(value)


def _visible_mustache_variable_keys(variables: MustacheVariablesDict) -> List[str]:
    return [key for key in variables.keys() if key not in _RESERVED_VARIABLE_KEYS]


def _find_unresolved_mustache_variables(text: str) -> List[str]:
    return extract_template_variables(str(text))


def _raise_for_unresolved_mustache_variables(text: str, *, context: str) -> None:
    unresolved = _find_unresolved_mustache_variables(text)
    if unresolved:
        raise ValueError(
            f"{context} contains unresolved mustache variables: {', '.join(sorted(unresolved))}."
        )


def _mustache_template_spec_mapping(variables: MustacheVariablesDict) -> Dict[str, List[CompiledMustacheTemplate | None]]:
    raw_specs = variables.get(_VARIABLE_TEMPLATE_SPECS_KEY)
    if raw_specs is None:
        return {}
    if not isinstance(raw_specs, dict):
        raise ValueError("MUSTACHE_VARIABLES template metadata must be a dictionary.")
    return raw_specs


def _mustache_weight_mapping(variables: MustacheVariablesDict) -> Dict[str, List[float]]:
    raw_weights = variables.get(_VARIABLE_WEIGHTS_KEY)
    if raw_weights is None:
        return {}
    if not isinstance(raw_weights, dict):
        raise ValueError("MUSTACHE_VARIABLES weight metadata must be a dictionary.")
    return raw_weights


def _set_variable_weights(
    variables: MustacheVariablesDict,
    variable_name: str,
    weights: Sequence[float] | None,
) -> None:
    if weights is None:
        raw_weights = variables.get(_VARIABLE_WEIGHTS_KEY)
        if isinstance(raw_weights, dict):
            raw_weights.pop(variable_name, None)
            if not raw_weights:
                variables.pop(_VARIABLE_WEIGHTS_KEY, None)
        return

    if _VARIABLE_WEIGHTS_KEY not in variables:
        variables[_VARIABLE_WEIGHTS_KEY] = {}
    raw_weights = variables[_VARIABLE_WEIGHTS_KEY]
    if not isinstance(raw_weights, dict):
        raise ValueError("MUSTACHE_VARIABLES weight metadata must be a dictionary.")
    raw_weights[variable_name] = [float(weight) for weight in weights]


def _set_variable_template_specs(
    variables: MustacheVariablesDict,
    variable_name: str,
    template_specs: Sequence[CompiledMustacheTemplate | None] | None,
) -> None:
    if template_specs is None:
        raw_specs = variables.get(_VARIABLE_TEMPLATE_SPECS_KEY)
        if isinstance(raw_specs, dict):
            raw_specs.pop(variable_name, None)
            if not raw_specs:
                variables.pop(_VARIABLE_TEMPLATE_SPECS_KEY, None)
        return

    if _VARIABLE_TEMPLATE_SPECS_KEY not in variables:
        variables[_VARIABLE_TEMPLATE_SPECS_KEY] = {}
    raw_specs = variables[_VARIABLE_TEMPLATE_SPECS_KEY]
    if not isinstance(raw_specs, dict):
        raise ValueError("MUSTACHE_VARIABLES template metadata must be a dictionary.")
    raw_specs[variable_name] = list(template_specs)


def _get_variable_weights(
    variables: MustacheVariablesDict,
    variable_name: str,
) -> List[float] | None:
    raw_weights = _mustache_weight_mapping(variables)
    weights = raw_weights.get(variable_name)
    if weights is None:
        return None
    values = variables.get(variable_name)
    if not isinstance(values, list):
        raise ValueError(f"Mustache variable '{variable_name}' values must be stored as a list.")
    if len(weights) != len(values):
        raise ValueError(
            f"Mustache variable '{variable_name}' has {len(values)} values but {len(weights)} weights."
        )
    normalized = [float(weight) for weight in weights]
    for weight in normalized:
        if weight < 0.0 or weight > 1.0 + _WEIGHT_TOLERANCE:
            raise ValueError(
                f"Mustache variable '{variable_name}' weights must be between 0.0 and 1.0, got {weight}."
            )
    total = sum(normalized)
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(f"Mustache variable '{variable_name}' weights must sum to 1.0, got {total:.6f}.")
    return normalized


def _get_variable_template_specs(
    variables: MustacheVariablesDict,
    variable_name: str,
) -> List[CompiledMustacheTemplate | None] | None:
    raw_specs = _mustache_template_spec_mapping(variables)
    specs = raw_specs.get(variable_name)
    if specs is None:
        return None
    values = variables.get(variable_name)
    if not isinstance(values, list):
        raise ValueError(f"Mustache variable '{variable_name}' values must be stored as a list.")
    if len(specs) != len(values):
        raise ValueError(
            f"Mustache variable '{variable_name}' has {len(values)} values but {len(specs)} template specs."
        )
    return list(specs)


def _dependency_ordered_mustache_variable_keys(variables: MustacheVariablesDict) -> List[str]:
    visible_keys = _visible_mustache_variable_keys(variables)
    dependency_sets: Dict[str, set[str]] = {}
    for variable_name in visible_keys:
        dependencies: set[str] = set()
        template_specs = _get_variable_template_specs(variables, variable_name)
        if template_specs is not None:
            for template_spec in template_specs:
                if template_spec is None:
                    continue
                _, referenced_variables, _, _ = template_spec
                for referenced_variable in referenced_variables:
                    if referenced_variable != variable_name and referenced_variable in visible_keys:
                        dependencies.add(referenced_variable)
        dependency_sets[variable_name] = dependencies

    ordered_keys: List[str] = []
    emitted_keys: set[str] = set()
    while len(ordered_keys) < len(visible_keys):
        emitted_this_pass = False
        for variable_name in visible_keys:
            if variable_name in emitted_keys:
                continue
            if dependency_sets[variable_name].issubset(emitted_keys):
                ordered_keys.append(variable_name)
                emitted_keys.add(variable_name)
                emitted_this_pass = True
        if emitted_this_pass:
            continue

        unresolved_keys = [key for key in visible_keys if key not in emitted_keys]
        raise ValueError(
            "Mustache variable dependencies contain a cycle or unresolved ordering across: "
            f"{', '.join(unresolved_keys)}."
        )

    return ordered_keys


def _append_parsed_variable_values(
    variables: MustacheVariablesDict,
    *,
    variable_name: str,
    values: Sequence[str],
    weights: Sequence[float] | None,
    template_specs: Sequence[CompiledMustacheTemplate | None] | None,
) -> None:
    if variable_name in variables:
        existing_value_count = len(variables[variable_name])
        raw_specs = _mustache_template_spec_mapping(variables)
        existing_specs = raw_specs.get(variable_name)
        if existing_specs is not None and len(existing_specs) != existing_value_count:
            raise ValueError(
                f"Mustache variable '{variable_name}' has {existing_value_count} values but {len(existing_specs)} template specs."
            )
        if _get_variable_weights(variables, variable_name) is not None or weights is not None:
            raise ValueError(
                f"Mustache variable '{variable_name}' cannot be merged across multiple definitions when weighted values are in use."
            )
        variables[variable_name].extend(str(value) for value in values)
        if existing_specs is not None or template_specs is not None:
            combined_specs = [None] * existing_value_count if existing_specs is None else list(existing_specs)
            combined_specs.extend([None] * len(values) if template_specs is None else list(template_specs))
            _set_variable_template_specs(variables, variable_name, combined_specs)
        return

    _raise_for_unresolved_mustache_variables(
        variable_name,
        context="Mustache variable name",
    )
    variables[variable_name] = [str(value) for value in values]
    _set_variable_weights(variables, variable_name, weights)
    _set_variable_template_specs(variables, variable_name, None if template_specs is None else list(template_specs))


def _split_weighted_variable_value(value, *, variable_name: str) -> tuple[str, float | None]:
    text = _coerce_yaml_scalar_to_string(value, variable_name=variable_name)
    match = _WEIGHTED_VALUE_PATTERN.match(text)
    if match is None:
        return text, None

    weight = float(match.group(2))
    if weight < 0.0 or weight > 1.0 + _WEIGHT_TOLERANCE:
        raise ValueError(
            f"Mustache variable '{variable_name}' weight suffix must be between 0.0 and 1.0, got {weight}."
        )
    return match.group(1), weight


def _compile_local_template_value(
    template_text: str,
    *,
    variable_name: str,
    variables: MustacheVariablesDict,
) -> CompiledMustacheTemplate | None:
    compiled_format, referenced_variables, field_name_to_variable, field_name_to_setting = _compile_mustache_template(template_text)
    if not referenced_variables:
        return None

    available_variables = set(_visible_mustache_variable_keys(variables))
    missing = [name for name in referenced_variables if name not in available_variables]
    if missing:
        raise ValueError(
            f"Mustache variable '{variable_name}' references undefined variables: {', '.join(sorted(missing))}. "
            "Local template references must be defined earlier in the YAML or supplied through the input variables."
        )
    return compiled_format, referenced_variables, field_name_to_variable, field_name_to_setting


def _parse_variable_values(
    raw_values,
    *,
    variable_name: str,
    variables: MustacheVariablesDict,
) -> tuple[List[str], List[float] | None, List[CompiledMustacheTemplate | None] | None]:
    items = raw_values if isinstance(raw_values, list) else [raw_values]
    values: List[str] = []
    template_specs: List[CompiledMustacheTemplate | None] = []
    explicit_weights: List[float | None] = []
    explicit_weight_count = 0

    for item in items:
        value_text, explicit_weight = _split_weighted_variable_value(item, variable_name=variable_name)
        template_spec = _compile_local_template_value(
            value_text,
            variable_name=variable_name,
            variables=variables,
        )
        values.append(value_text)
        template_specs.append(template_spec)
        explicit_weights.append(explicit_weight)
        if explicit_weight is not None:
            explicit_weight_count += 1

    if not values:
        raise ValueError(f"Mustache variable '{variable_name}' must contain at least one value.")

    if 0 < explicit_weight_count < len(values):
        raise ValueError(
            f"Mustache variable '{variable_name}' mixes weighted and unweighted values; "
            "every value must provide a trailing :probability when any one does."
        )

    derived_weights: List[float] | None = None
    if explicit_weight_count == len(values):
        explicit_weight_total = sum(float(explicit_weight) for explicit_weight in explicit_weights if explicit_weight is not None)
        if abs(explicit_weight_total - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                f"Mustache variable '{variable_name}' weights must sum to 1.0, got {explicit_weight_total:.6f}."
            )
        derived_weights = [float(explicit_weight) for explicit_weight in explicit_weights if explicit_weight is not None]

    if all(template_spec is None for template_spec in template_specs):
        template_specs = None

    return values, derived_weights, template_specs


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
        if key in _RESERVED_VARIABLE_KEYS:
            raise ValueError(f"Mustache variable name '{key}' is reserved.")
        _raise_for_unresolved_mustache_variables(
            key,
            context="Mustache variable name",
        )

        values, weights, template_specs = _parse_variable_values(raw_values, variable_name=key, variables=variables)
        _append_parsed_variable_values(
            variables,
            variable_name=key,
            values=values,
            weights=weights,
            template_specs=template_specs,
        )


def _quote_mustache_yaml_scalars(yaml_text: str) -> str:
    quoted_lines: List[str] = []
    for line in str(yaml_text).splitlines():
        if "{{" not in line and "}}" not in line:
            quoted_lines.append(line)
            continue

        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            quoted_lines.append(line)
            continue

        prefix = None
        remainder = None
        list_match = re.match(r"^(\s*-\s+)(.+)$", line)
        if list_match is not None:
            prefix = list_match.group(1)
            remainder = list_match.group(2)
        else:
            mapping_match = re.match(r"^(\s*[^:\n]+:\s+)(.+)$", line)
            if mapping_match is not None:
                prefix = mapping_match.group(1)
                remainder = mapping_match.group(2)

        if prefix is None or remainder is None:
            quoted_lines.append(line)
            continue

        if remainder.startswith(("'", '"', "|", ">", "[")):
            quoted_lines.append(line)
            continue

        escaped_remainder = remainder.replace("\\", "\\\\").replace('"', '\\"')
        quoted_lines.append(f'{prefix}"{escaped_remainder}"')

    quoted_text = "\n".join(quoted_lines)
    if str(yaml_text).endswith("\n"):
        quoted_text += "\n"
    return quoted_text


def _construct_yaml_value(loader: yaml.Loader, node):
    if isinstance(node, ScalarNode):
        return loader.construct_object(node, deep=True)
    if isinstance(node, SequenceNode):
        return [_construct_yaml_value(loader, item) for item in node.value]
    if isinstance(node, MappingNode):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            mapping[key] = _construct_yaml_value(loader, value_node)
        return mapping
    raise ValueError(f"Unsupported YAML node type: {type(node).__name__}.")


def _load_yaml_preserving_top_level_order(yaml_text: str):
    quoted_text = _quote_mustache_yaml_scalars(yaml_text)
    loader = _YAML_SAFE_LOADER(quoted_text)
    try:
        root = loader.get_single_node()
        if root is None:
            return None
        if isinstance(root, MappingNode):
            return [
                {
                    loader.construct_object(key_node, deep=True): _construct_yaml_value(loader, value_node)
                }
                for key_node, value_node in root.value
            ]
        return _construct_yaml_value(loader, root)
    finally:
        loader.dispose()


def parse_mustache_variables_yaml(yaml_text: str) -> MustacheVariablesDict:
    text = str(yaml_text or "").strip()
    if not text:
        logger.debug("Mustache variables YAML input was empty; returning an empty variable set.")
        return {}

    try:
        parsed = _load_yaml_preserving_top_level_order(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse Mustache variables YAML: {exc}") from exc

    if parsed is None:
        logger.debug("Mustache variables YAML parsed to None; returning an empty variable set.")
        return {}

    variables: MustacheVariablesDict = {}
    _merge_parsed_mustache_variables(parsed, variables=variables)

    logger.debug(
        "Parsed %d mustache variables from YAML: %s",
        len(_visible_mustache_variable_keys(variables)),
        tuple(_visible_mustache_variable_keys(variables)),
    )
    return variables


def parse_mustache_variables_inputs(yaml_inputs: Sequence[str] | str) -> MustacheVariablesDict:
    if isinstance(yaml_inputs, str):
        inputs = [yaml_inputs]
    else:
        inputs = [str(item) for item in yaml_inputs]

    merged_variables: MustacheVariablesDict = {}
    for input_index, yaml_text in enumerate(inputs):
        parsed_variables = parse_mustache_variables_yaml(yaml_text)
        for key in _visible_mustache_variable_keys(parsed_variables):
            _append_parsed_variable_values(
                merged_variables,
                variable_name=key,
                values=parsed_variables[key],
                weights=_get_variable_weights(parsed_variables, key),
                template_specs=_get_variable_template_specs(parsed_variables, key),
            )
        logger.debug(
            "Merged mustache variables input %d with %d keys into aggregate key count %d.",
            input_index,
            len(_visible_mustache_variable_keys(parsed_variables)),
            len(_visible_mustache_variable_keys(merged_variables)),
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
        template_text = str(yaml_text)
        for setting_index, mapping in enumerate(variables):
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"MUSTACHE_VARIABLE_LIST items must be dictionaries, got {type(mapping).__name__} at index {setting_index}."
                )
            rendered_inputs.append(
                _MUSTACHE_VARIABLE_PATTERN.sub(
                    lambda match: mapping.get(_parse_template_variable_reference(match.group(1))[0], match.group(0)),
                    template_text,
                )
            )
    return rendered_inputs


class _FormatMapView:
    __slots__ = ("variables", "field_name_to_variable")

    def __init__(self, variables: Dict[str, str], field_name_to_variable: Dict[str, str]):
        self.variables = variables
        self.field_name_to_variable = field_name_to_variable

    def __getitem__(self, field_name: str) -> str:
        return self.variables[self.field_name_to_variable.get(field_name, field_name)]


class _LazyFormatMapView:
    __slots__ = (
        "root_variables",
        "resolved_variables",
        "field_name_to_variable",
        "field_name_to_setting",
        "rng",
        "repeated_choices",
    )

    def __init__(
        self,
        root_variables: MustacheVariablesDict,
        resolved_variables: Dict[str, str],
        field_name_to_variable: Dict[str, str],
        field_name_to_setting: Dict[str, str],
        rng: random.Random,
    ):
        self.root_variables = root_variables
        self.resolved_variables = resolved_variables
        self.field_name_to_variable = field_name_to_variable
        self.field_name_to_setting = field_name_to_setting
        self.rng = rng
        self.repeated_choices: Dict[str, str] = {}

    def __getitem__(self, field_name: str) -> str:
        variable_name = self.field_name_to_variable.get(field_name, field_name)
        setting = self.field_name_to_setting.get(field_name)

        if setting == "randomize":
            value = _choose_random_lazy_variable_value(
                variable_name,
                variables=self.root_variables,
                resolved_variables=self.resolved_variables,
                rng=self.rng,
            )
            self.repeated_choices[variable_name] = value
            return value

        if setting == "repeat":
            if variable_name in self.repeated_choices:
                return self.repeated_choices[variable_name]
            if variable_name in self.resolved_variables:
                value = self.resolved_variables[variable_name]
                self.repeated_choices[variable_name] = value
                return value
            value = _choose_random_lazy_variable_value(
                variable_name,
                variables=self.root_variables,
                resolved_variables=self.resolved_variables,
                rng=self.rng,
            )
            self.repeated_choices[variable_name] = value
            return value

        if variable_name not in self.resolved_variables:
            raise KeyError(variable_name)

        value = self.resolved_variables[variable_name]
        self.repeated_choices[variable_name] = value
        return value


def _escape_format_literal(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _can_use_direct_format_field(variable_name: str) -> bool:
    return variable_name.isidentifier()


def _find_last_unescaped_colon(text: str) -> int:
    escape = False
    last_colon_index = -1
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == ":":
            last_colon_index = index
    return last_colon_index


def _parse_template_variable_reference(reference_text: str) -> tuple[str, str | None]:
    stripped_reference = str(reference_text).strip()
    split_index = _find_last_unescaped_colon(stripped_reference)
    variable_text = stripped_reference
    setting = None

    if split_index >= 0:
        maybe_setting = stripped_reference[split_index + 1 :].strip().lower()
        if maybe_setting:
            if maybe_setting not in _TEMPLATE_INSTANCE_SETTINGS:
                raise ValueError(
                    f"Unsupported mustache instance setting '{maybe_setting}'. "
                    f"Expected one of {tuple(sorted(_TEMPLATE_INSTANCE_SETTINGS))}."
                )
            variable_text = stripped_reference[:split_index]
            setting = maybe_setting

    variable_name = variable_text.replace("\\:", ":").strip()
    if not variable_name:
        raise ValueError("Mustache template variable names must be non-empty.")
    return variable_name, setting


def _compile_mustache_template(template_text: str) -> tuple[str, List[str], Dict[str, str], Dict[str, str]]:
    format_parts: List[str] = []
    referenced_variables: List[str] = []
    field_name_to_variable: Dict[str, str] = {}
    field_name_to_setting: Dict[str, str] = {}
    field_name_by_variable: Dict[str, str] = {}
    last_index = 0
    synthetic_field_index = 0

    for match in _MUSTACHE_VARIABLE_PATTERN.finditer(template_text):
        format_parts.append(_escape_format_literal(template_text[last_index:match.start()]))
        variable_name, setting = _parse_template_variable_reference(match.group(1))
        if variable_name not in referenced_variables:
            referenced_variables.append(variable_name)
        if setting is None:
            if variable_name not in field_name_by_variable:
                if _can_use_direct_format_field(variable_name):
                    field_name = variable_name
                else:
                    field_name = f"mustache_{synthetic_field_index}"
                    synthetic_field_index += 1
                    field_name_to_variable[field_name] = variable_name
                field_name_by_variable[variable_name] = field_name
            format_parts.append("{")
            format_parts.append(field_name_by_variable[variable_name])
            format_parts.append("}")
        else:
            field_name = f"mustache_{synthetic_field_index}"
            synthetic_field_index += 1
            field_name_to_variable[field_name] = variable_name
            field_name_to_setting[field_name] = setting
            format_parts.append("{")
            format_parts.append(field_name)
            format_parts.append("}")
        last_index = match.end()

    format_parts.append(_escape_format_literal(template_text[last_index:]))
    return "".join(format_parts), referenced_variables, field_name_to_variable, field_name_to_setting


def extract_template_variables(template: str) -> List[str]:
    _, referenced_variables, _, _ = _compile_mustache_template(str(template))
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


def _render_lazy_compiled_template(
    compiled_format: str,
    field_name_to_variable: Dict[str, str],
    field_name_to_setting: Dict[str, str],
    *,
    variables: MustacheVariablesDict,
    resolved_variables: Dict[str, str],
    rng: random.Random,
) -> str:
    return compiled_format.format_map(
        _LazyFormatMapView(
            variables,
            resolved_variables,
            field_name_to_variable,
            field_name_to_setting,
            rng,
        )
    )


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


def _selection_indices_for_index(
    value_sets: Sequence[Sequence[str]],
    permutation_index: int,
) -> tuple[int, ...]:
    index = permutation_index
    chosen_indices = [0] * len(value_sets)

    for position in range(len(value_sets) - 1, -1, -1):
        values = value_sets[position]
        value_index = index % len(values)
        index //= len(values)
        chosen_indices[position] = int(value_index)

    return tuple(chosen_indices)


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


def _weighted_choice_index(cumulative_weights: Sequence[float], rng: random.Random) -> int:
    draw = rng.random()
    return min(bisect_left(cumulative_weights, draw), len(cumulative_weights) - 1)


def _materialized_weighted_sample_without_replacement(
    value_sets: Sequence[Sequence[str]],
    weight_sets: Sequence[Sequence[float]],
    sample_count: int,
    rng: random.Random,
) -> List[tuple[int, ...]]:
    weighted_settings: List[tuple[float, tuple[int, ...]]] = []
    index_sets = [range(len(values)) for values in value_sets]
    for combination in itertools.product(*index_sets):
        probability = 1.0
        for position, value_index in enumerate(combination):
            probability *= float(weight_sets[position][value_index])
        if probability <= 0.0:
            continue
        weighted_settings.append((rng.random() ** (1.0 / probability), tuple(int(index) for index in combination)))

    if len(weighted_settings) < sample_count:
        raise ValueError(
            "Weighted random sampling does not have enough positive-probability combinations to satisfy the requested limit."
        )

    return [selection_indices for _, selection_indices in heapq.nlargest(sample_count, weighted_settings, key=lambda item: item[0])]


def _weighted_random_sample_mustache_variable_list(
    value_sets: Sequence[Sequence[str]],
    weight_sets: Sequence[Sequence[float]],
    *,
    total_permutations: int,
    sample_count: int,
    rng: random.Random,
) -> List[tuple[int, ...]]:
    positive_permutations = 1
    cumulative_weight_sets: List[List[float]] = []
    for values, weights in zip(value_sets, weight_sets, strict=True):
        if len(values) != len(weights):
            raise ValueError("Weighted mustache variable sampling requires a weight for each value.")
        positive_count = sum(1 for weight in weights if float(weight) > 0.0)
        positive_permutations *= positive_count
        cumulative: List[float] = []
        running_total = 0.0
        for weight in weights:
            running_total += float(weight)
            cumulative.append(running_total)
        cumulative_weight_sets.append(cumulative)

    if sample_count > positive_permutations:
        raise ValueError(
            "Weighted random sampling does not have enough positive-probability combinations to satisfy the requested limit."
        )

    if total_permutations <= _MAX_UNBOUNDED_VARIABLE_SETTINGS:
        return _materialized_weighted_sample_without_replacement(
            value_sets,
            weight_sets,
            sample_count,
            rng,
        )

    sampled_indices: List[tuple[int, ...]] = []
    seen_combinations: set[tuple[int, ...]] = set()
    consecutive_duplicates = 0
    max_consecutive_duplicates = max(sample_count * 50, 1000)
    while len(sampled_indices) < sample_count:
        chosen_indices: List[int] = []
        for values, cumulative_weights in zip(value_sets, cumulative_weight_sets, strict=True):
            chosen_index = _weighted_choice_index(cumulative_weights, rng)
            chosen_indices.append(chosen_index)

        encoded_indices = tuple(chosen_indices)
        if encoded_indices in seen_combinations:
            consecutive_duplicates += 1
            if consecutive_duplicates > max_consecutive_duplicates:
                raise ValueError(
                    "Weighted random sampling could not find enough unique combinations efficiently. "
                    "Reduce limit or flatten the weighted variable space."
                )
            continue

        consecutive_duplicates = 0
        seen_combinations.add(encoded_indices)
        sampled_indices.append(encoded_indices)

    return sampled_indices


def _choose_random_lazy_variable_value(
    variable_name: str,
    *,
    variables: MustacheVariablesDict,
    resolved_variables: Dict[str, str],
    rng: random.Random,
) -> str:
    available_values = variables.get(variable_name)
    if not isinstance(available_values, list) or not available_values:
        raise ValueError(f"Mustache variable '{variable_name}' does not contain any values to randomize from.")

    weights = _get_variable_weights(variables, variable_name)
    if weights is None:
        choice_index = rng.randrange(len(available_values))
    else:
        cumulative_weights: List[float] = []
        running_total = 0.0
        for weight in weights:
            running_total += float(weight)
            cumulative_weights.append(running_total)
        choice_index = _weighted_choice_index(cumulative_weights, rng)

    template_specs = _get_variable_template_specs(variables, variable_name)
    template_spec = None if template_specs is None else template_specs[choice_index]
    return _render_lazy_variable_value(
        available_values[choice_index],
        template_spec,
        variable_name=variable_name,
        resolved_variables=resolved_variables,
        variables=variables,
        rng=rng,
    )


def _render_lazy_variable_value(
    raw_value: str,
    template_spec: CompiledMustacheTemplate | None,
    *,
    variable_name: str,
    resolved_variables: Dict[str, str],
    variables: MustacheVariablesDict,
    rng: random.Random,
) -> str:
    if template_spec is None:
        template_spec = _compile_local_template_value(
            raw_value,
            variable_name=variable_name,
            variables=resolved_variables,
        )
        if template_spec is None:
            return raw_value

    compiled_format, _, field_name_to_variable, field_name_to_setting = template_spec
    try:
        return _render_lazy_compiled_template(
            compiled_format,
            field_name_to_variable,
            field_name_to_setting,
            variables=variables,
            resolved_variables=resolved_variables,
            rng=rng,
        )
    except KeyError as exc:
        missing_variable = str(exc.args[0])
        raise ValueError(
            f"Mustache variable '{variable_name}' references undefined variables during sampling: {missing_variable}."
        ) from exc


def _render_variable_setting_from_indices(
    ordered_keys: Sequence[str],
    value_sets: Sequence[Sequence[str]],
    template_spec_sets: Sequence[Sequence[CompiledMustacheTemplate | None]],
    selection_indices: Sequence[int],
    *,
    variables: MustacheVariablesDict,
    rng: random.Random,
) -> Dict[str, str]:
    rendered_setting: Dict[str, str] = {}
    for position, variable_name in enumerate(ordered_keys):
        value_index = int(selection_indices[position])
        rendered_setting[variable_name] = _render_lazy_variable_value(
            value_sets[position][value_index],
            template_spec_sets[position][value_index],
            variable_name=variable_name,
            resolved_variables=rendered_setting,
            variables=variables,
            rng=rng,
        )
    return rendered_setting


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

    ordered_keys = _dependency_ordered_mustache_variable_keys(variables)
    has_lazy_dependencies = any(
        _get_variable_template_specs(variables, key) is not None
        for key in ordered_keys
    )
    if normalized_sampling_mode == "random" and not has_lazy_dependencies:
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
    template_spec_sets: List[List[CompiledMustacheTemplate | None]] = []
    for key in ordered_keys:
        values = list(variables[key])
        template_specs = _get_variable_template_specs(variables, key)
        if template_specs is None:
            template_specs = [None] * len(values)
        if normalized_sampling_mode == "random" and _get_variable_weights(variables, key) is None:
            paired_entries = list(zip(values, template_specs, strict=True))
            rng.shuffle(paired_entries)
            values = [value for value, _ in paired_entries]
            template_specs = [template_spec for _, template_spec in paired_entries]
        value_sets.append(values)
        template_spec_sets.append(template_specs)

    total_permutations = _count_permutations(value_sets)
    sample_count = _resolve_sample_count(total_permutations, normalized_limit, normalized_sampling_mode)

    sampled_selection_indices: List[tuple[int, ...]] = []
    if normalized_sampling_mode == "random":
        if any(_get_variable_weights(variables, key) is not None for key in ordered_keys):
            normalized_weight_sets: List[List[float]] = []
            for key, values in zip(ordered_keys, value_sets, strict=True):
                weights = _get_variable_weights(variables, key)
                if weights is None:
                    normalized_weight_sets.append([1.0 / len(values)] * len(values))
                else:
                    normalized_weight_sets.append(list(weights))
            sampled_selection_indices = _weighted_random_sample_mustache_variable_list(
                value_sets,
                normalized_weight_sets,
                total_permutations=total_permutations,
                sample_count=sample_count,
                rng=rng,
            )
        else:
            sampled_indices = _sample_unique_indices(total_permutations, sample_count, rng)
            sampled_selection_indices = [
                _selection_indices_for_index(value_sets, permutation_index)
                for permutation_index in sampled_indices
            ]
    else:
        sampled_selection_indices = [
            tuple(int(index) for index in combination)
            for combination in itertools.islice(
                itertools.product(*(range(len(values)) for values in value_sets)),
                sample_count,
            )
        ]

    sampled_variables = [
        _render_variable_setting_from_indices(
            ordered_keys,
            value_sets,
            template_spec_sets,
            selection_indices,
            variables=variables,
            rng=rng,
        )
        for selection_indices in sampled_selection_indices
    ]

    logger.debug(
        "Sampled %d/%d mustache variable settings using %s mode with seed=%d limit=%s lazy_dependencies=%s.",
        len(sampled_variables),
        total_permutations,
        normalized_sampling_mode,
        int(seed) & _SEED_MASK_64,
        "unbounded" if normalized_limit is None else normalized_limit,
        has_lazy_dependencies,
    )
    return sampled_variables


def render_mustache_template_list(
    template: str,
    variable_list: MustacheVariableList,
) -> List[str]:
    template_text = str(template)
    compiled_format, referenced_variables, field_name_to_variable, _ = _compile_mustache_template(template_text)

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
        logger.debug(
            "MustacheVariables node produced %d variable groups.",
            len(_visible_mustache_variable_keys(variables)),
        )
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
