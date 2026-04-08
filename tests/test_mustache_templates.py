import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mustache_templates import (  # noqa: E402
    MustacheTemplate,
    MustacheVariables,
    extract_template_variables,
    parse_mustache_variables_yaml,
    render_mustache_permutations,
)


def test_parse_mustache_variables_yaml_parses_lists():
    variables = parse_mustache_variables_yaml(
        "haircolor:\n"
        "  - brown\n"
        "  - blonde\n"
        "leglength:\n"
        "  - short\n"
        "  - long\n"
    )

    assert variables == {
        "haircolor": ["brown", "blonde"],
        "leglength": ["short", "long"],
    }


def test_parse_mustache_variables_yaml_wraps_scalar_values():
    variables = parse_mustache_variables_yaml(
        "haircolor: brown\n"
        "count: 3\n"
    )

    assert variables == {
        "haircolor": ["brown"],
        "count": ["3"],
    }


def test_extract_template_variables_preserves_first_appearance_order():
    variables = extract_template_variables(
        "A {{haircolor}} coat, {{leglength}} legs, and again {{haircolor}}."
    )

    assert variables == ["haircolor", "leglength"]


def test_render_mustache_permutations_generates_all_products_in_template_order():
    rendered = render_mustache_permutations(
        "The {{haircolor}} fox has {{leglength}} legs.",
        {
            "haircolor": ["brown", "blonde"],
            "leglength": ["short", "long", "weird"],
            "unused": ["ignored"],
        },
    )

    assert rendered == [
        "The brown fox has short legs.",
        "The brown fox has long legs.",
        "The brown fox has weird legs.",
        "The blonde fox has short legs.",
        "The blonde fox has long legs.",
        "The blonde fox has weird legs.",
    ]


def test_render_mustache_permutations_raises_for_missing_variable():
    with pytest.raises(ValueError, match="undefined variables: leglength"):
        render_mustache_permutations(
            "The {{haircolor}} fox has {{leglength}} legs.",
            {"haircolor": ["brown"]},
        )


def test_render_mustache_permutations_without_placeholders_returns_template_once():
    assert render_mustache_permutations("plain text", {"haircolor": ["brown"]}) == ["plain text"]


def test_mustache_variables_node_returns_typed_mapping():
    node = MustacheVariables()

    (variables,) = node.parse_variables("haircolor:\n  - brown\n  - blonde\n")

    assert variables == {"haircolor": ["brown", "blonde"]}


def test_mustache_template_node_returns_list_output():
    node = MustacheTemplate()

    (rendered,) = node.render(
        {"haircolor": ["brown", "blonde"], "leglength": ["short", "long"]},
        "The {{haircolor}} fox has {{leglength}} legs.",
    )

    assert node.OUTPUT_IS_LIST == (True,)
    assert rendered == [
        "The brown fox has short legs.",
        "The brown fox has long legs.",
        "The blonde fox has short legs.",
        "The blonde fox has long legs.",
    ]


def test_parse_mustache_variables_yaml_rejects_nested_values():
    with pytest.raises(ValueError, match="scalars or lists of scalars"):
        parse_mustache_variables_yaml("haircolor:\n  nested: value\n")
