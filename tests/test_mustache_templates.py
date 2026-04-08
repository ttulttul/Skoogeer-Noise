import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.mustache_templates as mustache_templates  # noqa: E402
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


def test_render_mustache_permutations_limit_truncates_sequential_order():
    rendered = render_mustache_permutations(
        "The {{haircolor}} fox has {{leglength}} legs.",
        {
            "haircolor": ["brown", "blonde"],
            "leglength": ["short", "long", "weird"],
        },
        limit=4,
        sampling_method="sequential",
    )

    assert rendered == [
        "The brown fox has short legs.",
        "The brown fox has long legs.",
        "The brown fox has weird legs.",
        "The blonde fox has short legs.",
    ]


def test_render_mustache_permutations_random_limit_samples_indices_without_full_list(monkeypatch):
    sampled_calls = []

    def fake_sample(population, sample_count):
        sampled_calls.append((population, sample_count))
        return [5, 0, 3]

    monkeypatch.setattr(mustache_templates.random, "sample", fake_sample)

    rendered = render_mustache_permutations(
        "The {{haircolor}} fox has {{leglength}} legs.",
        {
            "haircolor": ["brown", "blonde"],
            "leglength": ["short", "long", "weird"],
        },
        limit=3,
        sampling_method="random",
    )

    assert sampled_calls == [(range(0, 6), 3)]
    assert rendered == [
        "The blonde fox has weird legs.",
        "The brown fox has short legs.",
        "The blonde fox has short legs.",
    ]


def test_render_mustache_permutations_raises_for_missing_variable():
    with pytest.raises(ValueError, match="undefined variables: leglength"):
        render_mustache_permutations(
            "The {{haircolor}} fox has {{leglength}} legs.",
            {"haircolor": ["brown"]},
        )


def test_render_mustache_permutations_without_placeholders_returns_template_once():
    assert render_mustache_permutations("plain text", {"haircolor": ["brown"]}) == ["plain text"]


def test_render_mustache_permutations_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit must be >= 0"):
        render_mustache_permutations(
            "The {{haircolor}} fox.",
            {"haircolor": ["brown"]},
            limit=-1,
        )


def test_render_mustache_permutations_rejects_invalid_sampling_method():
    with pytest.raises(ValueError, match="sampling_method must be one of"):
        render_mustache_permutations(
            "The {{haircolor}} fox.",
            {"haircolor": ["brown"]},
            sampling_method="chaotic",
        )


def test_mustache_variables_node_returns_typed_mapping():
    node = MustacheVariables()

    (variables,) = node.parse_variables("haircolor:\n  - brown\n  - blonde\n")

    assert variables == {"haircolor": ["brown", "blonde"]}


def test_mustache_template_node_returns_list_output():
    node = MustacheTemplate()

    (rendered,) = node.render(
        {"haircolor": ["brown", "blonde"], "leglength": ["short", "long"]},
        "The {{haircolor}} fox has {{leglength}} legs.",
        0,
        "sequential",
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
