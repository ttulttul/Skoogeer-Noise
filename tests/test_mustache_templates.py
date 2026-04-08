import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mustache_templates import (  # noqa: E402
    MustacheTemplate,
    MustacheVariableSampler,
    MustacheVariables,
    extract_template_variables,
    parse_mustache_variables_yaml,
    render_mustache_permutations,
    sample_mustache_variables,
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
    )

    assert rendered == [
        "The brown fox has short legs.",
        "The brown fox has long legs.",
        "The brown fox has weird legs.",
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
    with pytest.raises(ValueError, match="limit must be >= -1"):
        render_mustache_permutations(
            "The {{haircolor}} fox.",
            {"haircolor": ["brown"]},
            limit=-2,
        )


def test_sample_mustache_variables_rejects_invalid_sampling_method():
    with pytest.raises(ValueError, match="sampling_method must be one of"):
        sample_mustache_variables(
            {"haircolor": ["brown"]},
            sampling_mode="chaotic",
        )


def test_sample_mustache_variables_random_is_deterministic_for_seed():
    variables = {
        "haircolor": ["brown", "blonde", "black", "red"],
        "leglength": ["short", "long", "weird", "impossible"],
    }

    first = sample_mustache_variables(variables, sampling_mode="random", seed=42)
    second = sample_mustache_variables(variables, sampling_mode="random", seed=42)
    third = sample_mustache_variables(variables, sampling_mode="random", seed=43)

    assert first == second
    assert first != third


def test_mustache_variables_node_returns_typed_mapping():
    node = MustacheVariables()

    (variables,) = node.parse_variables("haircolor:\n  - brown\n  - blonde\n")

    assert variables == {"haircolor": ["brown", "blonde"]}


def test_mustache_template_node_returns_list_output():
    node = MustacheTemplate()

    (rendered,) = node.render(
        {"haircolor": ["brown", "blonde"], "leglength": ["short", "long"]},
        "The {{haircolor}} fox has {{leglength}} legs.",
        -1,
    )

    assert node.OUTPUT_IS_LIST == (True,)
    assert rendered == [
        "The brown fox has short legs.",
        "The brown fox has long legs.",
        "The blonde fox has short legs.",
        "The blonde fox has long legs.",
    ]


def test_mustache_variable_sampler_sequential_returns_variables():
    node = MustacheVariableSampler()

    (variables,) = node.sample(
        {"haircolor": ["brown", "blonde"], "leglength": ["short", "long"]},
        "sequential",
        0,
    )

    assert variables == {"haircolor": ["brown", "blonde"], "leglength": ["short", "long"]}


def test_mustache_variable_sampler_random_reorders_values_deterministically():
    node = MustacheVariableSampler()
    inputs = {"haircolor": ["brown", "blonde"], "leglength": ["short", "long", "weird"]}
    (first,) = node.sample(
        inputs,
        "random",
        123,
    )
    (second,) = node.sample(
        inputs,
        "random",
        123,
    )
    (third,) = node.sample(
        inputs,
        "random",
        124,
    )

    assert first == second
    assert third != first
    assert sorted(first["haircolor"]) == sorted(inputs["haircolor"])
    assert sorted(first["leglength"]) == sorted(inputs["leglength"])


def test_mustache_variable_sampler_randomized_variables_affect_template_order():
    sampler = MustacheVariableSampler()
    template = MustacheTemplate()

    (sampled_variables,) = sampler.sample(
        {"haircolor": ["brown", "blonde"], "leglength": ["short", "long"]},
        "random",
        7,
    )
    (rendered,) = template.render(
        sampled_variables,
        "The {{haircolor}} fox has {{leglength}} legs.",
        3,
    )

    assert rendered == [
        "The blonde fox has long legs.",
        "The blonde fox has short legs.",
        "The brown fox has long legs.",
    ]


def test_parse_mustache_variables_yaml_rejects_nested_values():
    with pytest.raises(ValueError, match="scalars or lists of scalars"):
        parse_mustache_variables_yaml("haircolor:\n  nested: value\n")
