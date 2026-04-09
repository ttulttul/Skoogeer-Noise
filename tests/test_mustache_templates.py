import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mustache_templates import (  # noqa: E402
    ConcatenateLists,
    JoinTextList,
    MustacheTemplate,
    MustacheVariableSampler,
    MustacheVariables,
    ReorderList,
    extract_template_variables,
    parse_mustache_variables_inputs,
    parse_mustache_variables_yaml,
    render_mustache_template_list,
    sample_mustache_variable_list,
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


def test_parse_mustache_variables_yaml_accepts_list_of_mappings():
    variables = parse_mustache_variables_yaml(
        "- haircolor:\n"
        "  - brown\n"
        "  - blonde\n"
        "- leglength:\n"
        "  - short\n"
        "  - long\n"
        "- haircolor:\n"
        "  - black\n"
    )

    assert variables == {
        "haircolor": ["brown", "blonde", "black"],
        "leglength": ["short", "long"],
    }


def test_parse_mustache_variables_inputs_merges_multiple_yaml_strings():
    variables = parse_mustache_variables_inputs([
        "haircolor:\n  - brown\n  - blonde\n",
        "leglength:\n  - short\n  - long\n",
        "haircolor:\n  - black\n",
    ])

    assert variables == {
        "haircolor": ["brown", "blonde", "black"],
        "leglength": ["short", "long"],
    }


def test_parse_mustache_variables_yaml_rejects_list_items_that_are_not_mappings():
    with pytest.raises(ValueError, match="lists must contain mappings"):
        parse_mustache_variables_yaml("- just\n- strings\n")


def test_extract_template_variables_preserves_first_appearance_order():
    variables = extract_template_variables(
        "A {{haircolor}} coat, {{leglength}} legs, and again {{haircolor}}."
    )

    assert variables == ["haircolor", "leglength"]


def test_sample_mustache_variable_list_generates_all_products_in_sequential_order():
    sampled = sample_mustache_variable_list(
        {
            "haircolor": ["brown", "blonde"],
            "leglength": ["short", "long", "weird"],
            "unused": ["ignored"],
        },
        sampling_mode="sequential",
    )

    assert sampled == [
        {"haircolor": "brown", "leglength": "short", "unused": "ignored"},
        {"haircolor": "brown", "leglength": "long", "unused": "ignored"},
        {"haircolor": "brown", "leglength": "weird", "unused": "ignored"},
        {"haircolor": "blonde", "leglength": "short", "unused": "ignored"},
        {"haircolor": "blonde", "leglength": "long", "unused": "ignored"},
        {"haircolor": "blonde", "leglength": "weird", "unused": "ignored"},
    ]


def test_sample_mustache_variable_list_limit_truncates_at_sampler():
    sampled = sample_mustache_variable_list(
        {
            "haircolor": ["brown", "blonde"],
            "leglength": ["short", "long", "weird"],
        },
        sampling_mode="sequential",
        limit=4,
    )

    assert sampled == [
        {"haircolor": "brown", "leglength": "short"},
        {"haircolor": "brown", "leglength": "long"},
        {"haircolor": "brown", "leglength": "weird"},
        {"haircolor": "blonde", "leglength": "short"},
    ]


def test_sample_mustache_variable_list_random_is_deterministic_for_seed():
    variables = {
        "haircolor": ["brown", "blonde", "black", "red"],
        "leglength": ["short", "long", "weird", "impossible"],
    }

    first = sample_mustache_variable_list(variables, sampling_mode="random", seed=42, limit=5)
    second = sample_mustache_variable_list(variables, sampling_mode="random", seed=42, limit=5)
    third = sample_mustache_variable_list(variables, sampling_mode="random", seed=43, limit=5)

    assert first == second
    assert first != third


def test_sample_mustache_variable_list_random_samples_full_permutations():
    sampled = sample_mustache_variable_list(
        {
            "haircolor": ["brown", "blonde"],
            "leglength": ["short", "long"],
            "hat": ["cap", "none"],
        },
        sampling_mode="random",
        seed=7,
        limit=3,
    )

    assert len(sampled) == 3
    assert len({tuple(item.items()) for item in sampled}) == 3
    for item in sampled:
        assert set(item.keys()) == {"haircolor", "leglength", "hat"}


def test_sample_mustache_variable_list_random_handles_population_above_ssize_t():
    variables = {f"v{i}": ["0", "1"] for i in range(70)}

    first = sample_mustache_variable_list(
        variables,
        sampling_mode="random",
        seed=123,
        limit=4,
    )
    second = sample_mustache_variable_list(
        variables,
        sampling_mode="random",
        seed=123,
        limit=4,
    )

    assert first == second
    assert len(first) == 4
    assert len({frozenset(item.items()) for item in first}) == 4
    for item in first:
        assert set(item.keys()) == set(variables.keys())


def test_sample_mustache_variable_list_empty_variables_returns_single_empty_setting():
    assert sample_mustache_variable_list({}, limit=-1) == [{}]


def test_sample_mustache_variable_list_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit must be >= -1"):
        sample_mustache_variable_list(
            {"haircolor": ["brown"]},
            limit=-2,
        )


def test_sample_mustache_variable_list_rejects_invalid_sampling_method():
    with pytest.raises(ValueError, match="sampling_method must be one of"):
        sample_mustache_variable_list(
            {"haircolor": ["brown"]},
            sampling_mode="chaotic",
        )


def test_sample_mustache_variable_list_requires_limit_for_large_unbounded_space():
    variables = {f"v{i}": ["0", "1"] for i in range(17)}

    with pytest.raises(ValueError, match="Set a finite limit when the permutation space is large"):
        sample_mustache_variable_list(
            variables,
            sampling_mode="random",
            limit=-1,
        )


def test_render_mustache_template_list_renders_each_variable_setting():
    rendered = render_mustache_template_list(
        "The {{haircolor}} fox has {{leglength}} legs.",
        [
            {"haircolor": "brown", "leglength": "short"},
            {"haircolor": "blonde", "leglength": "long"},
        ],
    )

    assert rendered == [
        "The brown fox has short legs.",
        "The blonde fox has long legs.",
    ]


def test_render_mustache_template_list_repeats_plain_text_for_each_setting():
    rendered = render_mustache_template_list(
        "plain text",
        [
            {"haircolor": "brown"},
            {"haircolor": "blonde"},
        ],
    )

    assert rendered == ["plain text", "plain text"]


def test_render_mustache_template_list_raises_for_missing_variable():
    with pytest.raises(ValueError, match="undefined variables in entry 0: leglength"):
        render_mustache_template_list(
            "The {{haircolor}} fox has {{leglength}} legs.",
            [{"haircolor": "brown"}],
        )


def test_mustache_variables_node_returns_typed_mapping():
    node = MustacheVariables()

    (variables,) = node.parse_variables("haircolor:\n  - brown\n  - blonde\n")

    assert node.INPUT_IS_LIST is True
    assert variables == {"haircolor": ["brown", "blonde"]}


def test_mustache_variables_node_accepts_list_of_yaml_strings():
    node = MustacheVariables()

    (variables,) = node.parse_variables([
        "haircolor:\n  - brown\n",
        "haircolor:\n  - blonde\n",
        "leglength:\n  - short\n  - long\n",
    ])

    assert variables == {
        "haircolor": ["brown", "blonde"],
        "leglength": ["short", "long"],
    }


def test_mustache_variables_node_renders_yaml_against_variable_list_input():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal}}\n"
        "camera_lens:\n"
        "  - {{lens}}\n",
        [
            {"animal": "fox", "lens": "35mm"},
            {"animal": "wolf", "lens": "85mm"},
        ],
    )

    assert variables == {
        "subject_identity": ["fox", "wolf"],
        "camera_lens": ["35mm", "85mm"],
    }


def test_second_stage_mustache_variables_workflow_uses_upstream_variable_settings():
    first_stage_variables = {
        "animal": ["fox", "wolf"],
        "lens": ["35mm", "85mm"],
    }

    first_stage_sampled = sample_mustache_variable_list(
        first_stage_variables,
        sampling_mode="sequential",
        limit=2,
    )

    reordered_stage = ReorderList()
    (reordered_variables,) = reordered_stage.reorder(first_stage_sampled, ["reverse"], [0])

    second_stage_node = MustacheVariables()
    (second_stage_variables,) = second_stage_node.parse_variables(
        "subject_identity:\n"
        "  - {{animal}}\n"
        "camera_lens:\n"
        "  - {{lens}}\n"
        "prompt:\n"
        "  - portrait of a {{animal}}\n",
        reordered_variables,
    )

    assert second_stage_variables == {
        "subject_identity": ["fox", "fox"],
        "camera_lens": ["85mm", "35mm"],
        "prompt": ["portrait of a fox", "portrait of a fox"],
    }


def test_mustache_variable_sampler_returns_variable_list():
    node = MustacheVariableSampler()

    (variables,) = node.sample(
        {"haircolor": ["brown", "blonde"], "leglength": ["short", "long"]},
        "sequential",
        0,
        3,
    )

    assert variables == [
        {"haircolor": "brown", "leglength": "short"},
        {"haircolor": "brown", "leglength": "long"},
        {"haircolor": "blonde", "leglength": "short"},
    ]


def test_mustache_template_node_returns_list_output():
    node = MustacheTemplate()

    (rendered,) = node.render(
        [
            {"haircolor": "brown", "leglength": "short"},
            {"haircolor": "brown", "leglength": "long"},
            {"haircolor": "blonde", "leglength": "short"},
        ],
        "The {{haircolor}} fox has {{leglength}} legs.",
    )

    assert node.OUTPUT_IS_LIST == (True,)
    assert rendered == [
        "The brown fox has short legs.",
        "The brown fox has long legs.",
        "The blonde fox has short legs.",
    ]


def test_join_text_list_joins_all_items_for_preview():
    node = JoinTextList()

    joined, count = node.join_text([
        "The brown fox has short legs.",
        "The brown fox has long legs.",
        "The blonde fox has short legs.",
    ], "\\n")

    assert node.INPUT_IS_LIST == (True,)
    assert joined == (
        "The brown fox has short legs.\n"
        "The brown fox has long legs.\n"
        "The blonde fox has short legs."
    )
    assert count == 3


def test_join_text_list_accepts_custom_escaped_separator():
    node = JoinTextList()

    joined, count = node.join_text([
        "alpha",
        "beta",
        "gamma",
    ], "\\n===\\n")

    assert joined == "alpha\n===\nbeta\n===\ngamma"
    assert count == 3


def test_reorder_list_reverse_returns_reversed_items():
    node = ReorderList()

    (items,) = node.reorder(
        ["alpha", "beta", "gamma"],
        ["reverse"],
        [123],
    )

    assert node.INPUT_IS_LIST is True
    assert node.OUTPUT_IS_LIST == (True,)
    assert items == ["gamma", "beta", "alpha"]


def test_reorder_list_shuffle_is_deterministic_for_seed():
    node = ReorderList()

    (first,) = node.reorder([1, 2, 3, 4], ["shuffle"], [42])
    (second,) = node.reorder([1, 2, 3, 4], ["shuffle"], [42])
    (third,) = node.reorder([1, 2, 3, 4], ["shuffle"], [43])

    assert first == second
    assert sorted(first) == [1, 2, 3, 4]
    assert first != third


def test_concatenate_lists_joins_mustache_variable_list_batches():
    node = ConcatenateLists()

    (items,) = node.concatenate(
        [
            {"animal": "fox", "lens": "35mm"},
            {"animal": "wolf", "lens": "50mm"},
        ],
        [
            {"animal": "cat", "lens": "85mm"},
        ],
    )

    assert node.INPUT_IS_LIST == (True, True)
    assert node.OUTPUT_IS_LIST == (True,)
    assert items == [
        {"animal": "fox", "lens": "35mm"},
        {"animal": "wolf", "lens": "50mm"},
        {"animal": "cat", "lens": "85mm"},
    ]


def test_parse_mustache_variables_yaml_rejects_nested_values():
    with pytest.raises(ValueError, match="scalars or lists of scalars"):
        parse_mustache_variables_yaml("haircolor:\n  nested: value\n")
