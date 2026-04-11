import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mustache_templates import (  # noqa: E402
    ConcatenateLists,
    JoinTextList,
    MergeMustacheVariableLists,
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
    )

    assert variables == {
        "haircolor": ["brown", "blonde"],
        "leglength": ["short", "long"],
    }


def test_parse_mustache_variables_yaml_rejects_duplicate_top_level_keys():
    with pytest.raises(ValueError, match="defined more than once"):
        parse_mustache_variables_yaml(
            "base:\n"
            "  - x\n"
            "location:\n"
            "  - {{base}}\n"
            "middle:\n"
            "  - y\n"
            "location:\n"
            "  - {{middle}}\n"
        )


def test_parse_mustache_variables_yaml_rejects_duplicate_keys_in_mapping_list():
    with pytest.raises(ValueError, match="defined more than once"):
        parse_mustache_variables_yaml(
            "- haircolor:\n"
            "  - brown\n"
            "- haircolor:\n"
            "  - black\n"
        )


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


def test_parse_mustache_variables_yaml_expands_local_references_without_order_constraints():
    variables = parse_mustache_variables_yaml(
        "hairstyle:\n"
        "  - {{color:static}} hair in a {{hairarrangement:static}}\n"
        "color:\n"
        "  - brown\n"
        "  - blue\n"
        "hairarrangement:\n"
        "  - ponytail\n"
        "  - bun\n"
    )

    assert variables["color"] == ["brown", "blue"]
    assert variables["hairarrangement"] == ["ponytail", "bun"]
    assert variables["hairstyle"] == ["{{color:static}} hair in a {{hairarrangement:static}}"]

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        limit=-1,
    )
    assert sampled == [
        {
            "color": "brown",
            "hairarrangement": "ponytail",
            "hairstyle": "brown hair in a ponytail",
        },
        {
            "color": "brown",
            "hairarrangement": "bun",
            "hairstyle": "brown hair in a bun",
        },
        {
            "color": "blue",
            "hairarrangement": "ponytail",
            "hairstyle": "blue hair in a ponytail",
        },
        {
            "color": "blue",
            "hairarrangement": "bun",
            "hairstyle": "blue hair in a bun",
        },
    ]


def test_parse_mustache_variables_yaml_rejects_undefined_local_references():
    with pytest.raises(ValueError, match="references undefined variables"):
        parse_mustache_variables_yaml(
            "hairstyle:\n"
            "  - {{color}} hair\n"
        )


def test_parse_mustache_variables_yaml_rejects_unknown_instance_settings():
    with pytest.raises(ValueError, match="Unsupported mustache instance setting"):
        parse_mustache_variables_yaml(
            "hairstyle:\n"
            "  - {{color:sideways}} hair\n"
            "color:\n"
            "  - brown\n"
        )


def test_parse_mustache_variables_yaml_rejects_conflicting_case_settings():
    with pytest.raises(ValueError, match="cannot combine multiple case transforms"):
        parse_mustache_variables_yaml(
            "hairstyle:\n"
            "  - {{color:uppercase,lowercase}} hair\n"
            "color:\n"
            "  - brown\n"
        )


def test_parse_mustache_variables_yaml_rejects_multiple_selection_settings():
    with pytest.raises(ValueError, match="cannot combine multiple selection actions"):
        parse_mustache_variables_yaml(
            "hairstyle:\n"
            "  - {{color:repeat,randomize}} hair\n"
            "color:\n"
            "  - brown\n"
        )


def test_parse_mustache_variables_yaml_distributes_remaining_weight_across_unweighted_values():
    variables = parse_mustache_variables_yaml(
        "color:\n"
        "  - brown\n"
        "  - purple\n"
        "  - white:0.5\n"
    )

    assert variables["color"] == ["brown", "purple", "white"]

    weights = variables["__mustache_weights__"]["color"]
    assert weights == pytest.approx([0.25, 0.25, 0.5])


def test_parse_mustache_variables_yaml_accepts_quoted_weight_shorthand():
    variables = parse_mustache_variables_yaml(
        "lens_effect:\n"
        '  - " with shallow focus":0.2\n'
        '  - " with motion blur":0.1\n'
        '  - "":0.7\n'
    )

    assert variables["lens_effect"] == [" with shallow focus", " with motion blur", ""]

    weights = variables["__mustache_weights__"]["lens_effect"]
    assert weights == pytest.approx([0.2, 0.1, 0.7])


def test_parse_mustache_variables_yaml_rejects_weight_total_above_one():
    with pytest.raises(ValueError, match="weights must not exceed 1.0"):
        parse_mustache_variables_yaml(
            "color:\n"
            "  - brown:0.8\n"
            "  - blue:0.3\n"
            "  - black\n"
        )


def test_parse_mustache_variables_yaml_rejects_weight_sum_not_equal_to_one():
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        parse_mustache_variables_yaml(
            "color:\n"
            "  - brown:0.2\n"
            "  - blue:0.2\n"
            "  - black:0.2\n"
        )


def test_parse_mustache_variables_yaml_rejects_list_items_that_are_not_mappings():
    with pytest.raises(ValueError, match="lists must contain mappings"):
        parse_mustache_variables_yaml("- just\n- strings\n")


def test_extract_template_variables_preserves_first_appearance_order():
    variables = extract_template_variables(
        "A {{haircolor}} coat, {{leglength}} legs, and again {{haircolor}}."
    )

    assert variables == ["haircolor", "leglength"]


def test_extract_template_variables_ignores_instance_settings_and_unescapes_colons():
    variables = extract_template_variables(
        "A {{haircolor:randomize}} coat, {{haircolor:static}}, {{haircolor:repeat}}, {{haircolor:lowercase}}, {{haircolor:propercase}}, {{haircolor:uppercase}}, {{haircolor:notrim}}, and {{camera\\:lens}}."
    )

    assert variables == ["haircolor", "camera:lens"]


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


def test_sample_mustache_variable_list_random_respects_value_weights():
    weighted_variables = parse_mustache_variables_yaml(
        "color:\n"
        "  - brown:0.3\n"
        "  - blue:0.3\n"
        "  - black:0.4\n"
    )

    counts = {"brown": 0, "blue": 0, "black": 0}
    for seed in range(1000):
        sampled = sample_mustache_variable_list(
            weighted_variables,
            sampling_mode="random",
            seed=seed,
            limit=1,
        )
        counts[sampled[0]["color"]] += 1

    assert 0.24 <= counts["brown"] / 1000 <= 0.36
    assert 0.24 <= counts["blue"] / 1000 <= 0.36
    assert 0.33 <= counts["black"] / 1000 <= 0.47


def test_sample_mustache_variable_list_supports_randomize_and_repeat_settings():
    variables = parse_mustache_variables_yaml(
        "color:\n"
        "  - brown\n"
        "  - blue\n"
        "  - black\n"
        "pair:\n"
        "  - {{color:randomize}} and {{color:repeat}}\n"
    )

    first = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        seed=42,
        limit=-1,
    )
    second = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        seed=42,
        limit=-1,
    )

    assert first == second
    for entry in first:
        left, right = entry["pair"].split(" and ", maxsplit=1)
        assert left == right


def test_sample_mustache_variable_list_trims_by_default():
    variables = parse_mustache_variables_yaml(
        "TitleCase:\n"
        "  - '  Golden Hour  '\n"
        "label:\n"
        "  - {{TitleCase}}\n"
    )

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        limit=-1,
    )

    assert sampled == [
        {"TitleCase": "  Golden Hour  ", "label": "Golden Hour"},
    ]


def test_sample_mustache_variable_list_supports_notrim_setting():
    variables = parse_mustache_variables_yaml(
        "TitleCase:\n"
        "  - '  Golden Hour  '\n"
        "label:\n"
        "  - {{TitleCase:notrim}}\n"
    )

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        limit=-1,
    )

    assert sampled == [
        {"TitleCase": "  Golden Hour  ", "label": "  Golden Hour  "},
    ]


def test_sample_mustache_variable_list_supports_lowercase_setting():
    variables = parse_mustache_variables_yaml(
        "TitleCase:\n"
        "  - Golden Hour\n"
        "label:\n"
        "  - {{TitleCase:lowercase}}\n"
    )

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        limit=-1,
    )

    assert sampled == [
        {"TitleCase": "Golden Hour", "label": "golden hour"},
    ]


def test_sample_mustache_variable_list_supports_propercase_setting():
    variables = parse_mustache_variables_yaml(
        "word:\n"
        "  - hello\n"
        "label:\n"
        "  - {{word:propercase}}\n"
    )

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        limit=-1,
    )

    assert sampled == [
        {"word": "hello", "label": "Hello"},
    ]


def test_sample_mustache_variable_list_supports_uppercase_setting():
    variables = parse_mustache_variables_yaml(
        "word:\n"
        "  - hello\n"
        "label:\n"
        "  - {{word:uppercase}}\n"
    )

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        limit=-1,
    )

    assert sampled == [
        {"word": "hello", "label": "HELLO"},
    ]


def test_sample_mustache_variable_list_supports_default_randomize_setting():
    variables = parse_mustache_variables_yaml(
        "word:\n"
        "  - hello\n"
        "  - world\n"
        "label:\n"
        "  - {{word}}\n"
    )

    first = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        seed=42,
        limit=-1,
    )
    second = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        seed=42,
        limit=-1,
    )

    assert first == second
    assert {entry["label"] for entry in first} <= {"hello", "world"}
    assert any(entry["label"] != entry["word"] for entry in first)


def test_sample_mustache_variable_list_supports_static_setting():
    variables = parse_mustache_variables_yaml(
        "word:\n"
        "  - hello\n"
        "  - world\n"
        "label:\n"
        "  - {{word:static}}\n"
    )

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        seed=42,
        limit=-1,
    )

    assert sampled == [
        {"word": "hello", "label": "hello"},
        {"word": "world", "label": "world"},
    ]


def test_sample_mustache_variable_list_random_preserves_lazy_dependency_order():
    variables = parse_mustache_variables_yaml(
        "body_type2:\n"
        "  - slim\n"
        "  - athletic\n"
        "build2:\n"
        "  - {{body_type2:static}}\n"
    )

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="random",
        seed=42,
        limit=4,
    )

    assert len(sampled) == 2
    assert {entry["build2"] for entry in sampled} == {"slim", "athletic"}
    for entry in sampled:
        assert entry["build2"] == entry["body_type2"]


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


def test_render_mustache_template_list_renders_repeated_placeholders():
    rendered = render_mustache_template_list(
        "{{subject}} with {{subject}} energy",
        [
            {"subject": "fox"},
            {"subject": "wolf"},
        ],
    )

    assert rendered == [
        "fox with fox energy",
        "wolf with wolf energy",
    ]


def test_render_mustache_template_list_preserves_literal_braces():
    rendered = render_mustache_template_list(
        "literal {curly} braces and {{subject}}",
        [
            {"subject": "fox"},
        ],
    )

    assert rendered == ["literal {curly} braces and fox"]


def test_render_mustache_template_list_supports_variable_names_with_spaces():
    rendered = render_mustache_template_list(
        "Lens: {{camera lens}}",
        [
            {"camera lens": "35mm"},
        ],
    )

    assert rendered == ["Lens: 35mm"]


def test_render_mustache_template_list_trims_by_default():
    rendered = render_mustache_template_list(
        "Lens: {{camera lens}}",
        [
            {"camera lens": "  35mm  "},
        ],
    )

    assert rendered == ["Lens: 35mm"]


def test_render_mustache_template_list_supports_notrim_setting():
    rendered = render_mustache_template_list(
        "Lens: {{camera lens:notrim}}",
        [
            {"camera lens": "  35mm  "},
        ],
    )

    assert rendered == ["Lens:   35mm  "]


def test_render_mustache_template_list_supports_lowercase_setting():
    rendered = render_mustache_template_list(
        "Lens: {{camera lens:lowercase}}",
        [
            {"camera lens": "35MM"},
        ],
    )

    assert rendered == ["Lens: 35mm"]


def test_render_mustache_template_list_supports_propercase_setting():
    rendered = render_mustache_template_list(
        "Lens: {{camera lens:propercase}}",
        [
            {"camera lens": "hello"},
        ],
    )

    assert rendered == ["Lens: Hello"]


def test_render_mustache_template_list_supports_uppercase_setting():
    rendered = render_mustache_template_list(
        "Lens: {{camera lens:uppercase}}",
        [
            {"camera lens": "hello"},
        ],
    )

    assert rendered == ["Lens: HELLO"]


def test_render_mustache_template_list_supports_uppercase_and_trim_settings():
    rendered = render_mustache_template_list(
        "Lens: {{camera lens:uppercase}}",
        [
            {"camera lens": "  hello  "},
        ],
    )

    assert rendered == ["Lens: HELLO"]


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


def test_mustache_variables_node_renders_yaml_against_variable_list_input_with_instance_settings():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal:repeat}}\n",
        [
            {"animal": "fox"},
        ],
    )

    assert variables == {
        "subject_identity": ["fox"],
    }


def test_mustache_variables_node_renders_yaml_against_variable_list_input_trims_by_default():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal}}\n",
        [
            {"animal": "  FOX  "},
        ],
    )

    assert variables == {
        "subject_identity": ["FOX"],
    }


def test_mustache_variables_node_renders_yaml_against_variable_list_input_with_notrim_setting():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal:notrim}}\n",
        [
            {"animal": "  FOX  "},
        ],
    )

    assert variables == {
        "subject_identity": ["  FOX  "],
    }


def test_mustache_variables_node_renders_yaml_against_variable_list_input_with_lowercase_setting():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal:lowercase}}\n",
        [
            {"animal": "FOX"},
        ],
    )

    assert variables == {
        "subject_identity": ["fox"],
    }


def test_mustache_variables_node_renders_yaml_against_variable_list_input_with_propercase_setting():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal:propercase}}\n",
        [
            {"animal": "fox"},
        ],
    )

    assert variables == {
        "subject_identity": ["Fox"],
    }


def test_mustache_variables_node_renders_yaml_against_variable_list_input_with_uppercase_setting():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal:uppercase}}\n",
        [
            {"animal": "fox"},
        ],
    )

    assert variables == {
        "subject_identity": ["FOX"],
    }


def test_mustache_variables_node_expands_local_references_after_input_rendering():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "color:\n"
        "  - {{animal}}-brown\n"
        "  - {{animal}}-black\n"
        "hairstyle:\n"
        "  - {{color:static}} hair\n",
        [
            {"animal": "fox"},
        ],
    )

    assert variables["color"] == ["fox-brown", "fox-black"]
    assert variables["hairstyle"] == ["{{color:static}} hair"]

    sampled = sample_mustache_variable_list(
        variables,
        sampling_mode="sequential",
        limit=-1,
    )
    assert sampled == [
        {"color": "fox-brown", "hairstyle": "fox-brown hair"},
        {"color": "fox-black", "hairstyle": "fox-black hair"},
    ]


def test_mustache_variables_node_accepts_nested_variable_list_wrappers():
    node = MustacheVariables()

    (variables,) = node.parse_variables(
        "subject_identity:\n"
        "  - {{animal}}\n"
        "camera_lens:\n"
        "  - {{lens}}\n",
        [
            [
                {"animal": "fox", "lens": "35mm"},
                {"animal": "wolf", "lens": "50mm"},
            ],
            [
                {"animal": "cat", "lens": "85mm"},
            ],
        ],
    )

    assert variables == {
        "subject_identity": ["fox", "wolf", "cat"],
        "camera_lens": ["35mm", "50mm", "85mm"],
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


def test_join_text_list_unwraps_singleton_list_separator_values():
    node = JoinTextList()

    joined, count = node.join_text([
        "text entry 1",
        "text entry 2",
    ], ["\\n===\\n"])

    assert joined == "text entry 1\n===\ntext entry 2"
    assert count == 2


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


def test_merge_mustache_variable_lists_merges_entries_by_index():
    node = MergeMustacheVariableLists()

    (items,) = node.merge(
        [
            {"camera_lens": "85mm", "lighting_style": "gritty neon glow"},
            {"camera_lens": "35mm", "lighting_style": "soft overcast"},
        ],
        [
            {"person_masc": "40-year-old burly Black man, a mechanic", "outfit_masc": "grime-covered charcoal flannel polo"},
            {"person_masc": "22-year-old slim East Asian man, a student", "outfit_masc": "clean white tank top"},
        ],
    )

    assert items == [
        {
            "camera_lens": "85mm",
            "lighting_style": "gritty neon glow",
            "person_masc": "40-year-old burly Black man, a mechanic",
            "outfit_masc": "grime-covered charcoal flannel polo",
        },
        {
            "camera_lens": "35mm",
            "lighting_style": "soft overcast",
            "person_masc": "22-year-old slim East Asian man, a student",
            "outfit_masc": "clean white tank top",
        },
    ]


def test_merge_mustache_variable_lists_broadcasts_singleton_side():
    node = MergeMustacheVariableLists()

    (items,) = node.merge(
        [
            {"camera_lens": "85mm"},
            {"camera_lens": "35mm"},
        ],
        [
            {"quality_boosters": "masterpiece, photorealistic"},
        ],
    )

    assert items == [
        {"camera_lens": "85mm", "quality_boosters": "masterpiece, photorealistic"},
        {"camera_lens": "35mm", "quality_boosters": "masterpiece, photorealistic"},
    ]


def test_merge_mustache_variable_lists_rejects_incompatible_lengths():
    node = MergeMustacheVariableLists()

    with pytest.raises(ValueError, match="different lengths unless one side has exactly one entry"):
        node.merge(
            [{"a": "1"}, {"a": "2"}],
            [{"b": "x"}, {"b": "y"}, {"b": "z"}],
        )


def test_parse_mustache_variables_yaml_rejects_nested_values():
    with pytest.raises(ValueError, match="scalars or lists of scalars"):
        parse_mustache_variables_yaml("haircolor:\n  nested: value\n")
