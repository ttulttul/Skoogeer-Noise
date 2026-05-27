import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_list import ModelsList  # noqa: E402


def test_models_list_declares_list_valued_model_output():
    node = ModelsList()
    inputs = node.INPUT_TYPES()

    assert node.RETURN_TYPES == ("MODEL",)
    assert node.OUTPUT_IS_LIST == (True,)
    assert inputs["required"]["model_1"][0] == "MODEL"
    assert inputs["required"]["model_2"][0] == "MODEL"
    assert inputs["optional"]["model_50"][0] == "MODEL"


def test_models_list_returns_models_in_socket_order_and_skips_missing_optionals():
    node = ModelsList()
    model_1 = object()
    model_2 = object()
    model_4 = object()

    (models,) = node.models_list(model_1, model_2, model_4=model_4)

    assert models == [model_1, model_2, model_4]


def test_comfyui_ksampler_model_input_is_list_mappable():
    comfyui_root = PROJECT_ROOT.parent / "ComfyUI"
    execution_source = (comfyui_root / "execution.py").read_text(encoding="utf-8")
    nodes_source = (comfyui_root / "nodes.py").read_text(encoding="utf-8")

    assert 'input_is_list = getattr(obj, "INPUT_IS_LIST", False)' in execution_source
    assert "for i in range(max_len_input):" in execution_source
    assert "input_dict = slice_dict(input_data_all, i)" in execution_source
    assert "output_is_list = obj.OUTPUT_IS_LIST" in execution_source
    assert '"model": ("MODEL"' in nodes_source
    assert "class KSampler:" in nodes_source
    assert "INPUT_IS_LIST" not in nodes_source[
        nodes_source.index("class KSampler:"):nodes_source.index("class KSamplerAdvanced:")
    ]

    models = ["model-a", "model-b", "model-c"]
    input_data_all = {
        "model": models,
        "seed": [123],
        "steps": [20],
    }
    max_len_input = max(len(values) for values in input_data_all.values())

    def slice_dict(values_by_name, index):
        return {
            key: values[index if len(values) > index else -1]
            for key, values in values_by_name.items()
        }

    ksampler_calls = [slice_dict(input_data_all, index) for index in range(max_len_input)]

    assert [call["model"] for call in ksampler_calls] == models
    assert [call["seed"] for call in ksampler_calls] == [123, 123, 123]
    assert [call["steps"] for call in ksampler_calls] == [20, 20, 20]
