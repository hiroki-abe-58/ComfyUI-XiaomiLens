"""The node pack as ComfyUI loads it: registration, model discovery, ComfyUI's own validation of the shipped workflows, and node
execution on an image decoded by ComfyUI's own Load Image, compared with the runtime (which tests/test_runtime.py compares with the
official infer.py)."""

import asyncio
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import image

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"XiaomiLensLoader", "XiaomiLensSegment"}


def pack_module(comfy_nodes, name):
    """A module of the pack as ComfyUI imported it (under a path-based package name, so not the `xiaomi_lens_comfy` the tests import)."""
    return importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS["XiaomiLensSegment"].__module__.rsplit(".", 1)[0] + "." + name)


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, tiny_dir, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    folder_paths.set_output_directory(str(base / "output"))
    folder_paths.set_input_directory(str(base / "input"))
    (base / "input").mkdir()
    image(192, 256, seed=7).save(base / "input" / "example.png")
    models = base / "models"
    shutil.copytree(tiny_dir, models / "Xiaomi-Lens-0")  # the name the shipped workflows use
    (models / "not-a-model").mkdir()
    folder_paths.add_model_folder_path("xiaomi_lens", str(models), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    yield nodes
    pack_module(nodes, "runtime").release()


def _model(comfy_nodes, after_run="keep on device"):
    return comfy_nodes.NODE_CLASS_MAPPINGS["XiaomiLensLoader"].execute("Xiaomi-Lens-0", "cpu", "auto", "auto", after_run).args[0]


def _image(comfy_nodes):
    return comfy_nodes.NODE_CLASS_MAPPINGS["LoadImage"]().load_image("example.png")[0]


def test_registered_and_models_listed(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)
    schema = comfy_nodes.NODE_CLASS_MAPPINGS["XiaomiLensLoader"].GET_SCHEMA()
    options = next(i for i in schema.inputs if i.id == "model").options
    assert options == ["Xiaomi-Lens-0"]  # folders without a checkpoint are not offered


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_gui_workflows_match_api_workflows():
    """Every API workflow has a UI workflow with the same node types and widget values (exported by scripts/export_gui_workflows.py)."""
    for api_file in (ROOT / "workflows" / "api").glob("*.json"):
        api = json.loads(api_file.read_text(encoding="utf-8"))
        gui = json.loads((ROOT / "workflows" / "gui" / api_file.name).read_text(encoding="utf-8"))
        assert sorted(n["class_type"] for n in api.values()) == sorted(n["type"] for n in gui["nodes"])
        for node_id, node in api.items():
            ui = next(n for n in gui["nodes"] if str(n["id"]) == node_id)
            widgets = [v for v in node["inputs"].values() if not isinstance(v, list)]
            assert [w for w in ui.get("widgets_values", []) if w in widgets] == widgets, (api_file.name, node_id)


def test_segment_node_equals_runtime(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    img = _image(comfy_nodes)
    out = comfy_nodes.NODE_CLASS_MAPPINGS["XiaomiLensSegment"].execute(_model(comfy_nodes), img, "the cat", "find", 0.5, 64).args
    mask, instances, preview, text, boxes = out
    handle = _model(comfy_nodes)
    assert handle["decoder"] == "sdpa" and handle["attention"] == "sdpa"  # auto on the CPU
    model, processor, mask_id = runtime.load(handle["folder"], handle["attention"])
    pixels = (img[0].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    from PIL import Image

    r = runtime.predict(model, processor, mask_id, Image.fromarray(pixels), runtime.instruction("the cat", "find"), 64, decoder="sdpa")
    expected = runtime.to_image_size(r.logits, 192, 256, 0.5)
    assert torch.equal(mask[0], expected[0]) and torch.equal(instances, expected)
    assert preview.shape == (1, 192, 256, 3) and text == r.text
    assert json.loads(boxes) == [{"image": 0, "note": "0 boxes for 1 masks"}]
    assert np.array_equal(pixels, np.asarray(image(192, 256, seed=7)))  # Load Image gave the file's pixels


def test_move_to_cpu_after_run(comfy_nodes):
    model = _model(comfy_nodes, "move to CPU")
    comfy_nodes.NODE_CLASS_MAPPINGS["XiaomiLensSegment"].execute(model, _image(comfy_nodes), "x", "find", 0.5, 4)
    runtime = pack_module(comfy_nodes, "runtime")
    net = runtime.load(model["folder"], model["attention"])[0]
    assert next(net.parameters()).device.type == "cpu"


def test_invalid_inputs_are_rejected(comfy_nodes):
    loader = comfy_nodes.NODE_CLASS_MAPPINGS["XiaomiLensLoader"]
    with pytest.raises(FileNotFoundError):
        loader.execute("../Xiaomi-Lens-0", "cpu", "auto", "auto", "keep on device")
    with pytest.raises(FileNotFoundError):
        loader.execute("not-a-model", "cpu", "auto", "auto", "keep on device")
    with pytest.raises(ValueError):
        loader.execute("Xiaomi-Lens-0", "cuda:7", "auto", "auto", "keep on device")
    with pytest.raises(ValueError):
        loader.execute("Xiaomi-Lens-0", "cpu", "auto", "flash", "keep on device")
    segment = comfy_nodes.NODE_CLASS_MAPPINGS["XiaomiLensSegment"]
    with pytest.raises(ValueError):
        segment.execute(_model(comfy_nodes), _image(comfy_nodes), "   ", "find", 0.5, 4)
    with pytest.raises(ValueError):
        segment.execute(_model(comfy_nodes), _image(comfy_nodes), "x", "chat", 0.5, 4)
