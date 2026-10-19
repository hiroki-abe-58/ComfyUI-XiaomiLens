"""The pack's inference steps against the official infer.py functions, on the tiny checkpoint, bit for bit (CPU)."""

import json

import numpy as np
import pytest
import torch
from conftest import image

from xiaomi_lens_comfy import runtime

CPU = torch.device("cpu")


def official_run(official, model, processor, mask_id, img, text, max_new_tokens=64, kernel="flash"):
    """infer.py's steps on `model`. The decoder kernel is a setting of the model's module (upstream: XIAOMI_LENS_DECODER_ATTN at
    import, default "flash", which falls back to "mha" on the CPU); the model here is built from the vendored classes."""
    runtime.set_decoder_attention(kernel)
    inputs, _ = official.build_inputs(processor, img, text, CPU)
    ids, completion, total = official.generate_completion(model, processor, inputs, max_new_tokens, mask_id)
    _, thought = official.split_answer_content(completion)
    logits = official.run_forward(model, inputs, ids, mask_id, CPU)[:, thought:] if total - thought > 0 else None
    return inputs, ids, completion, logits


@pytest.mark.parametrize("size", [(256, 320), (300, 451), (512, 288)])
def test_predict_equals_official(official, tiny, size):
    model, processor, mask_id = tiny
    img = image(*size, seed=size[1])
    text = runtime.instruction("the cat on the left", "find")
    inputs, ids, completion, logits = official_run(official, model, processor, mask_id, img, text)
    r = runtime.predict(model, processor, mask_id, img, text, 64)
    assert r.completion_ids == ids[0].tolist() and r.text == completion
    assert completion == "</think><|mask|><|im_end|>"  # the scripted answer of the tiny model: one answer mask
    grid = inputs["image_grid_thw"][0].tolist()
    assert r.canvas == (grid[1] * 16, grid[2] * 16) == tuple(logits.shape[-2:])
    assert r.logits.dtype == torch.bfloat16 and torch.equal(r.logits, logits[0])


@pytest.mark.parametrize("kernel", ["sdpa", "mha"])
def test_decoder_kernels_equal_official(official, tiny, kernel):
    """The other decoder kernels upstream offers (XIAOMI_LENS_DECODER_ATTN=sdpa | mha); on the CPU "flash" is "mha"."""
    model, processor, mask_id = tiny
    img = image(224, 288, seed=5)
    text = runtime.instruction("the dog", "find")
    _, ids, completion, logits = official_run(official, model, processor, mask_id, img, text, kernel=kernel)
    r = runtime.predict(model, processor, mask_id, img, text, 64, decoder=kernel)
    assert r.text == completion and torch.equal(r.logits, logits[0])
    if kernel == "mha":
        assert torch.equal(runtime.predict(model, processor, mask_id, img, text, 64, decoder="flash").logits, r.logits)


def test_decoder_kernel_choice():
    assert runtime.decoder_attention("auto", CPU) == "sdpa" and runtime.decoder_attention("mha", CPU) == "mha"
    with pytest.raises(ValueError):
        runtime.decoder_attention("flash", CPU)
    with pytest.raises(ValueError):
        runtime.set_decoder_attention("fast")


def test_teacher_forced_masks_equal_official(official, tiny):
    """A completion with a mask inside <think>, two boxes and two answer masks: the same logits as infer.run_forward, and the
    thought mask is left out as infer.py leaves it out."""
    model, processor, mask_id = tiny
    img = image(seed=3)
    inputs = runtime.build_inputs(processor, img, runtime.instruction("coin", "find"), CPU)
    ref_inputs, _ = official.build_inputs(processor, img, runtime.instruction("coin", "find"), CPU)
    assert all(torch.equal(inputs[k], ref_inputs[k]) for k in ref_inputs if isinstance(ref_inputs[k], torch.Tensor))
    completion = ("first <|mask|> is the left one</think>\n\n<bbox>[10, 20, 300, 400]</bbox><mask><|mask|></mask>"
                  "<bbox>[500, 0, 1000, 999]</bbox><mask><|mask|></mask><|im_end|>")
    ids = processor.tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ours = runtime.mask_logits(model, inputs, ids, mask_id)
    ref = official.run_forward(model, ref_inputs, ids, mask_id, CPU)
    assert ours.shape[1] == 3 and torch.equal(ours, ref)
    answer, thought = runtime.split_answer(completion)
    assert (answer, thought) == official.split_answer_content(completion) and thought == 1
    assert runtime.answer_boxes(completion) == official.parse_answer_bboxes(completion) == [[10, 20, 300, 400], [500, 0, 1000, 999]]


@pytest.mark.parametrize("text", [
    "<bbox>[1, 2, 3, 4]</bbox>", "<bbox> [ 1 ,2, 3,4 ] </bbox> and <bbox>[5,6,7,8]</bbox>", "no boxes", "<bbox>[1, 2, 3]</bbox>",
    "<bbox>[9, 9, 9, 9]</bbox></think><bbox>[1, 1, 2, 2]</bbox>", "a</think>b</think><bbox>[3, 3, 4, 4]</bbox>",
])
def test_parsing_equals_official(official, text):
    assert runtime.answer_boxes(text) == official.parse_answer_bboxes(text)
    assert runtime.split_answer(text) == official.split_answer_content(text)


def test_find_prompt_is_the_readme_prompt():
    assert runtime.instruction(" the cat on the left ", "find") == (
        'Locate all instances in the image that match the description "the cat on the left". Report bbox coordinates and masks.')
    assert runtime.instruction("What can be squeezed onto seafood?", "instruction") == "What can be squeezed onto seafood?"
    with pytest.raises(ValueError):
        runtime.instruction("  ", "find")
    with pytest.raises(ValueError):
        runtime.instruction("x", "chat")


def test_masks_at_canvas_size_are_infer_py_binarization():
    logits = (torch.randn(3, 64, 96) * 4).bfloat16()
    logits[0, 0, :4] = torch.tensor([-0.001, 0.0, 0.001, -0.0005])  # 0.5 in bfloat16: foreground for infer.py, not in float32
    probs = torch.sigmoid(logits).float()  # infer.visualize_pixel_mask: torch.sigmoid(pred_masks[0, k]).float()
    masks = runtime.to_image_size(logits, 64, 96, 0.5)
    assert torch.equal(masks.bool(), probs >= 0.5) and masks[0, 0, :4].all()
    assert torch.equal(runtime.to_image_size(logits, 64, 96, 0.0), probs)  # threshold 0: infer.py's soft overlay


def test_masks_are_resized_to_the_image():
    logits = torch.full((1, 64, 96), -10.0)
    logits[0, 16:48, 24:72] = 10.0  # the middle half of the canvas
    masks = runtime.to_image_size(logits, 120, 200, 0.5)
    assert masks.shape == (1, 120, 200)
    assert masks[0, 35:85, 55:145].all() and not masks[0, :25].any() and not masks[0, :, :45].any()


def test_boxes_to_pixels():
    assert runtime.boxes_to_pixels([[0, 0, 1000, 1000], [250, 500, 750, 1000]], 200, 400) == [[0, 0, 400, 200], [100, 100, 300, 200]]


def test_segment_images_batch(tiny):
    model, processor, mask_id = tiny
    batch = torch.stack([torch.from_numpy(np.asarray(image(160, 224, seed=s))).float() / 255 for s in (1, 2)])
    seen = []
    out = runtime.segment_images(model, processor, mask_id, batch, "the cat", "find", 0.5, 64, image_callback=seen.append)
    assert out["mask"].shape == (2, 160, 224) and out["instance_masks"].shape == (2, 160, 224)
    assert out["preview"].shape == (2, 160, 224, 3) and seen == [1, 2] and out["instances"] == 2
    assert torch.equal(out["mask"], out["instance_masks"])  # one instance per image: the union is that mask
    assert out["text"] == "</think><|mask|><|im_end|>\n\n</think><|mask|><|im_end|>"
    notes = json.loads(out["boxes"])
    assert notes == [{"image": 0, "note": "0 boxes for 1 masks"}, {"image": 1, "note": "0 boxes for 1 masks"}]
    for b in range(2):
        r = runtime.predict(model, processor, mask_id, image(160, 224, seed=b + 1), runtime.instruction("the cat", "find"), 64)
        assert torch.equal(out["mask"][b], runtime.to_image_size(r.logits, 160, 224, 0.5)[0])


def test_no_answer_mask_gives_empty_masks(tiny):
    """max_new_tokens=1 stops before the <|mask|> token: empty masks, not an error."""
    model, processor, mask_id = tiny
    batch = torch.from_numpy(np.asarray(image(128, 128))).float()[None] / 255
    out = runtime.segment_images(model, processor, mask_id, batch, "the cat", "find", 0.5, 1)
    assert out["text"] == "</think>" and out["instances"] == 0
    assert not out["mask"].any() and out["instance_masks"].shape == (1, 128, 128) and not out["instance_masks"].any()
    assert torch.allclose(out["preview"], batch, atol=1 / 255)


def test_step_callback_can_interrupt(tiny):
    model, processor, mask_id = tiny

    class Stop(Exception):
        pass

    def stop(steps):
        if steps >= 2:
            raise Stop

    with pytest.raises(Stop):
        runtime.predict(model, processor, mask_id, image(), "x", 64, step_callback=stop)


def test_overlay_colors_and_boxes():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    masks = torch.zeros((2, 40, 60))
    masks[0, :10, :10] = 1
    masks[1, 30:, 50:] = 1
    out = runtime.overlay(img, masks, [[20, 15, 40, 25]])
    assert tuple(out[5, 5]) == (127, 34, 34)  # #ff4444 at alpha int(0.5 * 255) = 127 over black, as infer.py composites
    assert tuple(out[35, 55]) == (34, 127, 34)  # #44ff44
    assert tuple(out[15, 30]) == (255, 68, 68) and tuple(out[20, 30]) == (0, 0, 0)  # box outline in the first color


def test_attention_choice():
    assert runtime.attention_implementation("auto", CPU) == "sdpa"
    assert runtime.attention_implementation("eager", CPU) == "eager"
    with pytest.raises(ValueError):
        runtime.attention_implementation("flash", CPU)


def test_loader_rejects_a_checkpoint_without_mask_token(tiny_dir, tmp_path):
    import shutil

    folder = tmp_path / "plain"
    shutil.copytree(tiny_dir, folder)
    tok = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
    tok["added_tokens"] = [t for t in tok["added_tokens"] if t["content"] != "<|mask|>"]
    (folder / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    cfg = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg.pop("extra_special_tokens", None)
    (folder / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="no <\\|mask\\|> token"):
            runtime.load(str(folder), "sdpa")
    finally:
        runtime.release()
