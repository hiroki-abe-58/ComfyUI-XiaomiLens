"""Xiaomi-Lens-0 inference without ComfyUI: the steps of the official infer.py, plus masks and boxes at the input image's size.

The official usage (README and infer.py of SeerRay-Lab/Xiaomi-Lens-0) is

    model, processor, mask_token_id = load_model_and_processor(<folder>, device)       # bfloat16, FlashAttention-2
    inputs, _ = build_inputs(processor, image, text, device)                          # chat template, image first
    completion_ids, text, n = generate_completion(model, processor, inputs, 16384, mask_token_id)   # greedy
    _, thought = split_answer_content(text)                                           # masks inside <think> are not answers
    logits = run_forward(model, inputs, completion_ids, mask_token_id, device)[:, thought:]        # [1, K, 16*gh, 16*gw]
    masks = logits.sigmoid() >= 0.5                                                   # on the canvas the vision encoder saw

`predict()` runs the same steps with the same calls. The canvas is the input image resized to multiples of 32 pixels (the Qwen
image processor), so `to_image_size()` resizes infer.py's probability map to the input image's size before the threshold; on an
image that already has the canvas size it is infer.py's binarization exactly. The model is loaded from a local folder only; nothing is downloaded at run time.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw

from .vendor.xiaomi_lens.constants import BOX_END_TOKEN, BOX_START_TOKEN, DEFAULT_MAX_NEW_TOKENS, MASK_TOKEN

MIN_TRANSFORMERS = (5, 4, 0)  # Qwen3.5 in Transformers; the version upstream tested
FIND_PROMPT = 'Locate all instances in the image that match the description "{}". Report bbox coordinates and masks.'  # README
PROMPT_MODES = ("find", "instruction")
ATTENTION = ("auto", "sdpa", "eager")
DECODER_ATTENTION = ("auto", "sdpa", "mha")  # the kernels of the mask decoder's patch self-attention (upstream: flash | sdpa | mha)
PALETTE = ("#ff4444", "#44ff44", "#4488ff", "#ffaa00", "#aa44ff", "#00cccc", "#ff44aa", "#aaff44")  # infer.py's overlay colors
_BOX_RE = re.compile(rf"{re.escape(BOX_START_TOKEN)}\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*{re.escape(BOX_END_TOKEN)}")  # infer.py

_cache: dict = {}


@dataclass
class Result:
    text: str  # the generated completion (special tokens kept, as infer.py prints it)
    completion_ids: list[int]
    thought_masks: int  # <|mask|> tokens inside <think> (not answers)
    logits: torch.Tensor  # [K, canvas_h, canvas_w] in the model's dtype, on the CPU: one per answer <|mask|>
    boxes_1000: list[list[int]]  # answer boxes, normalized to 0-1000 (x1, y1, x2, y2)
    canvas: tuple[int, int]  # (height, width) the vision encoder saw


def transformers_version() -> tuple[int, ...]:
    import transformers

    return tuple(int(p) for p in re.findall(r"\d+", transformers.__version__)[:3])


def attention_implementation(choice: str, device: torch.device) -> str:
    """auto: FlashAttention-2 (upstream's choice) when flash-attn is installed and the model runs on CUDA, otherwise PyTorch SDPA."""
    if choice not in ATTENTION:
        raise ValueError(f"attention must be one of {ATTENTION}")
    if choice != "auto":
        return choice
    from transformers.utils import is_flash_attn_2_available

    return "flash_attention_2" if torch.device(device).type == "cuda" and is_flash_attn_2_available() else "sdpa"


def decoder_attention(choice: str, device: torch.device) -> str:
    """The mask decoder's kernel. auto: "flash" (upstream's default) when flash-attn is installed and the model runs on CUDA, otherwise
    "sdpa". Upstream falls back to "mha" (nn.MultiheadAttention) without flash-attn, which holds the whole patch-by-patch attention
    matrix: 16 heads x patches^2 in bfloat16, about 2 GB for a 2 MP image and 32 GB at the processor's 8 MP limit. "sdpa" is upstream's
    PyTorch scaled_dot_product_attention path with the same weights (results equal up to rounding, docs/VERIFICATION.md)."""
    if choice not in DECODER_ATTENTION:
        raise ValueError(f"decoder attention must be one of {DECODER_ATTENTION}")
    if choice != "auto":
        return choice
    from transformers.utils import is_flash_attn_2_available

    return "flash" if torch.device(device).type == "cuda" and is_flash_attn_2_available() else "sdpa"


def set_decoder_attention(kernel: str) -> None:
    """Select the vendored decoder's kernel: the module setting upstream reads from XIAOMI_LENS_DECODER_ATTN at import."""
    from .vendor.xiaomi_lens.model import pixel_mask_decoder

    if kernel not in ("flash", "sdpa", "mha"):
        raise ValueError(f"unknown decoder kernel {kernel!r}")
    pixel_mask_decoder._DECODER_ATTN = kernel


def load(folder: str, attention: str = "sdpa"):
    """(model, processor, mask_token_id) of the checkpoint in `folder`, bfloat16 on the CPU (as upstream loads it, without
    device_map); cached per folder and attention implementation, one model at a time."""
    if transformers_version() < MIN_TRANSFORMERS:
        import transformers

        raise RuntimeError(f"Xiaomi-Lens-0 needs transformers >= 5.4.0 (Qwen3.5); installed: {transformers.__version__}")
    from transformers import AutoProcessor

    from .vendor.xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration

    key = (os.path.realpath(folder), attention)
    if key not in _cache:
        _cache.clear()
        model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(folder, dtype=torch.bfloat16, attn_implementation=attention,
                                                                         local_files_only=True)
        processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        mask_token_id = processor.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        if not isinstance(mask_token_id, int) or mask_token_id == processor.tokenizer.unk_token_id:
            raise ValueError(f"{folder}: the tokenizer has no {MASK_TOKEN} token (not a Xiaomi-Lens-0 checkpoint)")
        _cache[key] = (model.eval(), processor, mask_token_id)
    return _cache[key]


def release() -> None:
    _cache.clear()


def model_bytes(model: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()])


def build_inputs(processor, image: Image.Image, text: str, device) -> dict:
    """infer.build_inputs: the checkpoint's chat template with the image before the text."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _stopping_criteria(step_callback):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Progress(StoppingCriteria):
        """Reports every generated token; never stops generation (the callback may raise to interrupt it)."""

        def __init__(self):
            self.steps = 0

        def __call__(self, input_ids, scores, **kwargs):
            self.steps += 1
            step_callback(self.steps)
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([Progress()])


def generate(model, processor, inputs: dict, max_new_tokens: int, step_callback=None) -> tuple[torch.Tensor, str]:
    """infer.generate_completion with do_sample=False (greedy): (completion ids [1, n], decoded text with special tokens)."""
    extra = {} if step_callback is None else {"stopping_criteria": _stopping_criteria(step_callback)}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **extra)
    completion_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return completion_ids, processor.tokenizer.decode(completion_ids[0], skip_special_tokens=False)


def mask_logits(model, inputs: dict, completion_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """infer.run_forward: replay prompt + completion and decode every <|mask|> -> logits [1, K, canvas_h, canvas_w]."""
    prompt_ids, prompt_mask = inputs["input_ids"], inputs["attention_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, torch.ones_like(completion_ids)], dim=1)
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = torch.cat([mm_token_type_ids, torch.zeros(completion_ids.shape, dtype=mm_token_type_ids.dtype,
                                                                      device=completion_ids.device)], dim=1)
    positions = (completion_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("the completion has no <|mask|> token")
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=mm_token_type_ids,
                    mask_positions=(positions + prompt_ids.shape[1]).unsqueeze(0))
    if out.get("pred_masks") is None:
        raise RuntimeError("the model returned no masks")
    return out["pred_masks"]


def split_answer(text: str) -> tuple[str, int]:
    """infer.split_answer_content: (answer text after the first </think>, <|mask|> count inside the thought)."""
    thought, separator, answer = text.partition("</think>")
    if not separator:
        return text, 0
    return answer, thought.count(MASK_TOKEN)


def answer_boxes(text: str) -> list[list[int]]:
    """infer.parse_answer_bboxes: the <bbox>[x1, y1, x2, y2]</bbox> boxes of the answer, normalized to 0-1000."""
    answer, _ = split_answer(text)
    return [[int(v) for v in m.groups()] for m in _BOX_RE.finditer(answer)]


def predict(model, processor, mask_token_id: int, image: Image.Image, text: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
            step_callback=None, decoder: str = "flash") -> Result:
    """One image and one instruction through the official steps, on the device the model is on. `decoder` is the mask decoder's
    kernel (default: upstream's "flash", which falls back to "mha" without flash-attn or CUDA)."""
    set_decoder_attention(decoder)
    device = next(model.parameters()).device
    inputs = build_inputs(processor, image, text, device)
    completion_ids, completion = generate(model, processor, inputs, max_new_tokens, step_callback)
    total = int((completion_ids == mask_token_id).sum())
    _, thought = split_answer(completion)
    grid = [int(v) for v in inputs["image_grid_thw"][0].tolist()]
    canvas = (grid[1] * 16, grid[2] * 16)  # the decoder's MASK_CANVAS_SCALE: masks match the vision encoder's input pixels
    if total - thought > 0:
        logits = mask_logits(model, inputs, completion_ids, mask_token_id)[0, thought:].cpu()
    else:
        logits = torch.zeros((0, *canvas), dtype=model.dtype)
    return Result(completion, completion_ids[0].tolist(), thought, logits, answer_boxes(completion), canvas)


def to_image_size(logits: torch.Tensor, height: int, width: int, threshold: float) -> torch.Tensor:
    """[K, canvas_h, canvas_w] logits (model dtype) -> [K, height, width] float masks. infer.py's probabilities
    (`torch.sigmoid(pred_masks[0, k]).float()`, the sigmoid in the model's dtype), resized bilinearly to the image when the canvas
    differs from it, then `probs >= threshold` as infer.py binarizes (threshold 0: the probabilities, infer.py's soft overlay)."""
    out = torch.zeros((logits.shape[0], height, width))
    for k in range(logits.shape[0]):
        probs = torch.sigmoid(logits[k]).float()
        if tuple(probs.shape) != (height, width):
            probs = F.interpolate(probs[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        out[k] = probs if threshold <= 0 else (probs >= threshold).float()
    return out


def boxes_to_pixels(boxes_1000: list[list[int]], height: int, width: int) -> list[list[float]]:
    """0-1000 boxes -> pixel coordinates of a height x width image, computed as infer.py draws them."""
    return [[b[0] / 1000.0 * width, b[1] / 1000.0 * height, b[2] / 1000.0 * width, b[3] / 1000.0 * height] for b in boxes_1000]


def overlay(image: np.ndarray, masks: torch.Tensor, boxes_px: list[list[float]], alpha: float = 0.5) -> np.ndarray:
    """infer.visualize_pixel_mask at the input image's size: the masks in the palette colors at `alpha`, the boxes outlined."""
    canvas = Image.fromarray(image).convert("RGBA")
    height, width = image.shape[:2]
    for k in range(masks.shape[0]):
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = ImageColor.getrgb(PALETTE[k % len(PALETTE)])
        rgba[..., 3] = (masks[k].numpy() * int(alpha * 255)).astype(np.uint8)
        canvas = Image.alpha_composite(canvas, Image.fromarray(rgba, mode="RGBA"))
    draw = ImageDraw.Draw(canvas)
    for k, box in enumerate(boxes_px):
        draw.rectangle(box, outline=ImageColor.getrgb(PALETTE[k % len(PALETTE)]) + (255,), width=max(2, width // 320))
    return np.array(canvas.convert("RGB"))


def instruction(query: str, mode: str) -> str:
    if mode not in PROMPT_MODES:
        raise ValueError(f"prompt mode must be one of {PROMPT_MODES}")
    query = query.strip()
    if not query:
        raise ValueError("the query is empty")
    return FIND_PROMPT.format(query) if mode == "find" else query


def segment_images(model, processor, mask_token_id: int, images: torch.Tensor, query: str, mode: str = "find", threshold: float = 0.5,
                   max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, step_callback=None, image_callback=None, decoder: str = "flash") -> dict:
    """ComfyUI IMAGE [B, H, W, 3] -> union masks [B, H, W], instance masks [N, H, W], previews [B, H, W, 3], answers, boxes.
    `step_callback(tokens)` is called for every generated token, `image_callback(images_done)` after every image."""
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"expected an IMAGE batch [B, H, W, C], got shape {tuple(images.shape)}")
    text = instruction(query, mode)
    batch, height, width = images.shape[0], images.shape[1], images.shape[2]
    unions, instances, previews, answers, boxes = [], [], [], [], []
    for b in range(batch):
        pixels = (images[b, ..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        result = predict(model, processor, mask_token_id, Image.fromarray(pixels), text, max_new_tokens, step_callback, decoder)
        masks = to_image_size(result.logits, height, width, threshold)
        px = boxes_to_pixels(result.boxes_1000, height, width)
        unions.append(masks.amax(0) if masks.shape[0] else torch.zeros((height, width)))
        instances.extend(masks)
        previews.append(torch.from_numpy(overlay(pixels, masks, px)).float() / 255)
        answers.append(result.text)
        boxes.extend({"image": b, "box": [round(v, 1) for v in box], "box_1000": box_1000} for box, box_1000 in zip(px, result.boxes_1000))
        if len(result.boxes_1000) != masks.shape[0]:
            boxes.append({"image": b, "note": f"{len(result.boxes_1000)} boxes for {masks.shape[0]} masks"})
        if image_callback is not None:
            image_callback(b + 1)
    return {
        "mask": torch.stack(unions),
        "instance_masks": torch.stack(instances) if instances else torch.zeros((1, height, width)),
        "preview": torch.stack(previews),
        "text": "\n\n".join(answers),
        "boxes": json.dumps(boxes),
        "instances": len(instances),
    }
