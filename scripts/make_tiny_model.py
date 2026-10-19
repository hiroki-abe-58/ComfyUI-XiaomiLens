"""Build a tiny Xiaomi-Lens-0 checkpoint for the CPU tests (maintainer/CI tool; nothing of it ships).

    python scripts/make_tiny_model.py --source <models/xiaomi_lens/Xiaomi-Lens-0> --out <models/xiaomi_lens/Xiaomi-Lens-0-tiny>

`--source` needs only the configuration, tokenizer and processor files (tools/download_model.py --no-weights). The tiny model has
the real architecture (Qwen3.5 with linear and full attention layers, the vision encoder, the pixel mask decoder) and the real
tokenizer, with small widths and seeded random weights. Two edits make its greedy answer known in advance, so the tests and the
queue check run the whole mask path:
- the output projections of every text layer are zero, so the last hidden state is the embedding of the current token;
- the embeddings of "\\n" (the last prompt token), "</think>" and "<|mask|>" are unit vectors e0, e1, e2, and the (untied) output
  rows of "</think>", "<|mask|>" and "<|im_end|>" point along e0, e1, e2.
Greedy decoding then answers "</think><|mask|><|im_end|>": one mask, decoded from the image features by the (random) mask decoder.
The pack and the official code are compared on this model; its masks mean nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COPY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "generation_config.json")


def tiny_config(config: dict) -> dict:
    text = dict(config["text_config"])
    text.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, full_attention_interval=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, tie_word_embeddings=False)
    text["rope_parameters"] = dict(text["rope_parameters"], mrope_section=[3, 3, 2])  # 64 * 0.25 / 2 = 8 rotary pairs
    vision = dict(config["vision_config"])
    vision.update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    out = dict(config, text_config=text, vision_config=vision, tie_word_embeddings=False)
    out["mask_decoder"] = {"llm_hidden_size": 64, "llm_intermediate_size": 128, "llm_num_heads": 4, "vision_hidden_size": 32,
                           "mask_decoder_channels": 8, "mask_decoder_embed_dim": 4}
    return out


def main(argv=None) -> int:
    from transformers import AutoTokenizer

    from xiaomi_lens_comfy.vendor.xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copyfile(args.source / name, args.out / name)
    config = tiny_config(json.loads((args.source / "config.json").read_text(encoding="utf-8")))
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    from transformers import AutoConfig

    torch.manual_seed(0)
    model = XiaomiLensQwen3_5ForConditionalGeneration(AutoConfig.from_pretrained(args.out)).to(torch.bfloat16).eval()
    ids = [AutoTokenizer.from_pretrained(args.out).convert_tokens_to_ids(t) for t in ("<|im_end|>", "</think>", "<|mask|>")]
    newline = AutoTokenizer.from_pretrained(args.out)("\n", add_special_tokens=False)["input_ids"]
    if len(newline) != 1:
        raise SystemExit(f"unexpected tokenization of a newline: {newline}")
    chain = [newline[0], ids[1], ids[2], ids[0]]
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("model.language_model.layers.") and name.endswith(("o_proj.weight", "out_proj.weight", "down_proj.weight")):
                p.zero_()
        embed, head = model.get_input_embeddings().weight, model.lm_head.weight
        for i, (token, successor) in enumerate(zip(chain[:-1], chain[1:])):
            embed[token] = 0
            embed[token, i] = 1
            head[successor] = 0
            head[successor, i] = 30
    model.save_pretrained(args.out, safe_serialization=True)
    print(f"tiny checkpoint in {args.out} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters); answer chain {chain}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
