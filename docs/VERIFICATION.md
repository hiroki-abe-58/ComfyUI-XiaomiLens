# Verification

What was compared, how, and the raw results. Machine: Windows 11 Pro (26200), NVIDIA RTX 5090 32 GB (driver 610.88),
Python 3.12.13, PyTorch 2.14.1+cu130, 24 CPU threads.

| | |
| --- | --- |
| Upstream code | SeerRay-Lab/Xiaomi-Lens-0 `157b7bcb0b5010e81df858f5e9e4aa55a2fa9819` |
| Model | SeerRay-Lab/Xiaomi-Lens-0 revision `9452efab96fa8b886e7a0f89e390219ba59deddb`; every file's SHA-256 equals `tools/model_files.sha256` (the Hub's LFS hashes) |
| Environments | "5.4": Transformers 5.4.0 (the version upstream tested), accelerate 1.15.0, tokenizers 0.22.2; "5.20": Transformers 5.20.1, tokenizers 0.23.3 (the ComfyUI v0.42.1 environment has the same) |
| ComfyUI | v0.42.1 (`63be2f9`) for the real-queue checks; CI also v0.38.0 |

## The official reference

Upstream's `infer.py` steps, imported from the checkout and called as its `main()` calls them: `load_model_and_processor`,
`build_inputs`, `generate_completion` (greedy), `split_answer_content`, `run_forward`, `parse_answer_bboxes`. The only change:
`attn_implementation="sdpa"` instead of the script's default `flash_attention_2`, because flash-attn is not available on Windows.
For the same reason the mask decoder's default kernel (`flash`) falls back to `mha` (nn.MultiheadAttention); with
`XIAOMI_LENS_DECODER_ATTN=sdpa` upstream uses its SDPA path. Each reference ran in its own process; the pack ran in another
process, loaded through its own loader (`from_pretrained` on the CPU, then `.to("cuda")`, instead of `device_map`).

Cases (scikit-image sample photos; "find" = the README prompt `Locate all instances in the image that match the description
"<query>". Report bbox coordinates and masks.`):

| Case | Image | Text | Answer |
| --- | --- | --- | --- |
| cat | chelsea (451 x 300) | find "the cat" | 1 box + mask |
| cup | coffee (600 x 400) | find "the coffee cup" | 1 |
| coins | coins (384 x 303, gray) | find "coin" | 23 (of 24 coins) |
| rocket | rocket (640 x 427) | find "the rocket" | 1 |
| astronaut_flag | astronaut (512 x 512) | find "the flag" | 1 |
| reason_drink | coffee | "Which object in the image would you use to stir the drink? Report bbox coordinates and masks." | a sentence naming the spoon, then 1 |

## 1. The official code repeats itself

Two separate runs of the reference ("5.4", default decoder): completion ids, boxes and the mask logits (bfloat16, at the
processor's canvas) **identical in 6 of 6 cases**.

## 2. The pack equals the official code

Completion ids, decoded text, answer boxes, and every mask logit compared byte for byte (`np.array_equal`):

| Environment | Decoder kernel | Result |
| --- | --- | --- |
| 5.4 | upstream default (`flash` → `mha`) vs pack `decoder="flash"` | **6 / 6 bit-identical** |
| 5.20 | upstream default vs pack `decoder="flash"` | **6 / 6 bit-identical** |
| 5.4 | `XIAOMI_LENS_DECODER_ATTN=sdpa` vs pack `decoder="sdpa"` | **6 / 6 bit-identical** |
| 5.20 | `XIAOMI_LENS_DECODER_ATTN=sdpa` vs pack `decoder="sdpa"` | **6 / 6 bit-identical** |

The binarization and the preview are infer.py's: `torch.sigmoid(logits).float() >= threshold` (the sigmoid in bfloat16, as
infer.py computes it) and `visualize_pixel_mask`'s colors, alpha (`int(0.5 * 255)`) and box lines. When the image is not the
canvas size, the probabilities are resized bilinearly to the image (the pack's step; upstream draws on the canvas).

## 3. How much the decoder kernel and the Transformers version change

Same answers and boxes unless noted; mask IoU at the 0.5 threshold, the smallest over the case's masks.

| Case | SDPA vs MHA decoder (5.4) | Transformers 5.20.1 vs 5.4.0 (default decoder) |
| --- | --- | --- |
| cat | IoU 0.9997 | IoU 0.9988 |
| cup | 0.9998 | 0.9987 |
| coins | 0.9976 | 0.9749; other tokens: 23 boxes in both, 18 differ by at most 8 units of 1000 |
| rocket | 0.9981 | 0.9863; one box coordinate 483 → 481 |
| astronaut_flag | 0.9991 | 0.9984 |
| reason_drink | 0.9990 | 0.9968 |

The image processor's output is byte-identical in both Transformers versions (checked on `coins`), so the difference comes from
the model's numerics in Transformers.

## 4. Memory: why the pack does not default to upstream's fallback

`coffee` upscaled to 3840 x 2560 (9.8 MP; the processor resizes it to 8.4 MP, 32,768 patches), "the coffee cup" and "the spoon":

| Decoder kernel | Result |
| --- | --- |
| `mha` (upstream without flash-attn) | out of memory: nn.MultiheadAttention tried to allocate **30.75 GiB** for one attention matrix (16 heads x 32,768² in bfloat16) |
| `sdpa` (the pack's `auto` without flash-attn) | 6.5 s and 5.9 s, **15.05 GB** peak; one mask each at 3840 x 2560 |

By image size (upstream's own infer.py steps, "the coffee cup" on `coffee` resized; Transformers 5.4.0), the VRAM the mask pass
adds over the loaded model, and the request's peak:

| Image | Patches | MHA: mask pass / peak | SDPA: mask pass / peak | Boxes |
| --- | --- | --- | --- | --- |
| 0.5 MP | 1,944 | +0.86 / 9.51 GB | +0.46 / 9.11 GB | same |
| 1 MP | 3,952 | +2.58 / 11.24 GB | +0.83 / 9.50 GB | same |
| 2 MP | 7,776 | +8.53 / 17.22 GB | +1.55 / 10.24 GB | same |
| 3 MP | 11,616 | +18.02 / 26.73 GB | +2.28 / 10.99 GB | same |

The Loader's `decoder_attention` therefore defaults to flash-attn when it is installed (upstream's default) and to SDPA
otherwise; `mha` stays available for upstream's exact bytes on smaller images. The same change was proposed upstream
(SeerRay-Lab/Xiaomi-Lens-0#3, with these measurements).

## 5. Speed and memory (RTX 5090, bfloat16, model loaded, reference runs)

| Case | Seconds | VRAM peak |
| --- | --- | --- |
| cat | 1.97 | 8.82 GB |
| cup | 1.75 | 8.94 GB |
| coins (23 instances, long answer) | 29.92 | 9.14 GB |
| rocket | 1.64 | 9.14 GB |
| astronaut_flag | 1.89 | 9.14 GB |
| reason_drink (reasoning sentence first) | 3.25 | 9.14 GB |

Loading: 7–11 s from an SSD. Transformers runs Qwen3.5's linear-attention layers in plain PyTorch here (causal-conv1d and
flash-linear-attention are not installed), which makes long answers slower. CPU (24 threads, "5.20", pack, SDPA decoder): `cat`
in 6.6 s after 8.4 s of loading, the same box as on the GPU.

## 6. Real ComfyUI queue (RTX 5090)

`scripts/ci_queue_check.py --gpu` on ComfyUI v0.42.1 with the real model: a fresh ComfyUI base directory with the pack in
`custom_nodes`, the three shipped API workflows posted to `/prompt` on ComfyUI's own `example.png` (768 x 768, a size the
processor keeps, so the comparison is exact). The reference is the official model class and infer.py's functions in the check's
own process, with the decoder kernel the Loader's `auto` picks (SDPA here).

| Workflow | Query | Through the queue |
| --- | --- | --- |
| segment_object | "the pink dress" | Mask To Image → Save Image equals infer.py's mask (0 of 589,824 pixels differ); the preview equals `visualize_pixel_mask` with the box drawn; `boxes` holds infer.py's box |
| cut_out | "the cartoon character" | the PNG's RGB equals the input, its alpha equals infer.py's mask |
| reasoning_segment | "Which part of the picture would you replace to change the season to winter? ..." | the answer text equals infer.py's (it names the grass); the preview equals infer.py's |

Also: the same job again (after other jobs, with `after_run` = `move to CPU`) gave the same mask; `../Xiaomi-Lens-0` as the
model and an unknown prompt mode were rejected by ComfyUI's validation; an empty query failed with "the query is empty".
The same check passed on the CPU with the tiny checkpoint (ComfyUI v0.42.1), as CI runs it.

## 7. Clean install

A fresh ComfyUI v0.42.1 checkout and a fresh Python 3.12 environment (PyTorch 2.14.1+cu130, ComfyUI's requirements, which
bring Transformers 5.20.1), then `git clone` of this repository into `custom_nodes` and `pip install -r requirements.txt` (nothing
new to install), then `tools/download_model.py --models-dir models/xiaomi_lens`: all 12 files downloaded from the Hub at the pinned
revision and verified. The GPU real-queue check of section 6 against that installation (`--base-dir` = the ComfyUI folder, the
pack as cloned, the model as downloaded) passed with the same results; the cloned working tree was unchanged afterwards.

The Registry package (`comfy node validate` and `comfy node pack`, comfy-cli 1.22.0, on the committed tree): validation passed;
the archive holds only the node (no tests, scripts, images or weights) and none of the scanned patterns (environment, network or
process code, absolute paths, tokens).

## 8. Tests (CPU, tiny checkpoint)

`tests/` (41 tests) with ComfyUI v0.42.1 and Transformers 5.20.1, every reference required: the pack against infer.py's
functions in the same process on the tiny checkpoint (three image sizes; teacher-forced completions with a mask inside
`<think>`, two answer masks and two boxes; all three decoder kernels), infer.py's parsing on edge cases, the threshold at exactly
0.5 in bfloat16, resizing, empty answers, interruption, ComfyUI's validation of every API workflow, the UI workflows against the
API ones, node execution against the runtime, the vendored tree against the pinned commit, and the package scan (no environment,
network or process code in what ships).

The tiny checkpoint (`scripts/make_tiny_model.py`) has the real architecture and tokenizer with small widths; two weight edits
make its greedy answer `</think><|mask|><|im_end|>` so that the whole mask path runs. Its masks mean nothing; it only serves to
compare the pack with the official code.

CI (GitHub Actions, CPU) runs the same 41 tests and the real-queue check with the tiny checkpoint on four jobs: ComfyUI v0.38.0
on Ubuntu and Windows (Python 3.12, PyTorch 2.14.1, Transformers 5.20.1), ComfyUI v0.38.0 on Ubuntu with Python 3.10, PyTorch
2.8.0 and Transformers 5.4.0, and ComfyUI v0.42.1 on Ubuntu; all passed with nothing skipped.

## Not checked

Segmentation quality itself (see the authors' benchmarks), flash-attn and FlashAttention-2 (not on Windows), Linux and macOS
outside CI, AMD and Apple GPUs, GPUs with less than 32 GB (the measured peaks are above), sampling (the pack decodes greedily
like infer.py), multiple images in one request (upstream supports one image per request in infer.py; batches are run image by
image).
