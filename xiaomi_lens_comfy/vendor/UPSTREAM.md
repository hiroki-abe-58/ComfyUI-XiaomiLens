# Vendored code: Xiaomi-Lens-0

| | |
| --- | --- |
| Upstream | https://github.com/SeerRay-Lab/Xiaomi-Lens-0 |
| Revision | `157b7bcb0b5010e81df858f5e9e4aa55a2fa9819` (2026-10-19) |
| License | Apache-2.0 (`LICENSE`, copied from the upstream repository); Copyright 2026 Xiaomi Corporation |
| Paper | Pei Fu et al., "Xiaomi-Lens-0 Technical Report", arXiv:2610.23279 |

Only the model package that inference with Hugging Face Transformers needs is vendored: `src/xiaomi_lens` without
`integrations/` (the vLLM serving code). `MANIFEST.sha256` lists every vendored file with its SHA-256;
`scripts/vendor_sync.py` rebuilds the tree from the revision above and `tests/test_vendor.py` checks it in CI.

## Copied unchanged (byte for byte)

`xiaomi_lens/__init__.py`, `xiaomi_lens/constants.py`, `xiaomi_lens/model/__init__.py`, `xiaomi_lens/model/config.py`,
`xiaomi_lens/model/modeling_xiaomi_lens.py`, `LICENSE`.

## Modified

`xiaomi_lens/model/pixel_mask_decoder.py` carries a notice at the top. One line changed: upstream picks the mask decoder's
attention kernel at import time from the `XIAOMI_LENS_DECODER_ATTN` variable of the process (`flash`, `sdpa` or `mha`; default
`flash`, which falls back to `mha` without flash-attn). The vendored copy starts with the default (`flash`) and reads no
variables; the pack sets the same module setting (`_DECODER_ATTN`) from the Loader's `decoder_attention` input before each run
(`runtime.set_decoder_attention`). The kernels are upstream's own code paths.

## Not vendored

`infer.py` (upstream's command-line script): `xiaomi_lens_comfy/runtime.py` repeats its steps (`build_inputs`,
`generate_completion` with greedy decoding, `run_forward`, `split_answer_content`, `parse_answer_bboxes`, and the binarization and
overlay of `visualize_pixel_mask`); `tests/test_runtime.py` and `scripts/ci_queue_check.py` compare them with infer.py itself.

## Model files (not in this repository)

`tools/download_model.py` fetches them at a pinned revision and checks the SHA-256 in `tools/model_files.sha256`.

| Model | Repository and revision | Size | License |
| --- | --- | --- | --- |
| Xiaomi-Lens-0 (4B, bfloat16) | `SeerRay-Lab/Xiaomi-Lens-0` `9452efab96fa8b886e7a0f89e390219ba59deddb` | 9.2 GB | Apache-2.0 (model card and `LICENSE`) |

The model card: Xiaomi-Lens-0 derives from Qwen3.5-4B (Apache-2.0; Copyright 2026 Alibaba Cloud) and adds the pixel-mask decoder
(Copyright 2026 Xiaomi Corporation).
