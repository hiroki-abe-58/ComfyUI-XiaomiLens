"""Real-queue check for CI, for the GPU verification and for a clean-install test: start a real ComfyUI server with this node pack
installed, and run the shipped workflows through POST /prompt on ComfyUI's own example.png.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --models-dir <folder holding Xiaomi-Lens-0/>
                                     --upstream <Xiaomi-Lens-0 checkout> [--gpu] [--workdir DIR] [--base-dir DIR]

CI runs it on the CPU with the tiny checkpoint of scripts/make_tiny_model.py (named Xiaomi-Lens-0 in its models folder); with
--gpu it runs the real model on ComfyUI's GPU. The reference is computed first, in this process, by the official code of the
upstream checkout: its model class, loaded from the same folder, and infer.py's build_inputs, generate_completion, run_forward and
visualize_pixel_mask. example.png is 768 x 768, a size the image processor keeps, so the masks need no resizing and the check is
exact: through the queue
- Segment -> Mask To Image -> Save Image gives infer.py's binarized masks (their union), and the preview is infer.py's overlay
  (with the boxes drawn); the boxes output holds infer.py's answer boxes;
- the cut-out workflow's PNG has the input's RGB and the mask as alpha;
- the instruction workflow returns infer.py's completion text;
- a repeated job gives the same files, "move to CPU" works, and invalid inputs are rejected.
The server is started and stopped by this script only. --base-dir reuses a ComfyUI base directory that already has the pack in
custom_nodes (clean-install test); otherwise a fresh one is made in the work directory with a copy of this checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_client  # noqa: E402

NODE_IDS = {"XiaomiLensLoader", "XiaomiLensSegment"}
FIND = 'Locate all instances in the image that match the description "{}". Report bbox coordinates and masks.'


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def references(upstream: Path, folder: Path, image_path: Path, texts: dict, device: str) -> dict:
    """{name: (completion, union mask [H, W] bool, overlay [H, W, 3] uint8, answer boxes)} from the official code."""
    import torch
    from PIL import Image
    from transformers.utils import is_flash_attn_2_available

    # the decoder kernel the Loader's "auto" picks, set the way upstream reads it (at import)
    os.environ["XIAOMI_LENS_DECODER_ATTN"] = "flash" if device == "cuda" and is_flash_attn_2_available() else "sdpa"
    sys.path.insert(0, str(upstream))
    import infer  # puts the checkout's src/ on sys.path
    from xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration
    from transformers import AutoProcessor

    dev = torch.device(device)
    model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(str(folder), dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    processor = AutoProcessor.from_pretrained(str(folder))
    mask_id = processor.tokenizer.convert_tokens_to_ids("<|mask|>")
    image = Image.open(image_path).convert("RGB")
    out = {}
    for name, text in texts.items():
        inputs, _ = infer.build_inputs(processor, image, text, dev)
        ids, completion, total = infer.generate_completion(model, processor, inputs, 16384, mask_id)
        _, thought = infer.split_answer_content(completion)
        boxes = infer.parse_answer_bboxes(completion)
        grid = inputs["image_grid_thw"][0].tolist()
        union = np.zeros((grid[1] * 16, grid[2] * 16), dtype=bool)
        picture = np.array(image)
        if total - thought > 0:
            pred = infer.run_forward(model, inputs, ids, mask_id, dev)[:, thought:]
            union = (torch.sigmoid(pred[0]).float() >= 0.5).any(0).cpu().numpy()
            picture = np.array(infer.visualize_pixel_mask(image, pred, inputs["image_grid_thw"], threshold=0.5, bboxes=boxes, draw_bbox=True))
        out[name] = (completion, union, picture, boxes)
        print(json.dumps({"reference": name, "completion": completion[:200], "masks": total - thought, "foreground": int(union.sum())}))
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True, help="a folder holding Xiaomi-Lens-0/ (the real or the tiny checkpoint)")
    parser.add_argument("--upstream", type=Path, required=True, help="SeerRay-Lab/Xiaomi-Lens-0 checkout (the official code)")
    parser.add_argument("--gpu", action="store_true", help="run ComfyUI and the reference on the GPU (default: --cpu)")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None, help="a ComfyUI base directory with the pack installed (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    models = args.models_dir.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="xiaomi-lens-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    base = (args.base_dir or work / "base").resolve()
    pack = base / "custom_nodes" / "ComfyUI-XiaomiLens"
    if args.base_dir is None:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--base-dir: {pack} is not installed")
    inputs, output = base / "input", base / "output"
    inputs.mkdir(parents=True, exist_ok=True)
    if (inputs / "example.png").resolve() != (comfy / "input" / "example.png").resolve():  # --base-dir may be the ComfyUI folder
        shutil.copyfile(comfy / "input" / "example.png", inputs / "example.png")
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"xiaomi_lens_ci:\n  base_path: {models.parent.as_posix()}\n  xiaomi_lens: {models.name}\n", encoding="utf-8")

    workflows = {name: json.loads((ROOT / "workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
                 for name in ("segment_object", "cut_out", "reasoning_segment")}
    texts = {"segment_object": FIND.format(workflows["segment_object"]["3"]["inputs"]["query"]),
             "cut_out": FIND.format(workflows["cut_out"]["3"]["inputs"]["query"]),
             "reasoning_segment": workflows["reasoning_segment"]["3"]["inputs"]["query"]}
    ref = references(args.upstream.resolve(), models / "Xiaomi-Lens-0", inputs / "example.png", texts, "cuda" if args.gpu else "cpu")

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
           "--base-directory", str(base), "--extra-model-paths-config", str(yaml)]
    proc = subprocess.Popen(cmd + ([] if args.gpu else ["--cpu"]), cwd=comfy, stdout=log, stderr=subprocess.STDOUT)
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")

        def run(name: str, tag: str, expect_success: bool = True, **overrides) -> dict:
            wf = json.loads(json.dumps(workflows[name]))
            for node in wf.values():
                if "filename_prefix" in node["inputs"]:
                    node["inputs"]["filename_prefix"] = node["inputs"]["filename_prefix"].replace("xiaomi_lens/", f"ci/{tag}_")
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            try:
                result = queue_client.run(server, wf, None, 1800)
            except SystemExit as rejected:  # POST /prompt refused the workflow (validation error)
                result = {"status": "rejected", "wall_seconds": 0, "error": str(rejected)[:300]}
            error = result.get("error") or {}
            print(json.dumps({"run": tag, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:200] if isinstance(error, dict) else error}))
            if (result["status"] == "success") != expect_success:
                failures.append(f"{tag}: expected {'success' if expect_success else 'an error'}, got {result['status']} {error}")
            return result

        def saved(stem: str, mode: str = "RGB"):
            files = sorted((output / "ci").glob(f"{stem}_*.png"))
            if len(files) != 1:
                failures.append(f"{stem}: expected one saved .png, found {[f.name for f in files]}")
                return None
            return np.array(Image.open(files[0]).convert(mode))

        def check(tag: str, got, expected):
            same = got is not None and got.shape == expected.shape and np.array_equal(got, expected)
            print(json.dumps({"compare": tag, "equal": bool(same), "shape": None if got is None else list(got.shape),
                              "differing": None if got is None or got.shape != expected.shape else int((got != expected).sum())}))
            if not same:
                failures.append(f"{tag}: differs from the official output")

        example = np.array(Image.open(inputs / "example.png").convert("RGB"))
        result = run("segment_object", "seg_a")
        completion, union, picture, boxes = ref["segment_object"]
        check("seg_a mask", saved("seg_a_mask", "L"), union.astype(np.uint8) * 255)
        check("seg_a preview", saved("seg_a_preview"), picture)
        texts_out = [json.loads(v[0]) for v in (result.get("text") or {}).values() if v and v[0].lstrip().startswith("[")]
        got_boxes = [b["box_1000"] for b in (texts_out[0] if texts_out else []) if "box_1000" in b]
        if got_boxes != boxes:
            failures.append(f"seg_a boxes {got_boxes} != official {boxes}")

        run("cut_out", "cut")
        cut = saved("cut_cutout", "RGBA")
        completion, union, picture, boxes = ref["cut_out"]
        if cut is not None:
            check("cut rgb", cut[..., :3], example)
            check("cut alpha", cut[..., 3], union.astype(np.uint8) * 255)

        result = run("reasoning_segment", "reason")
        completion, union, picture, boxes = ref["reasoning_segment"]
        check("reason preview", saved("reason_reasoning"), picture)
        answers = [v[0] for v in (result.get("text") or {}).values() if v]
        if answers != [completion]:
            failures.append(f"reason text {answers} != official {completion!r}")

        run("segment_object", "seg_b", **{"2.after_run": "move to CPU"})  # after other jobs: ComfyUI's cache no longer holds seg_a
        a, b = saved("seg_a_mask", "L"), saved("seg_b_mask", "L")
        if a is not None and b is not None and not np.array_equal(a, b):
            failures.append("the same job gave a different mask")

        run("segment_object", "bad_model", expect_success=False, **{"2.model": "../Xiaomi-Lens-0"})
        run("segment_object", "bad_mode", expect_success=False, **{"3.prompt_mode": "chat"})
        run("segment_object", "empty_query", expect_success=False, **{"3.query": "  "})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
