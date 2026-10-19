"""ComfyUI nodes for Xiaomi-Lens-0: masks and boxes for the objects a text describes. Nothing is downloaded at run time."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

LensModel = io.Custom("XIAOMI_LENS_MODEL")
DEVICES = ("auto", "cpu")
AFTER_RUN = ("keep on device", "move to CPU")


def _device(choice: str):
    import comfy.model_management as mm
    import torch

    return torch.device("cpu") if choice == "cpu" else mm.get_torch_device()


class XiaomiLensLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XiaomiLensLoader",
            display_name="Xiaomi-Lens-0 Loader",
            category="loaders/xiaomi_lens",
            description=("Loads Xiaomi-Lens-0 (4B, bfloat16, about 9 GB) from models/xiaomi_lens/<name>/, the files of the Hugging Face "
                         "repository SeerRay-Lab/Xiaomi-Lens-0."),
            inputs=[
                io.Combo.Input("model", options=paths.model_names(), tooltip="a folder in models/xiaomi_lens/"),
                io.Combo.Input("device", options=list(DEVICES), default="auto",
                               tooltip="auto: ComfyUI's GPU. cpu: works, slower (bfloat16, about 10 GB of RAM)"),
                io.Combo.Input("attention", options=list(runtime.ATTENTION), default="auto",
                               tooltip="auto: FlashAttention-2 (what upstream uses) when flash-attn is installed and the model runs on CUDA, "
                                       "otherwise PyTorch SDPA"),
                io.Combo.Input("decoder_attention", options=list(runtime.DECODER_ATTENTION), default="auto",
                               tooltip="the mask decoder's attention. auto: flash-attn (upstream's default) when installed and on CUDA, "
                                       "otherwise PyTorch SDPA. mha: upstream's fallback without flash-attn (identical results to "
                                       "upstream there), but its memory grows with the square of the image size: about 32 GB for an 8 MP image"),
                io.Combo.Input("after_run", options=list(AFTER_RUN), default="keep on device",
                               tooltip="keep on device: fast repeated runs. move to CPU: frees about 9 GB of VRAM after each run"),
            ],
            outputs=[LensModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, device, attention, decoder_attention, after_run) -> io.NodeOutput:
        if device not in DEVICES or after_run not in AFTER_RUN:
            raise ValueError(f"unknown device {device!r} or after_run {after_run!r}")
        folder = paths.model_dir(model)
        dev = _device(device)
        impl = runtime.attention_implementation(attention, dev)
        decoder = runtime.decoder_attention(decoder_attention, dev)
        runtime.load(folder, impl)  # fail here, not later, when a file is missing or broken
        return io.NodeOutput({"folder": folder, "device": str(dev), "attention": impl, "decoder": decoder, "after_run": after_run})


class XiaomiLensSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XiaomiLensSegment",
            display_name="Xiaomi-Lens-0 Segment",
            category="mask/xiaomi_lens",
            description=("Finds what the text describes and returns a mask and a box per instance (greedy decoding, as upstream's "
                         "infer.py). find: the query is put into the README's grounding prompt. instruction: the text is sent as typed "
                         "(for questions such as 'Which object would you use to stir the drink? Report bbox coordinates and masks.')."),
            inputs=[
                LensModel.Input("model"),
                io.Image.Input("image"),
                io.String.Input("query", multiline=True, default="the cat"),
                io.Combo.Input("prompt_mode", options=list(runtime.PROMPT_MODES), default="find",
                               tooltip='find: Locate all instances in the image that match the description "<query>". Report bbox '
                                       "coordinates and masks. instruction: the query as typed"),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="mask probability threshold (upstream: 0.5). 0: soft masks (the probabilities)"),
                io.Int.Input("max_new_tokens", default=runtime.DEFAULT_MAX_NEW_TOKENS, min=16, max=runtime.DEFAULT_MAX_NEW_TOKENS,
                             tooltip="generation limit (upstream default 16384); a reasoning answer can be long"),
            ],
            outputs=[io.Mask.Output(display_name="mask"), io.Mask.Output(display_name="instance_masks"),
                     io.Image.Output(display_name="preview"), io.String.Output(display_name="text"),
                     io.String.Output(display_name="boxes")],
        )

    @classmethod
    def execute(cls, model, image, query, prompt_mode, threshold, max_new_tokens) -> io.NodeOutput:
        import comfy.model_management as mm
        import comfy.utils
        import torch

        mm.throw_exception_if_processing_interrupted()
        net, processor, mask_token_id = runtime.load(model["folder"], model["attention"])
        device = torch.device(model["device"])
        if next(net.parameters()).device != device:
            if device.type != "cpu":
                mm.free_memory(runtime.model_bytes(net) + (2 << 30), device)  # let ComfyUI unload its models to make room
            net.to(device)
        bar = comfy.utils.ProgressBar(image.shape[0])
        try:
            out = runtime.segment_images(net, processor, mask_token_id, image, query, prompt_mode, threshold, max_new_tokens,
                                         step_callback=lambda _: mm.throw_exception_if_processing_interrupted(),
                                         image_callback=bar.update_absolute, decoder=model["decoder"])
        finally:
            if model["after_run"] == "move to CPU" and device.type != "cpu":
                net.to("cpu")
                mm.soft_empty_cache()
        return io.NodeOutput(out["mask"], out["instance_masks"], out["preview"], out["text"], out["boxes"])


class XiaomiLensExtension(ComfyExtension):
    async def get_node_list(self):
        return [XiaomiLensLoader, XiaomiLensSegment]


async def comfy_entrypoint() -> XiaomiLensExtension:
    paths.register()
    return XiaomiLensExtension()
