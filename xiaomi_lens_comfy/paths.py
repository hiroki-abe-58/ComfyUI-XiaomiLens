"""Model location: `<ComfyUI models>/xiaomi_lens/<name>/` holding the files of the Hugging Face repository SeerRay-Lab/Xiaomi-Lens-0
(config.json, the safetensors shards, tokenizer and processor files), plus any path added for `xiaomi_lens` in
extra_model_paths.yaml. Nothing is downloaded at run time.
"""

from __future__ import annotations

import os

FOLDER = "xiaomi_lens"
REQUIRED = ("config.json", "tokenizer.json", "processor_config.json")


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def _is_checkpoint(folder: str) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in REQUIRED) and any(
        f.endswith(".safetensors") for f in os.listdir(folder))


def model_names() -> list[str]:
    """Sub-folders of the registered xiaomi_lens folders that hold a checkpoint."""
    import folder_paths

    names = set()
    for base in folder_paths.get_folder_paths(FOLDER):
        if os.path.isdir(base):
            names.update(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)) and _is_checkpoint(os.path.join(base, n)))
    return sorted(names)


def model_dir(name: str) -> str:
    """The folder of model `name`. Only names listed by model_names() are accepted, so a value cannot leave the models folders."""
    import folder_paths

    if name not in model_names():
        raise FileNotFoundError(f"no checkpoint named {name!r} in models/{FOLDER}/ (see the README for the download command)")
    for base in folder_paths.get_folder_paths(FOLDER):
        folder = os.path.join(base, name)
        if os.path.isdir(folder) and _is_checkpoint(folder):
            return folder
    raise FileNotFoundError(name)
