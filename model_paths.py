"""Register FASHN VTON weights with ComfyUI's Model Library / folder_paths."""

from __future__ import annotations

import os

import folder_paths

FASHN_VTON_FOLDER = "fashn-vton"
FASHN_VTON_EXTENSIONS = {".safetensors", ".onnx"}
DWPOSE_REQUIRED_FILES = ("yolox_l.onnx", "dw-ll_ucoco_384.onnx")


def _merge_folder_extensions(folder_name: str, extensions: set[str]) -> None:
    """Ensure registered folder lists the given file extensions in Model Library."""
    if folder_name not in folder_paths.folder_names_and_paths:
        return
    paths, current_exts = folder_paths.folder_names_and_paths[folder_name]
    folder_paths.folder_names_and_paths[folder_name] = (paths, set(current_exts) | set(extensions))


def register_fashn_vton_model_paths() -> str:
    """
    Register ``models/fashn-vton`` so weights appear in the Model Library sidebar
    and can be selected via ``folder_paths.get_filename_list``.

    Expected layout (managed via Model Library / manual install)::

        ComfyUI/models/fashn-vton/
            model.safetensors          # or any *.safetensors name
            dwpose/
                yolox_l.onnx
                dw-ll_ucoco_384.onnx
    """
    models_root = os.path.join(folder_paths.models_dir, FASHN_VTON_FOLDER)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(os.path.join(models_root, "dwpose"), exist_ok=True)

    folder_paths.add_model_folder_path(FASHN_VTON_FOLDER, models_root, is_default=True)
    _merge_folder_extensions(FASHN_VTON_FOLDER, FASHN_VTON_EXTENSIONS)
    return models_root


def list_fashn_vton_checkpoints() -> list[str]:
    """Return *.safetensors files registered under the fashn-vton Model Library folder."""
    try:
        names = folder_paths.get_filename_list(FASHN_VTON_FOLDER)
    except KeyError:
        register_fashn_vton_model_paths()
        names = folder_paths.get_filename_list(FASHN_VTON_FOLDER)
    return [n for n in names if n.lower().endswith(".safetensors")]


def resolve_weights_dir(checkpoint_filename: str) -> str:
    """
    Resolve the weights directory for a Model Library checkpoint selection.

    The TryOn pipeline expects ``model.safetensors`` (or a symlink/copy) plus a
    sibling ``dwpose/`` directory containing the ONNX pose models.
    """
    full_path = folder_paths.get_full_path(FASHN_VTON_FOLDER, checkpoint_filename)
    if full_path is None:
        raise FileNotFoundError(
            f"FASHN VTON checkpoint not found in Model Library folder "
            f"'{FASHN_VTON_FOLDER}': {checkpoint_filename}"
        )

    weights_dir = os.path.dirname(os.path.abspath(full_path))
    model_basename = os.path.basename(full_path)

    # Pipeline hard-codes model.safetensors; symlink/copy when the library name differs.
    expected = os.path.join(weights_dir, "model.safetensors")
    if model_basename != "model.safetensors" and not os.path.exists(expected):
        try:
            os.symlink(model_basename, expected)
        except OSError:
            # Fallback for filesystems that disallow symlinks.
            import shutil

            shutil.copy2(full_path, expected)

    return weights_dir


def missing_weight_files(weights_dir: str) -> list[str]:
    """Return absolute paths of required weight files that are missing."""
    missing = []
    tryon_path = os.path.join(weights_dir, "model.safetensors")
    if not os.path.exists(tryon_path):
        missing.append(tryon_path)
    dwpose_dir = os.path.join(weights_dir, "dwpose")
    for name in DWPOSE_REQUIRED_FILES:
        path = os.path.join(dwpose_dir, name)
        if not os.path.exists(path):
            missing.append(path)
    return missing


def model_library_install_hint(weights_dir: str | None = None) -> str:
    target = weights_dir or os.path.join(folder_paths.models_dir, FASHN_VTON_FOLDER)
    return (
        "Place FASHN VTON weights via ComfyUI Model Library (or manually) under:\n"
        f"  {target}/\n"
        "    model.safetensors          # from https://huggingface.co/fashn-ai/fashn-vton-1.5\n"
        "    dwpose/yolox_l.onnx\n"
        "    dwpose/dw-ll_ucoco_384.onnx  # from https://huggingface.co/fashn-ai/DWPose\n"
        "Then refresh the Model Library (r) and select the checkpoint in Load Fashn VTON."
    )
