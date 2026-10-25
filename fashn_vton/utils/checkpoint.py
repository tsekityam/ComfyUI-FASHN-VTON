"""Model checkpoint loading utilities."""

import os

import torch
from safetensors.torch import load_file


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load a local model checkpoint.

    Supports:
    - Local .pt, .pth files (PyTorch format)
    - Local .safetensors files

    HuggingFace Hub downloads are intentionally not performed here — install
    weights through ComfyUI Model Library into ``models/fashn-vton/``.

    Args:
        checkpoint_path: Local file path
        device: Device to load the checkpoint to

    Returns:
        The loaded state dictionary
    """
    if os.path.isfile(checkpoint_path):
        if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint_path.endswith(".safetensors"):
            return load_file(checkpoint_path, device=device)
        raise ValueError(f"Unknown checkpoint file format: {checkpoint_path}")

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. "
        "Install weights via ComfyUI Model Library under models/fashn-vton/."
    )
