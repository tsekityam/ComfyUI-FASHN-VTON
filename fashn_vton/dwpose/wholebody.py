"""
Wholebody pose detection adapted from DWPose (Apache-2.0):
https://github.com/IDEA-Research/DWPose
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnxruntime as ort

from .onnxdet import inference_detector
from .onnxpose import inference_pose


def _normalize_device(device) -> str:
    """Map torch devices / shorthand to an ORT backend hint."""
    if device is None:
        return "auto"
    if hasattr(device, "type"):
        # torch.device
        return str(device.type).lower()
    text = str(device).lower()
    if text.startswith("cuda"):
        return "cuda"
    if text in ("mps", "metal", "coreml"):
        return "mps"
    if text in ("cpu", "auto"):
        return text
    return text


def select_onnx_providers(device="auto"):
    """
    Choose ONNX Runtime execution providers for the active platform.

    - CUDA systems: CUDAExecutionProvider when available
    - macOS / Apple Silicon: CoreMLExecutionProvider when available (GPU/ANE)
    - Otherwise: CPUExecutionProvider

    Returns (providers, provider_options) suitable for InferenceSession.
    provider_options is None when providers are specified as (name, opts) tuples.
    """
    hint = _normalize_device(device)
    available = set(ort.get_available_providers())

    # Explicit CUDA request or auto on a CUDA-capable ORT build.
    if hint == "cuda" or (hint == "auto" and "CUDAExecutionProvider" in available):
        if "CUDAExecutionProvider" in available:
            device_id = 0
            if isinstance(device, str) and ":" in device:
                try:
                    device_id = int(device.split(":")[-1])
                except ValueError:
                    device_id = 0
            elif hasattr(device, "index") and device.index is not None:
                device_id = int(device.index)
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
                {"device_id": str(device_id)},
                {},
            ]

    # Apple Silicon / Darwin: prefer CoreML (covers GPU + Neural Engine).
    # onnxruntime-gpu is unavailable on macOS; plain onnxruntime ships CoreML EP.
    want_apple = hint in ("mps", "coreml", "metal") or (
        hint == "auto" and sys.platform == "darwin"
    )
    if want_apple and "CoreMLExecutionProvider" in available:
        return [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                },
            ),
            "CPUExecutionProvider",
        ], None

    return ["CPUExecutionProvider"], None


class Wholebody:
    def __init__(self, checkpoints_dir, device="auto"):
        providers, provider_options = select_onnx_providers(device)
        onnx_det = os.path.join(checkpoints_dir, "yolox_l.onnx")
        onnx_pose = os.path.join(checkpoints_dir, "dw-ll_ucoco_384.onnx")

        session_kwargs = {"providers": providers}
        if provider_options is not None:
            session_kwargs["provider_options"] = provider_options

        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, **session_kwargs)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, **session_kwargs)

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores
