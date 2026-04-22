import os
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch


_DEPTHANYTHINGV2_PROCESSOR = None
_DEPTHANYTHINGV2_MODEL = None
_DEPTHANYTHINGV2_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_DEPTHANYTHINGV2_CACHE_DIR = None


def _depth_path(depth_root: str, cache_key: str) -> str:
    return os.path.join(depth_root, f"{cache_key}.png")


def _codebook_path(depth_root: str, cache_key: str) -> str:
    return os.path.join(depth_root, f"{cache_key}.npy")


def _sanitize_cache_key(cache_key: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(cache_key))
    return safe or "frame"


def _model_cache_root(depth_root: str, model_id: str | None) -> str:
    model_key = _sanitize_cache_key(model_id or _DEPTHANYTHINGV2_MODEL_ID)
    return os.path.join(depth_root, model_key)


def _prepare_image(image_tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")

    image = image_tensor.detach().float().cpu()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.contiguous().numpy()
    return image


def _load_depthanythingv2(model_id: str | None = None, cache_dir: str | None = None):
    global _DEPTHANYTHINGV2_PROCESSOR, _DEPTHANYTHINGV2_MODEL, _DEPTHANYTHINGV2_CACHE_DIR, _DEPTHANYTHINGV2_MODEL_ID

    if model_id is None:
        model_id = _DEPTHANYTHINGV2_MODEL_ID

    if (
        _DEPTHANYTHINGV2_PROCESSOR is None
        or _DEPTHANYTHINGV2_MODEL is None
        or _DEPTHANYTHINGV2_CACHE_DIR != cache_dir
        or _DEPTHANYTHINGV2_MODEL_ID != model_id
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        kwargs = {}
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = cache_dir

        _DEPTHANYTHINGV2_PROCESSOR = AutoImageProcessor.from_pretrained(
            model_id, **kwargs
        )
        _DEPTHANYTHINGV2_MODEL = (
            AutoModelForDepthEstimation.from_pretrained(model_id, **kwargs)
            .cuda()
            .eval()
        )
        _DEPTHANYTHINGV2_CACHE_DIR = cache_dir
        _DEPTHANYTHINGV2_MODEL_ID = model_id

    return _DEPTHANYTHINGV2_PROCESSOR, _DEPTHANYTHINGV2_MODEL


def _save_quantized_depth(depth_root: str, cache_key: str, depth: torch.Tensor) -> None:
    Path(depth_root).mkdir(parents=True, exist_ok=True)

    depth = depth.detach().float().contiguous()
    codebook = depth.quantile(
        torch.linspace(0.0, 1.0, 65536, device=depth.device), interpolation="nearest"
    )
    depth_idx = torch.searchsorted(codebook, depth, side="right").clamp_max_(65535)
    prev_idx = (depth_idx - 1).clamp_min_(0)
    pick_prev = (depth - codebook[prev_idx]).abs() < (depth - codebook[depth_idx]).abs()
    depth_idx[pick_prev] = prev_idx[pick_prev]

    iio.imwrite(_depth_path(depth_root, cache_key), depth_idx.cpu().numpy().astype(np.uint16))
    np.save(_codebook_path(depth_root, cache_key), codebook.cpu().numpy().astype(np.float32))


def _load_quantized_depth(depth_root: str, cache_key: str) -> torch.Tensor:
    depth_idx = iio.imread(_depth_path(depth_root, cache_key))
    codebook = np.load(_codebook_path(depth_root, cache_key))
    return torch.tensor(codebook[depth_idx], dtype=torch.float32)


@torch.no_grad()
def load_or_infer_depthanythingv2(
    image_tensor: torch.Tensor,
    depth_root: str,
    cache_key: str,
    model_id: str | None = None,
    force_rerun: bool = False,
) -> torch.Tensor:
    cache_key = _sanitize_cache_key(cache_key)
    model_root = _model_cache_root(depth_root, model_id)
    depth_file = _depth_path(model_root, cache_key)
    codebook_file = _codebook_path(model_root, cache_key)

    if not force_rerun and os.path.exists(depth_file) and os.path.exists(codebook_file):
        return _load_quantized_depth(model_root, cache_key)

    hf_cache_dir = os.path.join(model_root, "_hf_cache")
    processor, model = _load_depthanythingv2(model_id=model_id, cache_dir=hf_cache_dir)
    image = _prepare_image(image_tensor)
    inputs = processor(images=image, return_tensors="pt", do_rescale=False)
    inputs["pixel_values"] = inputs["pixel_values"].cuda()
    outputs = model(**inputs)
    depth = outputs["predicted_depth"].squeeze()

    _save_quantized_depth(model_root, cache_key, depth)
    return depth.detach().cpu().float()
