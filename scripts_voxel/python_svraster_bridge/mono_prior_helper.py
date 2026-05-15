import os
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


_DEPTHANYTHINGV2_PROCESSOR = None
_DEPTHANYTHINGV2_MODEL = None
_DEPTHANYTHINGV2_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_DEPTHANYTHINGV2_CACHE_DIR = None
_METRIC3D_MODEL = None
_METRIC3D_MODEL_ID = "metric3d_vit_small"
_METRIC3D_CACHE_DIR = None


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


def _prepare_image_chw(image_tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")

    image = image_tensor.detach().float().cpu()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] != 3:
        raise ValueError(f"Expected 3-channel image tensor, got shape {tuple(image.shape)}")
    return image.contiguous()


def _is_metric3d_model_id(model_id: str | None) -> bool:
    if not model_id:
        return False
    return "metric3d" in str(model_id).lower()


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


def _load_metric3d(model_id: str | None = None, cache_dir: str | None = None):
    global _METRIC3D_MODEL, _METRIC3D_MODEL_ID, _METRIC3D_CACHE_DIR

    if model_id is None:
        model_id = _METRIC3D_MODEL_ID

    if (
        _METRIC3D_MODEL is None
        or _METRIC3D_CACHE_DIR != cache_dir
        or _METRIC3D_MODEL_ID != model_id
    ):
        hub_dir = None
        if cache_dir:
            hub_dir = os.path.join(cache_dir, "_torch_hub")
            Path(hub_dir).mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(hub_dir)

        _METRIC3D_MODEL = torch.hub.load("yvanyin/metric3d", model_id, pretrain=True)
        _METRIC3D_MODEL = _METRIC3D_MODEL.cuda().eval()
        _METRIC3D_CACHE_DIR = cache_dir
        _METRIC3D_MODEL_ID = model_id

    return _METRIC3D_MODEL


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
def _infer_metric3d_depth(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    fx: float | None,
) -> torch.Tensor:
    # WildGS-SLAM follows Metric3D's hubconf preprocessing:
    # resize while keeping aspect ratio, pad to (616, 1064), infer, unpad, upsample,
    # then convert from canonical depth to camera scale using fx / 1000.
    image_size = (616, 1064)
    image = _prepare_image_chw(image_tensor).unsqueeze(0).cuda()
    h, w = image.shape[-2:]
    scale = min(image_size[0] / h, image_size[1] / w)
    resized_h = max(1, int(h * scale))
    resized_w = max(1, int(w * scale))

    img_tensor = F.interpolate(
        image,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)[None, :, None, None]
    img_tensor = (img_tensor - mean) / std

    pad_h = image_size[0] - resized_h
    pad_w = image_size[1] - resized_w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    img_tensor = TF.pad(
        img_tensor,
        (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half),
        fill=0.0,
    )

    pred_depth, _, _ = model.inference({"input": img_tensor})
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half),
        pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half),
    ]
    pred_depth = F.interpolate(
        pred_depth[None, None, :, :],
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    if fx is None or not np.isfinite(float(fx)) or float(fx) <= 1e-6:
        raise ValueError("Metric3D inference requires a valid focal length fx")

    canonical_to_real_scale = float(fx) / 1000.0
    pred_depth = pred_depth * canonical_to_real_scale
    return torch.clamp(pred_depth, 0.0, 300.0).detach().cpu().float()


@torch.no_grad()
def load_or_infer_mono_prior(
    image_tensor: torch.Tensor,
    depth_root: str,
    cache_key: str,
    model_id: str | None = None,
    force_rerun: bool = False,
    fx: float | None = None,
) -> torch.Tensor:
    cache_key = _sanitize_cache_key(cache_key)
    model_root = _model_cache_root(depth_root, model_id)
    depth_file = _depth_path(model_root, cache_key)
    codebook_file = _codebook_path(model_root, cache_key)

    if not force_rerun and os.path.exists(depth_file) and os.path.exists(codebook_file):
        return _load_quantized_depth(model_root, cache_key)

    if _is_metric3d_model_id(model_id):
        metric3d_cache_dir = os.path.join(model_root, "_metric3d_cache")
        model = _load_metric3d(model_id=model_id, cache_dir=metric3d_cache_dir)
        depth = _infer_metric3d_depth(model, image_tensor, fx)
    else:
        hf_cache_dir = os.path.join(model_root, "_hf_cache")
        processor, model = _load_depthanythingv2(model_id=model_id, cache_dir=hf_cache_dir)
        image = _prepare_image(image_tensor)
        inputs = processor(images=image, return_tensors="pt", do_rescale=False)
        inputs["pixel_values"] = inputs["pixel_values"].cuda()
        outputs = model(**inputs)
        depth = outputs["predicted_depth"].squeeze()

    if force_rerun:
        return depth.detach().cpu().float()

    _save_quantized_depth(model_root, cache_key, depth)
    return depth.detach().cpu().float()


@torch.no_grad()
def load_or_infer_depthanythingv2(
    image_tensor: torch.Tensor,
    depth_root: str,
    cache_key: str,
    model_id: str | None = None,
    force_rerun: bool = False,
    fx: float | None = None,
) -> torch.Tensor:
    return load_or_infer_mono_prior(
        image_tensor=image_tensor,
        depth_root=depth_root,
        cache_key=cache_key,
        model_id=model_id,
        force_rerun=force_rerun,
        fx=fx,
    )
