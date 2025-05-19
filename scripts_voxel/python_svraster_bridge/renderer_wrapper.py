# python_svraster_bridge/renderer_wrapper.py
"""
Thin wrapper that converts Photo-SLAM “hyper-primitive” tensors to the
layout expected by svraster_cuda.renderer.rasterize_voxels(), invokes the
renderer, and returns the rendered RGB (+ optional depth / normal).

Nothing here is performance–critical – all the heavy work is still done
in the compiled CUDA extension.
"""
from __future__ import annotations
import os
import sys
import torch
import numpy as np
import imageio
import math
import inspect
import time
import faulthandler, threading, signal
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party"))
print("=== LOADING voxel_bridge ===")
# from third_party import svraster_cuda
# from third_party.svraster_cuda import renderer as svr
# import svraster_cuda.renderer as svr
# Drop stub path if present (optional)
THIRD_PARTY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../third_party"))
if THIRD_PARTY in sys.path:
    sys.path.remove(THIRD_PARTY)
# Dynamically load installed 'svraster_cuda.renderer'
spec = importlib.util.find_spec("svraster_cuda.renderer")
if spec is None:
    raise ImportError("Cannot find 'svraster_cuda.renderer'. Make sure it's pip-installed.")
svr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svr)

faulthandler.enable(file=sys.__stderr__)           # full C‑level dump on SIGUSR1
def watchdog():
    while True:
        time.sleep(5)
        print("[PY]  watchdog: threads =", threading.enumerate(), flush=True)
threading.Thread(target=watchdog, daemon=True).start()

# # A global optimizer, you may re-initialize it per scene
# _optimizer = None
# _loss_fn = torch.nn.MSELoss()

def safe_stat(x, name):
    try:
        print(f"[PY] {name}: shape={x.shape}, dtype={x.dtype}, min={x.min()}, max={x.max()}, device={x.device}")
    except Exception as e:
        print(f"[PY] {name}: ERROR: {e}")

def render(cam, voxel_data, rgb_image, output_dir="results/rendered"):
    """
    cam:      MiniCam object from C++ -> contains width, height, intrinsics, w2c, etc.
    voxel_data: dictionary with keys:
        - centers:    (N, 3)
        - sizes:      (N,) or (N, 1)
        - shs:        (N, 45)
        - colors:     (N, 3)
        - opacities:  (N, 8)
        - cov3D:      (N, 6)
    rgb_image: ground truth RGB image from C++ as a NumPy array (H, W, 3)
    output_dir: where to save renderings
    """

    # os.makedirs(output_dir, exist_ok=True)
    # allow '' or None → just don’t save anything
    if output_dir not in ("", None):
        os.makedirs(output_dir, exist_ok=True)
    # Determine proper CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Render: using device =", device)

    # Move camera matrices to CUDA if needed
    cam.w2c = cam.w2c.to(device)
    cam.c2w = cam.c2w.to(device)

    # Convert ground truth image
    gt_image = torch.from_numpy(np.array(rgb_image)).float() / 255.0
    gt_image = gt_image.permute(2, 0, 1).unsqueeze(0).to(device)

    if voxel_data["centers"].numel() == 0:
        print("[WARNING] No voxels to render!")
        return

    rs = svr.RasterSettings(
        color_mode='rgb',               # start simple; switch to 'sh' later
        vox_geo_mode='triinterp',
        density_mode='exp_linear_11',
        image_width=cam.image_width,
        image_height=cam.image_height,
        tanfovx=cam.tanfovx,  # <-- already computed in MiniCam Python class
        tanfovy=cam.tanfovy,
        cx=cam.cx,
        cy=cam.cy,
        w2c_matrix=cam.w2c.to(device),
        c2w_matrix=cam.c2w.to(device),
        background=torch.zeros(3, device=device),
        cam_mode="persp",
        need_depth=False,
        need_normal=False,
    )

    if voxel_data["subdiv_p"].requires_grad is False:
        print("[WARN] voxel_data['subdiv_p'] does not require grad.")

    def vox_fn(idx, cam_pos, mode):
        geo = voxel_data['cov3D']
        if geo.shape[1] == 6:
            pad = torch.zeros(geo.size(0), 2, device=device, dtype=geo.dtype)
            geos = torch.cat([geo, pad], dim=1)

        rgbs  = voxel_data['colors']              # (N,3)
        subdiv_p = voxel_data["subdiv_p"].view(-1,1)
        if not torch.isfinite(subdiv_p).all():
            print("[WARN] subdiv_p contains NaNs or infs; replacing with zeros")
            subdiv_p = torch.where(torch.isfinite(subdiv_p), subdiv_p, torch.zeros_like(subdiv_p))
        # Optional: clamp to [0,1]
        subdiv_p = subdiv_p.clamp(0.0, 1.0)
        dens = voxel_data["opacities"].view(-1, 1) * subdiv_p

        # return dict(geos=geos, rgbs=rgbs, densities=dens, subdiv_p=subdiv)
        return dict(geos=geos, rgbs=rgbs, densities=dens, subdiv_p=subdiv_p)

    # Move voxel data to device
    for k in voxel_data:
        if isinstance(voxel_data[k], torch.Tensor):
            voxel_data[k] = voxel_data[k].to(device, non_blocking=True).contiguous()

    try:
        out = svr.rasterize_voxels(
            rs,
            voxel_data["octpaths"],
            voxel_data["centers"],
            voxel_data["vox_lengths"].view(-1, 1),
            vox_fn
        )
    except Exception as e:
        print("[FATAL] Exception in rasterize_voxels:", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)    
    torch.cuda.synchronize()

    # rgb = out[0]  # (3, H, W)
    # depth = out[1]     # (3, H, W)
    # normal = out[2]    # (3, H, W)
    # T = out[3]         # (3, H, W)

    # return {
    #     "rgb": rgb,
    #     "depth": depth,
    #     "normal": normal,
    #     "T": T,
    #     "raw_T": T.clone(),  # used for T_concen or debugging
    # }

    # rgb = out[0].unsqueeze(0)  # (1, 3, H, W)
    # gt  = gt_image.clone()               # (1, 3, H, W)
    # depth = out[1] if len(out) > 1 else None
    # normal = out[2] if len(out) > 2 else None
    # T = out[3] if len(out) > 3 else None
    # return {
    #     "rgb": rgb,
    #     "gt": gt,  # ground-truth used for SSIM
    #     "depth": depth,
    #     "normal": normal,
    #     "T": T,
    #     "raw_T": T.clone() if T is not None else None
    # }

    rgb = out[0].unsqueeze(0)           # (1, 3, H, W)
    result = {"rgb": rgb}               # prediction
    if len(out) > 3:
        result["T"] = out[3]            # optional transmittance
    return result