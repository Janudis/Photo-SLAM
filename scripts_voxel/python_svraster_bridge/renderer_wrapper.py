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
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party"))

print("=== LOADING voxel_bridge!!! ===")

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

# -- Watchdog thread control --
_watchdog_thread: threading.Thread | None = None
_watchdog_stop = False
def watchdog():
    while not _watchdog_stop:
        time.sleep(5)
        print("[PY]  watchdog: threads =", threading.enumerate(), flush=True)
def start_watchdog():
    global _watchdog_thread, _watchdog_stop
    _watchdog_stop = False
    if _watchdog_thread is None or not _watchdog_thread.is_alive():
        _watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        _watchdog_thread.start()
def stop_watchdog():
    global _watchdog_stop, _watchdog_thread
    _watchdog_stop = True
    if _watchdog_thread is not None:
        _watchdog_thread.join(timeout=1.0)
        _watchdog_thread = None
# Start it once so that any render() call can rely on it
start_watchdog()
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
    # ensure watchdog is running
    # start_watchdog()
    # print("\n[PY-DBG] ===== ENTER render() =====")
    if output_dir not in ("", None):
        os.makedirs(output_dir, exist_ok=True)
    # Determine proper CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move camera matrices to CUDA if needed
    cam.w2c = cam.w2c.to(device)
    cam.c2w = cam.c2w.to(device)

    # Convert ground truth image
    if rgb_image is None:                   
        gt_image = None
    else:
        gt_image = torch.from_numpy(
            np.asarray(rgb_image, dtype=np.uint8)
        ).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    # imageio.imwrite("debug_input.png", rgb_image)

    if voxel_data["centers"].numel() == 0:
        print("[WARNING] No voxels to render!")
        return

    rs = svr.RasterSettings(
        color_mode='rgb',               # start simple; switch to 'sh' later
        vox_geo_mode='triinterp',
        density_mode='exp_linear_11',
        image_width=cam.image_width,
        image_height=cam.image_height,
        tanfovx=cam.tanfovx,  
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

    def vox_fn(idx, cam_pos, mode):
        # print(f"[PY-DBG] Running vox_fn with mode: {mode}, idx: {idx}")
        geos = voxel_data['geos']                  # (N, 8), passed directly
        # rgbs = svr.SH_eval.apply(
        #     2,                        # active_sh_degree   (use 2, 3, … as you like)
        #     None,                     # idx  (let SH_eval create empty tensor internally)
        #     voxel_data['centers'],    # vox_centers  (N,3)
        #     cam_pos,                  # cam_pos (3,) #cam.c2w[:3, 3],
        #     None,                        # viewdir
        #     voxel_data['colors'].squeeze(1),     # sh0  (N,3)
        #     voxel_data['shs']         # shs  (N,45,3)
        # )      
        rgbs = voxel_data['colors'].squeeze(1)
        # dens = voxel_data["opacities"].view(-1, 1)                        # (N, 3), computed SH colors

        return dict(
            geos=geos,
            rgbs=rgbs,
            subdiv_p=voxel_data['subdiv_p'].view(-1, 1)
        )

    # def vox_fn(idx, cam_pos, mode):
    #     geo_full = voxel_data['geos']           # (N,8) from C++
    #     cov6     = geo_full[:, :6]              # (N,6)  (σ_xx … σ_zz)
    #     density  = torch.full_like(cov6[:, :1], 1.5)  # log-density ~1.5 ⇒ exp≈4.5
    #     pad      = torch.zeros_like(cov6[:, :1])      # slot 7 = 0
    #     geos     = torch.cat([cov6, density, pad], 1) # (N,8)

    #     rgbs = voxel_data['colors'].squeeze(1)        # plain RGB for now
    #     return dict(
    #         geos      = geos,
    #         rgbs      = rgbs,
    #         subdiv_p  = voxel_data['subdiv_p'].view(-1,1),
    #     )
    
    # # Move voxel data to device
    # for k in voxel_data:
    #     if isinstance(voxel_data[k], torch.Tensor):
    #         voxel_data[k] = voxel_data[k].to(device, non_blocking=True).contiguous()
    for k in voxel_data:
        t = voxel_data[k]
        if isinstance(t, torch.Tensor) and t.device != device:
            # print(f"[PY-DBG] Moving {k} tensor to device: {device}")
            voxel_data[k] = t.to(device, non_blocking=True).contiguous()    # ← keep original leaf

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
        raise e
        # sys.exit(1)    

    torch.cuda.synchronize()

    rgb = out[0].unsqueeze(0)           # (1, 3, H, W)
    result = {"rgb": rgb}               # prediction
    # if len(out) > 3:
    #     result["T"] = out[3]            # optional transmittance
    return result