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

# Dynamically load installed 'svraster_cuda.renderer'
spec = importlib.util.find_spec("svraster_cuda.renderer")
if spec is None:
    raise ImportError("Cannot find 'svraster_cuda.renderer'. Make sure it's pip-installed.")
svr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svr)
print("svraster_cuda.renderer loaded from:", svr.__file__)
print(" underlying C extension:", svr._C.__file__)
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
print("=== LOADING voxel_bridge!!! ===")

def render(cam, voxel_data):
    # ensure watchdog is running
    start_watchdog()
    # print("\n[PY-DBG] ===== ENTER render() =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam.w2c = cam.w2c.to(device)
    cam.c2w = cam.c2w.to(device)

    if voxel_data["center"].numel() == 0:
        print("[WARNING] No voxels to render!")
        return
    for k in voxel_data:
        t = voxel_data[k]
        if isinstance(t, torch.Tensor) and t.device != device:
            voxel_data[k] = t.to(device, non_blocking=True).contiguous()    # ← keep original leaf

    def vox_fn(idx, cam_pos, color_mode=None, viewdir=None):
        # print(f"[PY-DBG] Running vox_fn with mode: {mode}, idx: {idx}")
        geos = svr.GatherGeoParams.apply(          # build it on-the-fly
            voxel_data['vox_key'],
            idx,
            voxel_data['geo_grid_pts'])

        rgbs = svr.SH_eval.apply(
            voxel_data['active_sh_degree'],                        # active_sh_degree   (use 2, 3, … as you like)
            idx,                     # idx  (let SH_eval create empty tensor internally)
            voxel_data['center'],    # vox_center  (N,3)
            cam_pos,                  # cam_pos (3,) #cam.c2w[:3, 3],
            viewdir,                        # viewdir
            # voxel_data['sh0'].squeeze(1),     # sh0  (N,3)
            voxel_data['sh0'],
            voxel_data['shs']         # shs  (N,45,3)
        )      
        return dict(
            geos=geos,
            rgbs=rgbs,
            subdiv_p=voxel_data['subdiv_p'].view(-1, 1)
        )
    
    rs = svr.RasterSettings(
            color_mode = 'sh',
            n_samp_per_vox = 1,  
            image_width = cam.image_width,
            image_height = cam.image_height,
            tanfovx = cam.tanfovx,
            tanfovy = cam.tanfovy,
            cx = cam.cx,
            cy = cam.cy,
            w2c_matrix = cam.w2c.to(device),
            c2w_matrix = cam.c2w.to(device),
            bg_color = 0.0,
            near = 0.02,
            track_max_w = True,
    )
    
    try:
        out = svr.rasterize_voxels(
            rs,
            voxel_data["octpath"],
            voxel_data["center"],
            voxel_data["vox_size"].view(-1, 1),
            vox_fn
        )
    except Exception as e:
        print("[FATAL] Exception in rasterize_voxels:", e)
        import traceback, sys
        traceback.print_exc()
        raise e

    torch.cuda.synchronize()
    color, depth, normal, T, max_w = out
    # SVRaster gives (H,W,3) color → make (1,3,H,W)
    color  = color .unsqueeze(0)
    depth  = depth .unsqueeze(0) if depth is not None  else None
    normal = normal.unsqueeze(0) if normal is not None else None
    T      = T     .unsqueeze(0) if T      is not None else None

    return {
      "color":  color,
      "depth":  depth,
      "normal": normal,
      "T":      T,
      "max_w":  max_w,
    }