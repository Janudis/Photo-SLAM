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

# def render(cam, voxel_data, gt_image, im_height, im_width, ss, track_max_w):
def render(
    voxel_data,
    cam,
    im_height, 
    im_width,
    gt_image = None, 
    color_mode=None,
    track_max_w=False,
    ss=None,
    output_depth=False,
    output_normal=False,
    output_T=False,
    rand_bg=False,
    use_auto_exposure=False,
    **other_opt):

    # print(f"other_opt {other_opt}")
    # ensure watchdog is running
    start_watchdog()
    # print("\n[PY-DBG] ===== ENTER render() =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam.w2c = cam.w2c.to(device)
    cam.c2w = cam.c2w.to(device)

    t_prep0 = time.perf_counter()

    if voxel_data["center"].numel() == 0:
        print("[WARNING] No voxels to render!")
        return
    for k in voxel_data:
        t = voxel_data[k]
        if isinstance(t, torch.Tensor) and t.device != device:
            voxel_data[k] = t.to(device, non_blocking=True).contiguous()    # ← keep original leaf

    torch.cuda.synchronize()  # make prep blocking for a clean split
    t_prep1 = time.perf_counter()
    prep_ms = (t_prep1 - t_prep0) * 1000.0
    seen = set()

    def resize_rendering(render, size, mode='bilinear', align_corners=False):
        return torch.nn.functional.interpolate(
            render[None], size=size, mode=mode, align_corners=align_corners, antialias=True)[0]
    
    # def freeze_vox_geo(self):
    #     '''
    #     Freeze grid points parameter and pre-gather them to each voxel.
    #     '''
    #     with torch.no_grad():
    #         voxel_data['frozen_vox_geo'] = svr.GatherGeoParams.apply(
    #             voxel_data['vox_key'],
    #             torch.arange(voxel_data['num_voxels'], device="cuda"),
    #             voxel_data['_geo_grid_pts']
    #         )
    #     voxel_data['_geo_grid_pts'].requires_grad = False

    # def unfreeze_vox_geo(self):
    #     '''
    #     Unfreeze grid points parameter.
    #     '''
    #     del voxel_data['frozen_vox_geo']
    #     voxel_data['_geo_grid_pts'].requires_grad = True

    # def vox_fn(idx, cam_pos, color_mode=None, viewdir=None):
    #     if isinstance(idx, torch.Tensor):
    #         seen.update(idx.detach().cpu().tolist())
    #     # print(f"[PY-DBG] Running vox_fn with mode: {mode}, idx: {idx}")
    #     geos = svr.GatherGeoParams.apply(          # build it on-the-fly
    #         voxel_data['vox_key'],
    #         idx,
    #         voxel_data['geo_grid_pts'])

    #     rgbs = svr.SH_eval.apply(
    #         voxel_data['active_sh_degree'],                        # active_sh_degree   (use 2, 3, … as you like)
    #         idx,                     # idx  (let SH_eval create empty tensor internally)
    #         voxel_data['center'],    # vox_center  (N,3)
    #         cam_pos,                  # cam_pos (3,) #cam.c2w[:3, 3],
    #         viewdir,                        # viewdir
    #         # voxel_data['sh0'].squeeze(1),     # sh0  (N,3)
    #         voxel_data['sh0'],
    #         voxel_data['shs']         # shs  (N,45,3)
    #     )      
    #     return dict(
    #         geos=geos,
    #         rgbs=rgbs,
    #         subdiv_p=voxel_data['subdiv_p'].view(-1, 1)
    #     )
    def vox_fn(idx, cam_pos, color_mode=None, viewdir=None):
        '''
        Per-frame voxel property processing. Two important operations:
        1. Gather grid points parameter into each voxel.
        2. Compute view-dependent color of each voxel.

        Input:
            @idx        Indices for active voxel for current frame.
            @cam_pos    Camera position.
        Output:
            @vox_params A dictionary of the pre-process voxel properties.
        '''
        # Gather the density values at the eight corners of each voxel.
        # It defined a trilinear density field.
        # The final tensor are in shape [#vox, 8]
        if 'frozen_vox_geo' in voxel_data and voxel_data['frozen_vox_geo'] is not None:
            # print("yes")
            # Match SVRaster: pass the full pre-gathered tensor (no index_select here)
            geos = voxel_data['frozen_vox_geo']
        else:
            geos = svr.GatherGeoParams.apply(
                voxel_data['vox_key'],
                idx,
                voxel_data['_geo_grid_pts']
            )
        # geos = svr.GatherGeoParams.apply(
        #     voxel_data['vox_key'],
        #     idx,
        #     voxel_data['_geo_grid_pts'])

        # Compute voxel colors
        if color_mode is None or color_mode == "sh":
            active_sh_degree = voxel_data['active_sh_degree']
            color_mode = "sh"
        elif color_mode.startswith("sh"):
            active_sh_degree = int(color_mode[2])
            color_mode = "sh"

        if color_mode == "sh":
            rgbs = svr.SH_eval.apply(
                active_sh_degree,
                idx,
                voxel_data['center'],
                cam_pos,
                viewdir, # Ignore above two when viewdir is not None
                voxel_data['sh0'],
                voxel_data['shs'],
            )
        elif color_mode == "rand":
            rgbs = torch.rand([voxel_data['num_voxels'], 3], dtype=torch.float32, device="cuda")
        elif color_mode == "dontcare":
            rgbs = torch.empty([voxel_data['num_voxels'], 3], dtype=torch.float32, device="cuda")
        else:
            raise NotImplementedError

        # Pack everything
        vox_params = {
            'geos': geos,
            'rgbs': rgbs,
            'subdiv_p': voxel_data['subdiv_p'].view(-1, 1), # Dummy param to record subdivision priority
        }
        if vox_params['subdiv_p'] is None:
            vox_params['subdiv_p'] = torch.ones([voxel_data['num_voxels'].view(-1, 1), 1], device="cuda")

        return vox_params
    
    # print(f"image size in renderer: {cam.image_width} x {cam.image_height}" " tanfovx, tanfovy:", cam.tanfovx, cam.tanfovy)
    # Preprocessing
    if ss is None:
        ss = voxel_data['ss'] 
    # print(f"ss {ss}")
        
    w_src, h_src = im_width, im_height
    w, h = round(w_src * ss), round(h_src * ss)
    w_ss, h_ss = w / w_src, h / h_src
    if ss != 1.0 and 'gt_color' in other_opt:
        other_opt['gt_color'] = resize_rendering(other_opt['gt_color'], size=(h, w))
    
    n_samp_per_vox = other_opt.pop('n_samp_per_vox', voxel_data['n_samp_per_vox'])
    # print(f"n_samp_per_vox {n_samp_per_vox}")

    # rs = svr.RasterSettings(
    #         color_mode = 'sh',
    #         n_samp_per_vox = 1,  
    #         # image_width = im_width,
    #         # image_height = im_height,
    #         image_width = w,
    #         image_height = h,
    #         tanfovx = cam.tanfovx,
    #         tanfovy = cam.tanfovy,
    #         # cx = cam.cx,
    #         # cy = cam.cy,
    #         cx = cam.cx * w_ss,
    #         cy = cam.cy * h_ss,
    #         w2c_matrix = cam.w2c.to(device),
    #         c2w_matrix = cam.c2w.to(device),
    #         bg_color = float(False), #0.0,
    #         near = 0.01,
    #         track_max_w = track_max_w
    #     )
    rs = svr.RasterSettings(
        color_mode=color_mode,
        n_samp_per_vox=n_samp_per_vox,
        image_width=w,
        image_height=h,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        cx=cam.cx * w_ss,
        cy=cam.cy * h_ss,
        w2c_matrix=cam.w2c,
        c2w_matrix=cam.c2w,
        bg_color=float(voxel_data['white_background']),
        near=0.01,
        need_depth=output_depth,
        need_normal=output_normal,
        track_max_w=track_max_w,
        **other_opt)
    torch.cuda.synchronize()
    t_rast0 = time.perf_counter()

    try:
        color, depth, normal, T, max_w = svr.rasterize_voxels(
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

    t_rast1 = time.perf_counter()
    raster_ms = (t_rast1 - t_rast0) * 1000.0
    # ---- PRINT: Python-side timing ----
    total_ms = prep_ms + raster_ms
    n_vox = int(voxel_data["center"].shape[0])
    H, W = int(cam.image_height), int(cam.image_width)
    # print(f"[PY][render] voxels={n_vox} res={W}x{H} "
    #       f"prep={prep_ms:.3f}ms raster={raster_ms:.3f}ms total={total_ms:.3f}ms",
    #       flush=True)
    used = len(seen)
    total = voxel_data["center"].shape[0]
    # print(f"[VOX][render] used={used} / total={total}")

    ###################################
    # Post-processing and pack output
    ###################################
    if rand_bg:
        color = color + T * torch.rand_like(color, requires_grad=False)
    elif not voxel_data['white_background'] and not voxel_data['black_background']:
        color = color + T * color.mean((1,2), keepdim=True)

    render_pkg = {
        'color': color,
        'depth': depth if output_depth else None,
        'normal': normal if output_normal else None,
        'T': T if output_T else None,
        'max_w': max_w,
    }

    for k in ['color', 'depth', 'normal', 'T']:
        render_pkg[f'raw_{k}'] = render_pkg[k]

        # Post process super-sampling
        if render_pkg[k] is not None and render_pkg[k].shape[-2:] != (h_src, w_src):
            render_pkg[k] = resize_rendering(render_pkg[k], size=(h_src, w_src))

    # Clip intensity
    render_pkg['color'] = render_pkg['color'].clamp(0, 1)
    
    return render_pkg