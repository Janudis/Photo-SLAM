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
import re
import faulthandler, threading, signal
import importlib.util
import imageio
from typing import NamedTuple

# Dynamically load installed 'svraster_cuda.renderer'
spec = importlib.util.find_spec("svraster_cuda.renderer")
if spec is None:
    raise ImportError("Cannot find 'svraster_cuda.renderer'. Make sure it's pip-installed.")
svr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svr)
_RASTERIZE_VOXELS_BACKWARD_DOC = getattr(svr._C.rasterize_voxels_backward, "__doc__", "") or ""
_BACKWARD_ARG_IDS = [
    int(m.group(1))
    for m in re.finditer(r"arg(\d+):", _RASTERIZE_VOXELS_BACKWARD_DOC)
]
_BACKWARD_SUPPORTS_SCALING_PENALTY = bool(_BACKWARD_ARG_IDS) and max(_BACKWARD_ARG_IDS) >= 34
# print("svraster_cuda.renderer loaded from:", svr.__file__)
# print(" underlying C extension:", svr._C.__file__)
faulthandler.enable(file=sys.__stderr__)           # full C‑level dump on SIGUSR1

# -- Watchdog thread control --
_watchdog_thread: threading.Thread | None = None
_watchdog_stop = False
def watchdog():
    while not _watchdog_stop:
        time.sleep(5)
        # print("[PY]  watchdog: threads =", threading.enumerate(), flush=True)
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
# print("=== LOADING voxel_bridge!!! ===")


class BridgeRasterSettings(NamedTuple):
    color_mode: str
    n_samp_per_vox: int
    image_width: int
    image_height: int
    tanfovx: float
    tanfovy: float
    cx: float
    cy: float
    w2c_matrix: torch.Tensor
    c2w_matrix: torch.Tensor
    bg_color: float = 0
    near: float = 0.02
    need_depth: bool = False
    need_normal: bool = False
    track_max_w: bool = False
    lambda_R_concen: float = 0
    lambda_ascending: float = 0
    lambda_scaling_penalty: float = 0
    min_voxel_size: float = 0
    lambda_dist: float = 0
    gt_color: torch.Tensor = torch.empty(0)
    vox_feats: torch.Tensor = torch.empty([0, 0])
    debug: bool = False


class _RasterizeVoxelsWithState(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raster_settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        vox_feats,
        subdiv_p,
    ):
        need_distortion = raster_settings.lambda_dist > 0

        args = (
            raster_settings.n_samp_per_vox,
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix,
            raster_settings.c2w_matrix,
            raster_settings.bg_color,
            raster_settings.need_depth,
            need_distortion,
            raster_settings.need_normal,
            raster_settings.track_max_w,
            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            vox_feats,
            geomBuffer,
            raster_settings.debug,
        )

        num_rendered, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w, out_feat = (
            svr._C.rasterize_voxels(*args)
        )
        _, _, n_contrib = svr._C.unpack_ImageState(
            raster_settings.image_width,
            raster_settings.image_height,
            imgBuffer,
        )

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            out_T,
            out_depth,
            out_normal,
        )
        ctx.mark_non_differentiable(max_w, n_contrib)
        return out_color, out_depth, out_normal, out_T, max_w, n_contrib, out_feat

    @staticmethod
    def backward(
        ctx,
        dL_dout_color,
        dL_dout_depth,
        dL_dout_normal,
        dL_dout_T,
        dL_dmax_w,
        dL_dn_contrib,
        dL_dout_feat,
    ):
        raster_settings = ctx.raster_settings
        num_rendered = ctx.num_rendered
        (
            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            out_T,
            out_depth,
            out_normal,
        ) = ctx.saved_tensors

        if _BACKWARD_SUPPORTS_SCALING_PENALTY:
            args = (
                num_rendered,
                raster_settings.n_samp_per_vox,
                raster_settings.image_width,
                raster_settings.image_height,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.cx,
                raster_settings.cy,
                raster_settings.w2c_matrix,
                raster_settings.c2w_matrix,
                raster_settings.bg_color,
                octree_paths,
                vox_centers,
                vox_lengths,
                geos,
                rgbs,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                out_T,
                dL_dout_color,
                dL_dout_depth,
                dL_dout_normal,
                dL_dout_T,
                raster_settings.lambda_R_concen,
                raster_settings.gt_color,
                raster_settings.lambda_ascending,
                raster_settings.lambda_scaling_penalty,
                raster_settings.min_voxel_size,
                raster_settings.lambda_dist,
                raster_settings.need_depth,
                raster_settings.need_normal,
                out_depth,
                out_normal,
                raster_settings.debug,
            )
        else:
            if abs(float(raster_settings.lambda_scaling_penalty)) > 0.0 or \
               abs(float(raster_settings.min_voxel_size)) > 0.0:
                raise RuntimeError(
                    "Loaded svraster_cuda backend does not support GeoSVR scaling penalty "
                    "in rasterize_voxels_backward(); rebuild /home/dimitris/svraster/cuda "
                    "or disable Optimization.lambda_scaling_penalty.")
            args = (
                num_rendered,
                raster_settings.n_samp_per_vox,
                raster_settings.image_width,
                raster_settings.image_height,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.cx,
                raster_settings.cy,
                raster_settings.w2c_matrix,
                raster_settings.c2w_matrix,
                raster_settings.bg_color,
                octree_paths,
                vox_centers,
                vox_lengths,
                geos,
                rgbs,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                out_T,
                dL_dout_color,
                dL_dout_depth,
                dL_dout_normal,
                dL_dout_T,
                raster_settings.lambda_R_concen,
                raster_settings.gt_color,
                raster_settings.lambda_ascending,
                raster_settings.lambda_dist,
                raster_settings.need_depth,
                raster_settings.need_normal,
                out_depth,
                out_normal,
                raster_settings.debug,
            )

        dL_dgeos, dL_drgbs, subdiv_p_bw = svr._C.rasterize_voxels_backward(*args)
        return (
            None,
            None,
            None,
            None,
            None,
            dL_dgeos,
            dL_drgbs,
            None,
            subdiv_p_bw,
        )


def rasterize_voxels_with_state(
    raster_settings,
    octree_paths,
    vox_centers,
    vox_lengths,
    vox_fn,
):
    if not isinstance(raster_settings, BridgeRasterSettings):
        raise Exception("Expect BridgeRasterSettings as first argument.")
    if raster_settings.n_samp_per_vox > svr._C.MAX_N_SAMP or raster_settings.n_samp_per_vox < 1:
        raise Exception(f"n_samp_per_vox should be in range [1, {svr._C.MAX_N_SAMP}].")

    N = octree_paths.numel()
    device = octree_paths.device
    if vox_centers.shape[0] != N or vox_lengths.numel() != N:
        raise Exception("Size mismatched.")
    if len(vox_centers.shape) != 2 or vox_centers.shape[1] != 3:
        raise Exception("Expect vox_centers in shape [N, 3].")
    if (
        raster_settings.w2c_matrix.device != device
        or raster_settings.c2w_matrix.device != device
        or vox_centers.device != device
        or vox_lengths.device != device
    ):
        raise Exception("Device mismatch.")

    n_duplicates, geomBuffer = svr._C.rasterize_preprocess(
        raster_settings.image_width,
        raster_settings.image_height,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.cx,
        raster_settings.cy,
        raster_settings.w2c_matrix,
        raster_settings.c2w_matrix,
        raster_settings.near,
        octree_paths,
        vox_centers,
        vox_lengths,
        raster_settings.debug,
    )
    in_frusts_idx = torch.where(n_duplicates > 0)[0]

    cam_pos = raster_settings.c2w_matrix[:3, 3]
    vox_params = vox_fn(in_frusts_idx, cam_pos, raster_settings.color_mode)
    geos = vox_params["geos"]
    rgbs = vox_params["rgbs"]
    subdiv_p = vox_params["subdiv_p"]

    if geos.shape != (N, 8):
        raise Exception(f"Expect geos in ({N}, 8) but got {geos.shape}")
    if rgbs.shape[0] != N:
        raise Exception(f"Expect rgbs in ({N}, 3) but got {rgbs.shape}")
    if subdiv_p.shape[0] != N:
        raise Exception(f"Expect subdiv_p in ({N}, 1) but got {subdiv_p.shape}")
    if geos.device != device:
        raise Exception("Device mismatch: geos.")
    if rgbs.device != device:
        raise Exception("Device mismatch: rgbs.")
    if subdiv_p.device != device:
        raise Exception("Device mismatch: subdiv_p.")

    vox_feats = raster_settings.vox_feats
    if not torch.is_tensor(vox_feats) or vox_feats.numel() == 0:
        vox_feats = torch.empty([N, 0], dtype=torch.float32, device=device)
    else:
        if vox_feats.shape[0] != N:
            raise Exception(f"Expect vox_feats in ({N}, C) but got {vox_feats.shape}")
        if vox_feats.device != device:
            raise Exception("Device mismatch: vox_feats.")
        vox_feats = vox_feats.contiguous()

    return _RasterizeVoxelsWithState.apply(
        raster_settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        vox_feats,
        subdiv_p,
    )

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
        kwargs = {
            "size": size,
            "mode": mode,
        }
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = align_corners
            kwargs["antialias"] = True
        return torch.nn.functional.interpolate(render[None], **kwargs)[0]
    
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
    vox_feats = other_opt.pop('vox_feats', torch.empty([0, 0], dtype=torch.float32, device=device))
    if torch.is_tensor(vox_feats):
        vox_feats = vox_feats.to(device, non_blocking=True).contiguous()

    rs = BridgeRasterSettings(
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
        vox_feats=vox_feats,
        **other_opt)
    torch.cuda.synchronize()
    t_rast0 = time.perf_counter()

    try:
        color, depth, normal, T, max_w, n_contrib, feat = rasterize_voxels_with_state(
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
        'n_contrib': n_contrib,
        'feat': feat if feat is not None and feat.numel() > 0 else None,
    }

    for k in ['color', 'depth', 'normal', 'T', 'n_contrib', 'feat']:
        render_pkg[f'raw_{k}'] = render_pkg[k]

        # Post process super-sampling
        if render_pkg[k] is not None and render_pkg[k].shape[-2:] != (h_src, w_src):
            if k == 'n_contrib':
                resized = resize_rendering(
                    render_pkg[k].to(torch.float32),
                    size=(h_src, w_src),
                    mode='nearest')
                render_pkg[k] = resized.round().to(torch.int32)
            elif k == 'feat':
                render_pkg[k] = resize_rendering(
                    render_pkg[k],
                    size=(h_src, w_src),
                    mode='nearest')
            else:
                render_pkg[k] = resize_rendering(render_pkg[k], size=(h_src, w_src))

    # Clip intensity
    render_pkg['color'] = render_pkg['color'].clamp(0, 1)
    
    return render_pkg
