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

# A global optimizer, you may re-initialize it per scene
_optimizer = None
_loss_fn = torch.nn.MSELoss()

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
        # tanfovx=math.tan(cam.fovx / 2),
        # tanfovy=math.tan(cam.fovy / 2),
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

    def vox_fn(idx, cam_pos, mode):
        geo = voxel_data['cov3D']
        if geo.shape[1] == 6:
            pad = torch.zeros(geo.size(0), 2, device=device, dtype=geo.dtype)
            geos = torch.cat([geo, pad], dim=1)
        else:
            geos = geo
        # rgbs = voxel_data['shs'][:, :3]
        # subdiv = torch.zeros(geos.shape[0], 1, device=device)
        rgbs  = voxel_data['colors']              # (N,3)
        dens  = voxel_data['opacities'].view(-1,1)  # (N,1)
        subdiv = torch.zeros_like(dens)           # (N,1)
        # return dict(geos=geos, rgbs=rgbs, subdiv_p=subdiv)
        return dict(geos=geos, rgbs=rgbs, densities=dens, subdiv_p=subdiv)

    # # — ensure all our voxel_data tensors are on the same CUDA device
    # for k in ("octpaths","centers","vox_lengths","cov3D","colors","shs","opacities"):
    #     t = voxel_data[k]
    #     if t.device.type != "cuda":
    #         voxel_data[k] = t.to(device, non_blocking=True)
    # # — call native signature: (settings, octree_paths, centers, lengths, vox_fn)
    # print('[PY]  launching kernel …');  t0 = time.time()

    # Move voxel data to device
    for k in voxel_data:
        if isinstance(voxel_data[k], torch.Tensor):
            voxel_data[k] = voxel_data[k].to(device, non_blocking=True).contiguous()
            # print(f"{k}: now on {voxel_data[k].device}")

    # # Debug print
    # print("=== voxel_data debug ===")
    # for k, v in voxel_data.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}, min={v.min().item()}, max={v.max().item()}")
    #     else:
    #         print(f"{k}: type={type(v)}")
    # print("Render cam.w2c.device:", cam.w2c.device)
    # print("Render cam.c2w.device:", cam.c2w.device)

    # # Move voxel data to CUDA
    # print("=== moving voxel_data to CUDA ===")
    # for k in voxel_data:
    #     if isinstance(voxel_data[k], torch.Tensor):
    #         voxel_data[k] = voxel_data[k].contiguous().to(device, non_blocking=False)
    #         # print(f"{k}: now on {voxel_data[k].device}")

    # print('[PY]  launching kernel …')
    t0 = time.time()
    try:
        out = svr.rasterize_voxels(
            rs,
            voxel_data["octpaths"],
            voxel_data["centers"],
            voxel_data["vox_lengths"].view(-1, 1),
            vox_fn
        )
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)    
        
    torch.cuda.synchronize()
    # print('[PY]  finished in %.2f s' % (time.time()-t0))

    rgb = out[0]  # (3, H, W)

    # loss = _loss_fn(rgb.unsqueeze(0), gt_image)
    # global _optimizer
    # if _optimizer is None:
    #     _optimizer = torch.optim.Adam([
    #         voxel_data["shs"].requires_grad_()
    #     ], lr=1e-2)

    # _optimizer.zero_grad()
    # loss.backward()
    # _optimizer.step()
    # ----------------------------------------------------------------
    # NOTE: Training disabled for now—pure rendering
    # If you later want to learn your SH coefficients, 
    # make sure voxel_data["shs"] has requires_grad=True *before* 
    # you call rasterize_voxels(), then uncomment the lines below.
    # ----------------------------------------------------------------
    # loss = _loss_fn(rgb.unsqueeze(0), gt_image)
    # global _optimizer
    # if _optimizer is None:
    #     # needs to be called once so shs.requires_grad=True is in the graph
    #     _optimizer = torch.optim.Adam([ voxel_data["shs"] ], lr=1e-2)
    # _optimizer.zero_grad()
    # loss.backward()
    # _optimizer.step()

    # out_img = (rgb.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # imageio.imwrite(os.path.join(output_dir, f"{cam.frame_id:06d}.png"), out_img)

    return {"rgb": rgb} 