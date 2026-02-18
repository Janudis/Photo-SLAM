#!/usr/bin/env python3
# scripts_planner/visualize.py
import os
import time
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
from scipy.spatial.transform import Rotation

import viser


# -----------------------------
# Small pose helpers
# -----------------------------
def matrix2wxyz(Rm: np.ndarray) -> np.ndarray:
    """3x3 -> wxyz quaternion"""
    return Rotation.from_matrix(Rm).as_quat()[[3, 0, 1, 2]]


def wxyz2matrix(wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3"""
    return Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()


def look_at_wxyz(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32),
) -> np.ndarray:
    """
    Right-handed look-at.

    Assumption:
      - camera +x right
      - camera +y up
      - camera +z forward
    """
    pos = position.astype(np.float32)
    tgt = target.astype(np.float32)

    z = tgt - pos
    z = z / (np.linalg.norm(z) + 1e-9)  # forward (+z)

    upv = up.astype(np.float32)
    if abs(float(np.dot(z, upv))) > 0.99:
        upv = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    x = np.cross(upv, z)
    x = x / (np.linalg.norm(x) + 1e-9)

    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)

    Rm = np.stack([x, y, z], axis=1)  # columns are axes

    if np.linalg.det(Rm) < 0:
        x = -x
        Rm = np.stack([x, y, z], axis=1)

    return matrix2wxyz(Rm)


# -----------------------------
# Small viser handle safety
# -----------------------------
def _safe_remove(h):
    """
    Different viser versions expose different handle APIs.
    We try to remove; if not available, hide.
    """
    if h is None:
        return
    try:
        h.remove()
        return
    except Exception:
        pass
    try:
        h.visible = False
    except Exception:
        pass


# -----------------------------
# SVRaster NPZ loader (full render state)
# -----------------------------
def _require_keys(npz, required, allow_aliases=None):
    keys = set(npz.files)
    missing = []
    for k in required:
        if k in keys:
            continue
        if allow_aliases and k in allow_aliases and any(a in keys for a in allow_aliases[k]):
            continue
        missing.append(k)
    return missing, keys


def load_svraster_npz(npz_path: Path):
    npz = np.load(str(npz_path), allow_pickle=True)

    required = [
        "scene_center", "scene_extent", "inside_extent",
        "octpath", "octlevel",
        "geo_grid_pts", "sh0", "shs", "subdiv_p",
        "active_sh_degree", "max_sh_degree",
    ]
    aliases = {
        "geo_grid_pts": ["_geo_grid_pts"],
        "sh0": ["_sh0"],
        "shs": ["_shs"],
        "subdiv_p": ["_subdiv_p"],
    }

    missing, keys = _require_keys(npz, required, allow_aliases=aliases)
    if missing:
        print("\n[ERROR] NPZ does not contain the SVRaster state needed to render.")
        print("Missing keys:", missing)
        print("Found keys:", sorted(list(keys)))
        raise SystemExit(2)

    def get(k):
        if k in npz.files:
            return npz[k]
        if k in aliases:
            for a in aliases[k]:
                if a in npz.files:
                    return npz[a]
        raise KeyError(k)

    data = {
        "scene_center": get("scene_center"),
        "scene_extent": get("scene_extent"),
        "inside_extent": get("inside_extent"),
        "octpath": get("octpath"),
        "octlevel": get("octlevel"),
        "geo_grid_pts": get("geo_grid_pts"),
        "sh0": get("sh0"),
        "shs": get("shs"),
        "subdiv_p": get("subdiv_p"),
        "active_sh_degree": int(np.array(get("active_sh_degree")).reshape(-1)[0]),
        "max_sh_degree": int(np.array(get("max_sh_degree")).reshape(-1)[0]),
    }
    return data


def load_planner_voxels_if_present(npz_path: Path):
    """
    Optional planner caches from the same NPZ:
      vox_center [N,3]
      vox_half_size [N] or [N,1] or [N,3]
    """
    z = np.load(str(npz_path), allow_pickle=True)
    if "vox_center" not in z.files or "vox_half_size" not in z.files:
        return None

    centers = np.asarray(z["vox_center"], dtype=np.float32)
    half = np.asarray(z["vox_half_size"], dtype=np.float32)

    if half.ndim == 2 and half.shape[1] == 1:
        half = half.reshape(-1)
    elif half.ndim == 2 and half.shape[1] == 3:
        half = half.max(axis=1)
    elif half.ndim != 1:
        print(f"[vox] unsupported vox_half_size shape {half.shape}, skipping voxel overlay.")
        return None

    if centers.ndim != 2 or centers.shape[1] != 3 or half.shape[0] != centers.shape[0]:
        print("[vox] invalid vox_center/vox_half_size shapes, skipping voxel overlay.")
        return None

    half = np.maximum(half.astype(np.float32), 1e-6)
    return centers, half


def torchify(arr: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)


def inject_into_sparse_voxel_model(voxel_model, data, device: torch.device):
    voxel_model.scene_center = torchify(data["scene_center"], device, torch.float32).view(3)
    voxel_model.scene_extent = torchify(data["scene_extent"], device, torch.float32).view(1)
    voxel_model.inside_extent = torchify(data["inside_extent"], device, torch.float32).view(1)

    op = torchify(data["octpath"], device, torch.int64)
    if op.ndim == 1:
        op = op.view(-1, 1)
    voxel_model.octpath = op.contiguous()

    lv = torchify(data["octlevel"], device, torch.int8)
    if lv.ndim == 1:
        lv = lv.view(-1, 1)
    voxel_model.octlevel = lv.contiguous()

    geo = torchify(data["geo_grid_pts"], device, torch.float32)
    if geo.ndim == 1:
        geo = geo.view(-1, 1)
    voxel_model._geo_grid_pts = geo.contiguous()

    sh0 = torchify(data["sh0"], device, torch.float32)
    if sh0.ndim == 3 and sh0.shape[1] == 1 and sh0.shape[2] == 3:
        sh0 = sh0.reshape(-1, 3)
    voxel_model._sh0 = sh0.contiguous()

    shs = torchify(data["shs"], device, torch.float32)
    voxel_model._shs = shs.contiguous()

    sp = torchify(data["subdiv_p"], device, torch.float32)
    if sp.ndim == 1:
        sp = sp.view(-1, 1)
    voxel_model._subdiv_p = sp.contiguous()

    try:
        voxel_model.max_sh_degree = int(data["max_sh_degree"])
    except Exception:
        pass
    try:
        voxel_model.active_sh_degree = int(data["active_sh_degree"])
    except Exception:
        pass

    if hasattr(voxel_model, "freeze_vox_geo"):
        voxel_model.freeze_vox_geo()


# -----------------------------
# Trajectory overlay helpers
# -----------------------------
def load_trajs_json(traj_path: Path):
    with open(traj_path, "r") as f:
        meta = json.load(f)
    datas = meta.get("total_data", [])
    return meta, datas


def colors_along_path(n_segments: int):
    """Return (n_segments,1,3) float32 colors in a jet-ish ramp."""
    if n_segments <= 0:
        return np.zeros((0, 1, 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_segments, dtype=np.float32)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    # cols = np.stack([r, g, b], axis=-1)
    # return cols.reshape(-1, 1, 3).astype(np.float32)
    cols = np.stack([r, g, b], axis=-1).astype(np.float32)   # (N,3)
    cols = np.repeat(cols[:, None, :], 2, axis=1)            # (N,2,3)
    return cols


def compute_traj_to_svr_transform(traj_meta: dict, scene_center: np.ndarray, inside_extent: float):
    """
    Similarity map: x_svr = c_svr + s * (x_traj - c_traj)
    Match planner AABB (lower/upper_bound) to SVRaster inside box.
    """
    if "lower_bound" not in traj_meta or "upper_bound" not in traj_meta:
        print("[traj-align] meta missing lower_bound/upper_bound -> NO alignment.")
        return None

    lb = np.asarray(traj_meta["lower_bound"], dtype=np.float32).reshape(3)
    ub = np.asarray(traj_meta["upper_bound"], dtype=np.float32).reshape(3)

    c_traj = 0.5 * (lb + ub)
    ext = (ub - lb)  # full extents
    e_traj = 0.5 * float(np.max(ext))  # half-extent (max axis)

    c_svr = np.asarray(scene_center, dtype=np.float32).reshape(3)
    e_svr = float(inside_extent)  # half-extent proxy

    print("\n[traj-align] -------- alignment meta --------")
    print(f"[traj-align] traj lower_bound={lb}, upper_bound={ub}")
    print(f"[traj-align] traj center c_traj={c_traj}")
    print(f"[traj-align] traj extents full={ext}, half(max) e_traj={e_traj:.6g}")
    print(f"[traj-align] svr scene_center c_svr={c_svr}")
    print(f"[traj-align] svr inside_extent e_svr={e_svr:.6g}")

    if not np.isfinite(e_traj) or e_traj < 1e-9:
        print("[traj-align] invalid e_traj -> NO alignment.")
        return None
    if not np.isfinite(e_svr) or e_svr < 1e-9:
        print("[traj-align] invalid e_svr -> NO alignment.")
        return None

    s = e_svr / e_traj
    print(f"[traj-align] scale s = e_svr/e_traj = {s:.6g}")

    def xf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return c_svr + s * (x - c_traj)

    return xf


def build_all_trajectory_handles(server, traj_datas, *, line_width, xform=None):
    """
    Create ALL /traj_i objects once.
    Later, we only toggle `handle.visible` (no delete/recreate loops).
    """
    handles = []
    for i, d in enumerate(traj_datas):
        traj = np.asarray(
            d.get("traj", d.get("traj_full", d.get("traj_quick", []))),
            dtype=np.float32
        )
        if traj.ndim != 2 or traj.shape[0] < 2:
            handles.append(None)
            continue

        pts3 = traj[:, :3]
        if xform is not None:
            pts3 = xform(pts3)

        segs = np.stack([pts3[:-1], pts3[1:]], axis=1)
        cols = colors_along_path(len(segs))

        h = server.scene.add_line_segments(
            name=f"/traj_{i}",
            points=segs,
            colors=cols,
            line_width=int(line_width),
        )
        handles.append(h)
    return handles


# -----------------------------
# Voxel overlay as AABB wireframe edges (world geometry)
# -----------------------------
def voxel_aabb_edges(centers: np.ndarray, half: np.ndarray):
    """
    Build line segments for voxel AABB edges.
    Returns points shape [M,2,3].
    """
    c = centers
    h = half.reshape(-1, 1).astype(np.float32)

    offsets = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float32,
    )
    corners = c[:, None, :] + offsets[None, :, :] * h[:, None, :]  # [N,8,3]

    edge_idx = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )

    segs = corners[:, edge_idx, :]
    segs = segs.reshape(-1, 2, 3).astype(np.float32)
    return segs


# -----------------------------
# Viewer
# -----------------------------
class SVRasterNPZViewer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

        svr_root = Path(args.svraster_root).expanduser().resolve()
        if not svr_root.exists():
            raise RuntimeError(f"--svraster_root does not exist: {svr_root}")

        import sys
        sys.path.insert(0, str(svr_root))  # enables `import src.*`

        from src.sparse_voxel_model import SparseVoxelModel
        from src.cameras import MiniCam
        from src.utils.image_utils import im_tensor2np, viz_tensordepth

        self.MiniCam = MiniCam
        self.im_tensor2np = im_tensor2np
        self.viz_tensordepth = viz_tensordepth

        self.npz_path = Path(args.npz).expanduser().resolve()
        data = load_svraster_npz(self.npz_path)

        self.voxel_model = SparseVoxelModel(
            n_samp_per_vox=args.n_samp_per_vox,
            sh_degree=int(data["max_sh_degree"]),
            ss=args.ss,
            white_background=args.white_background,
            black_background=args.black_background,
        )
        inject_into_sparse_voxel_model(self.voxel_model, data, self.device)

        self.server = viser.ViserServer(port=args.port)
        self.server.gui.set_panel_label("SVRaster viewer + trajectory overlay")

        # ------------- GUI -------------
        self.fps = self.server.gui.add_text("Rendering FPS", initial_value="-1", disabled=True)

        self.active_sh_degree_slider = self.server.gui.add_slider(
            "active_sh_degree",
            min=0,
            max=int(data["max_sh_degree"]),
            step=1,
            initial_value=int(data["active_sh_degree"]),
        )
        self.ss_slider = self.server.gui.add_slider("ss", min=0.5, max=2.0, step=0.05, initial_value=float(args.ss))
        self.width_slider = self.server.gui.add_slider("width", min=128, max=2048, step=8, initial_value=int(args.width))
        self.fovx_slider = self.server.gui.add_slider("fovx_deg", min=10, max=150, step=1, initial_value=int(args.fovx))
        self.near_slider = self.server.gui.add_slider("near", min=0.02, max=10, step=0.01, initial_value=float(args.near))

        self.render_dropdown = self.server.gui.add_dropdown(
            "render mode",
            options=["all", "rgb only", "depth only", "normal only"],
            initial_value="all",
        )
        self.output_dropdown = self.server.gui.add_dropdown(
            "output",
            options=["rgb", "alpha", "dmean", "dmed", "dmean2n", "dmed2n", "n"],
            initial_value="rgb",
        )
        self.show_svraster_checkbox = self.server.gui.add_checkbox(
            "show SVRaster render (background)",
            initial_value=bool(args.show_svraster),
        )

        # overlays
        self.show_traj_checkbox = self.server.gui.add_checkbox(
            "show trajectory overlay",
            initial_value=bool(args.traj is not None),
        )
        self.traj_all_checkbox = self.server.gui.add_checkbox(
            "traj: show ALL",
            initial_value=bool(args.traj_all),
        )
        self.traj_k_slider = self.server.gui.add_slider(
            "traj: show first K",
            min=0,
            max=0,
            step=1,
            initial_value=0,
            disabled=True,
        )
        self.traj_mode = self.server.gui.add_dropdown(
            "traj mode",
            options=["first K", "random K (seed)"],
            initial_value="first K",
        )
        self.traj_seed = self.server.gui.add_slider(
            "traj seed",
            min=0,
            max=9999,
            step=1,
            initial_value=int(args.traj_seed),
        )
        self.traj_index_slider = self.server.gui.add_slider(
            "traj index (single)",
            min=0,
            max=0,
            step=1,
            initial_value=int(args.traj_index),
            disabled=True,
        )
        self.traj_single_checkbox = self.server.gui.add_checkbox(
            "traj: show SINGLE index",
            initial_value=bool(args.traj_single),
        )
        self.traj_line_width = self.server.gui.add_slider(
            "traj line width",
            min=1, max=30, step=1,
            initial_value=int(args.traj_line_width),
        )

        self.show_vox_checkbox = self.server.gui.add_checkbox(
            "show voxel AABB wire overlay",
            initial_value=bool(args.show_voxels_wire),
        )
        self.vox_max_slider = self.server.gui.add_slider(
            "max voxel boxes",
            min=100, max=20000, step=100,
            initial_value=int(args.max_voxel_boxes),
        )
        self.vox_line_width = self.server.gui.add_slider(
            "voxel wire line width",
            min=1, max=10, step=1,
            initial_value=int(args.voxel_wire_width),
        )

        self.refresh_button = self.server.gui.add_button("Refresh overlays")
        self.download_button = self.server.gui.add_button("Download view")

        # ------------- world axes -------------
        self.server.scene.add_frame(
            name="/world",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([0.0, 0.0, 0.0]),
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.01,
        )

        # ------------- camera init -------------
        sc = self.voxel_model.scene_center.detach().cpu().numpy().reshape(3)
        se = float(self.voxel_model.scene_extent.detach().cpu().numpy().reshape(-1)[0])
        max_lv = int(self.voxel_model.octlevel.max().cpu())
        min_vox = se / (2.0 ** max_lv)
        cam_dist = max(0.5, 200.0 * float(min_vox))
        init_pos = sc + np.array([0.0, 0.0, cam_dist], dtype=np.float32)
        init_wxyz = look_at_wxyz(init_pos, sc)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            with client.atomic():
                client.camera.position = init_pos
                client.camera.wxyz = init_wxyz

        @self.download_button.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            im, _ = self.render_viser_camera(client.camera)
            client.send_file_download(
                "svraster_view.png",
                iio.imwrite("<bytes>", im, extension=".png"),
            )

        # ------------- overlays state -------------
        self.traj_meta = None
        self.traj_datas = []
        self.traj_handles = []  # list of line-seg handles
        self.traj_xform = None

        self._vox_handle = None

        # ------------- load trajs if provided -------------
        if args.traj is not None:
            self.traj_meta, self.traj_datas = load_trajs_json(Path(args.traj).expanduser().resolve())
            if len(self.traj_datas) > 0:
                self.traj_k_slider.max = len(self.traj_datas)
                self.traj_k_slider.value = min(int(args.traj_k), len(self.traj_datas))
                self.traj_k_slider.disabled = False

                self.traj_index_slider.max = len(self.traj_datas) - 1
                self.traj_index_slider.value = min(int(args.traj_index), len(self.traj_datas) - 1)
                self.traj_index_slider.disabled = False

            sc_np = self.voxel_model.scene_center.detach().cpu().numpy().reshape(3)
            ie = float(self.voxel_model.inside_extent.detach().cpu().numpy().reshape(-1)[0])
            # self.traj_xform = compute_traj_to_svr_transform(self.traj_meta, sc_np, ie)
            self.traj_xform = None

            print(f"[traj] creating {len(self.traj_datas)} trajectory objects...")
            self.traj_handles = build_all_trajectory_handles(
                self.server,
                self.traj_datas,
                line_width=int(args.traj_line_width),
                xform=self.traj_xform,
            )
            print("[traj] done creating trajectory objects.")

            # bbox debug
            try:
                all_pts = []
                for d in self.traj_datas:
                    t = np.asarray(d.get("traj", []), dtype=np.float32)
                    if t.ndim == 2 and t.shape[0] > 0:
                        all_pts.append(t[:, :3])
                if len(all_pts) > 0:
                    all_pts = np.concatenate(all_pts, axis=0)
                    raw_min, raw_max = all_pts.min(0), all_pts.max(0)
                    print("\n[traj-align] raw traj pts bbox:")
                    print("[traj-align]   min =", raw_min)
                    print("[traj-align]   max =", raw_max)

                    if self.traj_xform is not None:
                        all_pts2 = self.traj_xform(all_pts)
                        tr_min, tr_max = all_pts2.min(0), all_pts2.max(0)
                        print("[traj-align] transformed traj pts bbox:")
                        print("[traj-align]   min =", tr_min)
                        print("[traj-align]   max =", tr_max)
                        print("[traj-align] transformed bbox center =", 0.5 * (tr_min + tr_max))
                        print("[traj-align] scene_center             =", sc_np.reshape(3))
                        print("[traj-align] transformed bbox half(max) =", 0.5 * float(np.max(tr_max - tr_min)))
                        print("[traj-align] inside_extent              =", ie)
            except Exception as e:
                print("[traj-align] bbox debug failed:", repr(e))

        # ------------- wire up UI callbacks (NO recursion loops) -------------
        def _update_traj_width(_evt=None):
            # Viser doesn't reliably allow changing line_width in-place across versions.
            # Minimal change: keep geometry, only adjust visibility; width is applied at creation-time.
            # If you want width live-edit, we rebuild trajectories explicitly:
            self.rebuild_trajectories_with_width(int(self.traj_line_width.value))

        self.refresh_button.on_click(lambda _evt: self.rebuild_overlays())
        self.show_traj_checkbox.on_update(lambda _evt: self.apply_traj_visibility())
        self.traj_all_checkbox.on_update(lambda _evt: self.apply_traj_visibility())
        self.traj_k_slider.on_update(lambda _evt: self.apply_traj_visibility())
        self.traj_mode.on_update(lambda _evt: self.apply_traj_visibility())
        self.traj_seed.on_update(lambda _evt: self.apply_traj_visibility())
        self.traj_single_checkbox.on_update(lambda _evt: self.apply_traj_visibility())
        self.traj_index_slider.on_update(lambda _evt: self.apply_traj_visibility())
        self.traj_line_width.on_update(_update_traj_width)

        self.show_vox_checkbox.on_update(lambda _evt: self.rebuild_vox_overlay())
        self.vox_max_slider.on_update(lambda _evt: self.rebuild_vox_overlay())
        self.vox_line_width.on_update(lambda _evt: self.rebuild_vox_overlay())

        # initial overlays
        self.rebuild_overlays()

    # -----------------------------
    # overlay management
    # -----------------------------
    def rebuild_overlays(self):
        self.apply_traj_visibility()
        self.rebuild_vox_overlay()

    def apply_traj_visibility(self):
        if len(self.traj_handles) == 0:
            return

        master = bool(self.show_traj_checkbox.value)
        if not master:
            for h in self.traj_handles:
                if h is not None:
                    h.visible = False
            print("[traj] overlay disabled")
            return

        N = len(self.traj_handles)

        # SINGLE takes priority (matches "traj_index" semantics)
        if bool(self.traj_single_checkbox.value):
            idx0 = int(self.traj_index_slider.value)
            idx0 = max(0, min(idx0, N - 1))
            shown = 0
            for i, h in enumerate(self.traj_handles):
                if h is None:
                    continue
                vis = (i == idx0)
                h.visible = vis
                shown += int(vis)
            print(f"[traj] showing SINGLE trajectory idx={idx0} (1/{N})")
            return

        # ALL takes priority over K
        if bool(self.traj_all_checkbox.value):
            shown = 0
            for h in self.traj_handles:
                if h is None:
                    continue
                h.visible = True
                shown += 1
            print(f"[traj] showing ALL trajectories ({shown}/{N})")
            return

        # otherwise: show K
        K = int(self.traj_k_slider.value)
        K = max(0, min(K, N))

        if self.traj_mode.value == "random K (seed)":
            rng = np.random.default_rng(int(self.traj_seed.value))
            sel = rng.choice(N, size=K, replace=False).tolist() if K > 0 else []
            sel = set(sel)
        else:
            sel = set(range(K))

        shown = 0
        for i, h in enumerate(self.traj_handles):
            if h is None:
                continue
            vis = (i in sel)
            h.visible = vis
            shown += int(vis)

        print(f"[traj] showing {shown}/{N} (K={K}, mode={self.traj_mode.value})")

    def rebuild_trajectories_with_width(self, new_width: int):
        if self.args.traj is None or len(self.traj_datas) == 0:
            return

        # Remove old
        for h in self.traj_handles:
            _safe_remove(h)
        self.traj_handles = []

        # Recreate with same names (/traj_i). This matches SplatNav’s “create once then toggle”,
        # but we only rebuild when width changes (minimal + stable).
        self.traj_handles = build_all_trajectory_handles(
            self.server,
            self.traj_datas,
            line_width=int(new_width),
            xform=self.traj_xform,
        )
        self.apply_traj_visibility()

    def rebuild_vox_overlay(self):
        # delete previous
        if self._vox_handle is not None:
            _safe_remove(self._vox_handle)
            self._vox_handle = None

        if not bool(self.show_vox_checkbox.value):
            print("[vox] voxel wire overlay disabled")
            return

        out = load_planner_voxels_if_present(self.npz_path)
        if out is None:
            print("[vox] no vox_center/vox_half_size in NPZ, skipping voxel wire overlay.")
            return

        centers, half = out
        N = centers.shape[0]
        max_vox = int(self.vox_max_slider.value)
        if max_vox is not None and N > max_vox:
            rng = np.random.default_rng(0)
            idx = rng.choice(N, size=max_vox, replace=False)
            centers = centers[idx]
            half = half[idx]
            N = centers.shape[0]

        segs = voxel_aabb_edges(centers, half)

        hn = (half - half.min()) / (half.max() - half.min() + 1e-9) if half.size > 0 else half
        base = np.stack([hn, 0.2 * np.ones_like(hn), 1.0 - hn], axis=-1).astype(np.float32)
        # colors = np.repeat(base[:, None, :], 12, axis=1).reshape(-1, 1, 3)
        # base: (N,3)
        edge_cols = np.repeat(base[:, None, :], 12, axis=1)   # (N,12,3)
        edge_cols = edge_cols.reshape(-1, 3).astype(np.float32)  # (N*12,3)
        # expand to (Nsegs,2,3)
        colors = np.repeat(edge_cols[:, None, :], 2, axis=1)  # (N*12,2,3)


        self._vox_handle = self.server.scene.add_line_segments(
            name="/voxels_wire",
            points=segs,
            colors=colors,
            line_width=int(self.vox_line_width.value),
        )
        print(f"[vox] added voxel AABB wire overlay: vox={N}, segs={segs.shape[0]}")

    # -----------------------------
    # rendering
    # -----------------------------
    @torch.no_grad()
    def render_viser_camera(self, camera: viser.CameraHandle):
        width = int(self.width_slider.value)
        height = int(round(width / camera.aspect))
        fovx_deg = float(self.fovx_slider.value)
        fovy_deg = fovx_deg * float(height) / float(width)
        near = float(self.near_slider.value)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = wxyz2matrix(np.array(camera.wxyz, dtype=np.float32))
        c2w[:3, 3] = np.array(camera.position, dtype=np.float32)

        minicam = self.MiniCam(
            c2w,
            fovx=np.deg2rad(fovx_deg),
            fovy=np.deg2rad(fovy_deg),
            width=width,
            height=height,
            near=near,
        )

        try:
            self.voxel_model.active_sh_degree = int(self.active_sh_degree_slider.value)
        except Exception:
            pass

        render_opt = {
            "ss": float(self.ss_slider.value),
            "output_T": True,
            "output_depth": True,
            "output_normal": True,
        }

        if self.render_dropdown.value == "rgb only":
            render_opt["output_depth"] = False
            render_opt["output_normal"] = False
        elif self.render_dropdown.value == "depth only":
            render_opt["color_mode"] = "dontcare"
            render_opt["output_normal"] = False
        elif self.render_dropdown.value == "normal only":
            render_opt["color_mode"] = "dontcare"
            render_opt["output_depth"] = False

        t0 = time.time()
        render_pkg = self.voxel_model.render(minicam, **render_opt)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        eps = time.time() - t0

        out = self.output_dropdown.value
        if out == "dmean":
            im = self.viz_tensordepth(render_pkg["depth"][0])
        elif out == "dmed":
            im = self.viz_tensordepth(render_pkg["depth"][2])
        elif out == "dmean2n":
            depth2normal = minicam.depth2normal(render_pkg["depth"][0])
            im = self.im_tensor2np(depth2normal * 0.5 + 0.5)
        elif out == "dmed2n":
            depth_med2normal = minicam.depth2normal(render_pkg["depth"][2])
            im = self.im_tensor2np(depth_med2normal * 0.5 + 0.5)
        elif out == "n":
            im = self.im_tensor2np(render_pkg["normal"] * 0.5 + 0.5)
        elif out == "alpha":
            im = self.im_tensor2np(1 - render_pkg["T"].repeat(3, 1, 1))
        else:
            im = self.im_tensor2np(render_pkg["color"])

        del render_pkg
        return im, eps

    def update(self):
        clients = list(self.server.get_clients().values())
        if len(clients) == 0:
            return

        # If disabled, don't render; just keep overlays visible.
        if not bool(self.show_svraster_checkbox.value):
            self.fps.value = "off"
            return

        times = []
        for client in clients:
            im, eps = self.render_viser_camera(client.camera)
            times.append(eps)
            client.scene.set_background_image(im, format="jpeg")

        fps = 1.0 / (float(np.mean(times)) + 1e-9)
        self.fps.value = f"{int(round(fps)):4d}"


def main():
    parser = argparse.ArgumentParser("SVRaster viewer + trajectory overlay.")
    parser.add_argument("--show_svraster", action="store_true",
                    help="Render SVRaster photorealistic background. If off, only overlays are shown.")
    parser.add_argument("--npz", required=True, help="Path to exported SVRaster model npz.")
    parser.add_argument("--svraster_root", default="/home/dimitris/svraster",
                        help="Path to SVRaster repo root (contains src/).")
    parser.add_argument("--port", type=int, default=8080)

    # Rendering knobs
    parser.add_argument("--cpu", action="store_true", help="Force CPU rendering (slow).")
    parser.add_argument("--n_samp_per_vox", type=int, default=1)
    parser.add_argument("--ss", type=float, default=1.0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--black_background", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--fovx", type=int, default=70)
    parser.add_argument("--near", type=float, default=0.02)

    # Trajectory overlay (from run_rrt.py output)
    parser.add_argument("--traj", default=None, help="Path to trajs/*.json from run_rrt.py")
    parser.add_argument("--traj_all", action="store_true", help="Show ALL trajectories")
    parser.add_argument("--traj_k", type=int, default=10, help="Show first K trajectories when not --traj_all")
    parser.add_argument("--traj_index", type=int, default=0, help="Index for SINGLE trajectory mode")
    parser.add_argument("--traj_single", action="store_true", help="Show SINGLE trajectory index (traj_index)")
    parser.add_argument("--traj_line_width", type=int, default=10)
    parser.add_argument("--traj_seed", type=int, default=0)

    # Voxel overlay (wireframe AABB edges, from vox_center/vox_half_size if present)
    parser.add_argument("--show_voxels_wire", action="store_true", help="Draw voxel AABB wire overlay (subsampled)")
    parser.add_argument("--max_voxel_boxes", type=int, default=5000)
    parser.add_argument("--voxel_wire_width", type=int, default=2)

    args = parser.parse_args()

    viewer = SVRasterNPZViewer(args)
    while True:
        viewer.update()
        time.sleep(0.003)


if __name__ == "__main__":
    main()
