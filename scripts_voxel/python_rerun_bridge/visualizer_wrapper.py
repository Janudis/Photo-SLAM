# scripts_voxel/python_rerun_bridge/visualizer_wrapper.py

from typing import Dict, Deque, Optional
from collections import deque

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
import open3d as o3d
import torch
import os
import cv2

class RerunVisualizer:
    """
    Rerun visualizer for Photo-SLAM + SVRaster.

    This is intentionally very close to nvblox_torch's RerunVisualizer:
    - world/camera_0: pose, axes, image, observations
    - world/trajectory: camera trajectory
    - world/mesh: color mesh (TSDF or voxel mesh)
    """

    def __init__(self, app_id: str = "PhotoSLAM-SVRaster", spawn: bool = True) -> None:
        self._start_rerun_visualizer(app_id, spawn)
        # Parameters
        self.camera_pose_axis_scale = 0.1
        self.trajectory_length = 500
        # State
        self.track_colors: Dict[int, npt.NDArray] = {}
        self.t_W_C_history: Deque[npt.NDArray] = deque(maxlen=self.trajectory_length)

    def _start_rerun_visualizer(self, app_id: str, spawn: bool) -> None:
        rr.init(app_id, spawn=spawn)    
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.4, 0.6],
                    contents=[
                        # 2D view showing keyframe images:
                        rrb.Spatial2DView(
                            origin="world/keyframes_2d",
                            name="Keyframe images",
                        ),
                        # 3D view showing world, voxels, meshes, and cameras:
                        rrb.Spatial3DView(
                            origin="world",
                            name="Scene 3D",
                        ),
                    ],
                ),
            ),
            make_active=True,
        )

    def save_recording(self, path: str) -> None:
        """
        Save the current Rerun recording to a .rrd file.

        Typically called once from C++ at the end of the run.
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            rr.save(path)
            print(f"[RERUN] Saved recording to: {path}")
        except Exception as e:
            print(f"[RERUN] Failed to save recording to {path}: {e}")

    # ----------------------------------------------------------------------
    #  Low-level helpers (similar to nvblox RerunVisualizer)
    # ----------------------------------------------------------------------

    def _log_rig_pose(self, t_W_C: npt.NDArray, q_W_C_xyzw: npt.NDArray) -> None:
        """Log rig pose to Rerun.

        Args:
            t_W_C: (3,) translation.
            q_W_C_xyzw: (4,) quaternion (x, y, z, w).
        """
        rr.log(
            "world/camera_0",
            rr.Transform3D(
                translation=t_W_C,
                quaternion=q_W_C_xyzw,
            ),
            rr.Arrows3D(
                vectors=np.eye(3) * self.camera_pose_axis_scale,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB for XYZ axes
            ),
        )

    def _log_observations(
        self,
        points_uv: Optional[npt.NDArray],
        track_ids: Optional[npt.NDArray],
        image: npt.NDArray,
    ) -> None:
        """
        Log 2D observations for a specific camera with consistent colors per track.

        Args:
            points_uv: (N, 2) array of [u,v] pixel coords, or None.
            track_ids: (N,) int array of track IDs, or None.
            image:     H x W x C uint8 image (RGB).
        """
        if points_uv is None or len(points_uv) == 0:
            # Still log the image, just no points.
            rr.log("world/camera_0/image", rr.Image(image).compress())
            return

        points = np.asarray(points_uv, dtype=np.float32)

        colors = np.empty((points.shape[0], 3), dtype=np.uint8)
        if track_ids is not None and len(track_ids) == len(points):
            ids = np.asarray(track_ids, dtype=int)
            for i, tid in enumerate(ids):
                if tid not in self.track_colors:
                    self.track_colors[tid] = np.random.randint(0, 256, size=3, dtype=np.uint8)
                colors[i] = self.track_colors[tid]
        else:
            # No IDs: random color per point
            colors = np.random.randint(0, 256, size=(points.shape[0], 3), dtype=np.uint8)

        rr.log(
            "world/camera_0/observations",
            rr.Points2D(positions=points, colors=colors, radii=5.0),
        )
        rr.log("world/camera_0/image", rr.Image(image).compress())

    def _log_trajectory(self) -> None:
        """Log the trajectory to Rerun."""
        if len(self.t_W_C_history) == 0:
            return
        traj = np.stack(self.t_W_C_history, axis=0)  # [N,3]
        rr.log("world/trajectory", rr.LineStrips3D([traj]), static=True)

    def _log_pinhole(
        self,
        entity_path: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> None:
        """Log a pinhole camera (frustum) at the given entity."""
        rr.log(
            entity_path,
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                resolution=[width, height],
                camera_xyz=rr.ViewCoordinates.RDF,  # Camera convention: x-right, y-down, z-forward
                image_plane_distance=0.1,
                color=[255, 128, 0],
                line_width=0.003,
            ),
        )

    def _log_camera_entity(
        self,
        entity_path: str,
        t_W_C: npt.NDArray,
        q_W_C_xyzw: npt.NDArray,
        image: npt.NDArray,
        points_uv: Optional[npt.NDArray] = None,
        track_ids: Optional[npt.NDArray] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> None:
        """Log one camera pose + image into a given entity path."""

        t_W_C = np.asarray(t_W_C, dtype=np.float32).reshape(3)
        q_W_C_xyzw = np.asarray(q_W_C_xyzw, dtype=np.float32).reshape(4)
        image = np.asarray(image, dtype=np.uint8)
        H, W, _ = image.shape

        # Pose
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=t_W_C,
                quaternion=q_W_C_xyzw,
                relation=rr.TransformRelation.ParentFromChild,
            ),
        )

        # Optional 3D axes for that camera
        rr.log(
            entity_path + "/axes",
            rr.Arrows3D(
                vectors=np.eye(3) * self.camera_pose_axis_scale,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        # Pinhole camera frustum
        if fx is not None and fy is not None and cx is not None and cy is not None:
            self._log_pinhole(entity_path, fx, fy, cx, cy, W, H)

        # # 2D observations + image
        # if points_uv is None or len(points_uv) == 0:
        #     rr.log(entity_path + "/image", rr.Image(image).compress())
        # else:
        #     pts = np.asarray(points_uv, dtype=np.float32)
        #     colors = np.random.randint(0, 256, size=(pts.shape[0], 3), dtype=np.uint8)
        #     rr.log(
        #         entity_path + "/observations",
        #         rr.Points2D(positions=pts, colors=colors, radii=5.0),
        #     )
        #     rr.log(entity_path + "/image", rr.Image(image).compress())

    def visualize_cuvslam(
        self,
        t_W_C: npt.NDArray,
        q_W_C_xyzw: npt.NDArray,
        image: npt.NDArray,
        points_uv: Optional[npt.NDArray] = None,
        track_ids: Optional[npt.NDArray] = None,
        iteration: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> None:
        """
        Visualize:
        1) Camera pose (one entity per iteration/keyframe),
        2) Tracking image and 2D features,
        3) Global trajectory.
        """
        if iteration is not None:
            rr.set_time_sequence("iter", int(iteration))
            kf_name = f"kf_{int(iteration):06d}"
            cam_entity = f"world/keyframes/{kf_name}"
            img2d_entity = f"world/keyframes_2d/{kf_name}"
        else:
            cam_entity = "world/camera_0"
            img2d_entity = "world/keyframes_2d/cam_0"

        # Log this keyframe as its own camera
        self._log_camera_entity(
            cam_entity,
            t_W_C,
            q_W_C_xyzw,
            image,
            points_uv,
            track_ids,
            fx, fy, cx, cy,
        )

        # 2) Pure 2D image copy (NO Transform3D parent) for “straight” keyframe view
        image = np.asarray(image, dtype=np.uint8)
        rr.log(img2d_entity + "/image", rr.Image(image).compress())

        # For trajectory we just keep the translation history
        t_W_C = np.asarray(t_W_C, dtype=np.float32).reshape(3)
        self.t_W_C_history.append(t_W_C)
        self._log_trajectory()

    def visualize_nvblox_ply(self, ply_path: str, iteration: int):
        """
        Called from C++:
            impl_->visualizer.attr("visualize_nvblox_ply")(py::str(ply_path), iteration)
        Loads the NVBlox color mesh PLY and logs it to Rerun.
        """
        # print("[RERUN] visualize_nvblox_ply called")
        # print("         ply_path:", ply_path)
        # print("         iteration:", iteration)

        if not os.path.exists(ply_path):
            print("[RERUN] visualize_nvblox_ply: file does not exist")
            return

        # 1) Load mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if mesh.is_empty():
            print("[RERUN] visualize_nvblox_ply: loaded mesh is EMPTY")
            return

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces    = np.asarray(mesh.triangles, dtype=np.int32)

        v_colors = np.asarray(mesh.vertex_colors)
        if v_colors.size == 0:
            colors = None
            print("[RERUN] visualize_nvblox_ply: no vertex colors in mesh")
        else:
            if v_colors.shape[0] != vertices.shape[0]:
                print(
                    "[RERUN] visualize_nvblox_ply: colors/vertices mismatch, "
                    f"vertices={vertices.shape[0]}, colors={v_colors.shape[0]} – ignoring colors"
                )
                colors = None
            else:
                colors = v_colors.astype(np.float32, copy=False)

        # 2) Time
        rr.set_time_sequence("iter", int(iteration))

        # 3) Make it a child of "world" so it is visible in the Spatial3DView(origin="world")
        entity_path = f"world/tsdf_mesh/iter_{iteration:05d}"

        # print("[RERUN] loaded mesh:")
        # print("         vertices shape:", vertices.shape)
        # print("         faces    shape:", faces.shape)
        # if colors is not None:
        #     print("         colors   shape:", colors.shape)
        #     print("         colors   dtype:", colors.dtype)
        #     print("         colors   min:", colors.min(), "max:", colors.max())
        # else:
        #     print("         colors   = None")

        # 4) Log
        self._visualize_mesh(vertices, faces, colors, entity_path=entity_path)

    def _visualize_mesh(
        self,
        vertices,
        faces,
        colors=None,
        entity_path: str = "tsdf_mesh",
    ):
        """
        Visualize a triangular mesh in Rerun.

        Called as: self._visualize_mesh(vertices, faces, colors, entity_path=...)
        """

        # Ensure CPU numpy arrays:
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        else:
            vertices = np.asarray(vertices)

        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        else:
            faces = np.asarray(faces)

        if colors is not None:
            if isinstance(colors, torch.Tensor):
                colors = colors.detach().cpu().numpy()
            else:
                colors = np.asarray(colors)

            # Normalize/clamp to [0,1] then convert to uint8 [0,255]
            if colors.dtype != np.uint8:
                colors = np.clip(colors, 0.0, 1.0)
                colors = (colors * 255.0).astype(np.uint8)

        # Cast to types Rerun likes
        vertices = vertices.astype(np.float32, copy=False)
        faces    = faces.astype(np.int32,   copy=False)

        if vertices.size == 0 or faces.size == 0:
            print("[RERUN] _visualize_mesh: empty vertices/faces -> nothing to log")
            return

        # Build Rerun Mesh3D
        if colors is not None and colors.size > 0:
            mesh = rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=colors,
            )
        else:
            mesh = rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
            )

        # print("[RERUN] logging Mesh3D to", entity_path)
        rr.log(entity_path, mesh)

    def visualize_nvblox(
        self,
        vertices: npt.NDArray,
        colors: Optional[npt.NDArray],
        triangles: npt.NDArray,
        iteration: Optional[int] = None,
    ) -> None:
        """
        Visualize a mesh (e.g. from nvblox TSDF or a voxel mesh exported from SVRaster).

        Args:
            vertices:  (N,3) float32
            colors:    (N,3) uint8 or float32, or None
            triangles: (M,3) int32/int64
            iteration: optional training / SLAM iteration used as 'iter' timeline.
        """
        if iteration is not None:
            rr.set_time_sequence("iter", int(iteration))

        v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        f = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)

        c = None
        if colors is not None:
            c = np.asarray(colors)
            if c.shape[0] == v.shape[0]:
                # Rerun accepts uint8 [0,255] or float [0,1].
                if c.dtype != np.uint8:
                    c = np.clip(c, 0.0, 1.0)
                    c = (c * 255.0).astype(np.uint8)
            else:
                c = None

        self._visualize_mesh(v, f, c, entity_path="world/tsdf_mesh/live")

    def visualize_camera(
        self,
        T_W_C: npt.NDArray,
        image: npt.NDArray,
        points_uv: Optional[npt.NDArray] = None,
        track_ids: Optional[npt.NDArray] = None,
    ) -> None:
        """
        Simple camera visualization entry-point called from C++:
        - T_W_C: 4x4 float32 world-from-camera matrix.
        - image: HxWx3 uint8 RGB image.
        """
        T_W_C = np.asarray(T_W_C, dtype=np.float32).reshape(4, 4)
        image = np.asarray(image, dtype=np.uint8)

        t_W_C = T_W_C[:3, 3]
        R_W_C = T_W_C[:3, :3]

        # Pose (we use mat3x3 instead of quaternion for simplicity).
        rr.log(
            "world/camera_0",
            rr.Transform3D(
                translation=t_W_C,
                mat3x3=R_W_C,
            ),
        )

        # Log image (and optionally 2D points).
        self._log_observations(points_uv, track_ids, image)

        # Trajectory history.
        self.t_W_C_history.append(t_W_C)
        self._log_trajectory()

    def visualize_voxels_boxes(
        self,
        centers: npt.NDArray,
        half_sizes: npt.NDArray,
        colors: Optional[npt.NDArray] = None,
        entity_path: str = "world/voxels",
        max_boxes: int = 1000000,
        iteration: Optional[int] = None,
    ) -> None:
        """
        Visualize voxel grid as 3D boxes.

        Args:
            centers:    (N,3) float32 centers.
            half_sizes: (N,3) float32 half-sizes (already half, from C++).
            colors:     (N,3) uint8 or float32 in [0,1], or None.
        """
        if iteration is not None:
            rr.set_time_sequence("iter", int(iteration))

        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        half_sizes = np.asarray(half_sizes, dtype=np.float32)
        if half_sizes.ndim == 1:
            half_sizes = half_sizes[:, None].repeat(3, axis=1)
        else:
            half_sizes = half_sizes.reshape(-1, 3)

        N = centers.shape[0]
        if N == 0:
            print("[PY][voxels] empty centers, nothing to log")
            return

        if colors is not None:
            colors = np.asarray(colors)
            # Accept float [0,1] or uint8 [0,255], shape [N,3] or [N,4]
            if colors.dtype != np.uint8:
                colors = np.clip(colors, 0.0, 1.0)
                colors = (colors * 255.0).astype(np.uint8)
            colors = colors.reshape(N, -1)
            if colors.shape[1] not in (3, 4):
                print("[PY][voxels] unexpected color shape, ignoring colors")
                colors = None
            elif colors.shape[0] != N:
                print("[PY][voxels] colors length mismatch, ignoring colors")
                colors = None

        # Downsample if too many
        # print(f"[PY][voxels] visualizing {N} boxes")
        if N > max_boxes:
            print(f"[PY][voxels] too many boxes ({N}), downsampling to {max_boxes}")
            idx = np.linspace(0, N - 1, max_boxes, dtype=np.int64)
            centers = centers[idx]
            half_sizes = half_sizes[idx]
            if colors is not None:
                colors = colors[idx]
            N = max_boxes

        # print(
        #     f"[PY][voxels] logging {N} boxes to '{entity_path}' "
        #     f"center_min={centers.min(axis=0)}, center_max={centers.max(axis=0)}, "
        #     f"hs_min={half_sizes.min()}, hs_max={half_sizes.max()}"
        # )

        rr.log(
            entity_path,
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                colors=colors,
                fill_mode="solid",
            ),
        )

    def visualize_voxels_mesh(
        self,
        centers: npt.NDArray,
        half_sizes: npt.NDArray,
        colors: Optional[npt.NDArray] = None,
        max_voxels: int = 500000,
        iteration: Optional[int] = None,
    ) -> None:
        """
        Turn SVRaster voxels into a triangle mesh (cubes) and log as Mesh3D.
        """

        if iteration is not None:
            rr.set_time_sequence("iter", int(iteration))

        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        half_sizes = np.asarray(half_sizes, dtype=np.float32)
        if half_sizes.ndim == 1:
            half_sizes = half_sizes[:, None].repeat(3, axis=1)
        else:
            half_sizes = half_sizes.reshape(-1, 3)

        N = centers.shape[0]
        if N == 0:
            print("[PY][voxels_mesh] no voxels to visualize")
            return

        # Downsample to keep mesh size reasonable
        if N > max_voxels:
            idx = np.linspace(0, N - 1, max_voxels, dtype=np.int64)
            centers    = centers[idx]
            half_sizes = half_sizes[idx]
            if colors is not None:
                colors = np.asarray(colors)[idx]
            N = centers.shape[0]

        # Prepare per-voxel color → per-vertex color
        if colors is not None:
            c = np.asarray(colors)
            if c.dtype != np.uint8:
                c = np.clip(c, 0.0, 1.0)
                c = (c * 255.0).astype(np.uint8)
            if c.shape[0] != N:
                print("[PY][voxels_mesh] color length mismatch, ignoring colors")
                c = None
        else:
            c = None

        # Offsets of the 8 corners of a unit cube centered at 0
        base_offsets = np.array([
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1],
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
        ], dtype=np.float32) * 0.5  # will be scaled per-voxel

        # 12 triangles (indices into the 8 corners)
        base_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [1, 2, 6], [1, 6, 5],  # right
            [3, 0, 4], [3, 4, 7],  # left
        ], dtype=np.int32)

        all_vertices = []
        all_faces    = []
        all_vcolors  = [] if c is not None else None

        for i in range(N):
            center = centers[i]
            hs     = half_sizes[i]  # (3,)

            # Scale the base cube to this voxel
            voxel_offsets = base_offsets * (2.0 * hs[None, :])
            verts_i = center[None, :] + voxel_offsets  # (8,3)
            all_vertices.append(verts_i)

            faces_i = base_faces + 8 * i
            all_faces.append(faces_i)

            if all_vcolors is not None:
                color_i = c[i]
                all_vcolors.append(
                    np.tile(color_i[None, :], (8, 1))
                )

        vertices = np.concatenate(all_vertices, axis=0)
        faces    = np.concatenate(all_faces, axis=0)
        vcolors  = None
        if all_vcolors is not None and len(all_vcolors) > 0:
            vcolors = np.concatenate(all_vcolors, axis=0)

        self._visualize_mesh(
            vertices,
            faces,
            vcolors,
            entity_path="world/svraster_mesh",
        )

    #path planning   
    def visualize_points3d(
        self,
        points_xyz: npt.NDArray,
        colors: Optional[npt.NDArray] = None,
        radii: float = 0.02,
        entity_path: str = "world/points",
        iteration: Optional[int] = None,
    ) -> None:
        if iteration is not None:
            rr.set_time_sequence("iter", int(iteration))

        pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return

        col = None
        if colors is not None:
            col = np.asarray(colors)
            if col.dtype != np.uint8:
                col = np.clip(col, 0.0, 1.0)
                col = (col * 255.0).astype(np.uint8)
            col = col.reshape(pts.shape[0], -1)
            if col.shape[1] not in (3, 4):
                col = None

        rr.log(entity_path, rr.Points3D(pts, colors=col, radii=radii))

    def visualize_linestrip3d(
        self,
        points_xyz: npt.NDArray,
        color: Optional[npt.NDArray] = None,
        radius: float = 0.01,
        entity_path: str = "world/path",
        iteration: Optional[int] = None,
    ) -> None:
        if iteration is not None:
            rr.set_time_sequence("iter", int(iteration))

        pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] < 2:
            return

        col = None
        if color is not None:
            c = np.asarray(color)
            if c.dtype != np.uint8:
                c = np.clip(c, 0.0, 1.0)
                c = (c * 255.0).astype(np.uint8)
            c = c.reshape(-1)
            if c.size >= 3:
                col = c[:3]

        rr.log(
            entity_path,
            rr.LineStrips3D([pts], colors=[col] if col is not None else None, radii=radius),
        )
