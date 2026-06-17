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
import struct

class RerunVisualizer:
    """
    Rerun visualizer for Photo-SLAM + SVRaster.

    This is intentionally very close to nvblox_torch's RerunVisualizer:
    - world/camera_0: pose, axes, image, observations
    - world/trajectory: camera trajectory
    - world/mesh: color mesh (TSDF or voxel mesh)
    """

    def __init__(self, app_id: str = "PhotoSLAM-SVRaster", spawn: bool = True) -> None:
        # Parameters
        self.camera_pose_axis_scale = 0.1
        self.trajectory_length = 500
        # State
        self.track_colors: Dict[int, npt.NDArray] = {}
        self.t_W_C_history: Deque[npt.NDArray] = deque(maxlen=self.trajectory_length)
        self._gt_sdf_scene_path: Optional[str] = None
        self._gt_sdf_scene_key = None
        self._gt_sdf_scene = None
        self._gt_sdf_mesh_data = None
        self._gt_sdf_mesh_logged = False
        self._slam_centers_by_frame: Dict[int, npt.NDArray] = {}
        self._gt_traj_centers_path: Optional[str] = None
        self._gt_traj_centers: Optional[npt.NDArray] = None
        self._last_sdf_alignment_pairs = 0
        self._debug_recordings = {}
        self._debug_binary_streams = {}
        self._debug_trajectory_history = {}
        self._debug_gt_mesh_logged = set()
        self._main_binary_stream = None
        self._main_recording_bytes_cache = None
        self._start_rerun_visualizer(app_id, spawn)

    def _set_iter_time(self, iteration: Optional[int]) -> None:
        if iteration is None:
            return
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("iter", int(iteration))
        else:
            rr.set_time("iter", sequence=int(iteration))

    def _set_keyframe_time(self, keyframe_id: Optional[int]) -> None:
        if keyframe_id is None:
            return
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("keyframe_id", int(keyframe_id))
        else:
            rr.set_time("keyframe_id", sequence=int(keyframe_id))

    def _start_rerun_visualizer(self, app_id: str, spawn: bool) -> None:
        rr.init(app_id, spawn=spawn)    
        if hasattr(rr, "binary_stream"):
            self._main_binary_stream = rr.binary_stream()
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Spatial3DView(
                    origin="world",
                    name="Scene 3D",
                ),
            ),
            make_active=True,
        )

    def _send_default_blueprint(self) -> None:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Spatial3DView(
                    origin="world",
                    name="Scene 3D",
                ),
            ),
            make_active=True,
        )

    def _send_maps_blueprint(self) -> None:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        try:
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.TimePanel(state="collapsed"),
                    rrb.Tabs(
                        rrb.Grid(
                            rrb.Spatial2DView(origin="maps/rgb/gt", name="GT RGB"),
                            rrb.Spatial2DView(origin="maps/rgb/rendered", name="Rendered RGB"),
                            rrb.Spatial2DView(origin="maps/rgb/error", name="RGB Error"),
                            name="RGB",
                        ),
                        rrb.Grid(
                            rrb.Spatial2DView(origin="maps/depth/gt", name="GT Depth"),
                            rrb.Spatial2DView(origin="maps/depth/rendered", name="Rendered Depth"),
                            rrb.Spatial2DView(origin="maps/depth/error", name="Depth Error"),
                            rrb.Spatial2DView(origin="maps/depth/gaps", name="Depth Gaps"),
                            name="Depth",
                        ),
                        rrb.Grid(
                            rrb.Spatial2DView(origin="maps/normal/gt", name="GT Normal"),
                            rrb.Spatial2DView(origin="maps/normal/rendered", name="Rendered Normal"),
                            rrb.Spatial2DView(origin="maps/normal/error", name="Normal Error"),
                            name="Normal",
                        ),
                        rrb.TimeSeriesView(origin="maps/metrics", name="Metrics"),
                    ),
                ),
                make_active=True,
            )
        except Exception:
            self._send_default_blueprint()

    def _ensure_debug_recording(self, name: str):
        if not hasattr(rr, "RecordingStream") or not hasattr(rr, "binary_stream"):
            return None
        if name in self._debug_recordings:
            return self._debug_recordings[name]
        rec = rr.RecordingStream(
            f"PhotoSLAM-SVRaster-{name}",
            recording_id=name,
            make_default=False,
            make_thread_default=False,
            default_enabled=True,
        )
        stream = rr.binary_stream(recording=rec)
        self._debug_recordings[name] = rec
        self._debug_binary_streams[name] = stream
        with rec:
            if name == "maps":
                self._send_maps_blueprint()
            else:
                self._send_default_blueprint()
        return rec

    def save_recording(self, path: str) -> None:
        """
        Save the current Rerun recording to a .rrd file.

        Typically called once from C++ at the end of the run.
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if self._main_binary_stream is not None:
                if self._main_recording_bytes_cache is None:
                    data = self._main_binary_stream.read(flush=True)
                    self._main_recording_bytes_cache = data if data is not None else b""
                with open(path, "wb") as f:
                    f.write(self._main_recording_bytes_cache)
            else:
                rr.save(path)
            print(f"[RERUN] Saved recording to: {path}")
        except Exception as e:
            print(f"[RERUN] Failed to save recording to {path}: {e}")

    def save_debug_recording(self, name: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = self._ensure_debug_recording(name)
        stream = self._debug_binary_streams.get(name)
        if rec is None or stream is None:
            print(f"[RERUN] Debug recording unsupported, skipping: {name}")
            return
        try:
            data = stream.read(flush=True)
            if data is None:
                data = b""
            with open(path, "wb") as f:
                f.write(data)
            print(f"[RERUN] Saved debug recording '{name}' to: {path}")
        except Exception as e:
            print(f"[RERUN] Failed to save debug recording {name} to {path}: {e}")

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
        source_frame_id: Optional[int] = None,
        image_mode: str = "child",
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

        # Pinhole camera frustum
        if fx is not None and fy is not None and cx is not None and cy is not None:
            self._log_pinhole(entity_path, fx, fy, cx, cy, W, H)

        if image_mode == "child":
            rr.log(entity_path + "/image", rr.Image(image).compress())
        elif image_mode == "sibling":
            rr.log(entity_path + "_image/image", rr.Image(image).compress())

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
        keyframe_id: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        source_frame_id: Optional[int] = None,
    ) -> None:
        """
        Visualize:
        1) Camera pose (one entity per iteration/keyframe),
        2) Tracking image and 2D features,
        3) Global trajectory.
        """
        self._set_iter_time(iteration)
        if keyframe_id is not None:
            kf_name = f"kf_{int(keyframe_id):06d}"
            cam_entity = f"world/keyframes/{kf_name}"
            cam_image_entity = f"world/keyframes_with_images/{kf_name}"
        elif iteration is not None:
            kf_name = f"kf_{int(iteration):06d}"
            cam_entity = f"world/keyframes/{kf_name}"
            cam_image_entity = f"world/keyframes_with_images/{kf_name}"
        else:
            cam_entity = "world/camera_0"
            cam_image_entity = None

        # FOV/debug camera: clean Pinhole entity with no image child for
        # keyframes. Select this as eye-tracked to view the 3D scene through the
        # keyframe frustum. Non-keyframe camera_0 keeps the old image-child
        # behavior.
        clean_image_mode = "none" if cam_image_entity is not None else "child"
        self._log_camera_entity(
            cam_entity,
            t_W_C,
            q_W_C_xyzw,
            image,
            points_uv,
            track_ids,
            fx, fy, cx, cy,
            source_frame_id,
            image_mode=clean_image_mode,
        )

        # Image-plane camera: old behavior where the image is a child of the
        # camera/pinhole entity, useful for inspecting the keyframe image in 3D.
        if cam_image_entity is not None:
            self._log_camera_entity(
                cam_image_entity,
                t_W_C,
                q_W_C_xyzw,
                image,
                points_uv,
                track_ids,
                fx, fy, cx, cy,
                source_frame_id,
                image_mode="child",
            )

        # For trajectory we just keep the translation history
        t_W_C = np.asarray(t_W_C, dtype=np.float32).reshape(3)
        self.t_W_C_history.append(t_W_C)
        if source_frame_id is not None and int(source_frame_id) >= 0:
            self._slam_centers_by_frame[int(source_frame_id)] = t_W_C.copy()
        self._log_trajectory()

    def visualize_cuvslam_recording(
        self,
        recording_name: str,
        t_W_C: npt.NDArray,
        q_W_C_xyzw: npt.NDArray,
        image: npt.NDArray,
        points_uv: Optional[npt.NDArray] = None,
        track_ids: Optional[npt.NDArray] = None,
        iteration: Optional[int] = None,
        keyframe_id: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        source_frame_id: Optional[int] = None,
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return

        with rec:
            self._set_iter_time(iteration)
            if keyframe_id is not None:
                kf_name = f"kf_{int(keyframe_id):06d}"
                cam_entity = f"world/keyframes/{kf_name}"
                cam_image_entity = f"world/keyframes_with_images/{kf_name}"
            elif iteration is not None:
                kf_name = f"kf_{int(iteration):06d}"
                cam_entity = f"world/keyframes/{kf_name}"
                cam_image_entity = f"world/keyframes_with_images/{kf_name}"
            else:
                cam_entity = "world/camera_0"
                cam_image_entity = None

            clean_image_mode = "none" if cam_image_entity is not None else "child"
            self._log_camera_entity(
                cam_entity,
                t_W_C,
                q_W_C_xyzw,
                image,
                points_uv,
                track_ids,
                fx, fy, cx, cy,
                source_frame_id,
                image_mode=clean_image_mode,
            )

            if cam_image_entity is not None:
                self._log_camera_entity(
                    cam_image_entity,
                    t_W_C,
                    q_W_C_xyzw,
                    image,
                    points_uv,
                    track_ids,
                    fx, fy, cx, cy,
                    source_frame_id,
                    image_mode="child",
                )

            t_W_C_np = np.asarray(t_W_C, dtype=np.float32).reshape(3)
            hist = self._debug_trajectory_history.setdefault(
                str(recording_name),
                deque(maxlen=self.trajectory_length),
            )
            hist.append(t_W_C_np)
            if len(hist) > 0:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D([np.stack(hist, axis=0)]),
                    static=True,
                )

    def visualize_gt_sdf_mesh_recording(
        self,
        recording_name: str,
        mesh_path: str,
        align_gt_to_slam: bool,
        gt_traj_path: str,
        align_min_pairs: int,
        iteration: Optional[int] = None,
        entity_path: str = "world/gt/mesh",
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return

        key = (
            str(recording_name),
            mesh_path,
            bool(align_gt_to_slam),
            gt_traj_path if align_gt_to_slam else "",
            entity_path,
        )
        if key in self._debug_gt_mesh_logged:
            return

        with rec:
            self._set_iter_time(iteration)
            self._get_gt_sdf_scene(
                mesh_path,
                align_gt_to_slam=bool(align_gt_to_slam),
                gt_traj_path=gt_traj_path,
                align_min_pairs=int(align_min_pairs),
            )
            mesh_data = self._gt_sdf_mesh_data
            if mesh_data is not None:
                verts, faces, colors = mesh_data
                self._visualize_mesh(verts, faces, colors, entity_path=entity_path)
                self._debug_gt_mesh_logged.add(key)

    def visualize_nvblox_ply(
        self,
        ply_path: str,
        iteration: int,
        entity_path: str = "world/tsdf_mesh/live",
    ):
        """
        Called from C++:
            impl_->visualizer.attr("visualize_nvblox_ply")(py::str(ply_path), iteration, entity_path)
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
        self._set_iter_time(iteration)

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

    def visualize_nvblox_ply_recording(
        self,
        recording_name: str,
        ply_path: str,
        iteration: int,
        entity_path: str = "world/nvblox_mesh/reference",
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return
        with rec:
            self.visualize_nvblox_ply(ply_path, iteration, entity_path)

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
        self._set_iter_time(iteration)

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

    def visualize_nvblox_recording(
        self,
        recording_name: str,
        vertices: npt.NDArray,
        colors: Optional[npt.NDArray],
        triangles: npt.NDArray,
        iteration: Optional[int] = None,
        entity_path: str = "world/mesh",
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return

        with rec:
            self._set_iter_time(iteration)
            v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
            f = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)

            c = None
            if colors is not None:
                c = np.asarray(colors)
                if c.shape[0] == v.shape[0]:
                    if c.dtype != np.uint8:
                        c = np.clip(c, 0.0, 1.0)
                        c = (c * 255.0).astype(np.uint8)
                else:
                    c = None

            self._visualize_mesh(v, f, c, entity_path=entity_path)

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
        self._set_iter_time(iteration)

        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        half_sizes = np.asarray(half_sizes, dtype=np.float32)
        if half_sizes.ndim == 1:
            half_sizes = half_sizes[:, None].repeat(3, axis=1)
        else:
            half_sizes = half_sizes.reshape(-1, 3)

        N = centers.shape[0]
        if N == 0:
            rr.log(
                entity_path,
                rr.Boxes3D(
                    centers=np.zeros((0, 3), dtype=np.float32),
                    half_sizes=np.zeros((0, 3), dtype=np.float32),
                    colors=None,
                    fill_mode="solid",
                ),
            )
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
        if max_boxes is not None and int(max_boxes) > 0 and N > int(max_boxes):
            print(f"[PY][voxels] too many boxes ({N}), downsampling to {max_boxes}")
            idx = np.linspace(0, N - 1, int(max_boxes), dtype=np.int64)
            centers = centers[idx]
            half_sizes = half_sizes[idx]
            if colors is not None:
                colors = colors[idx]
            N = int(max_boxes)

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

        self._set_iter_time(iteration)

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
        labels: Optional[list[str]] = None,
    ) -> None:
        self._set_iter_time(iteration)

        pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            rr.log(entity_path, rr.Points3D(pts))
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

        point_labels = labels if labels is not None and len(labels) == pts.shape[0] else None
        kwargs = {"colors": col, "labels": point_labels}
        if radii is not None and float(radii) > 0.0:
            kwargs["radii"] = radii
        rr.log(entity_path, rr.Points3D(pts, **kwargs))

    def visualize_points3d_recording(
        self,
        recording_name: str,
        points_xyz: npt.NDArray,
        colors: Optional[npt.NDArray] = None,
        radii: float = 0.02,
        entity_path: str = "world/points",
        iteration: Optional[int] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return
        with rec:
            self.visualize_points3d(
                points_xyz,
                colors=colors,
                radii=radii,
                entity_path=entity_path,
                iteration=iteration,
                labels=labels,
            )

    def visualize_linestrip3d(
        self,
        points_xyz: npt.NDArray,
        color: Optional[npt.NDArray] = None,
        radius: float = 0.01,
        entity_path: str = "world/path",
        iteration: Optional[int] = None,
    ) -> None:
        self._set_iter_time(iteration)

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

    def visualize_scalar(
        self,
        value: float,
        entity_path: str,
        iteration: Optional[int] = None,
    ) -> None:
        self._set_iter_time(iteration)
        rr.log(entity_path, rr.Scalars(float(value)))

    def visualize_scalar_recording(
        self,
        recording_name: str,
        value: float,
        entity_path: str,
        iteration: Optional[int] = None,
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return
        with rec:
            self.visualize_scalar(value, entity_path, iteration)

    def visualize_maps_frame_recording(
        self,
        recording_name: str,
        keyframe_id: int,
        iteration: int,
        gt_rgb: npt.NDArray,
        rendered_rgb: npt.NDArray,
        rgb_error: npt.NDArray,
        gt_depth_rgb: npt.NDArray,
        rendered_depth_rgb: npt.NDArray,
        depth_error_rgb: npt.NDArray,
        depth_gap_rgb: npt.NDArray,
        gt_normal_rgb: npt.NDArray,
        rendered_normal_rgb: npt.NDArray,
        normal_error_rgb: npt.NDArray,
        psnr: float,
        ssim: float,
        depth_l1_m: float,
        depth_gap_percent: float,
        normal_mean_deg: float,
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return

        def _log_image(entity_path: str, image: npt.NDArray) -> None:
            img = np.asarray(image)
            if img.size == 0:
                return
            if img.dtype != np.uint8:
                img = np.clip(img, 0.0, 255.0).astype(np.uint8)
            if img.ndim == 2:
                rr.log(entity_path, rr.Image(img).compress())
            elif img.ndim == 3 and img.shape[2] in (3, 4):
                rr.log(entity_path, rr.Image(np.ascontiguousarray(img)).compress())

        with rec:
            self._set_iter_time(iteration)
            self._set_keyframe_time(keyframe_id)

            _log_image("maps/rgb/gt", gt_rgb)
            _log_image("maps/rgb/rendered", rendered_rgb)
            _log_image("maps/rgb/error", rgb_error)
            _log_image("maps/depth/gt", gt_depth_rgb)
            _log_image("maps/depth/rendered", rendered_depth_rgb)
            _log_image("maps/depth/error", depth_error_rgb)
            _log_image("maps/depth/gaps", depth_gap_rgb)
            _log_image("maps/normal/gt", gt_normal_rgb)
            _log_image("maps/normal/rendered", rendered_normal_rgb)
            _log_image("maps/normal/error", normal_error_rgb)

            rr.log("maps/metrics/psnr", rr.Scalars(float(psnr)))
            rr.log("maps/metrics/ssim", rr.Scalars(float(ssim)))
            if float(depth_l1_m) >= 0.0:
                rr.log("maps/metrics/depth_l1_m", rr.Scalars(float(depth_l1_m)))
            if float(depth_gap_percent) >= 0.0:
                rr.log("maps/metrics/depth_gap_percent", rr.Scalars(float(depth_gap_percent)))
            if float(normal_mean_deg) >= 0.0:
                rr.log("maps/metrics/normal_mean_deg", rr.Scalars(float(normal_mean_deg)))

    def visualize_voxels_boxes_recording(
        self,
        recording_name: str,
        centers: npt.NDArray,
        half_sizes: npt.NDArray,
        colors: Optional[npt.NDArray] = None,
        entity_path: str = "world/voxels",
        max_boxes: int = 1000000,
        iteration: Optional[int] = None,
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return
        with rec:
            self.visualize_voxels_boxes(
                centers,
                half_sizes,
                colors,
                entity_path=entity_path,
                max_boxes=max_boxes,
                iteration=iteration,
            )

    def visualize_sdf_voxels_recording(
        self,
        recording_name: str,
        centers: npt.NDArray,
        sizes: npt.NDArray,
        corner_points: npt.NDArray,
        computed_sdf: npt.NDArray,
        sdf_weights: npt.NDArray,
        corner_density: npt.NDArray,
        gt_sdf: npt.NDArray,
        voxel_colors: Optional[npt.NDArray],
        voxel_ids: Optional[npt.NDArray],
        source_sdf_mask: Optional[npt.NDArray],
        source_svraster_mask: Optional[npt.NDArray],
        iteration: int,
        gt_mesh_path: str,
        align_gt_to_slam: bool,
        gt_traj_path: str,
        align_min_pairs: int,
        surface_band_m: float,
        min_weight: float,
        log_gt_mesh: bool,
        entity_path: str = "world/voxels_sdf_pruned",
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return

        with rec:
            self._set_iter_time(iteration)

            if log_gt_mesh:
                self.visualize_gt_sdf_mesh_recording(
                    recording_name,
                    gt_mesh_path,
                    align_gt_to_slam,
                    gt_traj_path,
                    align_min_pairs,
                    iteration,
                    "world/gt/mesh",
                )
                self._set_iter_time(iteration)

            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
            sizes = np.asarray(sizes, dtype=np.float32)
            if sizes.ndim == 1:
                full_sizes = sizes[:, None].repeat(3, axis=1)
            elif sizes.ndim == 2 and sizes.shape[1] == 1:
                full_sizes = sizes.repeat(3, axis=1)
            else:
                full_sizes = sizes.reshape(-1, 3)
            half_sizes = 0.5 * full_sizes

            corners = np.asarray(corner_points, dtype=np.float32).reshape(-1, 8, 3)
            comp_sdf = np.asarray(computed_sdf, dtype=np.float32).reshape(-1, 8)
            weights = np.asarray(sdf_weights, dtype=np.float32).reshape(-1, 8)
            density = np.asarray(corner_density, dtype=np.float32).reshape(-1, 8)
            gt_sdf = np.asarray(gt_sdf, dtype=np.float32).reshape(-1, 8)
            n = centers.shape[0]
            if (
                n == 0
                or corners.shape[0] != n
                or comp_sdf.shape[0] != n
                or gt_sdf.shape[0] != n
            ):
                return

            gt_mesh_distance_mode = False
            if gt_mesh_path:
                gt_corner_distance = self._compute_gt_surface_distance(
                    corners.reshape(-1, 3),
                    gt_mesh_path,
                    bool(align_gt_to_slam),
                    gt_traj_path,
                    int(align_min_pairs),
                )
                if (
                    gt_corner_distance is not None
                    and gt_corner_distance.size == n * 8
                ):
                    gt_sdf = gt_corner_distance.reshape(n, 8).astype(np.float32, copy=False)
                    gt_mesh_distance_mode = True

            ids = np.arange(n, dtype=np.int64)
            if voxel_ids is not None:
                ids_in = np.asarray(voxel_ids).reshape(-1)
                if ids_in.shape[0] == n:
                    ids = ids_in.astype(np.int64, copy=False)

            source_sdf = np.zeros((n,), dtype=bool)
            if source_sdf_mask is not None:
                mask_in = np.asarray(source_sdf_mask).reshape(-1)
                if mask_in.shape[0] == n:
                    source_sdf = mask_in.astype(bool, copy=False)
            source_svraster = np.zeros((n,), dtype=bool)
            if source_svraster_mask is not None:
                mask_in = np.asarray(source_svraster_mask).reshape(-1)
                if mask_in.shape[0] == n:
                    source_svraster = mask_in.astype(bool, copy=False)

            colors = None
            if voxel_colors is not None:
                vc = np.asarray(voxel_colors)
                if vc.size > 0:
                    vc = vc.reshape(n, -1)
                    if vc.shape[0] == n and vc.shape[1] >= 3:
                        colors = vc[:, :min(vc.shape[1], 4)].copy()
            if colors is None:
                colors = np.zeros((n, 4), dtype=np.uint8)
                colors[:, 2] = 255
                colors[:, 3] = 180

            band = max(float(surface_band_m), 1.0e-6)
            valid_comp = np.isfinite(comp_sdf) & np.isfinite(weights) & (weights >= float(min_weight))
            valid_gt = np.isfinite(gt_sdf)
            corner_pts = corners.reshape(-1, 3)

            comp_cls = np.zeros_like(comp_sdf, dtype=np.int8)
            gt_cls = np.zeros_like(gt_sdf, dtype=np.int8)
            comp_cls[comp_sdf > band] = 1
            comp_cls[np.abs(comp_sdf) <= band] = 2
            comp_cls[comp_sdf < -band] = 3
            if gt_mesh_distance_mode:
                gt_cls[gt_sdf > band] = 1
                gt_cls[gt_sdf <= band] = 2
            else:
                gt_cls[gt_sdf > band] = 1
                gt_cls[np.abs(gt_sdf) <= band] = 2
                gt_cls[gt_sdf < -band] = 3

            half_diag = np.linalg.norm(half_sizes, axis=1).astype(np.float32, copy=False)

            def voxel_stats(sdf: npt.NDArray, valid: npt.NDArray):
                all_valid = np.all(valid, axis=1)
                has_pos = np.any((sdf > 1.0e-6) & valid, axis=1)
                has_neg = np.any((sdf < -1.0e-6) & valid, axis=1)
                inf = np.full_like(sdf, np.inf, dtype=np.float32)
                min_abs = np.min(np.where(valid, np.abs(sdf), inf), axis=1)
                return all_valid, has_pos, has_neg, min_abs

            def classify_voxels(sdf: npt.NDArray, valid: npt.NDArray):
                all_valid, has_pos, has_neg, min_abs = voxel_stats(sdf, valid)
                strong_far = min_abs > (band + half_diag)
                free = all_valid & has_pos & (~has_neg) & strong_far
                occupied = all_valid & has_neg & (~has_pos) & strong_far
                surface = all_valid & (~(free | occupied))
                return free, occupied, surface

            def classify_surface_crossing(sdf: npt.NDArray, valid: npt.NDArray):
                if gt_mesh_distance_mode:
                    all_valid = np.all(valid, axis=1)
                    min_corner_distance = np.min(
                        np.where(valid, sdf, np.inf), axis=1
                    )
                    surface = all_valid & (min_corner_distance <= band)
                    free = all_valid & (~surface)
                    occupied = np.zeros_like(surface, dtype=bool)
                    return free, occupied, surface
                all_valid, has_pos, has_neg, min_abs = voxel_stats(sdf, valid)
                strong_far = min_abs > (band + half_diag)
                free = all_valid & has_pos & (~has_neg) & strong_far
                occupied = all_valid & has_neg & (~has_pos) & strong_far
                surface = all_valid & has_pos & has_neg
                return free, occupied, surface

            comp_voxel_free, comp_voxel_occupied, comp_voxel_surface = classify_voxels(comp_sdf, valid_comp)
            gt_voxel_free, gt_voxel_occupied, gt_voxel_surface = classify_surface_crossing(gt_sdf, valid_gt)

            wrong_sdf_free_gt_surface = (source_sdf | source_svraster) & comp_voxel_free & gt_voxel_surface
            wrong_svraster_pruned_surface_agreement = (
                source_svraster & comp_voxel_surface & gt_voxel_surface
            )
            wrong_mesh = (
                wrong_sdf_free_gt_surface |
                wrong_svraster_pruned_surface_agreement
            )
            correct_mesh = (source_sdf | source_svraster) & comp_voxel_free & gt_voxel_free

            def log_voxel_subset(mask: npt.NDArray, subpath: str, override_rgba=None) -> None:
                mask = np.asarray(mask, dtype=bool).reshape(-1)
                subset_colors = colors[mask]
                if override_rgba is not None:
                    rgba = np.asarray(override_rgba, dtype=np.uint8).reshape(1, 4)
                    subset_colors = np.repeat(rgba, int(mask.sum()), axis=0)
                self.visualize_voxels_boxes(
                    centers[mask],
                    half_sizes[mask],
                    subset_colors,
                    entity_path=f"{entity_path}/{subpath}",
                    max_boxes=0,
                    iteration=iteration,
                )

            nvblox_reference_mode = str(entity_path).rstrip("/").endswith(
                "voxels_sdf_pruned_nvblox"
            )
            if not nvblox_reference_mode:
                log_voxel_subset(wrong_mesh, "wrong_voxels_mesh", [255, 140, 0, 220])
                log_voxel_subset(
                    wrong_sdf_free_gt_surface,
                    "wrong_voxels_mesh/sdf_free_gt_surface",
                    [255, 0, 0, 220],
                )
                log_voxel_subset(
                    wrong_svraster_pruned_surface_agreement,
                    "wrong_voxels_mesh/svraster_pruned_surface_agreement",
                    [255, 180, 0, 220],
                )
                log_voxel_subset(correct_mesh, "correct_voxels_mesh", None)

            class_names = {
                0: "unknown",
                1: "free_space",
                2: "surface_band",
                3: "occupied_side",
            }

            def voxel_class_name(vox_i: int, prefix: str) -> str:
                if prefix == "computed":
                    if comp_voxel_free[vox_i]:
                        return "free_space"
                    if comp_voxel_occupied[vox_i]:
                        return "occupied_side"
                    if comp_voxel_surface[vox_i]:
                        return "surface_band"
                    return "unknown"
                if gt_voxel_free[vox_i]:
                    return "free_space"
                if gt_voxel_occupied[vox_i]:
                    return "occupied_side"
                if gt_voxel_surface[vox_i]:
                    return "surface_band"
                return "unknown"

            if nvblox_reference_mode:
                wrong_nvblox = source_sdf & comp_voxel_free & (
                    gt_voxel_surface | gt_voxel_occupied
                )
                correct_nvblox = source_sdf & comp_voxel_free & gt_voxel_free
                missed_nvblox = source_svraster & (
                    comp_voxel_surface | comp_voxel_occupied
                ) & gt_voxel_free
                any_gt_known = gt_voxel_free | gt_voxel_surface | gt_voxel_occupied
                unknown_nvblox = (source_sdf | source_svraster) & (~any_gt_known)

                log_voxel_subset(
                    wrong_nvblox,
                    "wrong_voxels_nvblox",
                    [255, 0, 0, 220],
                )
                log_voxel_subset(
                    correct_nvblox,
                    "correct_voxels_nvblox",
                    None,
                )
                log_voxel_subset(
                    missed_nvblox,
                    "missed_voxels_nvblox",
                    [255, 180, 0, 220],
                )
                log_voxel_subset(
                    unknown_nvblox,
                    "unknown_nvblox",
                    [120, 120, 120, 180],
                )

            point_colors = np.zeros((corner_pts.shape[0], 4), dtype=np.uint8)
            point_colors[:, 0] = 60
            point_colors[:, 1] = 180
            point_colors[:, 2] = 255
            point_colors[:, 3] = 255
            dangerous_free_corner = (
                valid_comp
                & valid_gt
                & (comp_cls == 1)
                & ((gt_cls == 2) | (gt_cls == 3))
            )
            dangerous_free_flat = dangerous_free_corner.reshape(-1)
            point_colors[dangerous_free_flat, 0] = 255
            point_colors[dangerous_free_flat, 1] = 0
            point_colors[dangerous_free_flat, 2] = 0
            point_colors[dangerous_free_flat, 3] = 255

            labels = []
            meta_voxel_id = []
            meta_corner_id = []
            meta_pruned_by_tsdf = []
            meta_dangerous_free_corner = []
            meta_computed_sdf_m = []
            meta_computed_weight = []
            meta_gt_sdf_m = []
            meta_density_raw = []
            meta_computed_corner_class = []
            meta_gt_corner_class = []
            meta_computed_voxel_class = []
            meta_gt_voxel_class = []
            meta_wrong_voxel_mesh = []
            meta_wrong_sdf_free_gt_surface = []
            meta_wrong_svraster_pruned_surface_agreement = []
            meta_correct_voxel_mesh = []
            meta_pruned_by_sdf = []
            meta_pruned_by_svraster = []

            for vox_i in range(n):
                comp_voxel_name = voxel_class_name(vox_i, "computed")
                gt_voxel_name = voxel_class_name(vox_i, "gt")
                for c in range(8):
                    comp_name = class_names.get(int(comp_cls[vox_i, c]), "unknown")
                    gt_name = class_names.get(int(gt_cls[vox_i, c]), "unknown")
                    dangerous_corner = bool(dangerous_free_corner[vox_i, c])
                    labels.append(
                        "voxel_id={} corner={} pruned_by_tsdf={} dangerous_free_corner={} "
                        "computed_sdf_m={:.5f} computed_weight={:.3f} "
                        "gt_sdf_m={:.5f} density_raw={:.5f} "
                        "computed_corner_class={} gt_corner_class={} "
                        "computed_voxel_class={} gt_voxel_class={} "
                        "wrong_voxel_mesh={} wrong_sdf_free_gt_surface={} "
                        "wrong_svraster_pruned_surface_agreement={} "
                        "correct_voxel_mesh={} pruned_by_sdf={} pruned_by_svraster={}".format(
                            int(ids[vox_i]),
                            c,
                            int(source_sdf[vox_i]),
                            int(dangerous_corner),
                            float(comp_sdf[vox_i, c]),
                            float(weights[vox_i, c]),
                            float(gt_sdf[vox_i, c]),
                            float(density[vox_i, c]),
                            comp_name,
                            gt_name,
                            comp_voxel_name,
                            gt_voxel_name,
                            int(wrong_mesh[vox_i]),
                            int(wrong_sdf_free_gt_surface[vox_i]),
                            int(wrong_svraster_pruned_surface_agreement[vox_i]),
                            int(correct_mesh[vox_i]),
                            int(source_sdf[vox_i]),
                            int(source_svraster[vox_i]),
                        )
                    )
                    meta_voxel_id.append(int(ids[vox_i]))
                    meta_corner_id.append(c)
                    meta_pruned_by_tsdf.append(bool(source_sdf[vox_i]))
                    meta_dangerous_free_corner.append(dangerous_corner)
                    meta_computed_sdf_m.append(float(comp_sdf[vox_i, c]))
                    meta_computed_weight.append(float(weights[vox_i, c]))
                    meta_gt_sdf_m.append(float(gt_sdf[vox_i, c]))
                    meta_density_raw.append(float(density[vox_i, c]))
                    meta_computed_corner_class.append(comp_name)
                    meta_gt_corner_class.append(gt_name)
                    meta_computed_voxel_class.append(comp_voxel_name)
                    meta_gt_voxel_class.append(gt_voxel_name)
                    meta_wrong_voxel_mesh.append(bool(wrong_mesh[vox_i]))
                    meta_wrong_sdf_free_gt_surface.append(bool(wrong_sdf_free_gt_surface[vox_i]))
                    meta_wrong_svraster_pruned_surface_agreement.append(
                        bool(wrong_svraster_pruned_surface_agreement[vox_i])
                    )
                    meta_correct_voxel_mesh.append(bool(correct_mesh[vox_i]))
                    meta_pruned_by_sdf.append(bool(source_sdf[vox_i]))
                    meta_pruned_by_svraster.append(bool(source_svraster[vox_i]))

            def log_corner_subset(voxel_mask: npt.NDArray, subpath: str) -> None:
                voxel_mask = np.asarray(voxel_mask, dtype=bool).reshape(-1)
                voxel_idx = np.flatnonzero(voxel_mask)
                path = f"{entity_path}/{subpath}/corners"
                if voxel_idx.size == 0:
                    rr.log(path, rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
                    if hasattr(rr, "AnyValues"):
                        rr.log(
                            path,
                            rr.AnyValues(
                                voxel_id=np.asarray([], dtype=np.int64),
                                corner_id=np.asarray([], dtype=np.int64),
                                pruned_by_tsdf=np.asarray([], dtype=bool),
                                dangerous_free_corner=np.asarray([], dtype=bool),
                                computed_sdf_m=np.asarray([], dtype=np.float32),
                                computed_weight=np.asarray([], dtype=np.float32),
                                gt_sdf_m=np.asarray([], dtype=np.float32),
                                density_raw=np.asarray([], dtype=np.float32),
                                computed_corner_class=[],
                                gt_corner_class=[],
                                computed_voxel_class=[],
                                gt_voxel_class=[],
                                wrong_voxel_mesh=np.asarray([], dtype=bool),
                                wrong_sdf_free_gt_surface=np.asarray([], dtype=bool),
                                wrong_svraster_pruned_surface_agreement=np.asarray([], dtype=bool),
                                correct_voxel_mesh=np.asarray([], dtype=bool),
                                pruned_by_sdf=np.asarray([], dtype=bool),
                                pruned_by_svraster=np.asarray([], dtype=bool),
                                corner_info=[],
                            ),
                        )
                    return

                flat_idx = (voxel_idx[:, None] * 8 + np.arange(8, dtype=np.int64)[None, :]).reshape(-1)
                subset_labels = [labels[int(i)] for i in flat_idx]
                try:
                    subset_points = rr.Points3D(
                        corner_pts[flat_idx],
                        colors=point_colors[flat_idx],
                        labels=subset_labels,
                        show_labels=False,
                    )
                except TypeError:
                    subset_points = rr.Points3D(corner_pts[flat_idx], colors=point_colors[flat_idx])
                rr.log(path, subset_points)

                if hasattr(rr, "AnyValues"):
                    rr.log(
                        path,
                        rr.AnyValues(
                            voxel_id=np.asarray(meta_voxel_id, dtype=np.int64)[flat_idx],
                            corner_id=np.asarray(meta_corner_id, dtype=np.int64)[flat_idx],
                            pruned_by_tsdf=np.asarray(meta_pruned_by_tsdf, dtype=bool)[flat_idx],
                            dangerous_free_corner=np.asarray(meta_dangerous_free_corner, dtype=bool)[flat_idx],
                            computed_sdf_m=np.asarray(meta_computed_sdf_m, dtype=np.float32)[flat_idx],
                            computed_weight=np.asarray(meta_computed_weight, dtype=np.float32)[flat_idx],
                            gt_sdf_m=np.asarray(meta_gt_sdf_m, dtype=np.float32)[flat_idx],
                            density_raw=np.asarray(meta_density_raw, dtype=np.float32)[flat_idx],
                            computed_corner_class=[meta_computed_corner_class[int(i)] for i in flat_idx],
                            gt_corner_class=[meta_gt_corner_class[int(i)] for i in flat_idx],
                            computed_voxel_class=[meta_computed_voxel_class[int(i)] for i in flat_idx],
                            gt_voxel_class=[meta_gt_voxel_class[int(i)] for i in flat_idx],
                            wrong_voxel_mesh=np.asarray(meta_wrong_voxel_mesh, dtype=bool)[flat_idx],
                            wrong_sdf_free_gt_surface=np.asarray(
                                meta_wrong_sdf_free_gt_surface, dtype=bool
                            )[flat_idx],
                            wrong_svraster_pruned_surface_agreement=np.asarray(
                                meta_wrong_svraster_pruned_surface_agreement, dtype=bool
                            )[flat_idx],
                            correct_voxel_mesh=np.asarray(meta_correct_voxel_mesh, dtype=bool)[flat_idx],
                            pruned_by_sdf=np.asarray(meta_pruned_by_sdf, dtype=bool)[flat_idx],
                            pruned_by_svraster=np.asarray(meta_pruned_by_svraster, dtype=bool)[flat_idx],
                            corner_info=subset_labels,
                        ),
                    )

            if nvblox_reference_mode:
                log_corner_subset(wrong_nvblox, "wrong_voxels_nvblox")
                log_corner_subset(correct_nvblox, "correct_voxels_nvblox")
                log_corner_subset(missed_nvblox, "missed_voxels_nvblox")
                log_corner_subset(unknown_nvblox, "unknown_nvblox")
                return

            log_corner_subset(wrong_sdf_free_gt_surface, "wrong_voxels_mesh/sdf_free_gt_surface")
            log_corner_subset(
                wrong_svraster_pruned_surface_agreement,
                "wrong_voxels_mesh/svraster_pruned_surface_agreement",
            )
            log_corner_subset(correct_mesh, "correct_voxels_mesh")

    def _load_gt_trajectory_centers(self, traj_path: str) -> Optional[npt.NDArray]:
        if self._gt_traj_centers is not None and self._gt_traj_centers_path == traj_path:
            return self._gt_traj_centers
        if not traj_path or not os.path.exists(traj_path):
            print(f"[RERUN/gt_sdf] GT trajectory path missing: {traj_path}")
            return None

        centers = []
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = [float(x) for x in line.split()]
                if len(vals) < 16:
                    continue
                T = np.asarray(vals[:16], dtype=np.float64).reshape(4, 4)
                centers.append(T[:3, 3].copy())

        if len(centers) == 0:
            print(f"[RERUN/gt_sdf] no valid GT trajectory poses: {traj_path}")
            return None
        self._gt_traj_centers_path = traj_path
        self._gt_traj_centers = np.asarray(centers, dtype=np.float64)
        return self._gt_traj_centers

    def _estimate_similarity_umeyama(self, src: npt.NDArray, dst: npt.NDArray):
        src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
        if src.shape[0] < 4 or src.shape[0] != dst.shape[0]:
            return None

        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        xs = src - mu_src[None, :]
        yd = dst - mu_dst[None, :]
        var_src = np.mean(np.sum(xs * xs, axis=1))
        if var_src <= 1.0e-15:
            return None

        cov = (yd.T @ xs) / float(src.shape[0])
        U, singular, Vt = np.linalg.svd(cov)
        S = np.eye(3, dtype=np.float64)
        if np.linalg.det(U @ Vt) < 0.0:
            S[2, 2] = -1.0
        R = U @ S @ Vt
        scale = float(np.sum(singular * np.diag(S)) / var_src)
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        t = mu_dst - scale * (R @ mu_src)
        return scale, R, t

    def _estimate_gt_to_slam_transform(
        self,
        gt_traj_path: str,
        min_pairs: int,
    ):
        gt_centers = self._load_gt_trajectory_centers(gt_traj_path)
        if gt_centers is None:
            return None

        frame_ids = sorted(
            i for i in self._slam_centers_by_frame.keys()
            if 0 <= i < gt_centers.shape[0]
        )
        if len(frame_ids) < max(4, int(min_pairs)):
            if len(frame_ids) != self._last_sdf_alignment_pairs:
                self._last_sdf_alignment_pairs = len(frame_ids)
                print(
                    f"[RERUN/gt_sdf] waiting for GT alignment pairs: "
                    f"{len(frame_ids)}/{max(4, int(min_pairs))}"
                )
            return None

        slam = np.stack([self._slam_centers_by_frame[i] for i in frame_ids], axis=0).astype(np.float64)
        gt = gt_centers[frame_ids].astype(np.float64)
        sim = self._estimate_similarity_umeyama(slam, gt)
        if sim is None:
            return None

        slam_to_gt_scale, slam_to_gt_R, slam_to_gt_t = sim
        gt_to_slam_scale = 1.0 / slam_to_gt_scale
        gt_to_slam_R = slam_to_gt_R.T
        gt_to_slam_t = -gt_to_slam_scale * (gt_to_slam_R @ slam_to_gt_t)
        return {
            "scale": gt_to_slam_scale,
            "R": gt_to_slam_R,
            "t": gt_to_slam_t,
            "pairs": len(frame_ids),
            "slam_to_gt_scale": slam_to_gt_scale,
        }

    def _transform_vertices(self, vertices: npt.NDArray, transform) -> npt.NDArray:
        if transform is None:
            return np.asarray(vertices, dtype=np.float32)
        v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        out = transform["scale"] * (v @ transform["R"].T) + transform["t"][None, :]
        return out.astype(np.float32)

    def _get_gt_sdf_scene(
        self,
        mesh_path: str,
        align_gt_to_slam: bool = False,
        gt_traj_path: str = "",
        align_min_pairs: int = 10,
    ):
        transform = None
        align_pairs = 0
        if align_gt_to_slam:
            transform = self._estimate_gt_to_slam_transform(gt_traj_path, align_min_pairs)
            if transform is None:
                return None
            align_pairs = int(transform["pairs"])

        scene_key = (
            mesh_path,
            bool(align_gt_to_slam),
            gt_traj_path if align_gt_to_slam else "",
        )
        if self._gt_sdf_scene is not None and self._gt_sdf_scene_key == scene_key:
            return self._gt_sdf_scene
        if not mesh_path or not os.path.exists(mesh_path):
            print(f"[RERUN/gt_sdf] GT mesh path missing: {mesh_path}")
            return None

        mesh_data = self._load_gt_triangle_mesh(mesh_path)
        if mesh_data is None:
            print(f"[RERUN/gt_sdf] GT mesh failed to load or has no triangles: {mesh_path}")
            return None
        vertices, faces, _ = mesh_data
        vertices_sdf = self._transform_vertices(vertices, transform)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_sdf.astype(np.float64, copy=False))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False))
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)

        self._gt_sdf_scene_path = mesh_path
        self._gt_sdf_scene_key = scene_key
        self._gt_sdf_scene = scene
        self._gt_sdf_mesh_data = (vertices_sdf, faces, mesh_data[2])
        self._gt_sdf_mesh_logged = False
        # Keep GT SDF mesh loading quiet; this path is also used for
        # headless pruning statistics when Rerun recording is disabled.
        return scene

    def _ply_dtype(self, ply_type: str):
        table = {
            "char": "i1",
            "int8": "i1",
            "uchar": "u1",
            "uint8": "u1",
            "short": "<i2",
            "int16": "<i2",
            "ushort": "<u2",
            "uint16": "<u2",
            "int": "<i4",
            "int32": "<i4",
            "uint": "<u4",
            "uint32": "<u4",
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
        }
        return table.get(ply_type)

    def _load_binary_little_endian_ply_mesh(self, mesh_path: str):
        with open(mesh_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    return None
                header_lines.append(line.decode("ascii", errors="replace").strip())
                if header_lines[-1] == "end_header":
                    break

            if len(header_lines) < 2 or header_lines[0] != "ply":
                return None
            if header_lines[1] != "format binary_little_endian 1.0":
                return None

            vertex_count = 0
            face_count = 0
            vertex_props = []
            face_count_type = None
            face_index_type = None
            element = None
            for line in header_lines:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "element":
                    element = parts[1]
                    if element == "vertex":
                        vertex_count = int(parts[2])
                    elif element == "face":
                        face_count = int(parts[2])
                elif element == "vertex" and len(parts) == 3 and parts[0] == "property":
                    dtype = self._ply_dtype(parts[1])
                    if dtype is None:
                        return None
                    vertex_props.append((parts[2], dtype))
                elif element == "face" and len(parts) == 5 and parts[0] == "property" and parts[1] == "list":
                    face_count_type = parts[2]
                    face_index_type = parts[3]

            if vertex_count <= 0 or face_count <= 0 or not vertex_props:
                return None
            if face_count_type not in ("uchar", "uint8") or face_index_type not in ("int", "int32"):
                return None

            vertex_dtype = np.dtype(vertex_props)
            vertex_records = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
            if vertex_records.shape[0] != vertex_count:
                return None
            if not all(name in vertex_records.dtype.names for name in ("x", "y", "z")):
                return None

            vertices = np.stack(
                [vertex_records["x"], vertex_records["y"], vertex_records["z"]],
                axis=1,
            ).astype(np.float32, copy=False)

            colors = None
            if all(name in vertex_records.dtype.names for name in ("red", "green", "blue")):
                colors = np.stack(
                    [vertex_records["red"], vertex_records["green"], vertex_records["blue"]],
                    axis=1,
                ).astype(np.uint8, copy=False)

            faces = []
            for _ in range(face_count):
                count_raw = f.read(1)
                if not count_raw:
                    return None
                count = count_raw[0]
                idx_raw = f.read(4 * count)
                if len(idx_raw) != 4 * count:
                    return None
                idx = struct.unpack("<" + "i" * count, idx_raw)
                if count < 3:
                    continue
                for j in range(1, count - 1):
                    faces.append((idx[0], idx[j], idx[j + 1]))

            if not faces:
                return None
            faces_np = np.asarray(faces, dtype=np.int32)
            return vertices, faces_np, colors

    def _load_gt_triangle_mesh(self, mesh_path: str):
        # Replica meshes are binary PLYs with polygon faces; Open3D can reject
        # them before triangulation, so use a small local triangulating loader.
        if mesh_path.lower().endswith(".ply"):
            mesh_data = self._load_binary_little_endian_ply_mesh(mesh_path)
            if mesh_data is not None:
                vertices, faces, _ = mesh_data
                return mesh_data

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty() or len(mesh.triangles) == 0:
            return None
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        colors = None
        if len(mesh.vertex_colors) == vertices.shape[0]:
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        return vertices, faces, colors

    def _compute_gt_signed_distance(
        self,
        points: npt.NDArray,
        mesh_path: str,
        align_gt_to_slam: bool,
        gt_traj_path: str,
        align_min_pairs: int,
    ) -> Optional[npt.NDArray]:
        scene = self._get_gt_sdf_scene(
            mesh_path,
            align_gt_to_slam=align_gt_to_slam,
            gt_traj_path=gt_traj_path,
            align_min_pairs=align_min_pairs,
        )
        if scene is None:
            return None

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        out = np.empty((pts.shape[0],), dtype=np.float32)
        chunk = 500_000
        for start in range(0, pts.shape[0], chunk):
            stop = min(start + chunk, pts.shape[0])
            query = o3d.core.Tensor(pts[start:stop], dtype=o3d.core.Dtype.Float32)
            out[start:stop] = scene.compute_signed_distance(query).numpy().astype(np.float32, copy=False)
        return out

    def _compute_gt_surface_distance(
        self,
        points: npt.NDArray,
        mesh_path: str,
        align_gt_to_slam: bool,
        gt_traj_path: str,
        align_min_pairs: int,
    ) -> Optional[npt.NDArray]:
        scene = self._get_gt_sdf_scene(
            mesh_path,
            align_gt_to_slam=align_gt_to_slam,
            gt_traj_path=gt_traj_path,
            align_min_pairs=align_min_pairs,
        )
        if scene is None:
            return None

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        out = np.empty((pts.shape[0],), dtype=np.float32)
        chunk = 500_000
        for start in range(0, pts.shape[0], chunk):
            stop = min(start + chunk, pts.shape[0])
            query = o3d.core.Tensor(pts[start:stop], dtype=o3d.core.Dtype.Float32)
            out[start:stop] = scene.compute_distance(query).numpy().astype(np.float32, copy=False)
        return out

    def compute_gt_signed_distance(
        self,
        points: npt.NDArray,
        mesh_path: str,
        align_gt_to_slam: bool,
        gt_traj_path: str,
        align_min_pairs: int,
    ) -> Optional[npt.NDArray]:
        return self._compute_gt_signed_distance(
            points,
            mesh_path,
            bool(align_gt_to_slam),
            gt_traj_path,
            int(align_min_pairs),
        )

    def compute_gt_surface_distance(
        self,
        points: npt.NDArray,
        mesh_path: str,
        align_gt_to_slam: bool,
        gt_traj_path: str,
        align_min_pairs: int,
    ) -> Optional[npt.NDArray]:
        return self._compute_gt_surface_distance(
            points,
            mesh_path,
            bool(align_gt_to_slam),
            gt_traj_path,
            int(align_min_pairs),
        )

    def compute_gt_projective_sdf(
        self,
        points: npt.NDArray,
        Tcw: npt.NDArray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        mesh_path: str,
        align_gt_to_slam: bool,
        gt_traj_path: str,
        align_min_pairs: int,
    ) -> Optional[npt.NDArray]:
        scene = self._get_gt_sdf_scene(
            mesh_path,
            align_gt_to_slam=bool(align_gt_to_slam),
            gt_traj_path=gt_traj_path,
            align_min_pairs=int(align_min_pairs),
        )
        if scene is None:
            return None

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        Tcw_np = np.asarray(Tcw, dtype=np.float32).reshape(4, 4)
        Rcw = Tcw_np[:3, :3]
        tcw = Tcw_np[:3, 3]
        pts_cam = pts @ Rcw.T + tcw[None, :]
        z = pts_cam[:, 2]

        fx = float(fx)
        fy = float(fy)
        cx = float(cx)
        cy = float(cy)
        width = int(width)
        height = int(height)
        out = np.full((pts.shape[0],), np.nan, dtype=np.float32)
        if pts.shape[0] == 0 or fx <= 1.0e-6 or fy <= 1.0e-6 or width <= 0 or height <= 0:
            return out

        z_safe = np.maximum(z, 1.0e-6)
        u = fx * pts_cam[:, 0] / z_safe + cx
        v = fy * pts_cam[:, 1] / z_safe + cy
        valid = (
            np.isfinite(z) &
            (z > 1.0e-6) &
            (u >= 0.0) & (u <= float(width - 1)) &
            (v >= 0.0) & (v <= float(height - 1))
        )
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return out

        Rwc = Rcw.T
        cam_center_w = -(Rwc @ tcw).astype(np.float32)
        chunk = 250_000
        for start in range(0, valid_idx.size, chunk):
            stop = min(start + chunk, valid_idx.size)
            idx = valid_idx[start:stop]

            dirs_cam = np.stack(
                [
                    (u[idx] - cx) / fx,
                    (v[idx] - cy) / fy,
                    np.ones((idx.shape[0],), dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            dirs_cam_norm = np.linalg.norm(dirs_cam, axis=1, keepdims=True)
            dirs_cam = dirs_cam / np.maximum(dirs_cam_norm, 1.0e-12)
            dirs_world = (dirs_cam @ Rcw).astype(np.float32, copy=False)
            origins = np.repeat(cam_center_w[None, :], idx.shape[0], axis=0)
            rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32, copy=False)

            ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
            t_hit = ans["t_hit"].numpy().astype(np.float32, copy=False)
            hit_valid = np.isfinite(t_hit)
            if not np.any(hit_valid):
                continue

            hit_idx = idx[hit_valid]
            hit_points = origins[hit_valid] + dirs_world[hit_valid] * t_hit[hit_valid, None]
            hit_cam = hit_points @ Rcw.T + tcw[None, :]
            gt_depth = hit_cam[:, 2]
            good_depth = np.isfinite(gt_depth) & (gt_depth > 1.0e-6)
            if np.any(good_depth):
                out[hit_idx[good_depth]] = (
                    gt_depth[good_depth] - z[hit_idx[good_depth]]
                ).astype(np.float32, copy=False)

        return out
