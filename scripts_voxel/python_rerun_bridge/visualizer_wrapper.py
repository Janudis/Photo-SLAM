# scripts_voxel/python_rerun_bridge/visualizer_wrapper.py

from typing import Dict, Deque, Optional
from collections import deque
import queue

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
import os
import struct

class RerunVisualizer:
    """
    Rerun visualizer for Photo-SLAM + SVRecon.

    This is a thin mesh/voxel logging helper for Photo-SLAM Rerun recordings:
    - world/camera_0: pose, axes, image, observations
    - world/trajectory: camera trajectory
    - world/mesh: color mesh (TSDF or voxel mesh)
    """

    def __init__(self, app_id: str = "PhotoSLAM-SVRecon", spawn: bool = True) -> None:
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
        self._voxel_grid_levels = {}
        self._debug_job_queue = queue.Queue()
        self._main_binary_stream = None
        self._main_recording_bytes_cache = None
        self._app_id = str(app_id)
        self._spawn = bool(spawn)
        self._main_recording_started = False

    def _ensure_main_recording(self) -> None:
        if self._main_recording_started:
            return
        self._start_rerun_visualizer(self._app_id, self._spawn)
        self._main_recording_started = True

    def _drain_debug_jobs(self) -> None:
        while not self._debug_job_queue.empty():
            job = self._debug_job_queue.get()
            try:
                callback, args = job
                callback(*args)
            except Exception as error:
                print(f"[RERUN] Background debug logging failed: {error}")
            finally:
                self._debug_job_queue.task_done()

    def _submit_debug_job(self, callback, *args) -> None:
        self._debug_job_queue.put((callback, args))

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

    def _send_whole_run_blueprint(self) -> None:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        try:
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.TimePanel(state="collapsed"),
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            origin="world",
                            name="Scene 3D",
                        ),
                        rrb.Vertical(
                            rrb.Spatial2DView(
                                origin="depth/model",
                                name="Model Depth",
                            ),
                            rrb.Spatial2DView(
                                origin="depth/ground_truth",
                                name="GT Depth",
                            ),
                        ),
                        column_shares=[3, 1],
                    ),
                ),
                make_active=True,
            )
        except Exception:
            self._send_default_blueprint()

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
            f"PhotoSLAM-SVRecon-{name}",
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
            elif name == "whole_run":
                self._send_whole_run_blueprint()
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
            self._ensure_main_recording()
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
        try:
            # Debug recordings are consumed after the run. Deferring Rerun
            # serialization keeps its CPU work from changing online SLAM/mapping
            # cadence while preserving the exact ordered event timeline.
            self._drain_debug_jobs()
            rec = self._debug_recordings.get(name)
            stream = self._debug_binary_streams.get(name)
            if rec is None or stream is None:
                print(f"[RERUN] Debug recording unsupported, skipping: {name}")
                return
            data = stream.read(flush=True)
            if data is None:
                data = b""
            with open(path, "wb") as f:
                f.write(data)
            print(f"[RERUN] Saved debug recording '{name}' to: {path}")
        except Exception as e:
            print(f"[RERUN] Failed to save debug recording {name} to {path}: {e}")

    # ----------------------------------------------------------------------
    #  Low-level mesh helpers
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
        t_W_C = np.asarray(t_W_C, dtype=np.float32).reshape(3).copy()
        q_W_C_xyzw = np.asarray(q_W_C_xyzw, dtype=np.float32).reshape(4).copy()
        image = np.asarray(image).copy()
        points_uv = (
            None
            if points_uv is None
            else np.asarray(points_uv, dtype=np.float32).copy()
        )
        track_ids = (
            None
            if track_ids is None
            else np.asarray(track_ids).copy()
        )
        self._submit_debug_job(
            self._visualize_cuvslam_recording_now,
            str(recording_name),
            t_W_C,
            q_W_C_xyzw,
            image,
            points_uv,
            track_ids,
            iteration,
            keyframe_id,
            fx,
            fy,
            cx,
            cy,
            source_frame_id,
        )

    def _visualize_cuvslam_recording_now(
        self,
        recording_name: str,
        t_W_C: npt.NDArray,
        q_W_C_xyzw: npt.NDArray,
        image: npt.NDArray,
        points_uv: Optional[npt.NDArray],
        track_ids: Optional[npt.NDArray],
        iteration: Optional[int],
        keyframe_id: Optional[int],
        fx: Optional[float],
        fy: Optional[float],
        cx: Optional[float],
        cy: Optional[float],
        source_frame_id: Optional[int],
    ) -> None:
        rec = self._ensure_debug_recording(recording_name)
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

            self._update_debug_trajectory(
                recording_name, keyframe_id, iteration, t_W_C)

    def _update_debug_trajectory(
        self,
        recording_name: str,
        keyframe_id: Optional[int],
        iteration: Optional[int],
        t_W_C: npt.NDArray,
    ) -> None:
        t_W_C_np = np.asarray(t_W_C, dtype=np.float32).reshape(3)
        poses_by_keyframe = self._debug_trajectory_history.setdefault(
            str(recording_name),
            {},
        )
        trajectory_key = (
            int(keyframe_id)
            if keyframe_id is not None
            else int(iteration) if iteration is not None else len(poses_by_keyframe)
        )
        poses_by_keyframe[trajectory_key] = t_W_C_np.copy()
        ordered_ids = sorted(poses_by_keyframe)
        if self.trajectory_length > 0:
            ordered_ids = ordered_ids[-self.trajectory_length :]
        if ordered_ids:
            trajectory = np.stack(
                [poses_by_keyframe[kf_id] for kf_id in ordered_ids],
                axis=0,
            )
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([trajectory]),
            )

    def visualize_camera_pose_recording(
        self,
        recording_name: str,
        t_W_C: npt.NDArray,
        q_W_C_xyzw: npt.NDArray,
        iteration: Optional[int],
        keyframe_id: int,
    ) -> None:
        self._submit_debug_job(
            self._visualize_camera_pose_recording_now,
            str(recording_name),
            np.asarray(t_W_C, dtype=np.float32).reshape(3).copy(),
            np.asarray(q_W_C_xyzw, dtype=np.float32).reshape(4).copy(),
            iteration,
            int(keyframe_id),
        )

    def _visualize_camera_pose_recording_now(
        self,
        recording_name: str,
        t_W_C: npt.NDArray,
        q_W_C_xyzw: npt.NDArray,
        iteration: Optional[int],
        keyframe_id: int,
    ) -> None:
        rec = self._ensure_debug_recording(recording_name)
        if rec is None:
            return
        with rec:
            self._set_iter_time(iteration)
            kf_name = f"kf_{int(keyframe_id):06d}"
            translation = np.asarray(t_W_C, dtype=np.float32).reshape(3)
            quaternion = np.asarray(q_W_C_xyzw, dtype=np.float32).reshape(4)
            for entity_path in (
                f"world/keyframes/{kf_name}",
                f"world/keyframes_with_images/{kf_name}",
            ):
                rr.log(
                    entity_path,
                    rr.Transform3D(
                        translation=translation,
                        quaternion=quaternion,
                        relation=rr.TransformRelation.ParentFromChild,
                    ),
                )
            self._update_debug_trajectory(
                recording_name, keyframe_id, iteration, t_W_C)

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

    def visualize_ply_mesh(
        self,
        ply_path: str,
        iteration: int,
        entity_path: str = "world/sdf_mesh/live",
        static_mesh: bool = False,
    ):
        """
        Called from C++:
            impl_->visualizer.attr("visualize_ply_mesh")(py::str(ply_path), iteration, entity_path)
        Loads a triangle mesh PLY and logs it to Rerun.
        """
        # print("[RERUN] visualize_ply_mesh called")
        # print("         ply_path:", ply_path)
        # print("         iteration:", iteration)

        if not os.path.exists(ply_path):
            print("[RERUN] visualize_ply_mesh: file does not exist")
            return

        mesh_data = self._load_gt_triangle_mesh(ply_path)
        if mesh_data is None:
            print("[RERUN] visualize_ply_mesh: failed to load triangle mesh")
            return
        vertices, faces, colors = mesh_data

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
        self._visualize_mesh(
            vertices,
            faces,
            colors,
            entity_path=entity_path,
            static=bool(static_mesh),
        )

    def visualize_ply_mesh_recording(
        self,
        recording_name: str,
        ply_path: str,
        iteration: int,
        entity_path: str = "world/mesh/reference",
        static_mesh: bool = False,
    ) -> None:
        rec = self._ensure_debug_recording(str(recording_name))
        if rec is None:
            return
        with rec:
            self.visualize_ply_mesh(
                ply_path,
                iteration,
                entity_path,
                static_mesh=bool(static_mesh),
            )

    def _visualize_mesh(
        self,
        vertices,
        faces,
        colors=None,
        entity_path: str = "tsdf_mesh",
        static: bool = False,
    ):
        """
        Visualize a triangular mesh in Rerun.

        Called as: self._visualize_mesh(vertices, faces, colors, entity_path=...)
        """

        # Ensure CPU numpy arrays:
        import torch

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
        rr.log(entity_path, mesh, static=bool(static))

    def visualize_image_recording(
        self,
        recording_name: str,
        image: npt.NDArray,
        entity_path: str,
        iteration: int,
        keyframe_id: int,
    ) -> None:
        image_copy = np.asarray(image, dtype=np.uint8).copy()
        self._submit_debug_job(
            self._visualize_image_recording_now,
            str(recording_name),
            image_copy,
            str(entity_path),
            int(iteration),
            int(keyframe_id),
        )

    def _visualize_image_recording_now(
        self,
        recording_name: str,
        image: npt.NDArray,
        entity_path: str,
        iteration: int,
        keyframe_id: int,
    ) -> None:
        rec = self._ensure_debug_recording(recording_name)
        if rec is None:
            return
        image_rgb = np.asarray(image, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return
        with rec:
            self._set_iter_time(iteration)
            self._set_keyframe_time(keyframe_id)
            rr.log(entity_path, rr.Image(np.ascontiguousarray(image_rgb)).compress())

    def visualize_triangle_mesh(
        self,
        vertices: npt.NDArray,
        colors: Optional[npt.NDArray],
        triangles: npt.NDArray,
        iteration: Optional[int] = None,
    ) -> None:
        """
        Visualize a triangle mesh.

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

        self._visualize_mesh(v, f, c, entity_path="world/sdf_mesh/live")

    def visualize_triangle_mesh_recording(
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
        metadata: Optional[dict] = None,
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

        metadata_values = {}
        if metadata is not None:
            for key, value in dict(metadata).items():
                arr = np.asarray(value)
                if arr.shape[0] == N:
                    metadata_values[str(key)] = arr
                else:
                    print(f"[PY][voxels] metadata '{key}' length mismatch, ignoring it")

        # Downsample if too many
        # print(f"[PY][voxels] visualizing {N} boxes")
        if max_boxes is not None and int(max_boxes) > 0 and N > int(max_boxes):
            print(f"[PY][voxels] too many boxes ({N}), downsampling to {max_boxes}")
            idx = np.linspace(0, N - 1, int(max_boxes), dtype=np.int64)
            centers = centers[idx]
            half_sizes = half_sizes[idx]
            if colors is not None:
                colors = colors[idx]
            if metadata_values:
                metadata_values = {
                    key: np.asarray(value)[idx]
                    for key, value in metadata_values.items()
                }
            N = int(max_boxes)

        labels = None
        if metadata_values:
            label_parts = []
            for key in sorted(metadata_values.keys()):
                values = np.asarray(metadata_values[key]).reshape(-1)
                label_parts.append((key, values))
            labels = [
                " ".join(f"{key}={values[i]}" for key, values in label_parts)
                for i in range(N)
            ]

        # print(
        #     f"[PY][voxels] logging {N} boxes to '{entity_path}' "
        #     f"center_min={centers.min(axis=0)}, center_max={centers.max(axis=0)}, "
        #     f"hs_min={half_sizes.min()}, hs_max={half_sizes.max()}"
        # )

        box_kwargs = dict(
            centers=centers,
            half_sizes=half_sizes,
            colors=colors,
            fill_mode="solid",
        )
        if labels is not None:
            box_kwargs["labels"] = labels
        rr.log(entity_path, rr.Boxes3D(**box_kwargs))

    # Path planning
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
        points_copy = np.asarray(points_xyz, dtype=np.float32).copy()
        colors_copy = None if colors is None else np.asarray(colors).copy()
        labels_copy = None if labels is None else list(labels)
        self._submit_debug_job(
            self._visualize_points3d_recording_now,
            str(recording_name),
            points_copy,
            colors_copy,
            float(radii),
            str(entity_path),
            iteration,
            labels_copy,
        )

    def _visualize_points3d_recording_now(
        self,
        recording_name: str,
        points_xyz: npt.NDArray,
        colors: Optional[npt.NDArray],
        radii: float,
        entity_path: str,
        iteration: Optional[int],
        labels: Optional[list[str]],
    ) -> None:
        rec = self._ensure_debug_recording(recording_name)
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
        metadata: Optional[dict] = None,
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
                metadata=metadata,
            )

    def visualize_voxel_grid_map_recording(
        self,
        recording_name: str,
        centers: npt.NDArray,
        sizes: npt.NDArray,
        levels: npt.NDArray,
        colors: Optional[npt.NDArray],
        grid_origin: npt.NDArray,
        entity_path: str,
        iteration: Optional[int],
        opacity: float = 0.8,
    ) -> None:
        if not hasattr(rr, "VoxelGridMap"):
            raise RuntimeError(
                "rerun-sdk with VoxelGridMap support is required "
                "(Photo-SLAM pins version 0.34.1)"
            )

        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3).copy()
        sizes = np.asarray(sizes, dtype=np.float32).reshape(-1).copy()
        levels = np.asarray(levels, dtype=np.int32).reshape(-1).copy()
        origin = np.asarray(grid_origin, dtype=np.float32).reshape(3).copy()
        if sizes.shape[0] != centers.shape[0] or levels.shape[0] != centers.shape[0]:
            raise ValueError("VoxelGridMap centers, sizes, and levels must have equal length")

        rgba = None
        if colors is not None:
            rgba = np.asarray(colors)
            if rgba.dtype != np.uint8:
                rgba = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgba = rgba.reshape(centers.shape[0], -1)
            if rgba.shape[1] not in (3, 4):
                raise ValueError("VoxelGridMap colors must have three or four channels")
            if rgba.shape[1] == 4:
                # VoxelGridMap opacity is set explicitly below. Keep learned/fused
                # RGB here so per-color alpha is not multiplied into it a second time.
                rgba = rgba[:, :3]
            rgba = rgba.copy()

        self._submit_debug_job(
            self._visualize_voxel_grid_map_recording_now,
            str(recording_name),
            centers,
            sizes,
            levels,
            rgba,
            origin,
            str(entity_path),
            iteration,
            float(opacity),
        )

    def _visualize_voxel_grid_map_recording_now(
        self,
        recording_name: str,
        centers: npt.NDArray,
        sizes: npt.NDArray,
        levels: npt.NDArray,
        rgba: Optional[npt.NDArray],
        origin: npt.NDArray,
        entity_path: str,
        iteration: Optional[int],
        opacity: float,
    ) -> None:
        rec = self._ensure_debug_recording(recording_name)
        if rec is None:
            return
        state_key = (recording_name, entity_path)
        previous_levels = self._voxel_grid_levels.get(state_key, set())
        current_levels = set(int(level) for level in np.unique(levels))

        with rec:
            self._set_iter_time(iteration)
            for stale_level in sorted(previous_levels - current_levels):
                rr.log(
                    f"{entity_path}/level_{stale_level}",
                    rr.VoxelGridMap.cleared(),
                )

            for level in sorted(current_levels):
                mask = levels == level
                level_centers = centers[mask]
                level_sizes = sizes[mask]
                voxel_size = float(np.median(level_sizes))
                if not np.isfinite(voxel_size) or voxel_size <= 0.0:
                    raise ValueError(f"Invalid voxel size at octree level {level}")
                tolerance = max(1.0e-6, 1.0e-4 * voxel_size)
                if np.max(np.abs(level_sizes - voxel_size), initial=0.0) > tolerance:
                    raise ValueError(
                        f"Octree level {level} contains inconsistent voxel sizes"
                    )

                grid_coords = (level_centers - origin[None, :]) / voxel_size - 0.5
                voxel_indices = np.rint(grid_coords).astype(np.int32)
                reconstructed = (
                    origin[None, :] +
                    (voxel_indices.astype(np.float32) + 0.5) * voxel_size
                )
                max_error = float(
                    np.max(np.abs(reconstructed - level_centers), initial=0.0)
                )
                if max_error > tolerance:
                    raise ValueError(
                        f"Voxel centers at level {level} are not aligned to the "
                        f"scene grid (max error {max_error:g} m)"
                    )

                level_colors = rgba[mask] if rgba is not None else None
                rr.log(
                    f"{entity_path}/level_{level}",
                    rr.VoxelGridMap(
                        voxel_indices,
                        voxel_size=[voxel_size, voxel_size, voxel_size],
                        colors=level_colors,
                        translation=origin,
                        opacity=float(np.clip(opacity, 0.0, 1.0)),
                    ),
                )

        self._voxel_grid_levels[state_key] = current_levels

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

    @staticmethod
    def _load_tum_trajectory_with_timestamps(path: str):
        if not path or not os.path.exists(path):
            return None
        timestamps = []
        centers = []
        with open(path, "r", encoding="utf-8") as trajectory:
            for line in trajectory:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 8:
                    continue
                try:
                    values = [float(value) for value in fields[:8]]
                except ValueError:
                    continue
                if not np.all(np.isfinite(values[:4])):
                    continue
                timestamps.append(values[0])
                centers.append(values[1:4])
        if len(timestamps) < 4:
            return None
        order = np.argsort(np.asarray(timestamps, dtype=np.float64))
        return (
            np.asarray(timestamps, dtype=np.float64)[order],
            np.asarray(centers, dtype=np.float64)[order],
        )

    @staticmethod
    def _timestamp_matched_centers(source, target):
        source_t, source_xyz = source
        target_t, target_xyz = target
        positive_steps = np.diff(source_t)
        positive_steps = positive_steps[positive_steps > 1.0e-9]
        median_step = (
            float(np.median(positive_steps))
            if positive_steps.size else 1.0
        )
        tolerance = max(1.0e-6, 0.51 * median_step)

        source_matches = []
        target_matches = []
        used_source = set()
        for timestamp, target_center in zip(target_t, target_xyz):
            right = int(np.searchsorted(source_t, timestamp))
            candidates = []
            if right < source_t.size:
                candidates.append(right)
            if right > 0:
                candidates.append(right - 1)
            if not candidates:
                continue
            index = min(candidates, key=lambda i: abs(source_t[i] - timestamp))
            if index in used_source or abs(source_t[index] - timestamp) > tolerance:
                continue
            used_source.add(index)
            source_matches.append(source_xyz[index])
            target_matches.append(target_center)

        if len(source_matches) < 4:
            return None
        return (
            np.asarray(source_matches, dtype=np.float64),
            np.asarray(target_matches, dtype=np.float64),
            tolerance,
        )

    @staticmethod
    def _save_ascii_triangle_mesh(
        path: str,
        vertices: npt.NDArray,
        faces: npt.NDArray,
        colors: Optional[npt.NDArray],
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        has_colors = colors is not None and len(colors) == len(vertices)
        if has_colors:
            colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii") as mesh_file:
            mesh_file.write("ply\nformat ascii 1.0\n")
            mesh_file.write(f"element vertex {len(vertices)}\n")
            mesh_file.write("property float x\nproperty float y\nproperty float z\n")
            if has_colors:
                mesh_file.write(
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                )
            mesh_file.write(f"element face {len(faces)}\n")
            mesh_file.write("property list uchar int vertex_indices\nend_header\n")
            if has_colors:
                for vertex, color in zip(vertices, colors):
                    mesh_file.write(
                        f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                        f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                    )
            else:
                for vertex in vertices:
                    mesh_file.write(
                        f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n"
                    )
            for face in faces:
                mesh_file.write(
                    f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n"
                )

    def align_reference_ply_mesh(
        self,
        source_mesh_path: str,
        source_trajectory_tum_path: str,
        target_trajectory_tum_path: str,
        output_mesh_path: str,
        report_path: str,
    ) -> bool:
        source_trajectory = self._load_tum_trajectory_with_timestamps(
            source_trajectory_tum_path
        )
        target_trajectory = self._load_tum_trajectory_with_timestamps(
            target_trajectory_tum_path
        )
        if source_trajectory is None or target_trajectory is None:
            print("[RERUN/nvblox] missing or insufficient trajectory poses")
            return False

        matched = self._timestamp_matched_centers(
            source_trajectory, target_trajectory
        )
        if matched is None:
            print("[RERUN/nvblox] fewer than four timestamp-matched poses")
            return False
        source_centers, target_centers, tolerance = matched
        similarity = self._estimate_similarity_umeyama(
            source_centers, target_centers
        )
        if similarity is None:
            print("[RERUN/nvblox] trajectory Sim(3) estimation failed")
            return False
        scale, rotation, translation = similarity

        mesh = self._load_gt_triangle_mesh(source_mesh_path)
        if mesh is None:
            print(f"[RERUN/nvblox] failed to load mesh: {source_mesh_path}")
            return False
        vertices, faces, colors = mesh
        aligned_vertices = (
            scale * (np.asarray(vertices, dtype=np.float64) @ rotation.T)
            + translation[None, :]
        ).astype(np.float32)
        self._save_ascii_triangle_mesh(
            output_mesh_path, aligned_vertices, faces, colors
        )

        transformed_centers = scale * (source_centers @ rotation.T) + translation
        residuals = np.linalg.norm(transformed_centers - target_centers, axis=1)
        rmse = float(np.sqrt(np.mean(residuals * residuals)))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as report:
            report.write(f"source_mesh {source_mesh_path}\n")
            report.write(f"source_trajectory {source_trajectory_tum_path}\n")
            report.write(f"target_trajectory {target_trajectory_tum_path}\n")
            report.write(f"pairs {len(source_centers)}\n")
            report.write(f"timestamp_tolerance {tolerance:.9g}\n")
            report.write(f"scale {scale:.12g}\n")
            report.write("rotation\n")
            for row in rotation:
                report.write(" ".join(f"{value:.12g}" for value in row) + "\n")
            report.write(
                "translation "
                + " ".join(f"{value:.12g}" for value in translation)
                + "\n"
            )
            report.write(f"trajectory_rmse {rmse:.12g}\n")
        print(
            f"[RERUN/nvblox] aligned mesh: pairs={len(source_centers)} "
            f"scale={scale:.6g} trajectory_rmse={rmse:.6g} -> "
            f"{output_mesh_path}"
        )
        return True

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

        import open3d as o3d

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

    def _load_ascii_ply_mesh(self, mesh_path: str):
        with open(mesh_path, "r", encoding="ascii", errors="replace") as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    return None
                header_lines.append(line.strip())
                if header_lines[-1] == "end_header":
                    break

            if len(header_lines) < 2 or header_lines[0] != "ply":
                return None
            if header_lines[1] != "format ascii 1.0":
                return None

            vertex_count = 0
            face_count = 0
            vertex_properties = []
            element = None
            for line in header_lines:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "element":
                    element = parts[1]
                    if element == "vertex":
                        vertex_count = int(parts[2])
                    elif element == "face":
                        face_count = int(parts[2])
                elif (
                    element == "vertex" and len(parts) == 3 and
                    parts[0] == "property"
                ):
                    vertex_properties.append(parts[2])

            required = ("x", "y", "z")
            if vertex_count <= 0 or face_count <= 0 or not all(
                name in vertex_properties for name in required
            ):
                return None

            property_index = {
                name: index for index, name in enumerate(vertex_properties)
            }
            vertices = np.empty((vertex_count, 3), dtype=np.float32)
            has_colors = all(
                name in property_index for name in ("red", "green", "blue")
            )
            colors = (
                np.empty((vertex_count, 3), dtype=np.uint8)
                if has_colors else None
            )
            for row in range(vertex_count):
                values = f.readline().split()
                if len(values) < len(vertex_properties):
                    return None
                vertices[row] = [
                    float(values[property_index[name]]) for name in required
                ]
                if colors is not None:
                    colors[row] = [
                        int(values[property_index[name]])
                        for name in ("red", "green", "blue")
                    ]

            faces = []
            for _ in range(face_count):
                values = f.readline().split()
                if not values:
                    return None
                count = int(values[0])
                if len(values) < count + 1:
                    return None
                indices = [int(value) for value in values[1:count + 1]]
                for index in range(1, count - 1):
                    faces.append((indices[0], indices[index], indices[index + 1]))

            if not faces:
                return None
            return vertices, np.asarray(faces, dtype=np.int32), colors

    def _load_gt_triangle_mesh(self, mesh_path: str):
        # Replica meshes are binary PLYs with polygon faces; Open3D can reject
        # them before triangulation, so use a small local triangulating loader.
        if mesh_path.lower().endswith(".ply"):
            mesh_data = self._load_binary_little_endian_ply_mesh(mesh_path)
            if mesh_data is None:
                mesh_data = self._load_ascii_ply_mesh(mesh_path)
            if mesh_data is not None:
                return mesh_data

        import open3d as o3d

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

        import open3d as o3d

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

        import open3d as o3d

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

        import open3d as o3d

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
