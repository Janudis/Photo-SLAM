# viewer/rerun_visualization/visualizer_wrapper.py

from typing import Dict, Deque, Optional
from collections import deque
import queue

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from .map_visualization import MapVisualizationMixin
from .mesh_analysis import MeshAnalysisMixin

class RerunVisualizer(MapVisualizationMixin, MeshAnalysisMixin):
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
