from typing import Optional

import numpy as np
import numpy.typing as npt
import rerun as rr


class MapVisualizationMixin:
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
