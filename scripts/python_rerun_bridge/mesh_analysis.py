from typing import Optional

import os
import struct

import numpy as np
import numpy.typing as npt


class MeshAnalysisMixin:
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
