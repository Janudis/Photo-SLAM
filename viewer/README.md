# Viewer ownership

The two mapping systems have separate viewer implementations:

- `viewer/imgui_viewer.*`, `viewer/map_drawer.*`, and `viewer/drawer_utils.h`
  belong to the original Photo-SLAM Gaussian mapper.
- `include_voxel/viewer/` and `src_voxel/viewer/` belong to the SVRecon voxel
  mapper. Their public entry point is `VoxelImGuiViewer`.
- `viewer/imgui/` is the vendored Dear ImGui dependency shared by both viewers.

For a voxel-only Docker image, copy `include_voxel/viewer/`,
`src_voxel/viewer/`, and `viewer/imgui/`. The Photo-SLAM files in the root of
`viewer/` are not needed unless the Gaussian executables are also built.
The viewer target also requires GLFW, OpenGL, and GLM, plus access to a host
display server at runtime.

The voxel run scripts are headless by default. Set `VOXEL_VIEWER=1` to omit
the `no_viewer` executable argument and open the GLFW viewer:

```bash
VOXEL_VIEWER=1 ./run_tum_voxel.sh
```
