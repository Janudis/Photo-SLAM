cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(svr_slam LANGUAGES CXX CUDA)

option(PHOTOSLAM_ENABLE_RERUN "Build Rerun diagnostics" ON)
option(PHOTOSLAM_BUILD_REALSENSE "Build the live RealSense RGB-D runner" ON)
set(PHOTOSLAM_CUDA_ARCHITECTURES "75;86" CACHE STRING "CUDA architectures for project targets")
set(CMAKE_CUDA_ARCHITECTURES "${PHOTOSLAM_CUDA_ARCHITECTURES}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CAFFE2_USE_CUDNN ON)

if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES)

set(LIBTORCH_ROOT "" CACHE PATH "Root of an unpacked LibTorch distribution")
if(NOT LIBTORCH_ROOT AND DEFINED ENV{LIBTORCH_ROOT})
    set(LIBTORCH_ROOT "$ENV{LIBTORCH_ROOT}")
endif()
if(LIBTORCH_ROOT)
    if(NOT EXISTS "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
        message(FATAL_ERROR "Invalid LIBTORCH_ROOT: ${LIBTORCH_ROOT}")
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
    set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake" FORCE)
endif()
if(NOT Torch_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE PHOTOSLAM_TORCH_PREFIX_RESULT
        OUTPUT_VARIABLE PHOTOSLAM_TORCH_PREFIX
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PHOTOSLAM_TORCH_PREFIX_RESULT EQUAL 0 AND PHOTOSLAM_TORCH_PREFIX)
        list(PREPEND CMAKE_PREFIX_PATH "${PHOTOSLAM_TORCH_PREFIX}")
    endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

set(PHOTOSLAM_TORCH_LIBRARY_HINTS)
if(TORCH_INSTALL_PREFIX)
    list(APPEND PHOTOSLAM_TORCH_LIBRARY_HINTS "${TORCH_INSTALL_PREFIX}/lib")
endif()
if(Torch_DIR)
    get_filename_component(PHOTOSLAM_TORCH_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
    list(APPEND PHOTOSLAM_TORCH_LIBRARY_HINTS "${PHOTOSLAM_TORCH_ROOT}/lib")
endif()
find_library(TORCH_PYTHON_LIBRARY NAMES torch_python HINTS ${PHOTOSLAM_TORCH_LIBRARY_HINTS} REQUIRED)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

if(PHOTOSLAM_BUILD_REALSENSE)
    find_package(realsense2 CONFIG QUIET)
    if(TARGET realsense2::realsense2 OR realsense2_FOUND)
        set(PHOTOSLAM_REALSENSE_AVAILABLE ON)
    endif()
endif()

set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS})

set(SVRECON_NAMESPACE_REMAP
    PREPROCESS=SVRECON_PREPROCESS
    FORWARD=SVRECON_FORWARD
    BACKWARD=SVRECON_BACKWARD
    RASTER_STATE=SVRECON_RASTER_STATE
    GEO_PARAMS_GATHER=SVRECON_GEO_PARAMS_GATHER
    SH_COMPUTE=SVRECON_SH_COMPUTE
    TV_COMPUTE=SVRECON_TV_COMPUTE
    VG_COMPUTE=SVRECON_VG_COMPUTE
    GE_COMPUTE=SVRECON_GE_COMPUTE
    LS_COMPUTE=SVRECON_LS_COMPUTE
    PL_COMPUTE=SVRECON_PL_COMPUTE
    UTILS=SVRECON_UTILS
    ADAM_STEP=SVRECON_ADAM_STEP)

add_library(svrecon_cuda_core STATIC
    cuda_rasterizer/raster_state.cu
    cuda_rasterizer/preprocess.cu
    cuda_rasterizer/forward.cu
    cuda_rasterizer/backward.cu
    cuda_rasterizer/geo_params_gather.cu
    cuda_rasterizer/sh_compute.cu
    cuda_rasterizer/tv_compute.cu
    cuda_rasterizer/vg_compute.cu
    cuda_rasterizer/ge_compute.cu
    cuda_rasterizer/ls_compute.cu
    cuda_rasterizer/pl_compute.cu
    cuda_rasterizer/utils.cu
    cuda_rasterizer/adam_step.cu)
set_target_properties(svrecon_cuda_core PROPERTIES CUDA_ARCHITECTURES "${PHOTOSLAM_CUDA_ARCHITECTURES}")
target_compile_definitions(svrecon_cuda_core PRIVATE ${SVRECON_NAMESPACE_REMAP})
target_include_directories(svrecon_cuda_core PUBLIC
    ${PROJECT_SOURCE_DIR}/cuda_rasterizer
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS})
target_link_libraries(svrecon_cuda_core PUBLIC "${TORCH_LIBRARIES}")

add_library(imgui SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

add_library(tandem_mvs_runtime STATIC
    include_voxel/tandem_mvs_runtime.h
    src_voxel/tandem_mvs_runtime.cpp)
target_include_directories(tandem_mvs_runtime PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include_voxel)
target_link_libraries(tandem_mvs_runtime PRIVATE "${TORCH_LIBRARIES}" boost_system boost_thread)

add_library(voxel_core
    include_voxel/monocular_densification.cu
    src_voxel/mapper_depth_registry.cpp
    src_voxel/mesh_extraction.cpp
    src_voxel/rerun_utils.cpp
    src_voxel/rerun_utils_debug.cpp
    src_voxel/rerun_utils_mesh.cpp
    src_voxel/tandem_mvs_backend.cpp
    src_voxel/voxel_keyframe.cpp
    src_voxel/voxel_mapper.cpp
    src_voxel/voxel_mapper_config.cpp
    src_voxel/voxel_mapper_io.cpp
    src_voxel/voxel_mapper_mapping.cpp
    src_voxel/voxel_mapper_monocular_densification.cpp
    src_voxel/voxel_mapper_monocular_mvs_tsdf_evidence.cpp
    src_voxel/voxel_mapper_rerun.cpp
    src_voxel/voxel_mapper_rerun_map.cpp
    src_voxel/voxel_mapper_rerun_voxels.cpp
    src_voxel/voxel_mapper_rgbd.cpp
    src_voxel/voxel_mapper_rgbd_tsdf_evidence.cpp
    src_voxel/voxel_mapper_run.cpp
    src_voxel/voxel_mapper_state.cpp
    src_voxel/voxel_mapper_supervision.cpp
    src_voxel/voxel_mapper_supervision_monocular.cpp
    src_voxel/voxel_mapper_supervision_rgbd.cpp
    src_voxel/voxel_mapper_supervision_visualization.cpp
    src_voxel/voxel_mapper_tandem_mvs.cpp
    src_voxel/voxel_mapper_tandem_mvs_evidence.cpp
    src_voxel/voxel_mapper_topology_pruning.cpp
    src_voxel/voxel_mapper_topology_subdivision.cpp
    src_voxel/voxel_mapper_training.cpp
    src_voxel/voxel_mapper_tsdf.cpp
    src_voxel/voxel_mapper_utils.cpp
    src_voxel/voxel_model.cpp
    src_voxel/voxel_model_creation.cpp
    src_voxel/voxel_model_insertion.cpp
    src_voxel/voxel_model_io.cpp
    src_voxel/voxel_model_optimization.cpp
    src_voxel/voxel_model_regularization.cpp
    src_voxel/voxel_model_sdf.cpp
    src_voxel/voxel_parameters.cpp
    src_voxel/voxel_scene.cpp
    src_voxel/voxel_svrecon_rasterizer.cpp
    src_voxel/voxel_trainer.cpp
    third_party/tinyply/tinyply.cpp
    third_party/tinyply/tinyply.h)
set_target_properties(voxel_core PROPERTIES CUDA_ARCHITECTURES "${PHOTOSLAM_CUDA_ARCHITECTURES}")
target_include_directories(voxel_core PUBLIC
    include_voxel
    ${Python3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus)
target_link_libraries(voxel_core PUBLIC
    svrecon_cuda_core
    "${TORCH_LIBRARIES}"
    Python3::Python
    pybind11::embed
    ${OpenCV_LIBRARIES}
    tandem_mvs_runtime
    glog::glog)
target_compile_definitions(voxel_core PUBLIC PHOTOSLAM_ENABLE_RERUN=$<BOOL:${PHOTOSLAM_ENABLE_RERUN}>)

add_library(voxel_viewer SHARED
    include_voxel/viewer/voxel_drawer_utils.h
    include_voxel/viewer/voxel_imgui_viewer.h
    include_voxel/viewer/voxel_map_drawer.h
    src_voxel/viewer/voxel_imgui_viewer.cpp
    src_voxel/viewer/voxel_map_drawer.cpp)
target_include_directories(voxel_viewer PUBLIC ${PROJECT_SOURCE_DIR} ${Python3_INCLUDE_DIRS})
target_link_libraries(voxel_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    voxel_core
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

function(add_voxel_runner target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include_voxel pybind11::embed)
    target_link_libraries(${target}
        voxel_core
        voxel_viewer
        ${OpenCV_LIBRARIES}
        "${TORCH_LIBRARIES}"
        ${TORCH_PYTHON_LIBRARY}
        ${Python3_LIBRARIES}
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)
endfunction()

add_voxel_runner(replica_mono_voxel examples/replica_mono_voxel.cpp)
add_voxel_runner(replica_rgbd_voxel examples/replica_rgbd_voxel.cpp)
add_voxel_runner(tum_mono_voxel examples/tum_mono_voxel.cpp)
add_voxel_runner(tum_rgbd_voxel examples/tum_rgbd_voxel.cpp)
add_voxel_runner(scannet_mono_voxel examples/tum_mono_voxel.cpp)
add_voxel_runner(scannet_rgbd_voxel examples/tum_rgbd_voxel.cpp)

if(PHOTOSLAM_BUILD_REALSENSE)
    if(PHOTOSLAM_REALSENSE_AVAILABLE)
        add_voxel_runner(realsense_rgbd_voxel examples/realsense_rgbd_voxel.cpp)
        target_include_directories(realsense_rgbd_voxel PUBLIC ${realsense2_INCLUDE_DIR})
        target_link_libraries(realsense_rgbd_voxel realsense2::realsense2)
    else()
        message(WARNING "librealsense2 was not found; realsense_rgbd_voxel will not be built")
    endif()
endif()
