cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(photo-slam LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)
# FetchContent_Declare(rerun_sdk URL https://github.com/rerun-io/rerun/releases/download/0.24.1/rerun_cpp_sdk.zip)
FetchContent_Declare(rerun_sdk URL
https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CAFFE2_USE_CUDNN on)

# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")

# ------------------------------------------------------------ Python
# 1.  Tell CMake where to look **before** the first find_package().
#    (Conda installs everything under $CONDA_PREFIX.)
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# 2.  Now ask for the interpreter **and** the dev files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES)

# Path to your local LibTorch installation
set(CMAKE_PREFIX_PATH "/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch")
# Optional: explicitly set Torch_DIR, though not always needed with CMAKE_PREFIX_PATH
set(Torch_DIR "/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

# set(ENV{CUDA_LAUNCH_BLOCKING} 1)
# set(ENV{TORCH_USE_CUDA_DSA} 1)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

find_package(realsense2)

find_package(glog REQUIRED)

find_package(ompl REQUIRED)

set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# ------------------------------------------------------------------
# nvblox (TSDF/ESDF) – already built in third_party/nvblox/build
# ------------------------------------------------------------------
set(NVBLOX_ROOT_DIR  "${PROJECT_SOURCE_DIR}/third_party/nvblox")
set(NVBLOX_BUILD_DIR "${NVBLOX_ROOT_DIR}/build")
set(NVBLOX_SRC_DIR   "${NVBLOX_ROOT_DIR}/nvblox")

# Headers: third_party/nvblox/nvblox/include
set(NVBLOX_INCLUDE_DIR "${NVBLOX_SRC_DIR}/include")

# Library: third_party/nvblox/build/nvblox/libnvblox_lib.so
set(NVBLOX_LIB_DIR "${NVBLOX_BUILD_DIR}/nvblox")

# Make headers visible to all targets
include_directories(${NVBLOX_INCLUDE_DIR})

# Locate the core nvblox library built above
find_library(NVBLOX_LIB
    NAMES nvblox_lib
    PATHS "${NVBLOX_LIB_DIR}"
    NO_DEFAULT_PATH
)

if(NOT NVBLOX_LIB)
    message(FATAL_ERROR "Could not find nvblox_lib in ${NVBLOX_LIB_DIR}")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS})

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

# add_library(simple_knn SHARED
#     third_party/simple-knn/simple_knn.cu
#     third_party/simple-knn/simple_knn.h
#     third_party/simple-knn/spatial.cu
#     third_party/simple-knn/spatial.h)
# # target_compile_features(simple_knn PUBLIC cxx_std_17)
# target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

# add_library(cuda_rasterizer SHARED
#     include/operate_points.h
#     src/operate_points.cu
#     include/rasterize_points.h
#     src/rasterize_points.cu
#     include/stereo_vision.h
#     src/stereo_vision.cu
#     cuda_rasterizer/auxiliary.h
#     cuda_rasterizer/backward.cu
#     cuda_rasterizer/backward.h
#     cuda_rasterizer/config.h
#     cuda_rasterizer/forward.cu
#     cuda_rasterizer/forward.h
#     cuda_rasterizer/operate_points.h
#     cuda_rasterizer/rasterizer.h
#     cuda_rasterizer/rasterizer_impl.cu
#     cuda_rasterizer/rasterizer_impl.h
#     cuda_rasterizer/stereo_vision.h)
# set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
# # target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
# target_include_directories(cuda_rasterizer PRIVATE
#     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# target_link_libraries(cuda_rasterizer
#     glm::glm
#     "${TORCH_LIBRARIES}"
#     Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

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

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

# add_library(gaussian_mapper SHARED
#     third_party/tinyply/tinyply.h
#     third_party/tinyply/tinyply.cpp
#     include/gaussian_keyframe.h
#     include/gaussian_model.h
#     include/gaussian_parameters.h
#     include/gaussian_rasterizer.h
#     include/gaussian_renderer.h
#     include/gaussian_scene.h
#     include/gaussian_trainer.h
#     include/gaussian_mapper.h
#     include/general_utils.h
#     include/graphics_utils.h
#     include/loss_utils.h
#     include/sh_utils.h
#     include/tensor_utils.h
#     include/camera.h
#     include/point_cloud.h
#     include/point2d.h
#     include/point3d.h
#     include/types.h
#     src/gaussian_keyframe.cpp
#     src/gaussian_model.cpp
#     src/gaussian_parameters.cpp
#     src/gaussian_rasterizer.cpp
#     src/gaussian_renderer.cpp
#     src/gaussian_scene.cpp
#     src/gaussian_trainer.cpp
#     src/gaussian_mapper.cpp)
# target_link_libraries(gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
#     ${OpenCV_LIBRARIES}
#     jsoncpp_lib
#     "${TORCH_LIBRARIES}"
#     Eigen3::Eigen
#     simple_knn
#     cuda_rasterizer
#     ${Python3_LIBRARIES})

# target_include_directories(gaussian_mapper
#     PRIVATE
#         ${Python3_INCLUDE_DIRS}
# )

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

# add_library(gaussian_viewer SHARED
#     viewer/drawer_utils.h
#     viewer/imgui_viewer.cpp
#     viewer/imgui_viewer.h
#     viewer/map_drawer.cpp
#     viewer/map_drawer.h)
# target_link_libraries(gaussian_viewer
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
#     # gaussian_mapper
#     voxel_core
#     voxel_rasterizer
#     imgui
#     ${OpenCV_LIBRARIES}
#     jsoncpp_lib
#     "${TORCH_LIBRARIES}"
#     glm::glm
#     glfw
#     OpenGL::GL)

# target_include_directories(gaussian_viewer
#     PUBLIC
#     ${Python3_INCLUDE_DIRS}          # <--  add this line
# )
##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# This is a C++ libtorch implementation of gaussian-splatting (https://github.com/graphdeco-inria/gaussian-splatting)
# add_executable(train_colmap examples/train_colmap.cpp)
# target_link_libraries(train_colmap
#     gaussian_viewer
#     gaussian_mapper)

# add_executable(view_result examples/view_result.cpp)
# target_link_libraries(view_result
#     gaussian_viewer
#     gaussian_mapper)

###############################################################################<
# 3.  🧱 SPARSE‑VOXEL  – new code
###############################################################################

# ### 3.1  CUDA rasteriser  --------------------------------------------
# file (GLOB VX_SRCS cuda_voxel_rasterizer/*.cu cuda_voxel_rasterizer/*.cpp)
# add_library (voxel_rasterizer SHARED ${VX_SRCS})
# set_target_properties (voxel_rasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86;89")
# target_include_directories (voxel_rasterizer
#         PRIVATE
#             ${TORCH_INCLUDE_DIRS}
#             ${Python3_INCLUDE_DIRS})
# target_link_libraries (voxel_rasterizer "${TORCH_LIBRARIES}")

### 3.2  C++ core / optimiser  ---------------------------------------
add_library(voxel_core
        src_voxel/voxel_trainer.cpp
        src_voxel/voxel_mapper.cpp
        src_voxel/voxel_parameters.cpp
        src_voxel/voxel_scene.cpp
        src_voxel/voxel_keyframe.cpp
        src_voxel/voxel_model.cpp
        src_voxel/rerun_utils.cpp
        ${NVBLOX_SRC_DIR}/src/io/mesh_io.cpp
        third_party/tinyply/tinyply.h
        third_party/tinyply/tinyply.cpp
        src_voxel/voxel_planner.cpp)
        
target_include_directories(voxel_core
        PUBLIC
            include_voxel
            ${Python3_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
            ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
            ${OMPL_INCLUDE_DIRS} )
target_link_libraries(voxel_core
        PUBLIC
            # voxel_rasterizer           # call kernels directly if needed
            "${TORCH_LIBRARIES}"
            Python3::Python
            pybind11::embed
            ${OpenCV_LIBRARIES}
            rerun_sdk
            ${NVBLOX_LIB}
            glog::glog
            ${OMPL_LIBRARIES})

add_library(voxel_viewer SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h)

target_link_libraries(voxel_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    voxel_core      # <-- Make sure voxel_core is linked
    # voxel_rasterizer # <-- Ensure voxel_rasterizer is also linked
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

target_include_directories(voxel_viewer
    PUBLIC
    ${Python3_INCLUDE_DIRS}  # <-- Keep this if needed for Python integration
)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# # Replica Monocular
# add_executable(replica_mono examples/replica_mono.cpp)
# target_link_libraries(replica_mono
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# # Replica Monocular
# add_executable(replica_rgbd examples/replica_rgbd.cpp)
# target_link_libraries(replica_rgbd
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(replica_mono_voxel examples/replica_mono_voxel.cpp)
target_include_directories(replica_mono_voxel PRIVATE
        ${CMAKE_SOURCE_DIR}/include_voxel      # <‑‑ header path
        pybind11::embed
)
target_link_libraries(replica_mono_voxel
        voxel_core          # lib that implements LiveVoxelTrainer
        voxel_viewer
        # gaussian_viewer
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        torch_python
        ${Python3_LIBRARIES}
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)
target_link_directories(replica_mono_voxel PRIVATE /home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib)

set_target_properties(voxel_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# # TUM Monocular
# add_executable(tum_mono examples/tum_mono.cpp)
# target_link_libraries(tum_mono
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(tum_mono_voxel examples/tum_mono_voxel.cpp)
target_include_directories(tum_mono_voxel PRIVATE
        ${CMAKE_SOURCE_DIR}/include_voxel      # <‑‑ header path
        pybind11::embed
)
target_link_libraries(tum_mono_voxel
        voxel_core          # lib that implements LiveVoxelTrainer
        voxel_viewer
        # gaussian_viewer
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        torch_python
        ${Python3_LIBRARIES}
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)
target_link_directories(tum_mono_voxel PRIVATE /home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib)

set_target_properties(voxel_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(simple_knn PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(cuda_rasterizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

# TUM RGBD
add_executable(tum_rgbd_voxel examples/tum_rgbd_voxel.cpp)
target_link_libraries(tum_rgbd_voxel
        voxel_core          # lib that implements LiveVoxelTrainer
        voxel_viewer
        # gaussian_viewer
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        torch_python
        ${Python3_LIBRARIES}
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)
target_link_directories(tum_rgbd_voxel PRIVATE /home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib)
set_target_properties(voxel_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SplatNav / statues monocular (transforms.json -> rgb.txt -> ORB-SLAM3 TrackMonocular)
add_executable(statues_voxel examples/statues_voxel.cpp)
target_include_directories(statues_voxel PRIVATE
        ${CMAKE_SOURCE_DIR}/include_voxel
        ${Python3_INCLUDE_DIRS}
)
target_link_libraries(statues_voxel
        voxel_core
        voxel_viewer
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        torch_python
        ${Python3_LIBRARIES}
        jsoncpp_lib
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)
target_link_directories(statues_voxel PRIVATE /home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib)

# Mesh evaluation utility (reconstruction mesh vs GT mesh)
add_executable(mesh_eval
        examples/mesh_eval.cpp
        third_party/tinyply/tinyply.h
        third_party/tinyply/tinyply.cpp)
target_include_directories(mesh_eval PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mesh_eval
        ${OpenCV_LIBRARIES}
        jsoncpp_lib)

# # EuRoC Stereo
# add_executable(euroc_stereo examples/euroc_stereo.cpp)
# target_link_libraries(euroc_stereo
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# Intel Realsense
if(realsense2_FOUND)
    add_executable(realsense_rgbd examples/realsense_rgbd.cpp)
    target_include_directories(realsense_rgbd PUBLIC
        ${realsense_INCLUDE_DIR})
    target_link_libraries(realsense_rgbd
        gaussian_viewer    
        gaussian_mapper
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${realsense2_LIBRARY})
endif()
