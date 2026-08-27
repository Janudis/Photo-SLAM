ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.13.0
ARG OPENCV_BUILD_JOBS=4
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1

LABEL org.opencontainers.image.title="Photo-SLAM server build environment" \
      org.opencontainers.image.description="x86_64 CUDA environment for Photo-SLAM and SVRecon" \
      org.opencontainers.image.vendor="dimitris-photoslam-benchmark" \
      photoslam.owner="dimitris" \
      photoslam.project="server-evaluation"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        gcc-11 \
        g++-11 \
        git \
        git-lfs \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libboost-all-dev \
        libeigen3-dev \
        libgl1-mesa-dev \
        libglfw3-dev \
        libglm-dev \
        libgoogle-glog-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libjsoncpp-dev \
        libopenblas-dev \
        libopengl-dev \
        libpng-dev \
        libssl-dev \
        libswresample-dev \
        libswscale-dev \
        libtiff-dev \
        libx11-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev \
        mesa-utils \
        ninja-build \
        pkg-config \
        pybind11-dev \
        python3-dev \
        python3-numpy \
        python3-pip \
        unzip \
        wget \
        xauth \
        zip \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir "numpy==1.26.4" \
    && python3 -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128

# Photo-SLAM uses OpenCV CUDA for image warping and stereo support. Ubuntu's
# OpenCV packages are CPU-only, so build only the required modules for sm_120.
RUN mkdir -p /tmp/opencv-src \
    && cd /tmp/opencv-src \
    && wget -q \
        "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip" \
        -O opencv.zip \
    && wget -q \
        "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip" \
        -O opencv_contrib.zip \
    && unzip -q opencv.zip \
    && unzip -q opencv_contrib.zip \
    && cmake -S "opencv-${OPENCV_VERSION}" -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DOPENCV_EXTRA_MODULES_PATH="/tmp/opencv-src/opencv_contrib-${OPENCV_VERSION}/modules" \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DBUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,calib3d,features2d,flann,photo,video,cudev,cudaarithm,cudaimgproc,cudawarping,cudastereo,python3 \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_opencv_python3=ON \
        -DBUILD_opencv_sfm=OFF \
        -DBUILD_opencv_world=OFF \
        -DCUDA_ARCH_BIN=12.0 \
        -DCUDA_ARCH_PTX= \
        -DPYTHON3_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
        -DWITH_CUBLAS=ON \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GTK=ON \
        -DWITH_OPENGL=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_VTK=OFF \
    && cmake --build build --parallel "${OPENCV_BUILD_JOBS}" \
    && cmake --install build \
    && ldconfig \
    && rm -rf /tmp/opencv-src

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    OpenCV_DIR=/usr/local/lib/cmake/opencv4 \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig \
    PYTHONNOUSERSITE=1 \
    PHOTOSLAM_ENABLE_RERUN=OFF \
    PHOTOSLAM_BUILD_ORIGINAL=ON \
    PHOTOSLAM_BUILD_WAYMO=OFF \
    PHOTOSLAM_BUILD_REALSENSE=OFF \
    PHOTOSLAM_CUDA_ARCHITECTURES=120 \
    PHOTOSLAM_TORCH_CUDA_ARCH_LIST=12.0 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    CUDA_MODULE_LOADING=LAZY

RUN python3 -c \
    "import torch; print('PyTorch', torch.__version__); print('Torch CUDA', torch.version.cuda); print('Torch CMake', torch.utils.cmake_prefix_path)" \
    && python3 -c \
    "import cv2,re; info=cv2.getBuildInformation(); print('OpenCV', cv2.__version__); assert re.search(r'NVIDIA CUDA:\\s+YES', info), 'OpenCV was built without CUDA'"

WORKDIR /workspace
CMD ["/bin/bash"]
