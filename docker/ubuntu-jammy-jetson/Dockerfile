ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_NAME=photoslam
ARG USER_UID=1000
ARG USER_GID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
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
        python3-pip \
        sudo \
        unzip \
        wget \
        xauth \
        zip \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${USER_GID}" "${USER_NAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" \
        --create-home --shell /bin/bash "${USER_NAME}" \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" \
        > "/etc/sudoers.d/${USER_NAME}" \
    && chmod 0440 "/etc/sudoers.d/${USER_NAME}"

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONNOUSERSITE=1 \
    PHOTOSLAM_JETSON=1 \
    PHOTOSLAM_ENABLE_RERUN=OFF \
    PHOTOSLAM_BUILD_ORIGINAL=OFF \
    PHOTOSLAM_BUILD_WAYMO=OFF \
    PHOTOSLAM_CUDA_ARCHITECTURES=87 \
    PHOTOSLAM_TORCH_CUDA_ARCH_LIST=8.7 \
    TORCH_CUDA_ARCH_LIST=8.7

RUN python3 -c \
    "import torch; print('PyTorch', torch.__version__); print('Torch CMake', torch.utils.cmake_prefix_path)" \
    && python3 -c \
    "import cv2; print('OpenCV', cv2.__version__)"

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD ["/bin/bash"]
