ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG SOURCE_COMMIT=f8816c7d9a92b29e84e3d9055c2d3e28056e4a37
ARG MAX_JOBS=4

LABEL org.opencontainers.image.title="TANDEM Blackwell benchmark" \
      photoslam.baseline="tandem"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libboost-all-dev \
        libcxsparse3 \
        libeigen3-dev \
        libgl1-mesa-dev \
        libopencv-dev \
        libsuitesparse-dev \
        libtbb-dev \
        ninja-build \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/baseline-home && chmod 1777 /tmp/baseline-home

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
        torch==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

RUN git clone https://github.com/tum-vision/tandem.git /opt/tandem \
    && git -C /opt/tandem checkout "${SOURCE_COMMIT}"

COPY benchmark/baselines/patches/tandem-blackwell.patch /tmp/tandem-blackwell.patch
RUN git -C /opt/tandem apply --check /tmp/tandem-blackwell.patch \
    && git -C /opt/tandem apply /tmp/tandem-blackwell.patch \
    && rm /tmp/tandem-blackwell.patch

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=12.0 \
    CUDA_MODULE_LOADING=LAZY \
    PYTHONNOUSERSITE=1

RUN torch_prefix="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    && cmake -S /opt/tandem/tandem -B /opt/tandem/tandem/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DCMAKE_PREFIX_PATH="$torch_prefix" \
        -DCUDNN_INCLUDE_PATH=/usr/include \
        -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
    && cmake --build /opt/tandem/tandem/build --parallel "${MAX_JOBS}"

RUN test -x /opt/tandem/tandem/build/bin/tandem_dataset

COPY benchmark/baselines /opt/photoslam-benchmark
RUN chmod +x /opt/photoslam-benchmark/run.py

WORKDIR /opt/tandem/tandem
CMD ["/bin/bash"]
