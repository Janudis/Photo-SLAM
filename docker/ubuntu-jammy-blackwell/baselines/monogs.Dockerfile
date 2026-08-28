ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG SOURCE_COMMIT=6c9254c319d8bff5caeef65259e6bb0941a9b9f6
ARG MAX_JOBS=4

LABEL org.opencontainers.image.title="MonoGS Blackwell benchmark" \
      photoslam.baseline="monogs"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libx11-6 \
        ninja-build \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/baseline-home && chmod 1777 /tmp/baseline-home

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
        torch==2.7.1 \
        torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

RUN git clone --recursive https://github.com/muskie82/MonoGS.git /opt/MonoGS \
    && git -C /opt/MonoGS checkout "${SOURCE_COMMIT}" \
    && git -C /opt/MonoGS submodule update --init --recursive

COPY benchmark/baselines/patches/monogs-evo.patch /tmp/monogs-evo.patch
COPY benchmark/baselines/patches/monogs-blackwell.patch /tmp/monogs-blackwell.patch
RUN git -C /opt/MonoGS apply --check /tmp/monogs-evo.patch \
    && git -C /opt/MonoGS apply /tmp/monogs-evo.patch \
    && git -C /opt/MonoGS apply --check /tmp/monogs-blackwell.patch \
    && git -C /opt/MonoGS apply /tmp/monogs-blackwell.patch \
    && rm /tmp/monogs-evo.patch /tmp/monogs-blackwell.patch

RUN python3 -m pip install --no-cache-dir \
        evo==1.11.0 \
        glfw \
        imgviz \
        lpips \
        matplotlib \
        munch \
        numpy==1.26.4 \
        open3d==0.19.0 \
        opencv-python-headless==4.10.0.84 \
        plyfile \
        PyGLM \
        PyOpenGL \
        pyyaml \
        rich \
        scipy \
        torchmetrics==1.6.1 \
        tqdm \
        trimesh \
        wandb

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=12.0 \
    CUDA_MODULE_LOADING=LAZY \
    MAX_JOBS=${MAX_JOBS} \
    MPLBACKEND=Agg \
    WANDB_MODE=disabled \
    PYTHONNOUSERSITE=1

RUN python3 -m pip install --no-cache-dir --no-build-isolation \
        /opt/MonoGS/submodules/simple-knn \
    && python3 -m pip install --no-cache-dir --no-build-isolation \
        /opt/MonoGS/submodules/diff-gaussian-rasterization

RUN python3 -c "import torch; import diff_gaussian_rasterization; import simple_knn._C; print(torch.__version__, torch.version.cuda)"

COPY benchmark/baselines /opt/photoslam-benchmark
COPY evaluation/export_monogs_reconstruction.py /opt/photoslam-benchmark/export_monogs_reconstruction.py
RUN chmod +x \
        /opt/photoslam-benchmark/run.py \
        /opt/photoslam-benchmark/export_monogs_reconstruction.py \
        /opt/photoslam-benchmark/hi_slam2_tsdf_integrate.py

ENV MONOGS_ROOT=/opt/MonoGS \
    HI_SLAM2_ROOT=/opt/photoslam-benchmark \
    HI_SLAM2_TSDF=/opt/photoslam-benchmark/hi_slam2_tsdf_integrate.py \
    HI_SLAM2_PYTHON=/usr/bin/python3

WORKDIR /opt/MonoGS
CMD ["/bin/bash"]
