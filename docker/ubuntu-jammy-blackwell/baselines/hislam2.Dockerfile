ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG SOURCE_COMMIT=76c833c7d8ed474f0f3ba18056c1803e032a537f
ARG MAX_JOBS=4

LABEL org.opencontainers.image.title="HI-SLAM2 Blackwell benchmark" \
      photoslam.baseline="hislam2"

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
        libsuitesparse-dev \
        ninja-build \
        python3-dev \
        python3-pip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/baseline-home && chmod 1777 /tmp/baseline-home

RUN python3 -m pip install --no-cache-dir --upgrade pip wheel \
    && python3 -m pip install --no-cache-dir setuptools==69.5.1 \
    && python3 -m pip install --no-cache-dir \
        torch==2.7.1 \
        torchvision==0.22.1 \
        torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

RUN git clone --recursive https://github.com/Willyzw/HI-SLAM2.git /opt/HI-SLAM2 \
    && git -C /opt/HI-SLAM2 checkout "${SOURCE_COMMIT}" \
    && git -C /opt/HI-SLAM2 submodule update --init --recursive

COPY benchmark/baselines/patches/hislam2-output.patch /tmp/hislam2-output.patch
RUN git -C /opt/HI-SLAM2 apply --check /tmp/hislam2-output.patch \
    && git -C /opt/HI-SLAM2 apply /tmp/hislam2-output.patch \
    && rm /tmp/hislam2-output.patch

RUN python3 -m pip install --no-cache-dir \
        evo \
        glfw \
        imgviz \
        lightning \
        matplotlib \
        munch \
        numpy==1.26.4 \
        open3d==0.19.0 \
        opencv-python-headless==4.10.0.84 \
        plyfile \
        PyGLM \
        pyrender \
        pyyaml \
        rich \
        scipy \
        timm \
        torchmetrics \
        tqdm \
        trimesh \
    && python3 -m pip install --no-cache-dir \
        git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=12.0 \
    CUDA_MODULE_LOADING=LAZY \
    MAX_JOBS=${MAX_JOBS} \
    MPLBACKEND=Agg \
    PYTHONNOUSERSITE=1

# The upstream setup hard-codes pre-Ampere gencode flags. Let PyTorch's
# TORCH_CUDA_ARCH_LIST select only the Blackwell architecture instead.
RUN sed -i '/-gencode=arch=compute_/d' /opt/HI-SLAM2/setup.py \
    && cd /opt/HI-SLAM2 \
    && python3 setup.py install \
    && python3 -m pip install --no-cache-dir --no-build-isolation \
        thirdparty/simple-knn \
    && python3 -m pip install --no-cache-dir --no-build-isolation \
        thirdparty/diff-gaussian-rasterization \
    && python3 -m pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/rusty1s/pytorch_scatter.git

RUN wget -q \
        https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt \
        -O /opt/HI-SLAM2/pretrained_models/omnidata_dpt_normal_v2.ckpt \
    && wget -q \
        https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt \
        -O /opt/HI-SLAM2/pretrained_models/omnidata_dpt_depth_v2.ckpt

RUN python3 -c "import torch; import droid_backends; import lietorch; import torch_scatter; print(torch.__version__, torch.version.cuda)"

COPY benchmark/baselines /opt/photoslam-benchmark
RUN chmod +x /opt/photoslam-benchmark/run.py

WORKDIR /opt/HI-SLAM2
CMD ["/bin/bash"]
