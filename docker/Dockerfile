ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.4.0-devel-ubuntu22.04

FROM ${BASE_IMAGE}:${BASE_TAG} as base
ARG CUDA_ARCH
ENV CUDA_ARCH=${CUDA_ARCH}

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip openmpi-bin libopenmpi-dev git wget \
    xz-utils curl && \
    rm -rf /var/lib/apt/lists/*

FROM base as devel
WORKDIR /root/
RUN pip3 install --no-cache-dir -U tensorrt_llm==0.10.0 --extra-index-url https://pypi.nvidia.com
RUN git clone -b v0.10.0 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git && \
    mv TensorRT-LLM/examples ./TensorRT-LLM-examples && \
    rm -rf TensorRT-LLM

FROM devel as release
WORKDIR /root/
COPY scripts/setup-whisperfusion.sh /root/
RUN ./setup-whisperfusion.sh
