ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.2.2-devel-ubuntu22.04
ARG CUDA_ARCH

# Base environment stage
FROM ${BASE_IMAGE}:${BASE_TAG} as base
ENV CUDA_ARCH=${CUDA_ARCH}

WORKDIR /root

# Dependencies installation stage
FROM base as dependencies
COPY scripts/install-deps.sh /root/
RUN bash install-deps.sh && rm install-deps.sh

# TensorRT and LLM installation stage
FROM dependencies as trt-llm
COPY scripts/install-trt-llm.sh /root/
RUN bash install-trt-llm.sh && rm install-trt-llm.sh

# Whisperfusion installation stage
FROM trt-llm as whisperfusion
RUN apt-get update && apt-get install -y xz-utils
COPY scripts/setup-whisperfusion.sh /root/
RUN ./setup-whisperfusion.sh

FROM whisperfusion as final
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root
