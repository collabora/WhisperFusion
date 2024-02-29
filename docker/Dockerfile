ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.2.2-devel-ubuntu22.04

FROM ${BASE_IMAGE}:${BASE_TAG} as base
ARG CUDA_ARCH
ENV CUDA_ARCH=${CUDA_ARCH}

WORKDIR /root
COPY scripts/install-deps.sh /root
RUN bash install-deps.sh && rm install-deps.sh

COPY scripts/install-trt-llm.sh /root
RUN bash install-trt-llm.sh && rm install-trt-llm.sh

RUN apt-get update && apt-get install -y xz-utils

WORKDIR /root/
COPY scripts/setup-whisperfusion.sh /root/
RUN ./setup-whisperfusion.sh
