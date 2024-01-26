#ARG BASE_IMAGE=nvcr.io/nvidia/pytorch
#ARG BASE_TAG=23.10-py3
ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.2.2-devel-ubuntu22.04

FROM ${BASE_IMAGE}:${BASE_TAG} as base
ARG CUDA_ARCH
WORKDIR /root
COPY install-deps.sh /root
ENV CUDA_ARCH=${CUDA_ARCH}
RUN bash install-deps.sh && rm install-deps.sh

COPY install-trt-llm.sh /root
RUN bash install-trt-llm.sh && rm install-trt-llm.sh
