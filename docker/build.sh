#!/bin/bash -e

if [ -n "$1" ]; then
  CUDA_ARCH="$1"
  BASE_IMAGE_BUILD_ARG="--build-arg CUDA_ARCH=$CUDA_ARCH"
else
  BASE_IMAGE_BUILD_ARG=""
fi

[ -n "$VERBOSE" ] && ARGS="--progress plain"

(
  cd base-image &&
  docker build $ARGS $BASE_IMAGE_BUILD_ARG -t ghcr.io/collabora/whisperfusion-base:latest .
)

mkdir -p scratch-space
cp -r scripts/build-* scratch-space
docker run --gpus all --shm-size 64G -v "$PWD"/scratch-space:/root/scratch-space -w /root/scratch-space -it ghcr.io/collabora/whisperfusion-base:latest ./build-models.sh

docker build $ARGS -t ghcr.io/collabora/whisperfusion:latest .
