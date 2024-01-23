#!/bin/bash -e

export ENV=${ENV:-/etc/shinit_v2}
source $ENV

cd /root/TensorRT-LLM
python3 scripts/build_wheel.py --clean --cuda_architectures "89-real;90-real" --trt_root /usr/local/tensorrt
pip install build/tensorrt_llm-0.7.1-cp310-cp310-linux_x86_64.whl
mv examples ../TensorRT-LLM-examples
cd ..

rm -rf TensorRT-LLM
# we don't need static libraries and they take a lot of space
(cd /usr && find . -name '*static.a' | grep -v cudart_static | xargs rm -f)
