#!/bin/bash -e

cd /root/TensorRT-LLM-examples/llama

## Build TensorRT for Mistral with `fp16`

python build.py --model_dir teknium/OpenHermes-2.5-Mistral-7B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/mistral/7B/trt_engines/fp16/1-gpu/ \
                --max_input_len 5000 \
                --max_batch_size 1

mkdir -p /root/scratch-space/models
cp -r tmp/mistral/7B/trt_engines/fp16/1-gpu /root/scratch-space/models/mistral
