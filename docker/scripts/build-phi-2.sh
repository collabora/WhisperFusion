#!/bin/bash -e

## Note: Phi is only available in main branch and hasnt been released yet. So, make sure to build TensorRT-LLM from main branch.

cd /root/TensorRT-LLM-examples/phi

## Build TensorRT for Phi-2 with `fp16`
echo "Download Phi2 HUggingface models..."

git lfs install
phi_path=$(huggingface-cli download --repo-type model --revision 834565c23f9b28b96ccbeabe614dd906b6db551a microsoft/phi-2 > /dev/null 2>&1)
echo "Building Phi2 TensorRT Engine..."

python3 build.py --dtype=float16                    \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16          \
                 --max_batch_size=16                \
                 --max_input_len=1024               \
                 --max_output_len=1024              \
                 --output_dir=phi-2            \
                 --model_dir="$phi_path" > /dev/null 2>&1

dest=/root/scratch-space/models
mkdir -p "$dest/phi-2/tokenizer"
cp -r phi-2 "$dest"
(cd "$phi_path" && cp config.json tokenizer_config.json vocab.json merges.txt "$dest/phi-2/tokenizer")
cp -r "$phi_path" "$dest/phi-orig-model"
