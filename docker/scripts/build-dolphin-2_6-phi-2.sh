#!/bin/bash -e

## Note: Phi is only available in main branch and hasnt been released yet. So, make sure to build TensorRT-LLM from main branch.

cd /root/TensorRT-LLM-examples/phi

## Build TensorRT for [Dolphin Phi Finetuned](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2) ChatML format with `fp16`
echo "Download Phi2 Huggingface models..."

git lfs install
phi_path=$(huggingface-cli download --repo-type model cognitivecomputations/dolphin-2_6-phi-2)
name=dolphin-2_6-phi-2
echo "Building Phi2 TensorRT engine. This will take some time. Please wait ..."
python3 build.py --dtype=float16                    \
                 --log_level=error                  \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16          \
                 --max_batch_size=1                 \
                 --max_input_len=1024               \
                 --max_output_len=1024              \
                 --output_dir=$name                 \
                 --model_dir="$phi_path" >&1 | tee build.log

dest=/root/scratch-space/models
mkdir -p "$dest/$name/tokenizer"
cp -r "$name" "$dest"
(cd "$phi_path" && cp config.json tokenizer_config.json vocab.json merges.txt "$dest/$name/tokenizer")
cp -r "$phi_path" "$dest/phi-orig-model"
echo "Done building Phi2 TensorRT Engine. Starting the web server ..."
