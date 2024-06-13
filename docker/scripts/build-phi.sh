#!/bin/bash -e

## Note: Phi is only available in main branch and hasnt been released yet. So, make sure to build TensorRT-LLM from main branch.

cd /root/TensorRT-LLM-examples/phi

## Build TensorRT for Phi-2 with `fp16`

MODEL_TYPE=$1
echo "Download $MODEL_TYPE Huggingface models..."

phi_path=$(huggingface-cli download --repo-type model microsoft/$MODEL_TYPE)
echo "Building  TensorRT Engine..."
name=$1
pip install -r requirements.txt

python3 ./convert_checkpoint.py --model_type $MODEL_TYPE \
                    --model_dir $phi_path \
                    --output_dir ./phi-checkpoint \
                    --dtype float16

trtllm-build \
    --checkpoint_dir ./phi-checkpoint \
    --output_dir $name \
    --gpt_attention_plugin float16 \
    --context_fmha enable \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_output_len 1024 \
    --tp_size 1 \
    --pp_size 1

dest=/root/scratch-space/models
mkdir -p "$dest/$name/tokenizer"
cp -r "$name" "$dest"
(cd "$phi_path" && cp config.json tokenizer_config.json tokenizer.json special_token_map.json added_tokens.json vocab.json merges.txt "$dest/$name/tokenizer")
cp -r "$phi_path" "$dest/phi-orig-model"
