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
if [ -d "$dest/$name" ]; then
    rm -rf "$dest/$name"
fi
mkdir -p "$dest/$name/tokenizer"
cp -r "$name" "$dest"
(cd "$phi_path" && cp config.json tokenizer_config.json tokenizer.json special_tokens_map.json added_tokens.json "$dest/$name/tokenizer")
if [ "$MODEL_TYPE" == "phi-2" ]; then
    (cd "$phi_path" && cp vocab.json merges.txt "$dest/$name/tokenizer")
fi
