#!/bin/bash

download_and_build_phi_model() {
    git lfs install
    git clone https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2
    python3 build.py --dtype=float16                     \
                     --log_level=verbose                 \
                     --use_gpt_attention_plugin float16  \
                     --use_gemm_plugin float16           \
                     --max_batch_size=1                  \
                     --max_input_len=1024                \
                     --max_output_len=1024               \
                     --output_dir=phi_engine             \
                     --model_dir=dolphin-2_6-phi-2>&1 | tee build.log
    echo "Phi Engine Built."
    echo "==============================="
    echo "Model is located at: /home/TensorRT-LLM/examples/phi/phi_engine"
    
    # Cleanup, if you want to save space as this is not needed anymore
    rm -rf phi-2/model-0000*.safetensors
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path-to-tensorrt-examples-dir>"
    exit 1
fi

cd $1/phi

download_and_build_phi_model