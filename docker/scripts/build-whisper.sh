#!/bin/bash -e

## Change working dir to the [whisper example dir](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper) in TensorRT-LLM.
cd /root/TensorRT-LLM-examples/whisper

## Currently, by default TensorRT-LLM only supports `large-v2` and `large-v3`. In this repo, we use `small.en`.
## Download the required assets

# the sound filter definitions
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
# the small.en model weights
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt

## We have to patch the script to add support for out model size (`small.en`):
patch <<EOF
--- build.py.old  2024-01-17 17:47:47.508545842 +0100
+++ build.py  2024-01-17 17:47:41.404941926 +0100
@@ -58,6 +58,7 @@
                         choices=[
                             "large-v3",
                             "large-v2",
+                            "small.en",
                         ])
     parser.add_argument('--quantize_dir', type=str, default="quantize/1-gpu")
     parser.add_argument('--dtype',
EOF

## Finally we can build the TensorRT engine for the `small.en` Whisper model:
pip install -r requirements.txt
python3 build.py --output_dir whisper_small_en --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin --model_name small.en

mkdir -p /root/scratch-space/models
cp -r whisper_small_en /root/scratch-space/models
