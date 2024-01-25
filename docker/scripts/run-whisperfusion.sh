#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

cd WhisperFusion
if [ "$1" != "mistral" ]; then
  exec python3 main.py --phi \
                  --whisper_tensorrt_path /root/whisper_small_en \
                  --phi_tensorrt_path /root/dolphin-2_6-phi-2 \
                  --phi_tokenizer_path /root/dolphin-2_6-phi-2/tokenizer
else
  exec python3 main.py --mistral \
                  --whisper_tensorrt_path /root/models/whisper_small_en \
                  --mistral_tensorrt_path /root/models/mistral \
                  --mistral_tokenizer_path teknium/OpenHermes-2.5-Mistral-7B
fi
