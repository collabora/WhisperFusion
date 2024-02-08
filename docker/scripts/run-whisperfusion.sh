#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

cd WhisperFusion
if [ "$1" != "phi" ]; then
  exec python3 main.py --gpt \
                  --whisper_tensorrt_path /root/whisper_small_en
else
  exec python3 main.py --phi \
                  --whisper_tensorrt_path /root/whisper_small_en \
                  --phi_tensorrt_path /root/dolphin-2_6-phi-2 \
                  --phi_tokenizer_path /root/dolphin-2_6-phi-2/tokenizer
fi
