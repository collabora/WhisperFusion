#!/bin/bash -e

## Clone this repo and install requirements
[ -d "WhisperFusion" ] || git clone https://github.com/collabora/WhisperFusion.git

cd WhisperFusion
apt update
apt install ffmpeg portaudio19-dev -y

## Install torchaudio matching the PyTorch from the base image
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torchaudio

## Install all the other dependencies normally
pip install -r requirements.txt

## force update huggingface_hub (tokenizers 0.14.1 spuriously require and ancient <=0.18 version)
pip install -U huggingface_hub

huggingface-cli download collabora/whisperspeech t2s-small-en+pl.model s2a-q4-tiny-en+pl.model
huggingface-cli download charactr/vocos-encodec-24khz

mkdir -p /root/.cache/torch/hub/checkpoints/
curl -L -o /root/.cache/torch/hub/checkpoints/encodec_24khz-d7cc33bc.th https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
mkdir -p /root/.cache/whisper-live/
curl -L -o /root/.cache/whisper-live/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx

python -c 'from transformers.utils.hub import move_cache; move_cache()'
