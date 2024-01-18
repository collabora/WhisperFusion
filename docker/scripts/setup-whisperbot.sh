#!/bin/bash -e

## Clone this repo and install requirements
[ -d "WhisperBot" ] || git clone https://github.com/collabora/WhisperBot.git

cd WhisperBot
apt update
apt install ffmpeg portaudio19-dev -y

## NVidia containers are based on unreleased PyTorch versions so we have to manually install
## torchaudio from source (`pip install torchaudio` would pull all new PyTorch and CUDA versions)
#apt install -y cmake
#TORCH_CUDA_ARCH_LIST="8.9 9.0" pip install --no-build-isolation git+https://github.com/pytorch/audio.git

## Install all the other dependencies normally
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torchaudio
pip install -r requirements.txt
pip install openai-whisper whisperspeech soundfile

## force update huggingface_hub (tokenizers 0.14.1 spuriously require and ancient <=0.18 version)
pip install -U huggingface_hub

huggingface-cli download collabora/whisperspeech t2s-small-en+pl.model s2a-q4-tiny-en+pl.model
huggingface-cli download charactr/vocos-encodec-24khz

mkdir -p /root/.cache/torch/hub/checkpoints/
curl -L -o /root/.cache/torch/hub/checkpoints/encodec_24khz-d7cc33bc.th https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
mkdir -p /root/.cache/whisper-live/
curl -L -o /root/.cache/whisper-live/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx

python -c 'from transformers.utils.hub import move_cache; move_cache()'

