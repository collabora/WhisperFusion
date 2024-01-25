# WhisperFusion

<h2 align="center">
  <a href="https://www.youtube.com/watch?v=_PnaP0AQJnk"><img
src="https://img.youtube.com/vi/_PnaP0AQJnk/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperFusion"></a>
  <br><br>Doing math with WhisperFusion: Ultra-low latency conversations with an AI chatbot<br><br>
</h2>

Welcome to WhisperFusion. WhisperFusion builds upon the capabilities of
the [WhisperLive](https://github.com/collabora/WhisperLive) and
[WhisperSpeech](https://github.com/collabora/WhisperSpeech) by
integrating Mistral, a Large Language Model (LLM), on top of the
real-time speech-to-text pipeline. WhisperLive relies on OpenAI Whisper,
a powerful automatic speech recognition (ASR) system. Both Mistral and
Whisper are optimized to run efficiently as TensorRT engines, maximizing
performance and real-time processing capabilities.

## Features

- **Real-Time Speech-to-Text**: Utilizes OpenAI WhisperLive to convert
  spoken language into text in real-time.

- **Large Language Model Integration**: Adds Mistral, a Large Language
  Model, to enhance the understanding and context of the transcribed
  text.

- **TensorRT Optimization**: Both Mistral and Whisper are optimized to
  run as TensorRT engines, ensuring high-performance and low-latency
  processing.

## Prerequisites

Install
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md)
to build Whisper and Mistral TensorRT engines. The README builds a
docker image for TensorRT-LLM. Instead of building a docker image, we
can also refer to the README and the
[Dockerfile.multi](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/Dockerfile.multi)
to install the required packages in the base pytroch docker image. Just
make sure to use the correct base image as mentioned in the dockerfile
and everything should go nice and smooth.

### Build Whisper TensorRT Engine

> [!NOTE]
>
> These steps are included in `docker/scripts/build-whisper.sh`

Change working dir to the [whisper example
dir](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper)
in TensorRT-LLM.

``` bash
cd /root/TensorRT-LLM-examples/whisper
```

Currently, by default TensorRT-LLM only supports `large-v2` and
`large-v3`. In this repo, we use `small.en`.

Download the required assets

``` bash
# the sound filter definitions
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
# the small.en model weights
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt
```

We have to patch the script to add support for out model size
(`small.en`):

``` bash
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
```

Finally we can build the TensorRT engine for the `small.en` Whisper
model:

``` bash
pip install -r requirements.txt
python3 build.py --output_dir whisper_small_en --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin --model_name small.en
mkdir -p /root/scratch-space/models
cp -r whisper_small_en /root/scratch-space/models
```

### Build Mistral TensorRT Engine

> [!NOTE]
>
> These steps are included in `docker/scripts/build-mistral.sh`

``` bash
cd /root/TensorRT-LLM-examples/llama
```

Build TensorRT for Mistral with `fp16`

``` bash
python build.py --model_dir teknium/OpenHermes-2.5-Mistral-7B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/mistral/7B/trt_engines/fp16/1-gpu/ \
                --max_input_len 5000 \
                --max_batch_size 1
mkdir -p /root/scratch-space/models
cp -r tmp/mistral/7B/trt_engines/fp16/1-gpu /root/scratch-space/models/mistral
```

### Build Phi TensorRT Engine

> [!NOTE]
>
> These steps are included in `docker/scripts/build-phi-2.sh`

Note: Phi is only available in main branch and hasnt been released yet.
So, make sure to build TensorRT-LLM from main branch.

``` bash
cd /root/TensorRT-LLM-examples/phi
```

Build TensorRT for Phi-2 with `fp16`

``` bash
git lfs install
phi_path=$(huggingface-cli download --repo-type model --revision 834565c23f9b28b96ccbeabe614dd906b6db551a microsoft/phi-2)
python3 build.py --dtype=float16                    \
                 --log_level=verbose                \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16          \
                 --max_batch_size=16                \
                 --max_input_len=1024               \
                 --max_output_len=1024              \
                 --output_dir=phi-2            \
                 --model_dir="$phi_path" >&1 | tee build.log
dest=/root/scratch-space/models
mkdir -p "$dest/phi-2/tokenizer"
cp -r phi-2 "$dest"
(cd "$phi_path" && cp config.json tokenizer_config.json vocab.json merges.txt "$dest/phi-2/tokenizer")
cp -r "$phi_path" "$dest/phi-orig-model"
```

## Build WhisperFusion

> [!NOTE]
>
> These steps are included in `docker/scripts/setup-whisperfusion.sh`

Clone this repo and install requirements

``` bash
[ -d "WhisperFusion" ] || git clone https://github.com/collabora/WhisperFusion.git
cd WhisperFusion
apt update
apt install ffmpeg portaudio19-dev -y
```

Install torchaudio matching the PyTorch from the base image

``` bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torchaudio
```

Install all the other dependencies normally

``` bash
pip install -r requirements.txt
```

force update huggingface_hub (tokenizers 0.14.1 spuriously require and
ancient \<=0.18 version)

``` bash
pip install -U huggingface_hub
huggingface-cli download collabora/whisperspeech t2s-small-en+pl.model s2a-q4-tiny-en+pl.model
huggingface-cli download charactr/vocos-encodec-24khz
mkdir -p /root/.cache/torch/hub/checkpoints/
curl -L -o /root/.cache/torch/hub/checkpoints/encodec_24khz-d7cc33bc.th https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
mkdir -p /root/.cache/whisper-live/
curl -L -o /root/.cache/whisper-live/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
python -c 'from transformers.utils.hub import move_cache; move_cache()'
```

### Run WhisperFusion with Whisper and Mistral/Phi-2

Take the folder path for Whisper TensorRT model, folder_path and
tokenizer_path for Mistral/Phi-2 TensorRT from the build phase. If a
huggingface model is used to build mistral/phi-2 then just use the
huggingface repo name as the tokenizer path.

> [!NOTE]
>
> These steps are included in `docker/scripts/run-whisperfusion.sh`

``` bash
test -f /etc/shinit_v2 && source /etc/shinit_v2
cd WhisperFusion
if [ "$1" != "mistral" ]; then
  exec python3 main.py --phi \
                  --whisper_tensorrt_path /root/whisper_small_en \
                  --phi_tensorrt_path /root/phi-2 \
                  --phi_tokenizer_path /root/phi-2
else
  exec python3 main.py --mistral \
                  --whisper_tensorrt_path /root/models/whisper_small_en \
                  --mistral_tensorrt_path /root/models/mistral \
                  --mistral_tokenizer_path teknium/OpenHermes-2.5-Mistral-7B
fi
```

- On the client side clone the repo, install the requirements and
  execute `run_client.py`

``` bash
cd WhisperFusion
pip install -r requirements.txt
python3 run_client.py
```

## Contact Us

For questions or issues, please open an issue. Contact us at:
marcus.edel@collabora.com, jpc@collabora.com,
vineet.suryan@collabora.com
