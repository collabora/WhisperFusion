# WhisperBot
Welcome to WhisperBot. WhisperBot builds upon the capabilities of the [WhisperLive](https://github.com/collabora/WhisperLive) and [WhisperSpeech](https://github.com/collabora/WhisperSpeech) by integrating Mistral, a Large Language Model (LLM), on top of the real-time speech-to-text pipeline. WhisperLive relies on OpenAI Whisper, a powerful automatic speech recognition (ASR) system. Both Mistral and Whisper are optimized to run efficiently as TensorRT engines, maximizing performance and real-time processing capabilities.

## Features
- **Real-Time Speech-to-Text**: Utilizes OpenAI WhisperLive to convert spoken language into text in real-time.

- **Large Language Model Integration**: Adds Mistral, a Large Language Model, to enhance the understanding and context of the transcribed text.

- **TensorRT Optimization**: Both Mistral and Whisper are optimized to run as TensorRT engines, ensuring high-performance and low-latency processing.

## Prerequisites
Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md) to build Whisper and Mistral TensorRT engines. The README builds a docker image for TensorRT-LLM. 
Instead of building a docker image, we can also refer to the README and the [Dockerfile.multi](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/Dockerfile.multi) to install the required packages in the base pytroch docker image. Just make sure to use the correct base image as mentioned in the dockerfile and everything should go nice and smooth.

### Build Whisper TensorRT Engine
- Change working dir to the [whisper example dir](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper) in TensorRT-LLM.
```bash
cd TensorRT-LLM/examples/whisper
``` 
- Currently, by default TensorRT-LLM only supports `large-v2` and `large-v3`. In this repo, we use `small.en`.
- Download the required assets.
```bash
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

# small.en model
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt
```
- Edit `build.py` to support `small.en`. In order to do that, add `"small.en"` as an item in the list [`choices`](https://github.com/NVIDIA/TensorRT-LLM/blob/a75618df24e97ecf92b8899ca3c229c4b8097dda/examples/whisper/build.py#L58).
- Build `small.en` TensorRT engine.
```bash
pip install -r requirements.txt
python3 build.py --output_dir whisper_small_en --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin --model_name small.en
```

### Build Mistral TensorRT Engine
- Change working dir to [llama example dir](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) in TensorRT-LLM folder.
```bash
cd TensorRT-LLM/examples/llama
```
- Convert Mistral to `fp16` TensorRT engine.
```bash
python build.py --model_dir teknium/OpenHermes-2.5-Mistral-7B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/mistral/7B/trt_engines/fp16/1-gpu/ \
                --max_input_len 5000
                --max_batch_size 1
```

### Build Phi TensorRT Engine
Note: Phi is only available in main branch and hasnt been released yet. So, make sure to build TensorRT-LLM from main branch.
- Change working dir to [phi example dir](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/phi) in TensorRT-LLM folder.
```bash
cd TensorRT-LLM/examples/phi
```
- Build [Dolphin Phi Finetuned](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2) ChatML format
```bash
git lfs install
git clone https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2
python3 build.py --dtype=float16                    \
                 --log_level=verbose                \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16          \
                 --max_batch_size=1                 \
                 --max_input_len=1024               \
                 --max_output_len=1024              \
                 --output_dir=phi_engine            \
                 --model_dir=dolphin-2_6-phi-2>&1 | tee build.log
```

## Run WhisperBot
- Clone this repo and install requirements.
```bash
git clone https://github.com/collabora/WhisperBot.git
cd WhisperBot
apt update
apt install ffmpeg portaudio19-dev -y
pip install -r requirements.txt
```

### Whisper + Mistral
- Take the folder path for Whisper TensorRT model, folder_path and tokenizer_path for Mistral TensorRT from the build phase. If a huggingface model is used to build mistral then just use the huggingface repo name as the tokenizer path.
```bash
python3 main.py --mistral
                --whisper_tensorrt_path /root/TensorRT-LLM/examples/whisper/whisper_small_en \
                --mistral_tensorrt_path /root/TensorRT-LLM/examples/llama/tmp/mistral/7B/trt_engines/fp16/1-gpu/ \
                --mistral_tokenizer_path teknium/OpenHermes-2.5-Mistral-7B
```

### Whisper + Phi
- Take the folder path for Whisper TensorRT model, folder_path and tokenizer_path for Phi TensorRT from the build phase. If a huggingface model is used to build phi then just use the huggingface repo name as the tokenizer path.
```bash
python3 main.py --phi
                --whisper_tensorrt_path /root/TensorRT-LLM/examples/whisper/whisper_small_en \
                --phi_tensorrt_path /root/TensorRT-LLM/examples/phi/phi_engine \
                --phi_tokenizer_path /root/TensorRT-LLM/examples/phi/dolphin-2_6-phi-2
```

- On the client side clone the repo, install the requirements and execute `run_client.py`
```bash
cd WhisperBot
pip install -r requirements.txt
python3 run_client.py
```


## Contact Us
For questions or issues, please open an issue.
Contact us at: marcus.edel@collabora.com, jpc@collabora.com, vineet.suryan@collabora.com
