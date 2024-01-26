# WhisperFusion

<h2 align="center">
  <a href="https://www.youtube.com/watch?v=_PnaP0AQJnk"><img
src="https://img.youtube.com/vi/_PnaP0AQJnk/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperFusion"></a>
  <br><br>Seamless conversations with AI (with ultra-low latency)<br><br>
</h2>

Welcome to WhisperFusion. WhisperFusion builds upon the capabilities of
the [WhisperLive](https://github.com/collabora/WhisperLive) and
[WhisperSpeech](https://github.com/collabora/WhisperSpeech) by
integrating Mistral, a Large Language Model (LLM), on top of the
real-time speech-to-text pipeline. Both LLM and
Whisper are optimized to run efficiently as TensorRT engines, maximizing
performance and real-time processing capabilities. While WhiperSpeech is 
optimized with torch.compile.

## Features

- **Real-Time Speech-to-Text**: Utilizes OpenAI WhisperLive to convert
  spoken language into text in real-time.

- **Large Language Model Integration**: Adds Mistral, a Large Language
  Model, to enhance the understanding and context of the transcribed
  text.

- **TensorRT Optimization**: Both LLM and Whisper are optimized to
  run as TensorRT engines, ensuring high-performance and low-latency
  processing.
- **torch.compile**: WhisperSpeech uses torch.compile to speed up 
  inference which makes PyTorch code run faster by JIT-compiling PyTorch
  code into optimized kernels.

## Getting Started
- We provide a pre-built TensorRT-LLM docker container that has both whisper and
  phi converted to TensorRT engines and WhisperSpeech model is pre-downloaded to 
  quickly start interacting with WhisperFusion.
```bash
 docker run --gpus all --shm-size 64G -p 6006:6006 -p 8888:8888 -it ghcr.io/collabora/whisperfusion:latest
```

- Start Web GUI
```bash
 cd examples/chatbot/html
 python -m http.server
```

## Build Docker Image
- We provide the docker image for cuda-architecures 89 and 90. If you have a GPU
  with a different cuda architecture. For e.g. to build for RTX 3090 with cuda-
  architecture 86
```bash
 bash build.sh 86-real
```
This should build the `ghcr.io/collabora/whisperfusion:latest` for RTX 3090.

## Contact Us

For questions or issues, please open an issue. Contact us at:
marcus.edel@collabora.com, jpc@collabora.com,
vineet.suryan@collabora.com
