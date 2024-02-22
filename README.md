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

## Hardware Requirements

- A GPU with at least 24GB of RAM
- For optimal latency, the GPU should have a similar FP16 (half) TFLOPS as the RTX 4090. Here are the [hardware specifications](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) for the RTX 4090.

The demo was run on a single RTX 4090 GPU. WhisperFusion uses the Nvidia TensorRT-LLM library for CUDA optimized versions of popular LLM models. TensorRT-LLM supports multiple GPUs, so it should be possible to run WhisperFusion for even better performance on multiple GPUs.

## Getting Started
We provide a Docker Compose setup to streamline the deployment of the pre-built TensorRT-LLM docker container. This setup includes both Whisper and Phi converted to TensorRT engines, and the WhisperSpeech model is pre-downloaded to quickly start interacting with WhisperFusion. Additionally, we include a simple web server for the Web GUI.

- Build and Run with docker compose for RTX 3090 and RTX
```bash
mkdir docker/scratch-space
cp docker/scripts/build-* docker/scripts/run-whisperfusion.sh docker/scratch-space/

# Set the CUDA_ARCH environment variable based on your GPU
# Use '86-real' for RTX 3090, '89-real' for RTX 4090
CUDA_ARCH=86-real docker compose build
docker compose up
```

- Start Web GUI on `http://localhost:8000`

**NOTE**

## Contact Us

For questions or issues, please open an issue. Contact us at:
marcus.edel@collabora.com, jpc@collabora.com,
vineet.suryan@collabora.com
