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

### Backend

- A pre-built TensorRT-LLM docker container is provided that has both whisper and
  phi converted to TensorRT engines and WhisperSpeech model is pre-downloaded to 
  quickly start interacting with WhisperFusion. It is built for cuda-architectures 89 and 90.
```bash
 docker run --gpus all --shm-size 64G -p 6006:6006 -p 8888:8888 -it ghcr.io/collabora/whisperfusion:latest
```
#### Hosting on Google Cloud Compute

1. Configure an instance with a GPU with `>= 24GB` RAM and preferably cuda-architectures versions 89 and 90.
2. Configure the boot disk
	a) Pick the Operating System "Deep Learning on Linux"
	 b) Pick a Version which has "CUDA 12.1 Installed" or later.
	 c) Allocate ample disk space: `>= 256GB`
3. Create a Firewall Rule allowing ports tcp: 6006 and 8888. 
     a) Assign this Filewall Rule to the above instance
4. Launch the instance and, when prompted, agree to install the CUDA driver
5. Once finished, run the above docker command to start a container with `Whisper-Fusion`
    a) On the first run, it should download a few models. If no errors are printed, then it booted successfully.

### Frontend

#### Start Web GUI
1. If you are hosting the backend remotely, open an SSH tunnel for ports 6006 and 8888 from localhost to the External IP of the backend server. These tunnels must remain open for the duration of the frontend.
	- `ssh -N -n -L 6006:localhost:6006 <External IP>`
	- `ssh -N -n -L 8888:localhost:8888 <External IP>`

2. Run the frontend
```bash
 cd examples/chatbot/html
 python -m http.server
```

## Build Docker Image
- A docker image for cuda-architecures 89 and 90 is provided. If you have a GPU
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
