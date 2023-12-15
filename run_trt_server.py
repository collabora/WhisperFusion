from whisper_live.trt_server import TranscriptionServer

if __name__ == "__main__":
    server = TranscriptionServer()
    server.run("0.0.0.0", 6006)