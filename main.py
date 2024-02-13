import multiprocessing
import argparse
import threading
import ssl
import time
import sys
import functools
import ctypes

from multiprocessing import Process, Manager, Value, Queue

from whisper_live.trt_server import TranscriptionServer
from gpt_service import GPTEngine
from tts_service import WhisperSpeechTTS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper_tensorrt_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/whisper/whisper_small_en",
                        help='Whisper TensorRT model path')
    parser.add_argument('--mistral',
                        action="store_true",
                        help='Mistral')
    parser.add_argument('--mistral_tensorrt_path',
                        type=str,
                        default=None,
                        help='Mistral TensorRT model path')
    parser.add_argument('--mistral_tokenizer_path',
                        type=str,
                        default="teknium/OpenHermes-2.5-Mistral-7B",
                        help='Mistral TensorRT model path')
    parser.add_argument('--phi',
                        action="store_true",
                        help='Phi')
    parser.add_argument('--phi_tensorrt_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/phi/phi_engine",
                        help='Phi TensorRT model path')
    parser.add_argument('--phi_tokenizer_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/phi/phi-2",
                        help='Phi Tokenizer path')
    parser.add_argument('--gpt',
                        action="store_true",
                        help='GPT')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if not args.whisper_tensorrt_path:
        raise ValueError("Please provide whisper_tensorrt_path to run the pipeline.")
        import sys
        sys.exit(0)
    
    if args.mistral:
        if not args.mistral_tensorrt_path or not args.mistral_tokenizer_path:
            raise ValueError("Please provide mistral_tensorrt_path and mistral_tokenizer_path to run the pipeline.")
            import sys
            sys.exit(0)

    if args.phi:
        if not args.phi_tensorrt_path or not args.phi_tokenizer_path:
            raise ValueError("Please provide phi_tensorrt_path and phi_tokenizer_path to run the pipeline.")
            import sys
            sys.exit(0)

    multiprocessing.set_start_method('spawn')
    
    lock = multiprocessing.Lock()
    
    manager = Manager()
    shared_output = manager.list()
    should_send_server_ready = Value(ctypes.c_bool, False)
    transcription_queue = Queue()
    llm_queue = Queue()
    audio_queue = Queue()


    whisper_server = TranscriptionServer()
    whisper_process = multiprocessing.Process(
        target=whisper_server.run,
        args=(
            "0.0.0.0",
            6006,
            transcription_queue,
            llm_queue,
            args.whisper_tensorrt_path,
            should_send_server_ready
        )
    )
    whisper_process.start()

    llm_provider = GPTEngine()
    # llm_provider = MistralTensorRTLLMProvider()
    llm_process = multiprocessing.Process(
        target=llm_provider.run,
        args=(
            transcription_queue,
            llm_queue,
            audio_queue,
        )
    )
    llm_process.start()

    # audio process
    tts_runner = WhisperSpeechTTS()
    tts_process = multiprocessing.Process(target=tts_runner.run, args=("0.0.0.0", 8888, audio_queue, should_send_server_ready))
    tts_process.start()

    llm_process.join()
    whisper_process.join()
    tts_process.join()
