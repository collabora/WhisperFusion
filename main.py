from whisper_live.trt_server import TranscriptionServer
from llm_service import MistralTensorRTLLM
import multiprocessing
import threading
import ssl
import time
import sys
import functools

from multiprocessing import Process, Manager, Value, Queue


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    
    lock = multiprocessing.Lock()
    
    manager = Manager()
    shared_output = manager.list()

    transcription_queue = Queue()
    llm_queue = Queue()


    whisper_server = TranscriptionServer()
    whisper_process = multiprocessing.Process(target=whisper_server.run, args=("0.0.0.0", 6006, transcription_queue, llm_queue))
    whisper_process.start()

    llm_provider = MistralTensorRTLLM()
    # llm_provider = MistralTensorRTLLMProvider()
    llm_process = multiprocessing.Process(target=llm_provider.run, args=(transcription_queue, llm_queue))
    llm_process.start()

    llm_process.join()
    whisper_process.join()
