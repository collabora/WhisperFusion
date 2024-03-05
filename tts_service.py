import functools
import time
import logging
logging.basicConfig(level = logging.INFO)

from tqdm import tqdm
from websockets.sync.server import serve
from whisperspeech.pipeline import Pipeline


class WhisperSpeechTTS:
    def __init__(self):
        pass
    
    def initialize_model(self):
        self.pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model', torch_compile=True)
        self.last_llm_response = None

    def run(self, host, port, audio_queue=None, should_send_server_ready=None):
        # initialize and warmup model
        self.initialize_model()
        logging.info("\n[WhisperSpeech INFO:] Warming up torch compile model. Please wait ...\n")
        for _ in tqdm(range(3), desc="Warming up"):
            self.pipe.generate("Hello, I am warming up.")
        logging.info("[WhisperSpeech INFO:] Warmed up Whisper Speech torch compile model. Connect to the WebGUI now.")
        should_send_server_ready.value = True

        with serve(
            functools.partial(self.start_whisperspeech_tts, audio_queue=audio_queue), 
            host, port
            ) as server:
            server.serve_forever()

    def start_whisperspeech_tts(self, websocket, audio_queue=None):
        self.eos = False
        self.output_audio = None

        while True:
            llm_response = audio_queue.get()
            if audio_queue.qsize() != 0:
                continue

            # check if this websocket exists
            try:
                websocket.ping()
            except Exception as e:
                del websocket
                audio_queue.put(llm_response)
                break
            
            llm_output = llm_response["llm_output"][0]
            self.eos = llm_response["eos"]

            def should_abort():
                if not audio_queue.empty(): raise TimeoutError()

            # only process if the output updated
            if self.last_llm_response != llm_output.strip():
                try:
                    start = time.time()
                    audio = self.pipe.generate(llm_output.strip(), step_callback=should_abort)
                    inference_time = time.time() - start
                    logging.info(f"[WhisperSpeech INFO:] TTS inference done in {inference_time} ms.\n\n")
                    self.output_audio = audio.cpu().numpy()
                    self.last_llm_response = llm_output.strip()
                except TimeoutError:
                    pass

            if self.eos and self.output_audio is not None:
                try:
                    websocket.send(self.output_audio.tobytes())
                except Exception as e:
                    logging.error(f"[WhisperSpeech ERROR:] Audio error: {e}")

