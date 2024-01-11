import functools

from websockets.sync.server import serve
from whisperspeech.pipeline import Pipeline

class WhisperSpeechTTS:
    def __init__(self):
        pass
    
    def initialize_model(self):
        self.pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

    def run(self, host, port=6080, audio_queue=None):
        with serve(
            functools.partial(self.start_whisperspeech_tts, audio_queue=audio_queue), 
            host, port
            ) as server:
            server.serve_forever()

    def start_whisperspeech_tts(self, websocket, audio_queue=None):
        self.initialize_model()

        while True:
            if audio_queue.empty(): continue

            llm_output = audio_queue.get()[0]
            audio = self.pipe.vocoder.decode(self.pipe.generate_atoks(llm_output.strip()))
            audio = audio.cpu().numpy()
            audio = audio * 32768.0

            # send audio to client on another websocket
            try:
                websocket.send(audio.astype('int16').tobytes())
            except Exception as e:
                print("Audio error:", e)

