import os
import time
import logging
from multiprocessing import Queue

from openai import OpenAI

logging.basicConfig(level=logging.INFO)


class GPTEngine:
    def __init__(self):
        """The __init__ is instantiated outside of the Subprocess. Do nothing.
        Use `self.initialize` once the subprocess is running."""
        pass

    def initialize(self):
        self.last_prompt: str | None = None
        self.last_output: str | None = None
        self.infer_time = 0
        self.eos = False

        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        logging.info("[LLM INFO:] Connected to OpenAI.")

    def run(
        self,
        transcription_queue: Queue,
        llm_queue: Queue,
        audio_queue: Queue,
        streaming=False,
    ):
        self.initialize()

        conversation_history = {}

        while True:
            # Get the last transcription output from the queue
            transcription_output = transcription_queue.get()
            if transcription_queue.qsize() != 0:
                continue

            if transcription_output["uid"] not in conversation_history:
                conversation_history[transcription_output["uid"]] = []

            prompt = transcription_output["prompt"].strip()

            # If the `prompt` is same but EOS is True, we need
            # that to send outputs to websockets
            if self.last_prompt == prompt:
                if self.last_output is not None and transcription_output["eos"]:
                    self.eos = transcription_output["eos"]
                    llm_queue.put(
                        {
                            "uid": transcription_output["uid"],
                            # The `llm_queue` expects a list of possible outputs
                            "llm_output": [self.last_output],
                            "eos": self.eos,
                            "latency": self.infer_time,
                        }
                    )
                    # The `audio_queue` expects a list of possible outputs
                    audio_queue.put({"llm_output": [self.last_output], "eos": self.eos})
                    conversation_history[transcription_output["uid"]].append(
                        (
                            transcription_output["prompt"].strip(),
                            self.last_output.strip(),
                        )
                    )
                    continue

            input_messages = self.format_gpt_messages(
                conversation_history[transcription_output["uid"]],
                prompt,
                system_prompt="You are a masterful rhymer. Respond to all requests in a rhyming manner.",
            )
            self.eos = transcription_output["eos"]

            start = time.time()

            # Send a ChatCompletion request with the `input_messages`
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=input_messages,
            )

            self.infer_time = time.time() - start

            output = response.choices[0].message.content

            self.last_output = output
            self.last_prompt = prompt
            llm_queue.put(
                {
                    "uid": transcription_output["uid"],
                    # The `llm_queue` expects a list of possible `output`s
                    "llm_output": [output],
                    "eos": self.eos,
                    "latency": self.infer_time,
                }
            )
            # The `audio_queue` expects a list of possible `output`s
            audio_queue.put({"llm_output": [output], "eos": self.eos})
            logging.info(
                f"[LLM INFO:] Output: {output}\nLLM inference done in {self.infer_time} ms\n\n"
            )

            if self.eos:
                conversation_history[transcription_output["uid"]].append(
                    (transcription_output["prompt"].strip(), output.strip())
                )
                self.last_prompt = None
                self.last_output = None

    @staticmethod
    def format_gpt_messages(
        conversation_history: list[tuple[str, str]],
        prompt: str,
        system_prompt: str = "",
    ):
        messages = []

        # Add the `system_prompt` if it is non-empty
        if system_prompt != "":
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # Build up the conversation history
        for user_prompt, llm_response in conversation_history:
            messages += [
                {
                    "role": "user",
                    "content": user_prompt,
                },
                {
                    "role": "assistant",
                    "content": llm_response,
                },
            ]

        # Add the user `prompt` to the very end
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages
