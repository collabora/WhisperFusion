import time
import json
from pathlib import Path
from typing import Optional

import logging
logging.basicConfig(level = logging.INFO)

import numpy as np
import torch
from transformers import AutoTokenizer
import re

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def read_model_name(engine_dir: str):
    engine_version = tensorrt_llm.runtime.engine.get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name']

    return config['pretrained_config']['architecture']


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'gpt',
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    else:
        # For gpt-next, directly load from tokenizer.model
        assert model_name == 'gpt'
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left')

    if model_name == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        if chat_format == 'raw':
            pad_id = gen_config['pad_token_id']
            end_id = gen_config['eos_token_id']
        elif chat_format == 'chatml':
            pad_id = tokenizer.im_end_id
            end_id = tokenizer.im_end_id
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == 'glm_10b':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


class TensorRTLLMEngine:
    def __init__(self):
        pass
    
    def initialize_model(self, engine_dir, tokenizer_dir):
        self.log_level = 'error'
        self.runtime_rank = tensorrt_llm.mpi_rank()
        logger.set_level(self.log_level)
        model_name = read_model_name(engine_dir)
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=None,
            model_name=model_name,
            tokenizer_type=None,
        )
        self.prompt_template = None
        self.runner_cls = ModelRunner
        self.runner_kwargs = dict(engine_dir=engine_dir,
                         lora_dir=None,
                         rank=self.runtime_rank,
                         debug_mode=False,
                         lora_ckpt_source='hf')
        self.runner = self.runner_cls.from_dir(**self.runner_kwargs)
        self.last_prompt = None
        self.last_output = None

    def parse_input(
        self,
        input_text=None,
        add_special_tokens=True,
        max_input_length=923,
        pad_id=None,
    ):
        if self.pad_id is None:
            self.pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        for curr_text in input_text:
            if self.prompt_template is not None:
                curr_text = self.prompt_template.format(input_text=curr_text)
            input_ids = self.tokenizer.encode(
                curr_text,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=max_input_length
            )
            batch_input_ids.append(input_ids)

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]
        return batch_input_ids
    
    def decode_tokens(
        self,
        output_ids,
        input_lengths,
        sequence_lengths,
        transcription_queue
        ):
        batch_size, num_beams, _ = output_ids.size()
        for batch_idx in range(batch_size):
            if transcription_queue.qsize() != 0:
                return None

            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist()
            input_text = self.tokenizer.decode(inputs)
            output = []
            for beam in range(num_beams):
                if transcription_queue.qsize() != 0:
                    return None

                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = self.tokenizer.decode(outputs)
                output.append(output_text)
        return output
    
    def format_prompt_qa(self, prompt, conversation_history):
        formatted_prompt = ""
        for user_prompt, llm_response in conversation_history:
            formatted_prompt += f"Instruct: {user_prompt}\nOutput:{llm_response}\n"
        return f"{formatted_prompt}Instruct: {prompt}\nOutput:"
    
    def format_prompt_chat(self, prompt, conversation_history):
        formatted_prompt = ""
        for user_prompt, llm_response in conversation_history:
            formatted_prompt += f"Alice: {user_prompt}\nBob:{llm_response}\n"
        return f"{formatted_prompt}Alice: {prompt}\nBob:"

    def format_prompt_chatml(self, prompt, conversation_history, system_prompt=""):
        formatted_prompt = ("<|im_start|>system\n" + system_prompt + "<|im_end|>\n")
        for user_prompt, llm_response in conversation_history:
            formatted_prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            formatted_prompt += f"<|im_start|>assistant\n{llm_response}<|im_end|>\n"
        formatted_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        return formatted_prompt

    def run(
        self,
        model_path,
        tokenizer_path,
        transcription_queue=None,
        llm_queue=None,
        audio_queue=None,
        input_text=None, 
        max_output_len=50, 
        max_attention_window_size=4096, 
        num_beams=1, 
        streaming=False,
        streaming_interval=4,
        debug=False,
    ):
        self.initialize_model(
            model_path,
            tokenizer_path,
        )
        
        logging.info("[LLM INFO:] Loaded LLM TensorRT Engine.")

        conversation_history = {}

        while True:

            # Get the last transcription output from the queue
            transcription_output = transcription_queue.get()
            if transcription_queue.qsize() != 0:
                continue
            
            if transcription_output["uid"] not in conversation_history:
                conversation_history[transcription_output["uid"]] = []

            prompt = transcription_output['prompt'].strip()
                                
            # if prompt is same but EOS is True, we need that to send outputs to websockets
            if self.last_prompt == prompt:
                if self.last_output is not None and transcription_output["eos"]:
                    self.eos = transcription_output["eos"]
                    llm_queue.put({
                        "uid": transcription_output["uid"],
                        "llm_output": self.last_output,
                        "eos": self.eos,
                        "latency": self.infer_time
                    })
                    audio_queue.put({"llm_output": self.last_output, "eos": self.eos})
                    conversation_history[transcription_output["uid"]].append(
                        (transcription_output['prompt'].strip(), self.last_output[0].strip())
                    )
                    continue

            # input_text=[self.format_prompt_qa(prompt, conversation_history[transcription_output["uid"]])]
            input_text=[self.format_prompt_chatml(prompt, conversation_history[transcription_output["uid"]], system_prompt="You are Dolphin, a helpful AI assistant")]
            
            self.eos = transcription_output["eos"]

            batch_input_ids = self.parse_input(
                input_text=input_text,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
            )

            input_lengths = [x.size(0) for x in batch_input_ids]

            logging.info(f"[LLM INFO:] Running LLM Inference with WhisperLive prompt: {prompt}, eos: {self.eos}")
            start = time.time()
            with torch.no_grad():
                outputs = self.runner.generate(
                    batch_input_ids,
                    max_new_tokens=max_output_len,
                    max_attention_window_size=max_attention_window_size,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    temperature=1.0,
                    top_k=1,
                    top_p=0.0,
                    num_beams=num_beams,
                    length_penalty=1.0,
                    repetition_penalty=1.0,
                    stop_words_list=None,
                    bad_words_list=None,
                    lora_uids=None,
                    prompt_table_path=None,
                    prompt_tasks=None,
                    streaming=streaming,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()
            if streaming:
                for curr_outputs in throttle_generator(outputs, streaming_interval):
                    output_ids = curr_outputs['output_ids']
                    sequence_lengths = curr_outputs['sequence_lengths']
                    output = self.decode_tokens(
                        output_ids,
                        input_lengths,
                        sequence_lengths,
                        transcription_queue
                    )

                    if output is None:
                        break
                
                # Interrupted by transcription queue
                if output is None:
                    continue
            else:
                output_ids = outputs['output_ids']
                sequence_lengths = outputs['sequence_lengths']
                context_logits = None
                generation_logits = None
                if self.runner.gather_context_logits:
                    context_logits = outputs['context_logits']
                if self.runner.gather_generation_logits:
                    generation_logits = outputs['generation_logits']
                output = self.decode_tokens(
                    output_ids,
                    input_lengths,
                    sequence_lengths,
                    transcription_queue
                )
            self.infer_time = time.time() - start
            
            # if self.eos:
            if output is not None:
                output[0] = clean_llm_output(output[0])
                self.last_output = output
                self.last_prompt = prompt
                llm_queue.put({
                    "uid": transcription_output["uid"],
                    "llm_output": output,
                    "eos": self.eos,
                    "latency": self.infer_time
                })
                audio_queue.put({"llm_output": output, "eos": self.eos})
                logging.info(f"[LLM INFO:] Output: {output[0]}\nLLM inference done in {self.infer_time} ms\n\n")
            
            if self.eos:
                conversation_history[transcription_output["uid"]].append(
                    (transcription_output['prompt'].strip(), output[0].strip())
                )
                self.last_prompt = None
                self.last_output = None

def clean_llm_output(output):
    output = output.replace("\n\nDolphin\n\n", "")
    output = output.replace("\nDolphin\n\n", "")
    output = output.replace("Dolphin: ", "")
    output = output.replace("Assistant: ", "")

    if not output.endswith('.') and not output.endswith('?') and not output.endswith('!'):
        last_punct = output.rfind('.')
        last_q = output.rfind('?')
        if last_q > last_punct:
            last_punct = last_q
        
        last_ex = output.rfind('!')
        if last_ex > last_punct:
            last_punct = last_ex
        
        if last_punct > 0:
            output = output[:last_punct+1]

    return output
