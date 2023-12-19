import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def read_model_name(engine_dir: str):
    engine_version = tensorrt_llm.builder.get_engine_version(engine_dir)

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


class MistralTensorRTLLM:
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
            torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in batch_input_ids
        ]
        return batch_input_ids
    
    def decode_tokens(
        self,
        output_ids,
        input_lengths,
        sequence_lengths,
        ):
        batch_size, num_beams, _ = output_ids.size()
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = self.tokenizer.decode(inputs)
            output = []
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = self.tokenizer.decode(outputs)
                output.append(output_text)
        return output
    
    def run(
        self,
        transcription_queue=None,
        llm_queue=None,
        input_text=None, 
        max_output_len=20, 
        max_attention_window_size=4096, 
        num_beams=1, 
        streaming=True,
        streaming_interval=4,
        debug=False,
    ):
        self.initialize_model(
        "/root/TensorRT-LLM/examples/llama/tmp/mistral/7B/trt_engines/fp16/1-gpu",
        "teknium/OpenHermes-2.5-Mistral-7B",
        )
        print("Loaded LLM...")
        while True:
 
            # while transcription
            transcription_output = transcription_queue.get()
            input_text=transcription_output['prompt'].strip()
            
            print("Whisper: ", input_text)
            batch_input_ids = self.parse_input(
                input_text=input_text,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
            )

            input_lengths = [x.size(1) for x in batch_input_ids]
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
                        sequence_lengths
                    )
            else:
                output_ids = outputs['output_ids']
                sequence_lengths = outputs['sequence_lengths']
                context_logits = None
                generation_logits = None
                if runner.gather_all_token_logits:
                    context_logits = outputs['context_logits']
                    generation_logits = outputs['generation_logits']
                output = self.decode_tokens(
                    output_ids,
                    input_lengths,
                    sequence_lengths,
                )
            llm_queue.put({"uid": transcription_output["uid"], "llm_output": output})


if __name__=="__main__":
    llm = MistralTensorRTLLM()
    llm.initialize_model(
        "/root/TensorRT-LLM/examples/llama/tmp/mistral/7B/trt_engines/fp16/1-gpu",
        "teknium/OpenHermes-2.5-Mistral-7B",
    )
    print("intialized")
    for i in range(1):
        output = llm(
            ["Born in north-east France, Soyer trained as a"], streaming=True
        )
    print(output)


