from trl import DPOTrainer
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser
import yaml
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from typing import Dict
import torch
import csv
import functools
from datasets import Dataset
from peft import get_peft_model, PeftConfig, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass
import json


def readData(dataPath):
    reader = csv.reader(open(dataPath, "r", encoding="utf-8"))
    data = []
    for D in reader:
        data.append(D)

    return data[1:]

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # print(tokenizer.eos_token_id)
    # print(tokenizer.im_end_id)
    
    end_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.im_end_id
    # Or, depending on your model.

    assert end_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert end_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert end_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(end_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(end_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch

def collate_fn(batch, tokenizer):
    # first, pad everything to the same length
    tokenizer.pad_token_id = tokenizer.eod_id
    padded_batch = {}
    for k in batch[0].keys():
        if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
            if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
            else:
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
            if k.endswith('_input_ids'):
                padding_value = tokenizer.pad_token_id
            elif k.endswith('_labels'):
                padding_value = -100
            elif k.endswith('_attention_mask'):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")

            padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                padded_batch[k] = padded_batch[k].flip(dims=[1])
        else:
            padded_batch[k] = [ex[k] for ex in batch]

    return padded_batch


def test_dataSet(tokenizer, data):
    prompt = [D[0] for D in data]
    accept = [D[1] for D in data]
    reject = [D[2] for D in data]

    for P, A, R in zip(prompt, accept, reject):
        tokenize_data = tokenize_batch_element(P, chosen=A, rejected=R, truncation_mode="keep_start", tokenizer=tokenizer, max_length=2048, max_prompt_length=2048)
        # tokenize_data = collate_fn([tokenize_data], tokenizer)
        # print(tokenize_data)
        yield tokenize_data



def load_model(config, train_args):
    policy = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, local_files_only = True, torch_dtype = torch.bfloat16, trust_remote_code=True, cache_dir=config.local_dirs)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, local_files_only = True, torch_dtype = torch.bfloat16, trust_remote_code=True, cache_dir=config.local_dirs)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, use_fast=False, local_files_only = True, trust_remote_code=True, cache_dir=config.local_dirs)

    policy = policy.to("cuda:{}".format(train_args.local_rank))
    ref_model = ref_model.to("cuda:{}".format(train_args.local_rank))

    return policy, ref_model, tokenizer

def readJsonData(dataPath):
    data = []
    with open(dataPath, 'r', encoding="utf-8") as f:
        for D in f.readlines():
            jsonD = json.loads(D)
            data.append([jsonD["q"], jsonD["a"], jsonD["r"]])

    return data

def main():
    config = OmegaConf.load("./config.yaml")
    train_args = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()[0]

    policy, ref_model, tokenizer = load_model(config, train_args)

    data = readJsonData(config.dataset)
    dataset = Dataset.from_generator(lambda: test_dataSet(tokenizer, data))

    policy.gradient_checkpointing_enable() 
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    policy.enable_input_require_grads()
    # peft_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         inference_mode=False,
    #         r=8,
    #         lora_alpha=16,
    #         lora_dropout=0.1,
    #         target_modules = ['W_pack','down_proj','up_proj','gate_proj'] # 把model打印出来，找跟attention相关的模块
    #     )
        
    # policy = get_peft_model(policy, peft_config)

    if len(config.lora.previous_lora_weights) > 0:    # load lora
        policy = PeftModel.from_pretrained(policy, config.lora.previous_lora_weights)
        ref_model = PeftModel.from_pretrained(ref_model, config.lora.previous_lora_weights)
            # see: https://github.com/huggingface/peft/issues/184
        for name, param in policy.named_parameters():
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True


    ref_model = ref_model.eval().requires_grad_(False)
    print(train_args)
    trainer = DPOTrainer(policy, ref_model, args = train_args, train_dataset=dataset, data_collator= functools.partial(collate_fn, tokenizer = tokenizer))

    trainer.train()
    trainer.save_model(train_args.output_dir)

if __name__ == "__main__":
    main()