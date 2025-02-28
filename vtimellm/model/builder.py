import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from vtimellm.model import *
from peft import PeftModel
from vtimellm.constants import *


def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model

def load_pretrained_model(args, stage2=None, stage3=None):
    kwargs = {'torch_dtype': torch.float16}

    # model_path = os.path.expanduser(args.model_path)
    model_base = args.model_base


    # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    print('Loading VTimeLLM from base model...')
    if 'chatglm' in model_base:
        tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
        model = VTimeLLMChatGLMForCausalLM.from_pretrained(model_base)
    else:
        # tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
        model = VTimeLLMLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, model_args=args, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    if args.model_segment_format == "v1":
        num_new_tokens = tokenizer.add_tokens([SEG_START, SEG_END], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_segment_format == "v2":
        num_new_tokens = tokenizer.add_tokens([SEG_START, SEG_END, SEG_FIN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_segment_format == "v3":
        num_new_tokens = tokenizer.add_tokens([SEG_START], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_segment_format == "v4" or args.model_segment_format == "v5":
        segment_tokens = [SEG_START]
        for i in range(100):
            segment_tokens.append(SEGMENT_TOKEN_FORMAT.format(i))
        num_new_tokens = tokenizer.add_tokens(segment_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_segment_format == "v6":
        segment_tokens = [SEG_START, SEG_END, AND_SEG]
        for i in range(11):
            segment_tokens.append(IOU_FORMAT.format(i))
        for i in range(100):
            segment_tokens.append(SEGMENT_TOKEN_FORMAT.format(i))
        num_new_tokens = tokenizer.add_tokens(segment_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_segment_format == "v7":
        segment_tokens = [SEG_START, SEG_END, AND_SEG]
        for i in range(11):
            segment_tokens.append(IOU_FORMAT.format(i))
        num_new_tokens = tokenizer.add_tokens(segment_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_segment_format == "v8":
        # segment_tokens = [SEG_START, SEG_END, RETHINK, OFFSET]
        segment_tokens = [SEG_START, SEG_END, RETHINK]
        if args.add_temporal_tokens == "yes":
            for i in range(100):
                segment_tokens.append(SEGMENT_TOKEN_FORMAT.format(i))
            for i in range(100):
                segment_tokens.append(POSTIVE_VELOCITY.format(i))
            for i in range(100):
                segment_tokens.append(NEGATIVE_VELOCITY.format(i))
        num_new_tokens = tokenizer.add_tokens(segment_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    elif args.model_segment_format == "v9":
        segment_tokens = [SEG_END]
        if args.add_temporal_tokens == "yes":
            for i in range(100):
                segment_tokens.append(SEGMENT_TOKEN_FORMAT.format(i))
                segment_tokens.append(POSTIVE_VELOCITY.format(i))
                segment_tokens.append(NEGATIVE_VELOCITY.format(i))
            for i in range(1000):
                segment_tokens.append(TIMESTEP_FORMAT.format(i))
        num_new_tokens = tokenizer.add_tokens(segment_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    elif args.model_segment_format == "v10":
        segment_tokens = [SEG_START, SEG_END, RETHINK, "<OFFSET>"]
        if args.add_temporal_tokens == "yes":
            for i in range(100):
                segment_tokens.append(SEGMENT_TOKEN_FORMAT.format(i))
            for i in range(100):
                segment_tokens.append(POSTIVE_VELOCITY.format(i))
            for i in range(100):
                segment_tokens.append(NEGATIVE_VELOCITY.format(i))
        num_new_tokens = tokenizer.add_tokens(segment_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_segment_format == "v11":
        segment_tokens = [SEG_START, SEG_END, RETHINK, "<OFFSET>"]
        if args.add_temporal_tokens == "yes":
            for i in range(100):
                segment_tokens.append(SEGMENT_TOKEN_FORMAT.format(i))
            segment_tokens.append("<+>")
            segment_tokens.append("<->")
        elif args.add_temporal_tokens == "yes_sep":
            for i in range(10):
                segment_tokens.append(f"<{i}>")
            segment_tokens.append("<+>")
            segment_tokens.append("<->")
        num_new_tokens = tokenizer.add_tokens(segment_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    # load stage1:
    model.get_model().initialize_vision_modules(args)

    if stage2 is not None:
        print('Loading stage2 weights...')
        model = load_lora(model, stage2)
        print('Merging stage2 weights...')
        model = model.merge_and_unload()
        if stage3 is not None:
            print('Loading stage3 weights...')
            model = load_lora(model, stage3)
            print('Merging stage3 weights...')
            model = model.merge_and_unload()


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
