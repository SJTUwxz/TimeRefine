import os
import sys
import argparse
import torch
from vtimellm.constants import IMAGE_TOKEN_INDEX, SEG_START
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model, load_lora
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip
import re


def inference(model, image, query, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def temporal_segment_inference(model, image, query, tokenizer,  args, predict_center_offset, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        seg_start = None
        cached = None
        inputs = input_ids.clone()
        next_two_words = []
        predict_words = []
        for i in range(10): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            predict_words.append(word)

            if seg_start is None and word == SEG_START:
                seg_start = i
                continue

            # output_ids = torch.cat((output_ids, pred_id.unsqueeze(0)), dim=1)
            # if word == SEG_START and seg_start is None:
            #     seg_start = i
            #     # next_token_embedding = model.model.embed_tokens(pred_id.unsqueeze(0))
            #     continue

            if seg_start is not None and i == seg_start + 1:
                # print("should be t1 ", word, flush=True)
                seg_start_embedding = outputs.hidden_states[-1]
                if args.segment_head == "linear":
                    seg_start_value = model.segment_head(seg_start_embedding)
                elif args.segment_head == "mlp":
                    seg_start_embedding = model.segment_head_layer1(seg_start_embedding)
                    seg_start_embedding = torch.relu(seg_start_embedding)
                    seg_start_value = model.segment_head_layer2(seg_start_embedding)
                seg_start_value = torch.sigmoid(seg_start_value)
                
                next_two_words = int(seg_start_value.cpu().detach().item() * 100)
                next_two_words = f"{next_two_words:02}"
                first_word_id = torch.tensor([tokenizer.convert_tokens_to_ids(next_two_words[0])]).to(seg_start_value.device)
                second_word_id = torch.tensor([tokenizer.convert_tokens_to_ids(next_two_words[1])]).to(seg_start_value.device)

                pred_id = first_word_id.unsqueeze(0)
                continue

            if seg_start is not None and i == seg_start + 2:
                pred_id = second_word_id.unsqueeze(0)
                continue

            if seg_start is not None and i == seg_start + 3:
                pass

            elif seg_start is not None and i == seg_start + 4:
                # print("should be t2 ", word, flush=True)
                seg_end_embedding = outputs.hidden_states[-1]
                if args.segment_head == "linear":
                    seg_end_value = model.segment_head(seg_end_embedding)
                elif args.segment_head == "mlp":
                    seg_end_embedding = model.segment_head_layer1(seg_end_embedding)
                    seg_end_embedding = torch.relu(seg_end_embedding)
                    seg_end_value = model.segment_head_layer2(seg_end_embedding)
                seg_end_value = torch.sigmoid(seg_end_value)

    

    seg_start_value = seg_start_value.cpu().detach().item()
    seg_end_value = seg_end_value.cpu().detach().item()

    # next_token_start = float(predict_words[2] + predict_words[3])
    # next_token_end = float(predict_words[6] + predict_words[7])

    print(predict_words, next_two_words, seg_start_value, seg_end_value, flush=True)

    if predict_center_offset:
        center, offset = seg_start_value, seg_end_value
        seg_start_value = center - offset
        seg_end_value = center + offset
    outputs = [seg_start_value, seg_end_value]
    return outputs


def token_segment_inference(model, image, query, tokenizer,  args, predict_center_offset, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        seg_start = None
        cached = None
        inputs = input_ids.clone()
        predict_words = []
        for i in range(10): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            predict_words.append(word)

            if seg_start is None and word == SEG_START:
                seg_start = i
                continue

            # output_ids = torch.cat((output_ids, pred_id.unsqueeze(0)), dim=1)
            # if word == SEG_START and seg_start is None:
            #     seg_start = i
            #     # next_token_embedding = model.model.embed_tokens(pred_id.unsqueeze(0))
            #     continue

            if seg_start is not None and i == seg_start + 1:
                # print("should be t1 ", word, flush=True)
                seg_start_embedding = outputs.hidden_states[-1]
                seg_start_value = model.segment_head(seg_start_embedding)
                seg_start_value = torch.sigmoid(seg_start_value)
                
                next_token = int(seg_start_value.cpu().detach().item() * 100)
                next_token = f"<{next_token:02}>"
                pred_id = torch.tensor([tokenizer.convert_tokens_to_ids(next_token)]).to(seg_start_value.device).unsqueeze(0)

                continue

            if seg_start is not None and i == seg_start + 2:
                pass

            elif seg_start is not None and i == seg_start + 3:
                # print("should be t2 ", word, flush=True)
                seg_end_embedding = outputs.hidden_states[-1]
                seg_end_value = model.segment_head(seg_end_embedding)
                seg_end_value = torch.sigmoid(seg_end_value)

    

    seg_start_value = seg_start_value.cpu().detach().item()
    seg_end_value = seg_end_value.cpu().detach().item()

    # next_token_start = float(predict_words[2] + predict_words[3])
    # next_token_end = float(predict_words[6] + predict_words[7])

    print(predict_words, seg_start_value, seg_end_value, flush=True)

    if predict_center_offset:
        center, offset = seg_start_value, seg_end_value
        seg_start_value = center - offset
        seg_end_value = center + offset
    outputs = [seg_start_value, seg_end_value]
    return outputs

def v6_segment_inference(model, image, query, tokenizer,  args, predict_center_offset, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        seg_start = None
        cached = None
        inputs = input_ids.clone()
        predict_words = []
        for i in range(40): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            predict_words.append(word)
            if word == "<SEG_END>":
                break

            # if seg_start is None and word == SEG_START:
            #     seg_start = i
            #     continue
            #
            # if seg_start is not None and i == seg_start + 1:
            #     # print("should be t1 ", word, flush=True)
            #     seg_start_embedding = outputs.hidden_states[-1]
            #     seg_start_value = model.segment_head(seg_start_embedding)
            #     seg_start_value = torch.sigmoid(seg_start_value)
            #
            #     next_token = int(seg_start_value.cpu().detach().item() * 100)
            #     next_token = f"<{next_token:02}>"
            #     pred_id = torch.tensor([tokenizer.convert_tokens_to_ids(next_token)]).to(seg_start_value.device).unsqueeze(0)
            #
            #     continue
            #
            # if seg_start is not None and i == seg_start + 2:
            #     pass
            #
            # elif seg_start is not None and i == seg_start + 3:
            #     # print("should be t2 ", word, flush=True)
            #     seg_end_embedding = outputs.hidden_states[-1]
            #     seg_end_value = model.segment_head(seg_end_embedding)
            #     seg_end_value = torch.sigmoid(seg_end_value)
            #
            #

    # seg_start_value = seg_start_value.cpu().detach().item()
    # seg_end_value = seg_end_value.cpu().detach().item()

    # next_token_start = float(predict_words[2] + predict_words[3])
    # next_token_end = float(predict_words[6] + predict_words[7])

    start_ind = [float(predict_words[2 + i * 6][1:3]) / 100.0 for i in range(4)] 
    end_ind = [float(predict_words[4 + i * 6][1:3]) / 100.0 for i in range(4)] 
    iou_ind = [float(predict_words[6 + i * 6][5:7]) / 10.0 for i in range(4)] 
    if args.merge_segments == "first":
        outputs = [start_ind[0], end_ind[0]]
    elif args.merge_segments == "last":
        selected_start = [a for a, c in zip(start_ind, iou_ind) if c == 1.0]
        selected_end = [b for b, c in zip(end_ind, iou_ind) if c == 1.0]
        outputs = [np.mean(selected_start), np.mean(selected_end)]
    elif args.merge_segments == "avg":
        weighted_sum = sum(a * c for a, c in zip(start_ind, iou_ind))
        sum_of_weights = sum(iou_ind)
        weighted_start = weighted_sum / sum_of_weights

        weighted_sum = sum(a * c for a, c in zip(end_ind, iou_ind))
        weighted_end = weighted_sum / sum_of_weights

        outputs = [weighted_start, weighted_end]
    elif args.merge_segments == "simple_avg":
        outputs = [np.mean(start_ind), np.mean(end_ind)]
    else:
        raise NotImplementedError("merge segments need to be in first, last and avg")
    print(predict_words, outputs, flush=True)


    return outputs 

def v7_segment_inference(model, image, query, tokenizer,  args, predict_center_offset, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        seg_start = None
        cached = None
        inputs = input_ids.clone()
        predict_words = []
        for i in range(40): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            predict_words.append(word)
            if word == "<SEG_END>":
                break

            # if seg_start is None and word == SEG_START:
            #     seg_start = i
            #     continue
            #
            # if seg_start is not None and i == seg_start + 1:
            #     # print("should be t1 ", word, flush=True)
            #     seg_start_embedding = outputs.hidden_states[-1]
            #     seg_start_value = model.segment_head(seg_start_embedding)
            #     seg_start_value = torch.sigmoid(seg_start_value)
            #
            #     next_token = int(seg_start_value.cpu().detach().item() * 100)
            #     next_token = f"<{next_token:02}>"
            #     pred_id = torch.tensor([tokenizer.convert_tokens_to_ids(next_token)]).to(seg_start_value.device).unsqueeze(0)
            #
            #     continue
            #
            # if seg_start is not None and i == seg_start + 2:
            #     pass
            #
            # elif seg_start is not None and i == seg_start + 3:
            #     # print("should be t2 ", word, flush=True)
            #     seg_end_embedding = outputs.hidden_states[-1]
            #     seg_end_value = model.segment_head(seg_end_embedding)
            #     seg_end_value = torch.sigmoid(seg_end_value)
            #
            #

    # seg_start_value = seg_start_value.cpu().detach().item()
    # seg_end_value = seg_end_value.cpu().detach().item()

    # next_token_start = float(predict_words[2] + predict_words[3])
    # next_token_end = float(predict_words[6] + predict_words[7])
    start_ind = []
    end_ind = []
    iou_ind = []
    for i in range(len(predict_words)):
        if predict_words[i] in ["<SEG_START>", "<AND_SEG>"]:
            start = float(predict_words[i+1] + predict_words[i+2]) / 100.0
            end = float(predict_words[i+5] + predict_words[i+6]) / 100.0
            iou = float(predict_words[i+8][5:7]) / 10.0
            start_ind.append(start)
            end_ind.append(end)
            iou_ind.append(iou)

    # start_ind = [float(predict_words[2 + i * 6][1:3]) / 100.0 for i in range(4)] 
    # end_ind = [float(predict_words[4 + i * 6][1:3]) / 100.0 for i in range(4)] 
    # iou_ind = [float(predict_words[6 + i * 6][5:7]) / 10.0 for i in range(4)] 
    if args.merge_segments == "first":
        outputs = [start_ind[0], end_ind[0]]
    elif args.merge_segments == "last":
        selected_start = [a for a, c in zip(start_ind, iou_ind) if c == 1.0]
        selected_end = [b for b, c in zip(end_ind, iou_ind) if c == 1.0]
        outputs = [np.mean(selected_start), np.mean(selected_end)]
    elif args.merge_segments == "avg":
        weighted_sum = sum(a * c for a, c in zip(start_ind, iou_ind))
        sum_of_weights = sum(iou_ind)
        weighted_start = weighted_sum / sum_of_weights

        weighted_sum = sum(a * c for a, c in zip(end_ind, iou_ind))
        weighted_end = weighted_sum / sum_of_weights

        outputs = [weighted_start, weighted_end]
    elif args.merge_segments == "simple_avg":
        outputs = [np.mean(start_ind), np.mean(end_ind)]
    else:
        raise NotImplementedError("merge segments need to be in first, last and avg")


    return outputs 


def v8_segment_inference(model, image, query, tokenizer,  args, predict_center_offset, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        seg_start = None
        last_start = None
        last_end = None
        last_start_offset = None
        last_start_offset_symbol = None
        last_end_offset = None
        last_end_offset_symbol = None
        has_rethink = None 
        cached = None
        predicted_segments = []
        inputs = input_ids.clone()
        predict_words = []
        for i in range(300): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            if word == "<SEG_START>":
                seg_start = i
            
            if word == "<RETHINK>":
                has_rethink = i

            if seg_start is not None and has_rethink is None:
                if i == seg_start + 1:
                    # this is the very start segment prediction, no need to change the predicted word
                    last_start = int(word[1:-1])
                elif i == seg_start + 3:
                    # this is the end segment prediction
                    last_end = int(word[1:-1])
                elif i == seg_start + 6:
                    # this is the symbol
                    if '-' in word:
                        last_start_offset_symbol = -1
                    elif '+' in word:
                        last_start_offset_symbol = 1
                    else:
                        raise NotImplementedError("symbol has to be + or -.")
                elif i == seg_start + 7:
                    last_start_offset = int(word[1:-1]) * last_start_offset_symbol
                    last_start = last_start + last_start_offset
                elif i == seg_start +  9:
                    if '-' in word:
                        last_end_offset_symbol = -1
                    elif '+' in word:
                        last_end_offset_symbol = 1
                    else:
                        raise NotImplementedError("symbol has to be + or -.")
                    
                elif i == seg_start + 10:
                    last_end_offset = int(word[1:-1]) * last_end_offset_symbol 
                    last_end = last_end + last_end_offset
                    predicted_segments.append([last_start, last_end])

            elif has_rethink is not None:
                # if rethink has been met, then last_segments are all not None
                if i == has_rethink + 1:
                    word = last_start
                    word = min(round(word), 99)
                    word = f"<{word:02}>"
                    pred_id = torch.tensor([tokenizer.convert_tokens_to_ids(word)]).to(pred_id.device).unsqueeze(0)
                elif i == has_rethink + 3:
                    word = last_end
                    word = min(round(word), 99)
                    word = f"<{word:02}>"
                    pred_id = torch.tensor([tokenizer.convert_tokens_to_ids(word)]).to(pred_id.device).unsqueeze(0)

                elif i == has_rethink + 6:
                    if '-' in word:
                        last_start_offset_symbol = -1
                    elif '+' in word:
                        last_start_offset_symbol = 1
                    else:
                        raise NotImplementedError("symbol has to be + or -.")
                elif i == has_rethink + 7:
                    last_start_offset = int(word[1:-1]) * last_start_offset_symbol
                    last_start = last_start + last_start_offset
                elif i == has_rethink +  9:
                    if '-' in word:
                        last_end_offset_symbol = -1
                    elif '+' in word:
                        last_end_offset_symbol = 1
                    else:
                        raise NotImplementedError("symbol has to be + or -.")
                elif i == has_rethink + 10:
                    last_end_offset = int(word[1:-1]) * last_end_offset_symbol 
                    last_end = last_end + last_end_offset
                    predicted_segments.append([last_start, last_end])

            predict_words.append(word)
            if word == "<SEG_END>":
                break

    print(predict_words, flush=True)
    if args.merge_segments == "first":
        outputs = predicted_segments[0]
    elif args.merge_segments == "last":
        outputs = predicted_segments[-1]
    elif args.merge_segments == "simple_avg":
        # outputs = [np.mean(start_ind), np.mean(end_ind)]
        pass
    else:
        raise NotImplementedError("merge segments need to be in first, last and avg")


    return outputs 

def v9_segment_inference(model, image, query, tokenizer,  args, predict_center_offset, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    timestep = []
    start_segment = None
    end_segment = None
    start_velocity = None
    end_velocity = None
    predict_words = []

    actual_start_segments = []
    actual_end_segments = []
    middle_start_segments = []
    middle_end_segments = []
    cur_timestep = 800

    with torch.inference_mode():
        seg_start = None
        cached = None
        inputs = input_ids.clone()
        for i in range(300): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            # if word matches timestep, then this is the start of time
            timestep_pattern = r"^<TIME_\d{3}>$"
            segment_pattern = r"^<(\d{2})>$"
            velocity_pattern = r"^<([+-])(\d{2})>$"

            # modify timestep to the evenly sampled timestep
            if re.match(timestep_pattern, word):
                word = f"<TIME_{cur_timestep:03}>"
                timestep.append(cur_timestep / 1000)
                cur_timestep = cur_timestep - 200 
                pred_id = torch.tensor([tokenizer.convert_tokens_to_ids(word)]).to(pred_id.device).unsqueeze(0)

            # segment detected
            elif re.match(segment_pattern, word):
                match = re.match(segment_pattern, word)
                segment_time = float(match.group(1))
                # this is a start 
                if start_segment is None and end_segment is None: 
                    start_segment = segment_time
                    if len(middle_start_segments) > 0:
                        segment_time = middle_start_segments[-1] * 100
                # this is an end
                elif start_segment is not None and end_segment is None:
                    end_segment = segment_time
                    if len(middle_end_segments) > 0:
                        segment_time = middle_end_segments[-1] * 100
                # change the segment to last computed actual start and end segment
                word = min(round(segment_time), 99)
                word = f"<{word:02}>"
                pred_id = torch.tensor([tokenizer.convert_tokens_to_ids(word)]).to(pred_id.device).unsqueeze(0)

            elif re.match(velocity_pattern, word):
                match = re.match(velocity_pattern, word)
                velocity_sign = match.group(1)
                velocity_value = float(match.group(2))
                if velocity_sign == '-':
                    velocity_value = -velocity_value
                if start_velocity is None:
                    start_velocity = velocity_value
                elif end_velocity is None:

                    end_velocity = velocity_value

                    last_time = timestep[-1]
                    last_start = start_segment
                    last_end = end_segment
                    last_start_velo = start_velocity
                    last_end_velo = end_velocity

                    # check if the other three values are not None, otherwise continue
                    if start_segment is None or end_segment is None or start_velocity is None:
                        end_velocity = None
                        continue

                    tmp_start = last_start + last_time * last_start_velo
                    tmp_start = max(0, tmp_start)
                    tmp_end = last_end + last_time * last_end_velo

                    actual_start_segments.append(tmp_start / 100)
                    actual_end_segments.append(tmp_end / 100)

                    middle_start = last_start + 0.2 * last_start_velo
                    middle_start = max(0, middle_start)
                    middle_end = last_end + 0.2 * last_end_velo

                    middle_start_segments.append(middle_start / 100)
                    middle_end_segments.append(middle_end / 100)

                    # return the following variables to the initial state
                    start_segment = None
                    end_segment = None
                    start_velocity = None
                    end_velocity = None

            if word == "<SEG_END>" or (len(timestep) > 0 and timestep[-1] < 0):
                break
            predict_words.append(word)

    start_ind = actual_start_segments
    end_ind = actual_end_segments
    if args.merge_segments == "first":
        outputs = [start_ind[0], end_ind[0]]
    elif args.merge_segments == "last":
        outputs = [start_ind[-1], end_ind[-1]]
    elif args.merge_segments == "simple_avg":
        outputs = [np.mean(start_ind), np.mean(end_ind)]
    else:
        raise NotImplementedError("merge segments need to be in first, last and avg")


    return outputs, predict_words, start_ind, end_ind 



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--video_path", type=str, default="images/demo.mp4")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    # model.get_model().mm_projector.to(torch.float16)
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))

    query = "describe the video."
    print("query: ", query)
    print("answer: ", inference(model, features, "<video>\n " + query, tokenizer))


