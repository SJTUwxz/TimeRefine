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


    if predict_center_offset:
        center, offset = seg_start_value, seg_end_value
        seg_start_value = center - offset
        seg_end_value = center + offset
    outputs = [seg_start_value, seg_end_value]
    return outputs

def v10_segment_inference_parallel(model, image, query, tokenizer,  args, predict_center_offset, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    linear_prediction_start = []
    linear_prediction_end = []
    with torch.inference_mode():
        seg_start = None
        cached = None
        inputs = input_ids.clone()
        predict_words = []
        offset_index = None
        for i in range(200): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            if args.predict_goal is not None:
                if args.predict_goal == "offset":
                    if word == "<OFFSET>":
                        offset_index = i
                elif args.predict_goal == "rethink":
                    if word == "<SEG_START>" or word == "<RETHINK>":
                        offset_index = i
            if args.segment_head is not None:
                if offset_index is not None and i == offset_index + 1:
                    segment_embedding = outputs.hidden_states[-1]
                    if args.segment_head == "linear_2output":
                        segment = model.segment_head(segment_embedding)
                    elif args.segment_head == "mlp_2output":
                        segment_embedding = model.segment_head_layer1(segment_embedding)
                        segment_embedding = torch.relu(segment_embedding)
                        segment = model.segment_head_layer2(segment_embedding)
                    else:
                        raise NotImplementedError("segment head not implemented")
                    segment = torch.sigmoid(segment)
                    segment = segment.cpu().detach().numpy().flatten()
                    linear_start, linear_end = segment
                    linear_prediction_start.append(linear_start)
                    linear_prediction_end.append(linear_end)

            predict_words.append(word)
            if word == "<SEG_END>":
                break


    start_ind = []
    end_ind = []
    if args.add_temporal_tokens != "yes":
        for i in range(len(predict_words)):
            if predict_words[i] in ["<SEG_START>", "<RETHINK>"]:
                if args.use_fast == "no":
                    try:
                        start = float(predict_words[i+1].strip("_") + predict_words[i+2].strip("_")) / 100.0
                        end = float(predict_words[i+5].strip("_") + predict_words[i+6].strip("_")) / 100.0
                        start_offset_symbol = predict_words[i+8]
                        start_offset = float(predict_words[i+9].strip("_")  + predict_words[i+10].strip("_") ) / 100.0
                        if '-' in start_offset_symbol:
                            start_offset = - start_offset
                        end_offset_symbol = predict_words[i+12]
                        end_offset = float(predict_words[i+13].strip("_")  + predict_words[i+14].strip("_") ) / 100.0
                        if '-' in end_offset_symbol:
                            end_offset = - end_offset
                        start_ind.append(start + start_offset)
                        end_ind.append(end + end_offset)
                    except:
                        i = i + 14
                        continue
                else:
                    try:
                        start = float(predict_words[i+2] + predict_words[i+3]) / 100.0
                        end = float(predict_words[i+6] + predict_words[i+7]) / 100.0
                        start_offset_symbol = predict_words[i+11]
                        start_offset = float(predict_words[i+12] + predict_words[i+13] ) / 100.0
                        if '-' in start_offset_symbol:
                            start_offset = - start_offset
                        end_offset_symbol = predict_words[i+15]
                        end_offset = float(predict_words[i+16] + predict_words[i+17]) / 100.0
                        if '-' in end_offset_symbol:
                            end_offset = - end_offset
                        start_ind.append(start + start_offset)
                        end_ind.append(end + end_offset)
                    except:
                        i = i + 18
                        continue

    else:
        for i in range(len(predict_words)):
            if predict_words[i] in ["<SEG_START>", "<RETHINK>"]:
                if args.use_fast == "no":
                    try:
                        start = float(predict_words[i+1][1:-1])/ 100.0
                        end = float(predict_words[i+3][1:-1]) / 100.0
                        start_offset_symbol = 1.0 if '+' in predict_words[i+5] else -1.0
                        start_offset = float(predict_words[i+5][2:-1]) / 100.0 * start_offset_symbol
                        end_offset_symbol = 1.0 if '+' in predict_words[i+7] else -1.0
                        end_offset = float(predict_words[i+7][2:-1]) / 100.0 * end_offset_symbol
                        start_ind.append(start + start_offset)
                        end_ind.append(end + end_offset)
                    except:
                        i = i + 11
                        continue
                else:
                    try:
                        start = float(predict_words[i+2][1:-1])/ 100.0
                        end = float(predict_words[i+6][1:-1]) / 100.0
                        start_offset_symbol = 1.0 if '+' in predict_words[i+10] else -1.0
                        start_offset = float(predict_words[i+10][2:-1]) / 100.0 * start_offset_symbol
                        end_offset_symbol = 1.0 if '+' in predict_words[i+14] else -1.0
                        end_offset = float(predict_words[i+14][2:-1]) / 100.0 * end_offset_symbol
                        start_ind.append(start + start_offset)
                        end_ind.append(end + end_offset)
                    except:
                        i = i + 15
                        continue

    if len(start_ind) == 0 or len(end_ind) == 0:
        print(predict_words, flush=True)
        if len(start_ind) == 0:
            start_ind.append(0)
        if len(end_ind) == 0:
            end_ind.append(99)

    if args.merge_segments == "first":
        outputs = [start_ind[0], end_ind[0]]
    elif args.merge_segments == "last":
        outputs = [start_ind[-1], end_ind[-1]]
    elif args.merge_segments == "simple_avg":
        outputs = [np.mean(start_ind), np.mean(end_ind)]
    else:
        raise NotImplementedError("merge segments need to be in first, last and avg")

    if args.segment_head is not None:
        if len(linear_prediction_start) == 0:
            linear_start = 0
        else:
            linear_start = sum(linear_prediction_start) / len(linear_prediction_start)
        if len(linear_prediction_end) == 0:
            linear_end = 99
        else:
            linear_end = sum(linear_prediction_end) / len(linear_prediction_end)
        outputs.append(linear_start)
        outputs.append(linear_end)

    return outputs, predict_words

def offset_caption_inference_generate(model, image, query, tokenizer, args):
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
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    if outputs[-2:] != "}]":
        last_brace_index = outputs.rfind("}")
        outputs = outputs[:last_brace_index + 1] + "]"
        print(outputs, flush=True)
    return outputs


def offset_caption_inference(model, image, query, tokenizer, args):
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
        offset_index = None
        for i in range(1000): # 20 as the max length here, as an ex
            if cached is None:
                outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                cached = outputs.past_key_values
            else:
                outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                cached = outputs.past_key_values

            pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            word = tokenizer.convert_ids_to_tokens(int(pred_id.item()))
            if word == "<OFFSET>":
                offset_index = i
            if args.segment_head is not None:
                if offset_index is not None and i == offset_index + 1:
                    segment_embedding = outputs.hidden_states[-1]
                    if args.segment_head == "linear_2output":
                        segment = model.segment_head(segment_embedding)
                    elif args.segment_head == "mlp_2output":
                        segment_embedding = model.segment_head_layer1(segment_embedding)
                        segment_embedding = torch.relu(segment_embedding)
                        segment = model.segment_head_layer2(segment_embedding)
                    else:
                        raise NotImplementedError("segment head not implemented")
                    segment = torch.sigmoid(segment)
                    segment = segment.cpu().detach().numpy().flatten()
                    linear_start, linear_end = segment

            predict_words.append(word)
            if word == "}]":
                break

    predict_words = ''.join(predict_words)
    predict_words = predict_words.replace("▁", " ")

    print(predict_words, flush=True)


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


