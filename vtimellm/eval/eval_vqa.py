import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.append(root_dir)

import clip
import re
import argparse
import torch
import json
import numpy as np
import pickle as pkl
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import VideoExtractor
from vtimellm.inference import *
import pandas as pd

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="vtimellm/eval/data_example.json")
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--task", type=str, default='grounding', choices=['all', 'grounding', 'captioning', 'temporal_grounding', 'token_grounding', 'v6_grounding', 'v7_grounding', 'v8_grounding', 'v9_grounding', 'v10_grounding', 'v11_grounding', 'vqa'])
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log/example_log.txt')
    parser.add_argument("--model_segment_format", type=str, default=None)
    parser.add_argument("--segment_head", type=str, default=None)
    parser.add_argument("--predict_center_offset", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--merge_result", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--merge_segments", type=str, default="first")
    parser.add_argument("--add_temporal_tokens", type=str, default="no")
    parser.add_argument("--predict_goal", type=str, default=None)
    parser.add_argument("--parallel_inference", type=str, default="parallel")
    parser.add_argument("--use_fast", type=str, default="no")


    args = parser.parse_args()
    return args

def iou(outputs, gt):
    matches = re.search(r"(\d{2}) (to|and) (\d{2})", outputs)
    if not matches:
        return 0
    from_number = float(matches.group(1)) / 100
    to_number = float(matches.group(3)) / 100
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)

def token_iou(outputs, gt, merge_result=False):
    matches = re.search(r"(<(\d{2})>) (to|and) (<(\d{2})>)", outputs)
    if not matches:
        return 0
    from_number = float(matches.group(1).strip('<').strip('>')) / 100
    to_number = float(matches.group(4).strip('<').strip('>')) / 100
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)

def segment_iou(outputs, gt, merge_result=False):
    from_number = float(outputs[0]) 
    to_number = float(outputs[1])
    if merge_result:
        # merge result use 99 instead of 0.99
        from_number = from_number / 100
        to_number = to_number / 100
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)

def centeroffset_iou(outputs, gt):
    center = float(outputs[0]) 
    offset = float(outputs[1])
    from_number = center - offset
    to_number = center + offset
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)



def write_log(log_path, video_id, task, query_id, answer, predict_words=None, start_predictions=None, end_predictions=None, info=None):
    log = {
        'video_id': video_id,
        'task': task,
        'query_id': query_id,
        'answer': answer,
        'predict_words': predict_words,
        'start_predictions': start_predictions,
        'end_predictions': end_predictions
    }
    if info is not None:
        log['info'] = info
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')

questions = {
    'grounding': ['During which frames can we see {}?'],
    'captioning': ['Could you please describe the events in the video in detail? Be specific about the activities of individuals, their surroundings, and interactions with others. The output should be in JSON format, structured as follows: {"event": "xx", "timestamps": "from <SEG_START> xx to xx <OFFSET> xx and xx <RETHINK> xx to xx <OFFSET> xx and xx ... <SEG_END>"}.']
}

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)
    if args.predict_center_offset == "True":
        predict_center_offset = True
        print("predicting center and offset")
    else:
        predict_center_offset = False

    if args.merge_result == "True":
        merge_result = True
        merged_result = pkl.load(open("/home/fengchan/stor_sun/projects-ng/xizi/VTimeLLM_v3/checkpoints/v3_exps/vtimellm-stage2-angular-l1_loss-v3-10.0-checkpoint-6000-eval/merged_output.pkl","rb"))
    else:
        merge_result = False

    if args.video_folder is not None:
        clip_model, _ = clip.load(args.clip_path)
        clip_model.eval()
        clip_model = clip_model.cuda()

        video_loader = VideoExtractor(N=100)

        transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    # js = json.load(open(args.data_path))
    # for id, data in tqdm(js.items()):
    file = open(args.data_path, "r", encoding="utf-8")
    for line in file:
        obj = json.loads(line)
        id = obj["video_id"]
        question = obj["question"]
        answer = obj["answer"]
        features = None

        if args.feat_folder is not None:
            feat_path = os.path.join(args.feat_folder, f"{id}.npy")
            if os.path.isfile(feat_path):
                features = torch.from_numpy(np.load(feat_path)).cuda()

        if features is None and args.video_folder is not None:
            ext = 'mp4'
            # for ext in ['mp4', 'mkv', 'webm']:
            video_path = os.path.join(args.video_folder, f"{id}.{ext}")
            if os.path.isfile(video_path):
                _, images = video_loader.extract({'id': None, 'video': video_path})

                images = transform(images / 255.0)
                images = images.to(torch.float16)
                with torch.no_grad():
                    features = clip_model.encode_image(images.to('cuda'))

        if features is None:
            print(f'Can not find video {id}')
            continue

        query = "<video>\n Please answer the question {}?"
        if args.task in ['vqa', 'all']:
            answer = inference(model, features, query.format(question), tokenizer)
            write_log(args.log_path, id, query.format(question), answer, obj['answer'])

 
        # if args.task in ['captioning', 'all']:
        #     for query_id, query in enumerate(questions['captioning']):
        #         if args.add_temporal_tokens == "yes":
        #             answer = offset_caption_inference_generate(model, features, "<video>\n " + query, tokenizer, args)
        #         else:
        #             answer = offset_caption_inference_generate(model, features, "<video>\n " + query, tokenizer, args)
        #         write_log(args.log_path+"_captioning", id, 'captioning', query_id, answer)
        #
        # if args.task in ['grounding', 'all']:
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             answer = inference(model, features, "<video>\n" + query.format(sentence), tokenizer)
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             u = iou(answer, gt)
        #             write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
        #
        # if args.task in ['temporal_grounding', 'all']:
        #     assert args.model_segment_format in ["v3"], "temporal grounding task needs to specify model_segment_format"
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #
        #             else:
        #                 answer = temporal_segment_inference(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             u = segment_iou(answer, gt, merge_result)
        #             # print(num_features_per_video, gt, answer, u, flush=True)
        #             write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
        #
        # if args.task in ['token_grounding', 'all']:
        #     assert args.model_segment_format in ["v4", "v5"], "token grounding only takes input segment format v4 and v5."
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #             else:
        #                 if args.model_segment_format == "v4":
        #                     answer = inference(model, features, "<video>\n" + query.format(sentence), tokenizer)
        #                     u = token_iou(answer, gt, merge_result)
        #                     print(answer,u, flush=True)
        #                 elif args.model_segment_format == "v5":
        #                     answer = token_segment_inference(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                     u = segment_iou(answer, gt, merge_result)
        #                 else:
        #                     raise NotImplementedError("segment format not implemented")
        #             # print(num_features_per_video, gt, answer, u, flush=True)
        #             write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
        #
        # if args.task in ['v6_grounding', 'all']:
        #     assert args.model_segment_format in ["v6"], "v6 grounding only takes input segment format v6."
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #             else:
        #                 if args.model_segment_format == "v6":
        #                     answer = v6_segment_inference(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                     print(gt, flush=True)
        #                     u = segment_iou(answer, gt, merge_result)
        #                 else:
        #                     raise NotImplementedError("segment format not implemented")
        #             write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
        #
        # if args.task in ['v7_grounding', 'all']:
        #     assert args.model_segment_format in ["v7"], "v7 grounding only takes input segment format v7."
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #             else:
        #                 answer = v7_segment_inference(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                 u = segment_iou(answer, gt, merge_result)
        #             write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
        #
        # if args.task in ['v8_grounding', 'all']:
        #     assert args.model_segment_format in ["v8"], "v8 grounding only takes input segment format v8."
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #             else:
        #                 if args.parallel_inference == "parallel":
        #                     answer, predict_words = v8_segment_inference_parallel(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                 elif args.parallel_inference == "step":
        #                     answer, predict_words = v8_segment_inference_step_by_step(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                 u = segment_iou(answer, gt, merge_result)
        #             write_log(args.log_path, id, 'grounding', query_id, answer, predict_words, info={"sentence_id": sentence_id, 'iou': u})
        #
        #
        # if args.task in ['v9_grounding', 'all']:
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #             else:
        #                 answer, predict_words, start_predictions, end_predictions = v9_segment_inference(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset)
        #                 u = segment_iou(answer, gt, merge_result)
        #             write_log(args.log_path, id, 'grounding', query_id, answer, predict_words, start_predictions, end_predictions, info={"sentence_id": sentence_id, 'iou': u})
        #
        #
        # if args.task in ['v10_grounding', 'all']:
        #     assert args.model_segment_format in ["v10"], "v10 grounding only takes input segment format v10."
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #             else:
        #                 if args.parallel_inference == "parallel":
        #                     answer, predict_words = v10_segment_inference_parallel(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                 elif args.parallel_inference == "step":
        #                     raise NotImplementedError("step by step inference is not implemented yet.")
        #                     answer, predict_words = v10_segment_inference_step_by_step(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                 if args.segment_head is not None:
        #                     text_answer = answer[:2]
        #                     linear_answer = answer[2:]
        #                     start = (answer[0] + answer[2]) / 2
        #                     end = (answer[1] + answer[3]) / 2
        #                     merge_answer = [start, end]
        #                     text_u = segment_iou(text_answer, gt, merge_result)
        #                     write_log(args.log_path+"_text", id, 'grounding', query_id, text_answer, predict_words, info={"sentence_id": sentence_id, 'iou': text_u})
        #                     linear_u = segment_iou(linear_answer, gt, merge_result)
        #                     write_log(args.log_path+"_linear", id, 'grounding', query_id, linear_answer, predict_words, info={"sentence_id": sentence_id, 'iou': linear_u})
        #                     merge_u = segment_iou(merge_answer, gt, merge_result)
        #                     write_log(args.log_path+"_merge", id, 'grounding', query_id, merge_answer, predict_words, info={"sentence_id": sentence_id, 'iou': merge_u})
        #             if args.segment_head is None:
        #                 u = segment_iou(answer, gt, merge_result)
        #                 write_log(args.log_path, id, 'grounding', query_id, answer, predict_words, info={"sentence_id": sentence_id, 'iou': u})
        #
        # if args.task in ['v11_grounding', 'all']:
        #     assert args.model_segment_format in ["v11"], "v10 grounding only takes input segment format v11."
        #     ind = 0
        #     for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
        #         sentence = sentence.strip().lower()
        #         if sentence.endswith("."):
        #             sentence = sentence[:-1]
        #
        #         for query_id, query in enumerate(questions['grounding']):
        #             gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
        #             if merge_result:
        #                 answer = merged_result[ind]
        #                 ind = ind + 1
        #             else:
        #                 if args.parallel_inference == "parallel":
        #                     answer, predict_words = v11_segment_inference_parallel(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                 elif args.parallel_inference == "step":
        #                     raise NotImplementedError("step by step inference is not implemented yet.")
        #                     answer, predict_words = v10_segment_inference_step_by_step(model, features, "<video>\n" + query.format(sentence), tokenizer, args, predict_center_offset )
        #                 if args.segment_head is not None:
        #                     text_answer = answer[:2]
        #                     linear_answer = answer[2:]
        #                     start = (answer[0] + answer[2]) / 2
        #                     end = (answer[1] + answer[3]) / 2
        #                     merge_answer = [start, end]
        #                     text_u = segment_iou(text_answer, gt, merge_result)
        #                     write_log(args.log_path+"_text", id, 'grounding', query_id, text_answer, predict_words, info={"sentence_id": sentence_id, 'iou': text_u})
        #                     linear_u = segment_iou(linear_answer, gt, merge_result)
        #                     write_log(args.log_path+"_linear", id, 'grounding', query_id, linear_answer, predict_words, info={"sentence_id": sentence_id, 'iou': linear_u})
        #                     merge_u = segment_iou(merge_answer, gt, merge_result)
        #                     write_log(args.log_path+"_merge", id, 'grounding', query_id, merge_answer, predict_words, info={"sentence_id": sentence_id, 'iou': merge_u})
        #             if args.segment_head is None:
        #                 u = segment_iou(answer, gt, merge_result)
        #                 write_log(args.log_path, id, 'grounding', query_id, answer, predict_words, info={"sentence_id": sentence_id, 'iou': u})
        #
        #
        # if args.task in ['vqa', 'all']:
        #     for query_id, query in enumerate(questions['captioning']):
        #         if args.add_temporal_tokens == "yes":
        #             answer = offset_caption_inference_generate(model, features, "<video>\n " + query, tokenizer, args)
        #         else:
        #             answer = offset_caption_inference_generate(model, features, "<video>\n " + query, tokenizer, args)
        #         write_log(args.log_path+"_captioning", id, 'captioning', query_id, answer)
