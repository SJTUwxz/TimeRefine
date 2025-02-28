import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import json
from tqdm import tqdm
import re
import random
import numpy as np


def get_iou(segment1, segment2):
    start_1, end_1 = segment1
    start_2, end_2 = segment2
    
    # Calculate the intersection (overlap)
    intersection_start = max(start_1, start_2)
    intersection_end = min(end_1, end_2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate the union
    union = (end_1 - start_1) + (end_2 - start_2) - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

def propose_segment_iou(start, end, std_dev):
    proposed_segments = []
    not_found = True
    for std_ind, std in enumerate(std_dev):
        while not_found:
            new_start_tmp = np.random.normal(loc=start, scale=std)
            new_end_tmp = np.random.normal(loc=end, scale=std)
            new_start = min(new_start_tmp, new_end_tmp)
            new_start = max(0, new_start)
            new_end = max(new_start_tmp, new_end_tmp)
            new_end = min(duration, new_end)

            iou = get_iou([start, end], [new_start, new_end])
            iou = round(iou * 10.0)
            if new_end < new_start:
                continue
            else:
                proposed_segments.append([new_start, new_end, iou])
                break
    return proposed_segments

def propose_segment_offset(start, end, std_dev, gt_within=False, sort_increase=False):
    if not gt_within:
        proposed_segments = []
        not_found = True
        ious = []
        for std_ind, std in enumerate(std_dev):
            while not_found:
                new_start_tmp = np.random.normal(loc=start, scale=std)
                new_end_tmp = np.random.normal(loc=end, scale=std)
                new_start = min(new_start_tmp, new_end_tmp)
                new_start = max(0, new_start)
                new_end = max(new_start_tmp, new_end_tmp)
                new_end = min(duration, new_end)

                start_offset = start - new_start
                end_offset = end - new_end
                if new_end < new_start:
                    continue
                else:
                    proposed_segments.append([new_start, new_end, start_offset, end_offset])
                    ious.append(get_iou([start, end], [new_start, new_end]))
                    break
    else:
        proposed_segments = []
        ious = []
        not_found = True
        for std_ind, std in enumerate(std_dev):
            while not_found:
                new_start_tmp = np.random.normal(loc=start, scale=std)
                new_end_tmp = np.random.normal(loc=end, scale=std)
                new_start = min(new_start_tmp, new_end_tmp)
                new_start = max(0, new_start)
                new_end = max(new_start_tmp, new_end_tmp)
                new_end = min(duration, new_end)

                start_offset = start - new_start
                end_offset = end - new_end
                if new_end < new_start:
                    continue
                else:
                    if start_offset >= 0 and end_offset <= 0:
                        proposed_segments.append([new_start, new_end, start_offset, end_offset])
                        ious.append(get_iou([start, end], [new_start, new_end]))
                        break
                    else:
                        continue
    if sort_increase:
        proposed_segments = [x for x, _ in sorted(zip(proposed_segments, ious), key=lambda pair: pair[1], reverse=False)]

    return proposed_segments
    

def replace_question(sentence):
    # Regular expression to match <sX> ... <eX> where X is an integer and ... is any content
    pattern = r'(<s(\d+)>.*?<e\2>)'
    
    # Define a function to replace the matched subsentence
    def replace_with_tags(match):
        s_tag = f"<s{match.group(2)}>"  # The starting tag <sX>
        e_tag = f"<e{match.group(2)}>"  # The corresponding ending tag <eX>
        return f"<SEG_START> {s_tag} to {e_tag} <SEG_END>"
    
    # Use re.sub to replace the matched subsentences
    modified_sentence = re.sub(pattern, replace_with_tags, sentence)
    
    return modified_sentence

def process_answer(start_tag, end_tag):
    segments = []
    # Custom postprocessing: You can modify this logic to suit your needs
    processed_content = content.upper()  # Example: converting content to uppercase
    return f"{start_tag} {processed_content} {end_tag}"

def replace_answer(sentence, tokens, std_dev, gt_within=False, sort_increase=False):
    # Regular expression to match <sX> ... <eX> where X is an integer
    pattern = r'(<s(\d+)>).*?(<e(\d+)>)'
    new_segments = []
    cur_token_ind = len(tokens.keys()) // 2
    
    result = []
    
    current_position = 0
    
    while current_position < len(sentence):
        # Search for the pattern starting from current_position
        match = re.search(pattern, sentence[current_position:])
        
        if not match:
            # No more matches, append the remaining part of the sentence
            result.append(sentence[current_position:])
            break
        
        match_start = match.start() + current_position
        match_end = match.end() + current_position
        
        start_tag = match.group(1)  # Full <sX> tag
        end_tag = match.group(3)    # Full <eX> tag

        start = tokens[start_tag]
        end = tokens[end_tag]
        proposed_segments = propose_segment_offset(start, end, std_dev, gt_within, sort_increase)
        for std_ind, item in enumerate(proposed_segments):
            new_segments.append(start)
            new_segments.append(end)
            new_start, new_end, start_offset, end_offset = item

            tokens[f"<s{cur_token_ind}>"] = new_start
            tokens[f"<e{cur_token_ind}>"] = new_end
            tokens[f"<so{cur_token_ind}>"] = start_offset
            tokens[f"<eo{cur_token_ind}>"] = end_offset

            if std_ind == 0:
                segment_answer = f"<SEG_START> <s{cur_token_ind}> to <e{cur_token_ind}> <OFFSET> <so{cur_token_ind}> and <eo{cur_token_ind}>"
            elif std_ind == len(std_dev) - 1:
            #TODO: remove "." after <SEG_END> could cause issue
                segment_answer = f"{segment_answer} <RETHINK> <s{cur_token_ind}> to <e{cur_token_ind}> <OFFSET> <so{cur_token_ind}> and <eo{cur_token_ind}> <SEG_END>"
            else:
                segment_answer = f"{segment_answer} <RETHINK> <s{cur_token_ind}> to <e{cur_token_ind}> <OFFSET> <so{cur_token_ind}> and <eo{cur_token_ind}>"

            cur_token_ind = cur_token_ind + 1

        
        result.append(sentence[current_position:match_start])
        
        result.append(segment_answer)
        
        current_position = match_end
    
    modified_sentence = ''.join(result)
    
    return modified_sentence, new_segments, tokens 


if __name__ == "__main__":


    # if os.path.exists(output_json):
    #     print("file already exists!")
    #     sys.exit(0)
    # out_f = open(output_json, "w")
    conversation = {}

    output_annotation = []

    transform_dense_captioning = True 
    version = "v10"
    annotation = json.load(open('./data/vtimellm_train/stage3.json','r'))
    
    gt_within = True  # default = False
    sort_increase = False # default = False

    adaptive_std = False # default=False

    # number of std default to 4
    number_of_std = 4
    
    if number_of_std == 4:
        std_dev = [5, 3, 1, 0]
    elif number_of_std == 2:
        std_dev = [4, 0]
    elif number_of_std == 6:
        std_dev = [8, 6, 4, 2, 1, 0]
    elif number_of_std == 8:
        std_dev = [10, 8, 6, 5, 3, 2, 1, 0 ]

    num_segments = len(std_dev)

    for data in tqdm(annotation):
        id = data["id"]
        conversations = data["conversations"]
        meta = data.get('meta', None)
        source = data["source"]
        if meta is not None:
            tokens = meta["token"]
            duration = float(meta["duration"])
        else:
            new_data = data.copy()
            new_data["meta"] = {"token": {}, "segment": [], "duration": 1.0}
            new_data["conversations"] = conversations 
            output_annotation.append(new_data)
            continue
        length = len(conversations) // 2
        new_tokens = tokens.copy()
        cur_token_ind = len(new_tokens.keys()) // 2
        cur_token_length = len(new_tokens.keys()) // 2

        new_data = data.copy()
        new_conversations = []
        new_segments = []

        # match the template of question in the conversation to one type of templates 
        for i in range(length):
            question = conversations[2*i]["value"]
            answer = conversations[2*i+1]["value"]

            # check question first, if segment appears in the question, only add seg_start and seg_end to it
            # matches = re.findall(r'(<s(\d+)>)(.*?)(<e\2>)', question)
            question = replace_question(question)

            # find all the segments in answer, and replace it with offset
            answer, segments, new_tokens = replace_answer(answer, new_tokens, std_dev, gt_within, sort_increase)
            new_segments.extend(segments)

            conversations[2*i]["value"] = question
            conversations[2*i+1]["value"] = answer
            new_conversations.append(conversations[2*i])
            new_conversations.append(conversations[2*i+1])

        new_data["conversations"] = new_conversations
        new_data["meta"]["token"] = new_tokens
        new_data["meta"]["segment"] = new_segments 
        output_annotation.append(new_data)

    # json.dump(output_annotation, open(f'./data/vtimellm_train/stage3_offset.json', 'w'), indent = 6)
    filename = f'./data/vtimellm_train/stage3_offset_{version}_{number_of_std}-std'
    if gt_within:
        filename = filename + "_gtwithin"
    if sort_increase:
        filename = filename + "-sorted"
    if adaptive_std:
        filename = filename + "-adaptive_std"

    filename = filename + '.json'
    json.dump(output_annotation, open(filename, 'w'), indent = 6)
                    
                
"""


                conversations[2*i]["value"] = question
                conversations[2*i+1]["value"] = answer
                new_conversations.append(conversations[2*i])
                new_conversations.append(conversations[2*i+1])

            new_data["conversations"] = new_conversations
            new_data["meta"]["token"] = new_tokens
            if len(new_segments) == 0:
                new_data["meta"]["segment"] = list(tokens.values())
            else:
                new_data["meta"]["segment"] = new_segments 
            output_annotation.append(new_data)

     
    if transform_dense_captioning:
        if epoch_random:
            json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_{version}_epochrandom.json', 'w'), indent = 6)
        else:
            json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_{version}.json', 'w'), indent = 6)
    else:
        if epoch_random:
            json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_grounding_{version}_epochrandom.json', 'w'), indent = 6)
        else:
            json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_grounding_{version}.json', 'w'), indent = 6)




        """
