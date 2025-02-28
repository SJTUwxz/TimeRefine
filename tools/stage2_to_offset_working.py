import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import json
from tqdm import tqdm
import re
import random
import numpy as np


dense_captioning_templates = ["Could you please detail the events that took place during different time segments in the video?", 
                              "I'm curious about what happened at different points in the video. Could you please describe the events?",
                              "Could you provide a summary of the incidents that occurred at various timestamps in the video?",
                              "I'd like to know what events transpired during specific time intervals in the video. Could you please elaborate?",
                              "Can you give me a breakdown of the occurrences at different time stamps in the video?",
                              "I'm interested in understanding the events that unfolded at different points in the video. Could you please specify?",
                              "Could you outline the incidents that happened during different time periods in the video?",
                              "I'm trying to grasp the sequence of events in the video. Could you please outline what happened at different times?",
                              "Can you go through the video and describe what took place at different time intervals?",
                              "I'd appreciate it if you could provide a detailed account of the events that occurred at different timestamps in the video.",
                            ]
event_caption_templates = ["Can you describe what occurred from [S] to [E] in the video?",
                           "Could you tell me what happened from [S] to [E] in the video?",
                           "What transpired from [S] to [E] in the video?",
                           "Describe what took place from [S] to [E] in the video.",
                           "Tell me about the events from [S] to [E] in the video.",
                           "What was going on from [S] to [E] in the video?",
                           "Please recount what occurred from [S] to [E] in the video.",
                           "Explain what happened from [S] to [E] in the video.",
                           "Provide details about the events from [S] to [E] in the video.",
                           "Share what transpired from [S] to [E] in the video."
                           ]
temporal_grounding_templates = ["During which frames can we see [T] happening in the video?",
                                "Between which frames is [T] visible in the video?",
                                "At what point in the video can we observe [T] taking place?",
                                "Between which two frames can we witness [T] occurring in the video?",
                                "During which frames in the video can we observe [T] happening?",
                                "At which time interval in the video can we see [T] occurring?",
                                "Between which frames can we find [T] taking place in the video?",
                                "At what point in the video can we witness [T] happening?",
                                "Between which two frames in the video can we observe [T] taking place?",
                                "During which frames does [T] occur in the video?"
                                ]

def match_template(sentence, templates):
    for template in templates:
        template = template.strip()
        sentence = sentence.strip()
        if '[T]' in template:
            # a temporal_grounding template
            template_opener = template.split('[T]')[0]
            if template_opener in sentence:
                return template, "temporal_grounding"
        elif '[S]' in template:
            # an event captioning template
            template_opener = template.split('[S]')[0]
            if template_opener in sentence:
                return template, "event_caption"
        else:
            # a dense captioning template
            if template in sentence:
                return template, "dense_captioning"
        
        # Check if the sentence matches the template pattern
    
    return None, None  # No match found

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

def propose_segment_offset(start, end, std_dev, duration, gt_within=False, sort_increase=False):
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
    

if __name__ == "__main__":


    # if os.path.exists(output_json):
    #     print("file already exists!")
    #     sys.exit(0)
    # out_f = open(output_json, "w")
    conversation = {}

    output_annotation = []

    transform_dense_captioning = True 
    version = "v10"
    epoch_random = False

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

    annotation = json.load(open('./data/vtimellm_train/stage2.json','r'))
    for data in tqdm(annotation):
        id = data["id"]
        conversations = data["conversations"]
        meta = data["meta"]
        source = data["source"]
        duration = float(meta["duration"])
        tokens = meta["token"]
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

            pruned_question = question.strip().replace("<video>\n", "")
            matched_template, type_template = match_template(pruned_question, temporal_grounding_templates)
            if matched_template is None:
                matched_template, type_template = match_template(pruned_question, event_caption_templates)
            if matched_template is None:
                matched_template, type_template = match_template(pruned_question, dense_captioning_templates)

            assert matched_template is not None, f"{pruned_question} : question must match one of the templates"

            if type_template == "temporal_grounding":

                # first step is to find the start and end segments
                new_conversations.append(conversations[2*i])
                start, end = None, None
                for k, v in tokens.items():
                    if k in answer:
                        if start is None:
                            start = float(v)
                        else:
                            end = float(v)
                assert (start is not None and end is not None), "start and end needs to be found"

                if adaptive_std:

                    gt_duration = end - start

                    # choose std_dev based on duration, smaller duration gets smaller std
                    std_dev = np.linspace(duration*0.2, 0, number_of_std).tolist()

                proposed_segments = propose_segment_offset(start, end, std_dev, duration, gt_within, sort_increase)

                for std_ind, item in enumerate(proposed_segments):
                    new_segments.append(start)
                    new_segments.append(end)
                    new_start, new_end, start_offset, end_offset = item

                    new_tokens[f"<s{cur_token_ind}>"] = new_start
                    new_tokens[f"<e{cur_token_ind}>"] = new_end
                    new_tokens[f"<so{cur_token_ind}>"] = start_offset
                    new_tokens[f"<eo{cur_token_ind}>"] = end_offset

                    if std_ind == 0:
                        answer = f"From <SEG_START> <s{cur_token_ind}> to <e{cur_token_ind}> <OFFSET> <so{cur_token_ind}> and <eo{cur_token_ind}>"
                    elif std_ind == len(std_dev) - 1:
                        answer = f"{answer} <RETHINK> <s{cur_token_ind}> to <e{cur_token_ind}> <OFFSET> <so{cur_token_ind}> and <eo{cur_token_ind}> <SEG_END>."
                    else:
                        answer = f"{answer} <RETHINK> <s{cur_token_ind}> to <e{cur_token_ind}> <OFFSET> <so{cur_token_ind}> and <eo{cur_token_ind}>"

                    cur_token_ind = cur_token_ind + 1

                conversations[2*i+1]["value"] = answer
                new_conversations.append(conversations[2*i+1])

            elif type_template == "dense_captioning":
                # new_conversations.append(conversations[2*i])
                # new_conversations.append(conversations[2*i+1])
                # new_tokens = tokens
                # continue
                # get the events and corresponding temporal tokens
                pattern = r'(?P<event>.+?)\sfrom\s<(?P<start>s\d+)>\s+to\s+<(?P<end>e\d+)>'

                # Find all matches in the sentence
                matches = re.findall(pattern, answer)
                match_type = "v1"

                if len(matches) == 0:
                    pattern = r'From\s+<(?P<start>s\d+)>\s+to\s+<(?P<end>e\d+)>\s*,\s*(?P<event>.+?)(?:\.|$)'
                    matches = re.findall(pattern, answer)
                    match_type = "v2"

                assert len(matches) > 0, "have to find matches"
                assert length == 1, "round of conversations should be one for dense captioning"

                new_conversations.append(conversations[2*i])
                # Display extracted events and temporal segments
                # for each matching <event> <start> <end>, turn <start> <end> into lots of possible segments
                all_answers = ""
                for j, match in enumerate(matches):

                    if match_type == "v1":
                        event, start_str, end_str = match
                    elif match_type == "v2":
                        start_str, end_str, event = match
                    else:
                        raise NotImplementedError("match type has to be v1 or v2")

                    if "<" not in start_str:
                        start_str = f"<{start_str}>"
                    if "<" not in end_str:
                        end_str = f"<{end_str}>"
                        
                    start, end = None, None
                    for k, v in tokens.items():
                        if k == start_str:
                            if start is None:
                                start = float(v)
                        if k == end_str:
                            if end is None:
                                end = float(v)
                    assert (start is not None and end is not None), "start and end needs to be found"
                    # if random:
                    #
                    #     gt_duration = end - start
                    #
                    #     std_dev = np.random.uniform(0, gt_duration*0.5, [num_segments]).tolist()
                    #     std_dev.sort(reverse=True)
                    #
                    if transform_dense_captioning:

                        if adaptive_std:

                            gt_duration = end - start

                            # choose std_dev based on duration, smaller duration gets smaller std
                            std_dev = np.linspace(duration*0.2, 0, number_of_std).tolist()

                        proposed_segments = propose_segment_offset(start, end, std_dev, duration, gt_within, sort_increase)

                        for std_ind, item in enumerate(proposed_segments):
                            new_segments.append(start)
                            new_segments.append(end)
                            new_start, new_end, start_offset, end_offset = item

                            new_tokens[f"<s{cur_token_length}>"] = new_start
                            new_tokens[f"<e{cur_token_length}>"] = new_end
                            new_tokens[f"<so{cur_token_length}>"] = start_offset
                            new_tokens[f"<eo{cur_token_length}>"] = end_offset

                            if std_ind == 0:
                                if match_type == "v1":
                                    if event.startswith('. '):
                                        event = event[2:]
                                    answer = f"{event} from <SEG_START> <s{cur_token_length}> to <e{cur_token_length}> <OFFSET> <so{cur_token_length}> and <eo{cur_token_length}>"
                                else:
                                    answer = f"From <SEG_START> <s{cur_token_length}> to <e{cur_token_length}> <OFFSET> <so{cur_token_length}> and <eo{cur_token_length}>"

                            elif std_ind == len(std_dev) - 1:
                                if match_type == "v1":
                                    answer = f"{answer} <RETHINK> <s{cur_token_length}> to <e{cur_token_length}> <OFFSET> <so{cur_token_length}> and <eo{cur_token_length}> <SEG_END>."
                                else:
                                    answer = f"{answer} <RETHINK> <s{cur_token_length}> to <e{cur_token_length}> <OFFSET> <so{cur_token_length}> and <eo{cur_token_length}> <SEG_END>, {event}."
                            else:
                                answer = f"{answer} <RETHINK> <s{cur_token_length}> to <e{cur_token_length}> <OFFSET> <so{cur_token_length}> and <eo{cur_token_length}>"

                            cur_token_length = cur_token_length + 1

                    else:
                        new_tokens = tokens
                        if match_type == "v1":
                            if event.startswith('. '):
                                event = event[2:]
                            answer = f"{event} from <SEG_START> {start_str} to {end_str} <SEG_END>."
                        else:
                            answer = f"From <SEG_START> {start_str} to {end_str} <SEG_END>, {event}."

                    if j == 0:
                        all_answers = answer
                    else:
                        all_answers = all_answers + ' ' + answer


                conversations[2*i+1]["value"] = all_answers 
                new_conversations.append(conversations[2*i+1])

                    # the following codes are for transforming them to temporal grounding
                    # template = random.choice(temporal_grounding_templates)
                    # if event[-1] == ".":
                    #     event = event[:-1]
                    # event = event.lower()
                    # template = template.replace("[T]", event)
                    #
                    # if "<" not in start:
                    #     start = f"<{start}>"
                    # if "<" not in end:
                    #     end = f"<{end}>"
                    # answer = f"From {start} to {end}."
                    #
                    # if j == 0:
                    #     new_conversations.append({"from": "human", "value": f"<video>\n{template}"})
                    # else:
                    #     new_conversations.append({"from": "human", "value": f"{template}"})
                    #
                    # new_conversations.append({"from": "gpt", "value": answer})


            elif type_template == "event_caption":
                pattern = r'<(?P<start>s\d+)>\s+to\s+<(?P<end>e\d+)>'

                # Find the match in the sentence
                match = re.search(pattern, pruned_question)

                assert match, "s and e need to be in the question"

                start_str = match.group('start')
                end_str = match.group('end')
                event = answer

                if "<" not in start_str:
                    start_str = f"<{start_str}>"
                if "<" not in end_str:
                    end_str = f"<{end_str}>"

                question = question.replace(start_str, f"<SEG_START> {start_str}")
                question = question.replace(end_str, f"{end_str} <SEG_END>")
                conversations[2*i]["value"] =  question

                new_conversations.append(conversations[2*i])

                new_conversations.append(conversations[2*i+1])


                # template = random.choice(temporal_grounding_templates)
                # if event[-1] == ".":
                #     event = event[:-1]
                # event = event.lower()
                # template = template.replace("[T]", event)
                #
                # if "<" not in start:
                #     start = f"<{start}>"
                # if "<" not in end:
                #     end = f"<{end}>"
                # answer = f"From {start} to {end}."
                #
                # if i == 0:
                #     new_conversations.append({"from": "human", "value": f"<video>\n{template}"})
                # else:
                #     new_conversations.append({"from": "human", "value": f"{template}"})
                #
                # new_conversations.append({"from": "gpt", "value": answer})



        new_data["conversations"] = new_conversations
        new_data["meta"]["token"] = new_tokens
        if len(new_segments) == 0:
            new_data["meta"]["segment"] = list(tokens.values())
        else:
            new_data["meta"]["segment"] = new_segments 
        output_annotation.append(new_data)

     
    # if transform_dense_captioning:
    #     if epoch_random:
    #         json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_{version}_epochrandom_{number_of_std}-std.json', 'w'), indent = 6)
    #     else:
    #         json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_{version}_{number_of_std}-std.json', 'w'), indent = 6)
    # else:
    #     if epoch_random:
    #         json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_grounding_{version}_epochrandom_{number_of_std}-std.json', 'w'), indent = 6)
    #     else:
    #         json.dump(output_annotation, open(f'./data/vtimellm_train/stage2_offset_grounding_{version}_{number_of_std}-std.json', 'w'), indent = 6)

    filename = f'./data/vtimellm_train/stage2_offset_{version}_{number_of_std}-std'
    if gt_within:
        filename = filename + "_gtwithin"
    if sort_increase:
        filename = filename + "-sorted"
    if adaptive_std:
        filename = filename + "-adaptive_std"

    filename = filename + '.json'
    json.dump(output_annotation, open(filename, 'w'), indent = 6)



            
        


