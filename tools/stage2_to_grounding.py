import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import json
from tqdm import tqdm
import re
import random


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

if __name__ == "__main__":

    annotation = json.load(open('./data/vtimellm_train/stage2.json','r'))

    # if os.path.exists(output_json):
    #     print("file already exists!")
    #     sys.exit(0)
    # out_f = open(output_json, "w")
    conversation = {}

    output_annotation = []

    for data in tqdm(annotation):
        id = data["id"]
        conversations = data["conversations"]
        meta = data["meta"]
        source = data["source"]

        new_data = data.copy()
        new_conversations = []

        # match the template of question in the conversation to one type of templates 

        for i in range(len(conversations)//2):
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
                new_conversations.append(conversations[2*i])
                new_conversations.append(conversations[2*i+1])
            elif type_template == "dense_captioning":
                # get the events and corresponding temporal tokens
                pattern = r'(?P<event>.+?)\sfrom\s<(?P<start>s\d+)>\s+to\s+<(?P<end>e\d+)>'

                # Find all matches in the sentence
                matches = re.findall(pattern, answer)
                match_type = "v1"

                if len(matches) == 0:
                    pattern = r'From\s+<(?P<start>s\d+)>\s+to\s+<(?P<end>e\d+)>\s*,\s*(?P<event>.+?)(?:\.|$)'
                    matches = re.findall(pattern, answer)
                    match_type = "v2"

                print(answer, matches)

                assert len(matches) > 0, "have to find matches"

                # Display extracted events and temporal segments
                for j, match in enumerate(matches):

                    if match_type == "v1":
                        event, start, end = match
                    elif match_type == "v2":
                        start, end, event = match
                    template = random.choice(temporal_grounding_templates)
                    if event[-1] == ".":
                        event = event[:-1]
                    event = event.lower()
                    template = template.replace("[T]", event)

                    if "<" not in start:
                        start = f"<{start}>"
                    if "<" not in end:
                        end = f"<{end}>"
                    answer = f"From {start} to {end}."

                    if j == 0:
                        new_conversations.append({"from": "human", "value": f"<video>\n{template}"})
                    else:
                        new_conversations.append({"from": "human", "value": f"{template}"})

                    new_conversations.append({"from": "gpt", "value": answer})


            elif type_template == "event_caption":
                pattern = r'<(?P<start>s\d+)>\s+to\s+<(?P<end>e\d+)>'

                # Find the match in the sentence
                match = re.search(pattern, pruned_question)

                assert match, "s and e need to be in the question"

                start = match.group('start')
                end = match.group('end')
                event = answer

                template = random.choice(temporal_grounding_templates)
                if event[-1] == ".":
                    event = event[:-1]
                event = event.lower()
                template = template.replace("[T]", event)

                if "<" not in start:
                    start = f"<{start}>"
                if "<" not in end:
                    end = f"<{end}>"
                answer = f"From {start} to {end}."

                if i == 0:
                    new_conversations.append({"from": "human", "value": f"<video>\n{template}"})
                else:
                    new_conversations.append({"from": "human", "value": f"{template}"})

                new_conversations.append({"from": "gpt", "value": answer})



        new_data["conversations"] = new_conversations
        output_annotation.append(new_data)

    json.dump(output_annotation, open('./data/vtimellm_train/stage2_grounding.json', 'w'), indent = 6)


            
        


