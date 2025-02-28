import os
import json

file_path = '/mnt/mir/datasets/vlm-evaluation-datasets/QVHighlights/highlight_val_release.jsonl'

# List to store each JSON object
data = {}

# Open and read the file line by line
with open(file_path, 'r') as f:
    for line in f:
        # Parse the line as JSON and add to the list
        tmp = json.loads(line)
        duration = tmp["duration"]
        vid = tmp["vid"]
        sentences = tmp["query"]
        timestamps = tmp["relevant_windows"]
        if vid not in data:
            data[vid] = {"duration": duration, "timestamps": [], "sentences": []}
        data[vid]["timestamps"].append(timestamps)
        for timestamp in timestamps:
            data[vid]["sentences"].append(sentences)

json.dump(data, open('./data/vtimellm_eval/QVHighlights.json', 'w'))
