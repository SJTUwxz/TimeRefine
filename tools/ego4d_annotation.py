import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import json

if __name__ == "__main__":

    annotation = json.load(open("/mnt/mir/datasets/vlm-datasets/ego4d_em/v2/annotations/nlq_val.json", 'r'))['videos']

    data = {}

    for vid_info in annotation:
    
        filename = vid_info["video_uid"]
        id = os.path.basename(filename)

        for clip_info in vid_info["clips"]:
            clip_id = clip_info["clip_uid"]
            video_start_sec = clip_info['video_start_sec']
            video_end_sec = clip_info['video_end_sec']
            duration = video_end_sec - video_start_sec
            if clip_id not in data:
                data[clip_id] = {"duration": duration, "timestamps": [], "sentences": []}
            for ann in clip_info["annotations"]:
                for query in ann["language_queries"]:
                    query_response_start_time_sec = query['video_start_sec']
                    query_response_end_time_sec = query['video_end_sec']
                    query_end_time_sec = min(query_response_end_time_sec, duration)

                    language = query.get("query", None)
                    if language is None:
                        if len(data[clip_id]["sentences"]) > 0:
                            continue
                        else:
                            print(clip_info)


                    else:
                        language = language.strip("Query Text:")
                        data[clip_id]["sentences"].append(language)
                        data[clip_id]["timestamps"].append([query_response_start_time_sec-video_start_sec, query_response_end_time_sec-video_start_sec])
                    # anns.append({
                    #     "query_start_time_sec": clip["video_start_sec"],
                    #     "query_end_time_sec": clip["video_end_sec"],
                    #     "query_response_start_time_sec": query["video_start_sec"],
                    #     "query_response_end_time_sec": query["video_end_sec"],
                    #     "query_template": query.get("template", None),
                    #     "query": query.get("query", None),
                    # })
    json.dump(data, open('./data/vtimellm_eval/nlq_val.json', 'w'))


