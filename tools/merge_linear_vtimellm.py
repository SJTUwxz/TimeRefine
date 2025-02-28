import os
import json
import pickle as pkl

vtimellm_log = open("/home/fengchan/stor_sun/projects-ng/xizi/VTimeLLM_v3/checkpoints/v3_exps/vtimellm-stage2-angular-l1_loss-v3-10.0-checkpoint-6000-eval/released_checkpoint_output.log").readlines()
linear_log = open("/home/fengchan/stor_sun/projects-ng/xizi/VTimeLLM_v3/checkpoints/v3_exps/vtimellm-stage2-angular-l1_loss-v3-10.0-checkpoint-6000-eval-v3/released_checkpoint_output.log").readlines()

output = open("/home/fengchan/stor_sun/projects-ng/xizi/VTimeLLM_v3/checkpoints/v3_exps/vtimellm-stage2-angular-l1_loss-v3-10.0-checkpoint-6000-eval/merged_output.pkl", "wb")

merged_segments = []

for i in range(len(linear_log)):

    linear = json.loads(linear_log[i])
    start, end = linear["answer"]
    start = float(start * 100)
    end = float(end * 100)

    vtimellm = json.loads(vtimellm_log[i])
    answer = vtimellm["answer"].split()
    start_v = float(answer[2][:2])
    end_v = float(answer[4][:2])

    start = (start_v + start) / 2
    end = (end_v + end) / 2

    merged_segments.append([start, end])

pkl.dump(merged_segments, output)



