## stage 2 eval

```bash

# ActivityNet
for checkpoint in `seq 4000 1000 9000`; do
model_segment_format=v10
merge_segments='last'
add_temporal_tokens="no"
parallel_inference="parallel"
loss_type='l1_loss'
loss_weight=10.0
segment_head='linear_2output'
use_fast="yes"
exp_name=stage2-8gpu-${model_segment_format}-${add_temporal_tokens}-token-${loss_type}-${loss_weight}-${segment_head}-usefast
output_dir=./checkpoints/${exp_name}-checkpoint-${checkpoint}-${merge_segments}-${parallel_inference}-eval
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/val_2.json --feat_folder ./data/stage3_clip_feat/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin --stage2 ./checkpoints/${exp_name}/checkpoint-${checkpoint}/ --task ${model_segment_format}_grounding --model_segment_format ${model_segment_format} --merge_segments ${merge_segments} --add_temporal_tokens ${add_temporal_tokens} --parallel_inference ${parallel_inference} --segment_head ${segment_head} --use_fast ${use_fast}"
done

# Charades
# Please change the path of Charades-STA features to your Charades path.
for checkpoint in `seq 4000 1000 9000`; do
model_segment_format=v10
merge_segments='last'
add_temporal_tokens="no"
parallel_inference="parallel"
loss_type='l1_loss'
loss_weight=10.0
segment_head='linear_2output'
use_fast="yes"
exp_name=stage2-8gpu-${model_segment_format}-${add_temporal_tokens}-token-${loss_type}-${loss_weight}-${segment_head}-usefast
output_dir=./checkpoints/${exp_name}-checkpoint-${checkpoint}-${merge_segments}-${parallel_inference}-charades-eval
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-charades -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/charades_sta_test.json --feat_folder [Charades_path] --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin --stage2 ./checkpoints/${exp_name}/checkpoint-${checkpoint}/ --task ${model_segment_format}_grounding --model_segment_format ${model_segment_format} --merge_segments ${merge_segments} --add_temporal_tokens ${add_temporal_tokens} --parallel_inference ${parallel_inference} --segment_head ${segment_head} --use_fast ${use_fast}"
done

```

## stage 3 eval

```bash

# ANet
for checkpoint in `seq 400 400 1200`; do
model_segment_format=v10
merge_segments='last'
add_temporal_tokens="no"
parallel_inference="parallel"
loss_type='l1_loss'
loss_weight=10.0
segment_head='linear_2output'
use_fast="yes"
predict_goal="offset"
exp_name=stage3-8gpu-${model_segment_format}-${add_temporal_tokens}-token-${loss_type}-${loss_weight}-${segment_head}-usefast
output_dir=./checkpoints/${exp_name}-checkpoint-${checkpoint}-${merge_segments}-${parallel_inference}-eval
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/val_2.json --feat_folder ./data/stage3_clip_feat/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin --stage2 ./checkpoints/${exp_name}/checkpoint-${checkpoint}/ --task ${model_segment_format}_grounding --model_segment_format ${model_segment_format} --merge_segments ${merge_segments} --add_temporal_tokens ${add_temporal_tokens} --parallel_inference ${parallel_inference} --segment_head ${segment_head} --use_fast ${use_fast} --predict_goal ${predict_goal}"
done

# Charades
for checkpoint in `seq 400 400 1200`; do
model_segment_format=v10
merge_segments='last'
add_temporal_tokens="no"
parallel_inference="parallel"
loss_type='l1_loss'
loss_weight=10.0
segment_head='linear_2output'
use_fast="yes"
predict_goal="offset"
exp_name=stage3-8gpu-${model_segment_format}-${add_temporal_tokens}-token-${loss_type}-${loss_weight}-${segment_head}-usefast
output_dir=./checkpoints/v6_exps/${exp_name}-checkpoint-${checkpoint}-${merge_segments}-${parallel_inference}-charades-eval
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-charades -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/charades_sta_test.json --feat_folder [Charades_path] --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin --stage2 ./checkpoints/${exp_name}/checkpoint-${checkpoint}/ --task ${model_segment_format}_grounding --model_segment_format ${model_segment_format} --merge_segments ${merge_segments} --add_temporal_tokens ${add_temporal_tokens} --parallel_inference ${parallel_inference} --segment_head ${segment_head} --use_fast ${use_fast} --predict_goal ${predict_goal}"
done

```
