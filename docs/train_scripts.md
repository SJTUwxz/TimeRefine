## stage 2 training

```bash

ngpus=8
segment_format=v10
add_temporal_tokens="no"
train_data=./data/vtimellm_train/stage2_offset.json
lr=1e-4
loss_type='l1_loss'
loss_weight=10.0
segment_head='linear_2output'
other_args="--num_train_epochs 8 --save_steps 1000 --data_path ${train_data} --feat_folder ./data/intern_clip_feat/ --learning_rate ${lr} --segment_format ${segment_format} --add_temporal_tokens ${add_temporal_tokens} --loss_type ${loss_type} --loss_weight ${loss_weight} --segment_head ${segment_head}"
exp_name=stage2-${ngpus}gpu-${segment_format}-${add_temporal_tokens}-token-${loss_type}-${loss_weight}-${segment_head}-usefast-debug
output_dir=./checkpoints/${exp_name}
echo $output_dir
rm -rf $output_dir
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

```

## stage 3 training

```bash

# same annotation for every epoch
ngpus=8
segment_format=v10
add_temporal_tokens="no"
train_data=./data/vtimellm_train/stage3_offset.json
lr=1e-4
loss_type='l1_loss'
loss_weight=10.0
segment_head='linear_2output'
stage2_path=stage2-${ngpus}gpu-${segment_format}-${add_temporal_tokens}-token-${loss_type}-${loss_weight}-${segment_head}-usefast
other_args="--num_train_epochs 8 --save_steps 400 --data_path ${train_data} --feat_folder ./data/stage3_clip_feat/ --learning_rate ${lr} --segment_format ${segment_format} --add_temporal_tokens ${add_temporal_tokens} --loss_type ${loss_type} --loss_weight ${loss_weight} --segment_head ${segment_head} --resume_checkpoint ./checkpoints/${stage2_path}/checkpoint-5000/"
exp_name=stage3-${ngpus}gpu-${segment_format}-${add_temporal_tokens}-token-${loss_type}-${loss_weight}-${segment_head}-usefast
output_dir=./checkpoints/${exp_name}
echo $output_dir
rm -rf $output_dir
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""



```

