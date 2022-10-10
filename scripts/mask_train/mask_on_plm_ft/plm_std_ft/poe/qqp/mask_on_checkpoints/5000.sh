export TASK_NAME=QQP
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=7
export zero_rate=0.7
export ft_sstep=5000
export robust_type=poe
export output_dir=$ROOT_DIR/log/mask_train/mask_on_plm_std_ft/$robust_type/$TASK_NAME/mask_on_checkpoints/checkpoint-$ft_sstep/$zero_rate
export model_dir=$ROOT_DIR/log/full_bert/$TASK_NAME/epoch3_lr2e-5
export bias_dir=$ROOT_DIR/bias_model_preds/$TASK_NAME/log_probs.pkl

for seed in 1 2 3 4
do
	CUDA_VISIBLE_DEVICES=3 python3 $ROOT_DIR/mask_run_glue.py \
	  --model_type bert \
	  --model_name_or_path $model_dir/$seed/checkpoint-$ft_sstep \
	  --bias_dir $bias_dir \
	  --task_name $TASK_NAME \
	  --dataset_names qqp \
	  --best_eval_metric eval_f1 \
	  --do_train \
	  --do_eval \
	  --evaluate_during_training \
	  --data_dir $ROOT_DIR/data/$TASK_NAME \
	  --max_seq_length 128 \
	  --per_gpu_train_batch_size 32 \
	  --per_device_train_batch_size 32 \
	  --learning_rate 2e-5 \
	  --num_train_epochs $num_epoch \
	  --output_dir $output_dir/$seed \
          --logging_dir $output_dir/$seed/logging \
	  --root_dir $ROOT_DIR \
	  --logging_steps 1000 \
	  --warmup_steps 3400 \
	  --save_steps 0 \
	  --zero_rate $zero_rate \
	  --robust_training $robust_type \
	  --controlled_init magnitude \
	  --train_classifier false \
	  --mask_classifier true \
	  --global_prune false \
	  --global_grad_glip false \
	  --seed $seed 
done
