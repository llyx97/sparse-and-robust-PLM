export TASK_NAME=MNLI
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=5
export zero_rate=0.5
export logging_steps=1000
export warmup_steps=3600
export global_prune=false
export dataset_names=mnli
export best_metric=eval_acc
export robust_type=poe
export output_dir=$ROOT_DIR/log/mask_train/prune_after_ft/$robust_type/$TASK_NAME/$zero_rate
export model_dir=$ROOT_DIR/log/full_bert/$TASK_NAME/epoch3_lr5e-5
export bias_dir=$ROOT_DIR/bias_model_preds/$TASK_NAME/log_probs.npy

for seed in 1
do
	CUDA_VISIBLE_DEVICES= python3 $ROOT_DIR/mask_run_glue.py \
	  --model_type bert \
	  --model_name_or_path $model_dir/$seed \
	  --bias_dir $bias_dir \
	  --task_name $TASK_NAME \
	  --dataset_names $dataset_names \
	  --robust_training $robust_type \
	  --do_train \
	  --do_eval \
	  --evaluate_during_training \
	  --data_dir $ROOT_DIR/data/$TASK_NAME \
	  --max_seq_length 128 \
	  --per_gpu_train_batch_size 32 \
	  --per_device_train_batch_size 32 \
	  --learning_rate 5e-5 \
	  --warmup_steps $warmup_steps \
	  --num_train_epochs $num_epoch \
	  --output_dir $output_dir/$seed \
	  --logging_dir $output_dir/$seed/logging \
	  --root_dir $ROOT_DIR \
	  --logging_steps $logging_steps \
	  --save_steps 0 \
	  --zero_rate $zero_rate \
	  --controlled_init magnitude \
	  --train_classifier false \
	  --mask_classifier true \
	  --global_prune $global_prune \
	  --best_metric $best_metric \
	  --seed $seed
done
