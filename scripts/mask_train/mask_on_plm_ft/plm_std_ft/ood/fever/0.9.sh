export TASK_NAME=fever
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=5
export zero_rate=0.9
export start_step_ratio=0.7
export logging_steps=500
export warmup_steps=0
export global_prune=false
export dataset_names=fever,sym2
export set_types=train,dev
export duplicates=1,10
export best_metric=eval_acc
export output_dir=$ROOT_DIR/log/mask_train/mask_on_plm_std_ft/train_w_ood/$TASK_NAME/$zero_rate
export model_dir=$ROOT_DIR/log/full_bert/$TASK_NAME/epoch3_lr2e-5

for seed in 1
do
	CUDA_VISIBLE_DEVICES=3 python3 $ROOT_DIR/mask_run_glue.py \
	  --model_type bert \
	  --model_name_or_path $model_dir/$seed \
	  --task_name $TASK_NAME \
	  --dataset_names $dataset_names \
	  --set_types $set_types \
	  --duplicates $duplicates \
	  --do_train \
	  --do_eval \
	  --evaluate_during_training \
	  --data_dir $ROOT_DIR/data/$TASK_NAME \
	  --max_seq_length 128 \
	  --per_gpu_train_batch_size 32 \
	  --per_device_train_batch_size 32 \
	  --warmup_steps $warmup_steps \
	  --learning_rate 2e-5 \
	  --start_step_ratio $start_step_ratio \
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
