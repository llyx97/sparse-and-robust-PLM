export TASK_NAME=fever
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=3
export zero_rate=0.7
export logging_steps=500
export warmup_steps=0
export best_metric=eval_acc
export mask_dir=$ROOT_DIR/log/mask_train/mask_on_plm_std_ft/train_w_ood/$TASK_NAME/$zero_rate
export output_dir=$ROOT_DIR/log/mask_train/mask_on_plm_std_ft/train_w_ood/$TASK_NAME/retrain/$zero_rate
export model_dir=bert-base-uncased

for seed in 1
do
	CUDA_VISIBLE_DEVICES=3 python3 $ROOT_DIR/run_glue.py \
	  --model_type bert \
	  --model_name_or_path $model_dir \
	  --task_name $TASK_NAME \
	  --do_train \
	  --do_eval \
	  --evaluate_during_training \
	  --data_dir $ROOT_DIR/data/$TASK_NAME \
	  --max_seq_length 128 \
	  --per_gpu_train_batch_size 32 \
	  --per_device_train_batch_size 32 \
	  --warmup_steps $warmup_steps \
	  --learning_rate 2e-5 \
	  --num_train_epochs $num_epoch \
	  --mask_dir $mask_dir/$seed/best_eval_mask \
	  --output_dir $output_dir/$seed \
	  --logging_dir $output_dir/$seed/logging \
	  --root_dir $ROOT_DIR \
	  --logging_steps $logging_steps \
	  --save_steps 0 \
	  --zero_rate $zero_rate \
	  --is_prune true \
	  --mask_classifier false \
	  --best_metric $best_metric \
	  --seed $seed
done
