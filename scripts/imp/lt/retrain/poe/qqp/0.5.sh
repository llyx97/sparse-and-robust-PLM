export TASK_NAME=QQP
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=3
export zero_rate=0.5
export logging_steps=1000
export warmup_steps=3400
export robust_type=poe
export best_metric=eval_f1
export mask_dir=$ROOT_DIR/log/imp/lt/pruning/$TASK_NAME
export output_dir=$ROOT_DIR/log/imp/lt/retrain/$TASK_NAME/$local_or_global/$zero_rate
export model_dir=bert-base-uncased
export bias_dir=$ROOT_DIR/bias_model_preds/$TASK_NAME/log_probs.pkl

for seed in 1
do
	CUDA_VISIBLE_DEVICES=3 python3 $ROOT_DIR/run_glue.py \
	  --model_type bert \
	  --model_name_or_path $model_dir \
	  --bias_dir $bias_dir \
	  --task_name $TASK_NAME \
	  --robust_training $robust_type \
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
	  --mask_dir $mask_dir/$seed/$zero_rate \
	  --output_dir $output_dir/$seed \
	  --logging_dir $output_dir/$seed/logging \
	  --global_grad_clip false \
	  --root_dir $ROOT_DIR \
	  --logging_steps $logging_steps \
	  --save_steps 0 \
	  --zero_rate $zero_rate \
	  --is_prune true \
	  --mask_classifier false \
	  --best_metric $best_metric \
	  --seed $seed
done
