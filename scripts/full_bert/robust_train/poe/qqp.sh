export TASK_NAME=QQP
export ROOT_DIR=$HOME/srnet
export num_epoch=3
export lr=2e-5
export warmup_steps=3400
export robust_type=poe
export best_metric=eval_f1
export output_dir=$ROOT_DIR/log/full_bert/$TASK_NAME/robust_train/$robust_type
export bert_pt_dir=bert-base-uncased
export bias_dir=$ROOT_DIR/bias_model_preds/$TASK_NAME/log_probs.pkl

for seed in 1
do
	CUDA_VISIBLE_DEVICES=3 python3 $ROOT_DIR/run_glue.py \
	  --model_type bert \
	  --model_name_or_path $bert_pt_dir \
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
	  --learning_rate $lr \
	  --num_train_epochs $num_epoch \
	  --output_dir $output_dir/$seed \
	  --logging_dir $output_dir/$seed/logging \
	  --global_grad_clip false \
	  --root_dir $ROOT_DIR \
	  --best_metric $best_metric \
	  --logging_steps 1000 \
	  --save_steps 0 \
	  --seed $seed 
done
