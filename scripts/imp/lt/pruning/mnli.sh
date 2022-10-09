export TASK_NAME=MNLI
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=5
export robust_type=poe
export output_dir=$ROOT_DIR/log/imp/lt/pruning/$TASK_NAME
export bert_pt_dir=bert-base-uncased
export bias_dir=$ROOT_DIR/bias_model_preds/$TASK_NAME/log_probs.npy

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
	  --learning_rate 5e-5 \
	  --num_train_epochs $num_epoch \
	  --output_dir $output_dir/$seed \
	  --logging_dir $output_dir/$seed/logging \
	  --root_dir $ROOT_DIR \
	  --logging_steps 1000 \
	  --save_steps 0 \
	  --is_prune true \
	  --is_imp true \
	  --weight_rewind true \
	  --prune_global_imp false \
	  --seed $seed
done
