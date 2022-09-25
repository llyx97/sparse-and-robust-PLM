export TASK_NAME=fever
export ROOT_DIR=$HOME/srnet
export num_epoch=3
export lr=2e-5
export warmup_steps=0
export output_dir=$ROOT_DIR/log/full_bert/$TASK_NAME/epoch${num_epoch}_lr${lr}
export bert_pt_dir=bert-base-uncased

for seed in 1
do
	CUDA_VISIBLE_DEVICES=0 python3 $ROOT_DIR/run_glue.py \
	  --model_type bert \
	  --model_name_or_path $bert_pt_dir \
	  --task_name $TASK_NAME \
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
	  --root_dir $ROOT_DIR \
	  --logging_steps 500 \
	  --save_steps 0 \
	  --seed $seed
done
