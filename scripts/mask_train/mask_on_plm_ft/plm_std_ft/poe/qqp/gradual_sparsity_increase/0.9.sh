export TASK_NAME=QQP
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=7
export zero_rate=0.9
export init_sparsity=0.7
export robust_type=poe
export output_dir=$ROOT_DIR/log/mask_train/mask_on_plm_std_ft/$robust_type/$TASK_NAME/gradual_sparsity_increase/${init_sparsity}_$zero_rate
export model_dir=$ROOT_DIR/log/full_bert/$TASK_NAME/epoch3_lr2e-5
export bias_dir=$ROOT_DIR/bias_model_preds/$TASK_NAME/log_probs.pkl

for seed in 1 2 3 4
do
	CUDA_VISIBLE_DEVICES=3 python3 $ROOT_DIR/mask_run_glue.py \
	  --model_type bert \
	  --model_name_or_path $model_dir/$seed \
	  --bias_dir $bias_dir \
	  --task_name $TASK_NAME \
	  --dataset_names qqp \
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
	  --final_sparsity_epoch $num_epoch \
          --init_sparsity $init_sparsity \
	  --robust_training $robust_type \
	  --controlled_init magnitude_soft \
	  --best_metric eval_f1 \
	  --train_classifier false \
	  --mask_classifier true \
	  --global_prune false \
	  --global_grad_clip false \
	  --seed $seed 
done

