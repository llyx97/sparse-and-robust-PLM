export TASK_NAME=MNLI
export ROOT_DIR=/apdcephfs/share_47076/tmpv_xiuliu
export num_epoch=3
export zero_rate=0.9
export logging_steps=1000
export warmup_steps=3600
export best_metric=eval_acc
export robust_type=reweighting
export root_mask_dir=$ROOT_DIR/robust_compression/log/mask_train/$TASK_NAME/robust_train/$robust_type/known_bias
export mask_dir=$root_mask_dir/$zero_rate
export output_dir=$root_mask_dir/retrain/$zero_rate
export model_dir=$ROOT_DIR/LT/models/bert_pt

export seed=1
CUDA_VISIBLE_DEVICES=$(($seed-1)) python3 $ROOT_DIR/robust_compression/run_glue.py \
  --model_type bert \
  --model_name_or_path $model_dir \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir $ROOT_DIR/tinybert/glue/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --warmup_steps $warmup_steps \
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
  --seed $seed \
&

export seed=2
CUDA_VISIBLE_DEVICES=$(($seed-1)) python3 $ROOT_DIR/robust_compression/run_glue.py \
  --model_type bert \
  --model_name_or_path $model_dir \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir $ROOT_DIR/tinybert/glue/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --warmup_steps $warmup_steps \
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
  --seed $seed \
&

export seed=3
CUDA_VISIBLE_DEVICES=$(($seed-1)) python3 $ROOT_DIR/robust_compression/run_glue.py \
  --model_type bert \
  --model_name_or_path $model_dir \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir $ROOT_DIR/tinybert/glue/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --warmup_steps $warmup_steps \
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
  --seed $seed \
&

export seed=4
CUDA_VISIBLE_DEVICES=$(($seed-1)) python3 $ROOT_DIR/robust_compression/run_glue.py \
  --model_type bert \
  --model_name_or_path $model_dir \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir $ROOT_DIR/tinybert/glue/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --warmup_steps $warmup_steps \
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
  --seed $seed \
&

export seed=5
CUDA_VISIBLE_DEVICES=$(($seed-1)) python3 $ROOT_DIR/robust_compression/run_glue.py \
  --model_type bert \
  --model_name_or_path $model_dir \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir $ROOT_DIR/tinybert/glue/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --warmup_steps $warmup_steps \
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
