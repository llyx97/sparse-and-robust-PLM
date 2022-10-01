export TASK_NAME=MNLI
export ROOT_DIR=$HOME/sparse-and-robust-PLM
export num_epoch=3
export model_dir=$ROOT_DIR/robust_compression/log/full_bert/MNLI/epoch${num_epoch}_lr5e-5_warmup

for seed in 1 2 3 4
do
         python3 $ROOT_DIR/robust_compression/run_glue.py \
          --model_type bert \
          --model_name_or_path $model_dir/$seed \
          --task_name $TASK_NAME \
          --data_dir $ROOT_DIR/tinybert/glue/$TASK_NAME \
          --max_seq_length 128 \
          --per_device_train_batch_size 32 \
          --per_gpu_eval_batch_size 64 \
          --output_dir $model_dir/$seed/pred_probs \
          --logging_dir $model_dir/$seed/pred_probs/logging \
          --root_dir $ROOT_DIR \
          --seed $seed \
          --save_final_model false \
          --save_best_model false \
          --eval_ood false 
done
