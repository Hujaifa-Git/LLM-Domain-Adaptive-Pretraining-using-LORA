# pip3 install -r requirements.txt

echo ===== PRETRAINING =====
# Node that MODEL_PATH can be local folder path
MODEL_PATH=meta-llama/Meta-Llama-3-8B
TITLE=meta-llama/Meta-Llama-3-8B
DATA=data

OUTPUT_DIR=result
mkdir $OUTPUT_DIR

echo ===== current OUTPUT_DIR is $OUTPUT_DIR =====
echo ===== MODEL_PATH is $MODEL_PATH =====

torchrun --nproc_per_node=2 --master_port=9919 pretrain_in_k_bit.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA \
    --bf16 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --logging_steps 50 \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False 