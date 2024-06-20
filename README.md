# LLM-Pretrain

Scripts for LLM pretraining with LoRA and deepspeed.

**Supports LoRA & DeepSpeed**

This repository is built upon [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Supported LLMs

- [LLaMA 1 & 2](https://huggingface.co/meta-llama)
- [Mistral](https://huggingface.co/mistralai)
- Huggingface Models

## Continual Pretraining

1. **Model Selection**: Provide the model name (from Huggingface) or a local path to the model.

2. **Data Preparation**: Ensure your training data is in plain text format, either as **markdown or txt**. You can add additional text corpora to the `data` folder.

3. **Execution**:

    ```sh
    pip install -r requirements.txt
    cd llm_pretrain_custom
    ./pretrain_mistral_in_k_bits.sh
    ```

    Note that parameter settings may vary between different models.

## Detailed Pretraining Command

Below is an example command to run the pretraining script with specific parameters:

```sh
torchrun --nproc_per_node=2 --master_port=9919 pretrain_in_k_bit.py \
    --model_name_or_path $MODEL_PATH \
    --load_in_4bit True \
    --data_path $DATA \
    --bf16 True \
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

```

## File Structure

```
LLM-Pretrain-SFT/
├── data/                           # Training data
├── evaluation/                     # Evaluation scripts and results
├── utils/                          # Utility scripts and helper functions
├── generate_pretrain_data.py       # Script to generate pretraining data
├── pretrain.py                     # Main pretraining script
├── pretrain_in_k_bit.py            # Pretraining script with k-bit optimization
├── pretrain_llama_in_k_bits.sh     # Shell script for pretraining LLaMA model with k-bit optimization
├── pretrain_mistral.sh             # Shell script for pretraining Mistral model
├── pretrain_mistral_in_k_bits.sh   # Shell script for pretraining Mistral model with k-bit optimization
├── requirements.txt                # List of required Python packages

```
