cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="/mnt/task_runtime/outs/debug_log_2b.txt"
OUTPUT_DIR=/mnt/task_runtime/outs/

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path /mnt/task_runtime/ckpts/Qwen2-VL-2B-Instruct \
    --dataset_name /mnt/task_runtime/dses/clevr_cogen_a_train \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2 \
	--train_last_n_layers 1 \
    --train_lm_head
	# number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
    # --report_to \
    # --attn_implementation flash_attention_2 \
    # --num_generations 8 \
